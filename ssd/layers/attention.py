import torch
from torch import nn
import triton
import triton.language as tl
from ssd.utils.context import get_context


def _pad_varlen_q(q: torch.Tensor, cu_seqlens_q: torch.Tensor, max_seqlen_q: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = cu_seqlens_q.numel() - 1
    padded_q = q.new_zeros((batch_size, max_seqlen_q, q.shape[1], q.shape[2]))
    lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    token_indices = torch.arange(q.shape[0], device=q.device, dtype=cu_seqlens_q.dtype)
    batch_indices = torch.bucketize(token_indices, cu_seqlens_q[1:-1], right=False)
    token_offsets = token_indices - cu_seqlens_q[batch_indices]
    padded_q[batch_indices, token_offsets] = q
    return padded_q, lengths


def _unpad_varlen_o(padded_o: torch.Tensor, lengths: torch.Tensor, num_tokens: int) -> torch.Tensor:
    token_indices = torch.arange(num_tokens, device=padded_o.device, dtype=lengths.dtype)
    cu_seqlens_q = torch.cat([lengths.new_zeros(1), lengths.cumsum(dim=0)])
    batch_indices = torch.bucketize(token_indices, cu_seqlens_q[1:-1], right=False)
    token_offsets = token_indices - cu_seqlens_q[batch_indices]
    return padded_o[batch_indices, token_offsets]


def _gather_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = k_cache.shape[1]
    max_context_len = int(context_lens.max().item())
    token_positions = torch.arange(max_context_len, device=context_lens.device, dtype=context_lens.dtype)
    valid_mask = token_positions.unsqueeze(0) < context_lens.unsqueeze(1)
    block_indices = torch.div(token_positions, block_size, rounding_mode="floor")
    pos_in_block = token_positions % block_size
    page_ids = block_tables[:, block_indices]
    safe_page_ids = torch.where(valid_mask, page_ids, torch.zeros_like(page_ids))
    dense_k = k_cache[safe_page_ids, pos_in_block.view(1, -1)]
    dense_v = v_cache[safe_page_ids, pos_in_block.view(1, -1)]
    dense_k = dense_k.masked_fill(~valid_mask[:, :, None, None], 0)
    dense_v = dense_v.masked_fill(~valid_mask[:, :, None, None], 0)
    return dense_k.contiguous(), dense_v.contiguous()


def _flatten_dense_kv(
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    max_context_len = dense_k.shape[1]
    token_positions = torch.arange(max_context_len, device=lengths.device, dtype=lengths.dtype)
    valid_mask = token_positions.unsqueeze(0) < lengths.unsqueeze(1)
    flat_k = dense_k[valid_mask].contiguous()
    flat_v = dense_v[valid_mask].contiguous()
    cu_seqlens_k = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    cu_seqlens_k[1:] = lengths.to(torch.int32).cumsum(dim=0)
    return flat_k, flat_v, cu_seqlens_k, int(lengths.max().item())


def _sage_single_query_decode(
    q: torch.Tensor,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    from sageattention import sageattn_varlen

    cu_seqlens_q = torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device)
    flat_k, flat_v, cu_seqlens_k, max_seqlen_k = _flatten_dense_kv(dense_k, dense_v, context_lens)
    return sageattn_varlen(
        q,
        flat_k,
        flat_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        is_causal=False,
        sm_scale=softmax_scale,
    )


def _sage_decode_from_paged_kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    context,
    softmax_scale: float,
) -> torch.Tensor:
    assert context.block_tables is not None and context.context_lens is not None
    dense_k, dense_v = _gather_paged_kv_cache(k_cache, v_cache, context.block_tables, context.context_lens)
    return _sage_single_query_decode(q, dense_k, dense_v, context.context_lens, softmax_scale)


def _sage_verify_from_paged_kv(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    context,
    softmax_scale: float,
) -> torch.Tensor:
    assert context.block_tables is not None and context.context_lens is not None and context.cu_seqlens_q is not None
    padded_q, q_lengths = _pad_varlen_q(q, context.cu_seqlens_q, context.max_seqlen_q)
    dense_k, dense_v = _gather_paged_kv_cache(k_cache, v_cache, context.block_tables, context.context_lens)
    padded_o = padded_q.new_zeros(padded_q.shape)

    max_q_len = padded_q.shape[1]
    for token_idx in range(max_q_len):
        active = q_lengths > token_idx
        if not torch.any(active):
            continue
        step_q = padded_q[active, token_idx].contiguous()
        step_context_lens = context.context_lens[active] - q_lengths[active] + (token_idx + 1)
        step_dense_k = dense_k[active]
        step_dense_v = dense_v[active]
        step_o = _sage_single_query_decode(step_q, step_dense_k, step_dense_v, step_context_lens, softmax_scale)
        padded_o[active, token_idx] = step_o

    return _unpad_varlen_o(padded_o, q_lengths, q.shape[0])


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot.to(tl.int64) * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
        attn_backend: str = "sage",
        sparge_topk: float = 0.5,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None
        self.attn_backend = attn_backend
        self.sparge_topk = sparge_topk

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            k, v = k.view(-1, self.num_kv_heads, self.head_dim), v.view(-1, self.num_kv_heads, self.head_dim)

            if self.attn_backend in ("sparge", "sparge_sage"):
                from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
                if (
                    context.cu_seqlens_q is not None
                    and context.cu_seqlens_k is not None
                    and context.cu_seqlens_q.numel() == 2
                    and context.cu_seqlens_k.numel() == 2
                    and context.block_tables is None
                ):
                    q_s = q.unsqueeze(0).contiguous()
                    k_s = k.unsqueeze(0).contiguous()
                    v_s = v.unsqueeze(0).contiguous()
                    if self.num_heads != self.num_kv_heads:
                        kv_repeat = self.num_heads // self.num_kv_heads
                        k_s = k_s.repeat_interleave(kv_repeat, dim=2)
                        v_s = v_s.repeat_interleave(kv_repeat, dim=2)
                    o = spas_sage2_attn_meansim_topk_cuda(
                        q_s,
                        k_s,
                        v_s,
                        is_causal=True,
                        tensor_layout="NHD",
                        topk=self.sparge_topk,
                        pvthreshd=50.0,
                    ).squeeze(0)
                else:
                    from sageattention import sageattn_varlen

                    o = sageattn_varlen(
                        q,
                        k,
                        v,
                        cu_seqlens_q=context.cu_seqlens_q,
                        cu_seqlens_k=context.cu_seqlens_k,
                        max_seqlen_q=context.max_seqlen_q,
                        max_seqlen_k=context.max_seqlen_k,
                        is_causal=True,
                        tensor_layout="NHD",
                        sm_scale=self.scale,
                    )
            else:
                from sageattention import sageattn_varlen

                o = sageattn_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    max_seqlen_q=context.max_seqlen_q,
                    max_seqlen_k=context.max_seqlen_k,
                    is_causal=True,
                    tensor_layout="NHD",
                    sm_scale=self.scale,
                )
        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                o = _sage_verify_from_paged_kv(
                    q,
                    k_cache,
                    v_cache,
                    context=context,
                    softmax_scale=self.scale,
                )

            elif tree_decode:
                if self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                else:
                    mq_len = self.F * (self.K+1)
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(self.prefill_wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            else: # single query decode
                o = _sage_decode_from_paged_kv(
                    q,
                    k_cache,
                    v_cache,
                    context=context,
                    softmax_scale=self.scale,
                )

        o = o.reshape(-1, self.num_heads * self.head_dim)
        return o
