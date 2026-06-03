import torch
from torch import nn
import triton
import triton.language as tl
import inspect

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from ssd.utils.context import get_context


_FLASH_ATTN_WITH_KVCACHE_PARAMS = set(inspect.signature(flash_attn_with_kvcache).parameters)


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


def _flash_attn_with_kvcache_compat(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    context,
    softmax_scale: float,
    causal: bool,
) -> torch.Tensor:
    kwargs = {
        "cache_seqlens": context.context_lens,
        "softmax_scale": softmax_scale,
        "causal": causal,
    }

    if "page_table" in _FLASH_ATTN_WITH_KVCACHE_PARAMS:
        kwargs["page_table"] = context.block_tables
        if context.cu_seqlens_q is not None:
            kwargs["cu_seqlens_q"] = context.cu_seqlens_q
            kwargs["max_seqlen_q"] = context.max_seqlen_q
        return flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)

    if "block_table" in _FLASH_ATTN_WITH_KVCACHE_PARAMS:
        kwargs["block_table"] = context.block_tables
        if context.cu_seqlens_q is not None and q.dim() == 3:
            padded_q, lengths = _pad_varlen_q(q, context.cu_seqlens_q, context.max_seqlen_q)
            padded_o = flash_attn_with_kvcache(padded_q, k_cache, v_cache, **kwargs)
            return _unpad_varlen_o(padded_o, lengths, q.shape[0])
        return flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)

    return flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)


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
        attn_backend: str = "flash",
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

            if self.attn_backend == "sparge":
                from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
                # SpargeAttn expects (batch, seq_len, num_heads, head_dim) with NHD tensor_layout,
                # but its internal layout is HND (heads, seq, dim).
                # Reshape: (total_tokens, heads, dim) -> (1, total_tokens, heads, dim) for batch=1,
                # then permute to HND: (heads, total_tokens, dim)
                q_s = q.permute(1, 0, 2).contiguous()   # (num_heads, total_tokens, head_dim)
                k_s = k.permute(1, 0, 2).contiguous()   # (num_kv_heads, total_tokens, head_dim)
                v_s = v.permute(1, 0, 2).contiguous()   # (num_kv_heads, total_tokens, head_dim)
                o_s = spas_sage2_attn_meansim_topk_cuda(
                    q_s, k_s, v_s,
                    is_causal=True,
                    tensor_layout="HND",
                    topk=self.sparge_topk,
                )
                # Permute back to (total_tokens, num_heads, head_dim)
                o = o_s.permute(1, 0, 2).contiguous()
            else:
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True)
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
                o = _flash_attn_with_kvcache_compat(
                    q,
                    k_cache,
                    v_cache,
                    context=context,
                    softmax_scale=self.scale,
                    causal=True,
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
                q = q.unsqueeze(1)
                o = _flash_attn_with_kvcache_compat(
                    q,
                    k_cache,
                    v_cache,
                    context=context,
                    softmax_scale=self.scale,
                    causal=True,
                )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
