import torch
import torch.distributed as dist
import os
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.helpers.runner_helpers import prepare_prefill_payload
from ssd.engine.sequence import Sequence
from ssd.utils.misc import decode_tokens
from ssd.utils.async_helpers.nccl_pack import send_int64


class SpeculatorAsync(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        async_fan_out: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        kvcache_block_size: int,
        max_model_len: int,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        tokenizer: AutoTokenizer,
        verbose: bool,
        is_vlm: bool = False,
    ):
        super().__init__(lookahead, device)
        self.async_fan_out = async_fan_out
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.kvcache_block_size = kvcache_block_size
        self.max_model_len = max_model_len
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.K = lookahead
        self.is_vlm = is_vlm
        self._trace = os.environ.get("SSD_ASYNC_TRACE", "0") == "1"

        # Pre-allocate handshake send/recv buffers (reused every step)
        self._alloc_handshake_bufs(1)

        # Pre-allocate speculate() output buffers (avoid torch.tensor(device=cuda) sync)
        self._recovery_buf = torch.empty(1, dtype=torch.int64, device=device)
        self._speculations_buf = torch.empty(1, lookahead + 1, dtype=torch.int64, device=device)

    def _alloc_handshake_bufs(self, B):
        self._hs_B = B
        d = self.device
        self._cmd = torch.zeros(1, dtype=torch.int64, device=d)
        self._meta = torch.tensor([B, self.K, self.async_fan_out], dtype=torch.int64, device=d)
        self._cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self._num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self._temps_buf = torch.empty(B, dtype=torch.float32, device=d)
        self._block_tables_buf = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=d)
        self._fused_response = torch.empty(B + B * self.K, dtype=torch.int64, device=d)
        self._logits_q = torch.empty(B, self.K, self.vocab_size, dtype=self.draft_dtype, device=d)
        self._extend_counts = torch.zeros(B, dtype=torch.int64, device=d)

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        eagle_acts = verify_result.eagle_acts
        input_id_list = [seq.token_ids for seq in seqs]

        # EAGLE token-conditioning shift: token at position j gets conditioning
        # from target act at position j-1. Skip first token per seq and drop
        # last eagle_act per seq so they align correctly.
        if eagle_acts is not None:
            sliced = []
            offset = 0
            for ids in input_id_list:
                seq_len = len(ids)
                sliced.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = torch.cat(sliced, dim=0)
            input_id_list = [ids[1:] for ids in input_id_list]

        # Collect VLM inputs from sequences before they may be cleared by the target runner.
        # For non-EAGLE async, speculator.prefill() runs before verifier.prefill(), so
        # pixel_values etc. are still intact on the sequences at this point.
        # For EAGLE async, the sync fix in model_runner.py ensures the target does not
        # clear VLM inputs when speculate=True, so they are still present here too.
        vlm_inputs = None
        if self.is_vlm:
            all_pv = [seq.pixel_values for seq in seqs if seq.pixel_values is not None]
            all_thw = [seq.image_grid_thw for seq in seqs if seq.image_grid_thw is not None]
            all_mask = [seq.image_mask for seq in seqs if seq.image_mask is not None]
            if all_pv:
                vlm_inputs = {
                    "pixel_values": torch.cat(all_pv, dim=0),
                    "image_grid_thw": torch.cat(all_thw, dim=0) if all_thw else None,
                    "image_mask": torch.cat(all_mask, dim=0) if all_mask else None,
                }

        # Use the actual per-sequence draft block-table width for this batch.
        # Using a fixed width derived from max_model_len can desync the wire format
        # if seq.draft_block_table is wider.
        max_blocks = max(len(seq.draft_block_table) for seq in seqs)
        cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts, pixel_values, image_grid_thw, image_mask = prepare_prefill_payload(
            input_id_list, eagle_acts, self.device, max_blocks,
            [seq.draft_block_table for seq in seqs],
            vlm_inputs=vlm_inputs,
        )
        if self._trace:
            print(
                f"[async_prefill][rank0] send metadata={metadata.tolist()} input_ids={tuple(input_ids.shape)} "
                f"num_tokens={tuple(num_tokens.shape)} dbt={tuple(draft_block_table.shape)}",
                flush=True,
            )
        dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(metadata, dst=self.draft_runner_rank, group=self.async_pg)
        send_int64(self.async_pg, self.draft_runner_rank,
                   input_ids, num_tokens, draft_block_table.to(torch.int64))
        if self._trace:
            print("[async_prefill][rank0] sent fused int64 payload", flush=True)
        if eagle_acts is not None:
            dist.send(eagle_acts, dst=self.draft_runner_rank, group=self.async_pg)
            if self._trace:
                print(f"[async_prefill][rank0] sent eagle_acts={tuple(eagle_acts.shape)}", flush=True)
        if pixel_values is not None:
            pv_shape = torch.tensor(pixel_values.shape, dtype=torch.int64, device=self.device)
            dist.send(pv_shape, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(pixel_values.to(self.draft_dtype), dst=self.draft_runner_rank, group=self.async_pg)
            if self._trace:
                print(f"[async_prefill][rank0] sent pixel_values={tuple(pixel_values.shape)}", flush=True)
        if image_grid_thw is not None:
            dist.send(image_grid_thw.to(torch.int64), dst=self.draft_runner_rank, group=self.async_pg)
            if self._trace:
                print(f"[async_prefill][rank0] sent image_grid_thw={tuple(image_grid_thw.shape)}", flush=True)
        if image_mask is not None:
            dist.send(image_mask.to(torch.int8), dst=self.draft_runner_rank, group=self.async_pg)
            if self._trace:
                print(f"[async_prefill][rank0] sent image_mask={tuple(image_mask.shape)}", flush=True)
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.append_token(seq.recovery_token_id)

        if self.verbose:
            sep = '=' * 80
            print(f"\n{sep}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                trunk = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk: ...{decode_tokens(trunk, self.tokenizer)}", flush=True)
                print(f"    Recovery: {seq.recovery_token_id} ({decode_tokens([seq.recovery_token_id], self.tokenizer)})", flush=True)
            print(f"{sep}\n", flush=True)

        eagle = verify_result.eagle_acts is not None
        speculations_tokens, logits_q, cache_hits = self._speculation_request(seqs, eagle)

        # Build speculations using pre-allocated buffers (avoids torch.tensor(device=cuda) sync)
        B = len(seqs)
        if B != self._recovery_buf.shape[0]:
            self._recovery_buf = torch.empty(B, dtype=torch.int64, device=self.device)
            self._speculations_buf = torch.empty(B, self.K + 1, dtype=torch.int64, device=self.device)
        _rec_cpu = torch.tensor([seq.recovery_token_id for seq in seqs], dtype=torch.int64)
        self._recovery_buf.copy_(_rec_cpu, non_blocking=True)
        self._speculations_buf[:, 0] = self._recovery_buf
        self._speculations_buf[:, 1:] = speculations_tokens
        speculations = self._speculations_buf

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(speculations_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += len(speculations_tokens[i]) + 1

        return SpeculateResult(speculations, logits_q, cache_hits)

    def _speculation_request(self, seqs: list[Sequence], eagle: bool):
        B = len(seqs)
        if B != self._hs_B:
            self._alloc_handshake_bufs(B)

        # Fill send buffers in-place (avoids torch.tensor from Python lists)
        for i, seq in enumerate(seqs):
            self._cache_keys[i, 0] = seq.seq_id
            self._cache_keys[i, 1] = seq.last_spec_step_accepted_len - 1
            self._cache_keys[i, 2] = seq.recovery_token_id
            self._num_tokens_buf[i] = seq.num_tokens
            self._temps_buf[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            bt = seq.draft_block_table
            bt_len = len(bt)
            if bt_len > 0:
                self._block_tables_buf[i, :bt_len] = torch.tensor(bt, dtype=torch.int32, device=self.device)
            self._block_tables_buf[i, bt_len:] = -1

        # Send cmd + meta + fused payload (temps fused into int64 burst)
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        temps_as_int64 = self._temps_buf.view(torch.int32).to(torch.int64)
        send_int64(
            self.async_pg, self.draft_runner_rank,
            self._cache_keys, self._num_tokens_buf,
            self._block_tables_buf.to(torch.int64), temps_as_int64,
        )

        if eagle:
            recovery_activations = torch.stack(
                [seq.last_target_hidden_state for seq in seqs], dim=0,
            ).to(self.device)
            dist.send(recovery_activations.to(self.draft_dtype),
                      dst=self.draft_runner_rank, group=self.async_pg)

            # Send extend data for glue decode with fused extend
            K = self.K
            act_dim = recovery_activations.shape[-1]
            for i, seq in enumerate(seqs):
                self._extend_counts[i] = seq.extend_count
            extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.draft_dtype, device=self.device)
            extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
            for i, seq in enumerate(seqs):
                n = seq.extend_count
                if n > 0 and seq.extend_eagle_acts is not None:
                    extend_eagle_acts[i, :n] = seq.extend_eagle_acts[:n].to(self.draft_dtype)
                    extend_token_ids[i, :n] = seq.extend_token_ids[:n]
            dist.send(self._extend_counts, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(extend_eagle_acts, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(extend_token_ids, dst=self.draft_runner_rank, group=self.async_pg)

        # Recv into pre-allocated buffers
        dist.recv(self._fused_response, src=self.draft_runner_rank, group=self.async_pg)
        cache_hits = self._fused_response[:B]
        speculations = self._fused_response[B:].view(B, self.K)
        dist.recv(self._logits_q, src=self.draft_runner_rank, group=self.async_pg)

        return speculations, self._logits_q, cache_hits
