from ssd.engine.llm_engine import LLMEngine


class CustomSDVLM(LLMEngine):
    def __init__(
        self,
        target_model: str,
        draft_model: str,
        *,
        target_lora_path: str | None = None,
        draft_lora_path: str | None = None,
        spec_k: int = 4,
        max_model_len: int = 16384,
        enforce_eager: bool = False,
        verbose: bool = False,
        max_steps: int | None = None,
    ) -> None:
        super().__init__(
            target_model,
            draft=draft_model,
            lora_path=target_lora_path,
            draft_lora_path=draft_lora_path,
            is_vlm=True,
            speculate=True,
            draft_async=False,
            num_gpus=1,
            speculate_k=spec_k,
            max_model_len=max_model_len,
            use_prefix_caching=False,
            enforce_eager=True,
            verbose=verbose,
            max_steps=max_steps,
        )