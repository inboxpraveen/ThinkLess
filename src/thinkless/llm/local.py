"""Local generation with Hugging Face Transformers."""

from __future__ import annotations

import copy
import re
import threading
import time
from collections.abc import Sequence
from typing import Any

from ..decision import Usage
from ..logs import get_logger
from ..settings import resolve_device
from .base import LLM, Completion, Message

__all__ = ["TransformersLLM"]

logger = get_logger("llm.local")

_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class TransformersLLM(LLM):
    """Runs a chat model in-process with ``transformers``.

    This is the zero-setup option: no server, no API key. For throughput, run
    the same weights behind Ollama, vLLM or llama.cpp and use
    :class:`~thinkless.llm.OpenAICompatibleLLM` instead; a serving engine is
    several times faster than the plain ``generate`` loop used here.

    Args:
        model: Hugging Face model id or local path.
        device: ``auto``, ``cpu``, ``cuda``, ``cuda:N`` or ``mps``.
        dtype: ``auto`` picks bfloat16 on CUDA, float16 on MPS, float32 on CPU.
        enable_thinking: Passed to chat templates that support a thinking
            switch (Qwen3 and later). Off by default: decisions and short
            replies do not benefit from long reasoning traces.

    Requires the ``local-llm`` extra.
    """

    provider = "local"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-1.7B",
        *,
        device: str = "auto",
        dtype: str = "auto",
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self._device_request = device
        self._dtype_request = dtype
        self._enable_thinking = enable_thinking
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self.device: str | None = None

    def warmup(self) -> None:
        """Load the weights and generate one token, so the first request is not slow."""
        self._load()
        self.complete(
            [{"role": "user", "content": "Reply with a short greeting for a customer."}],
            max_tokens=16,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ImportError(
                    'TransformersLLM needs the local-llm extra: pip install "thinkless[local-llm]"'
                ) from exc
            device = resolve_device(self._device_request)
            dtype: Any
            if self._dtype_request != "auto":
                dtype = getattr(torch, self._dtype_request)
            elif device.startswith("cuda"):
                dtype = torch.bfloat16
            elif device == "mps":
                dtype = torch.float16
            else:
                dtype = torch.float32
            started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model: Any = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype)
            model.to(device)
            model.eval()
            config: Any = copy.deepcopy(model.generation_config)
            # Greedy decoding by default; clear sampling fields so transformers
            # does not warn about unused flags on every call.
            config.do_sample = False
            config.temperature = None
            config.top_p = None
            config.top_k = None
            if config.pad_token_id is None:
                config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            self._generation_config = config
            self._tokenizer = tokenizer
            self._model = model
            self.device = device
            logger.info(
                "loaded %s on %s in %.1f s", self.model, device, time.perf_counter() - started
            )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self._load()
        import torch

        chat: list[Message] = []
        if system:
            chat.append({"role": "system", "content": system})
        chat.extend(messages)
        started = time.perf_counter()
        with self._lock:
            inputs = self._tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=self._enable_thinking,
            ).to(self.device)
            config: Any = copy.deepcopy(self._generation_config)
            config.max_new_tokens = max_tokens
            if temperature:
                config.do_sample = True
                config.temperature = temperature
            with torch.inference_mode():
                # use_model_defaults=False stops transformers from swapping our
                # greedy settings for the checkpoint's sampling defaults.
                output = self._model.generate(
                    **inputs, generation_config=config, use_model_defaults=False
                )
            prompt_len = int(inputs["input_ids"].shape[1])
            new_tokens = output[0][prompt_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = _THINK.sub("", text).strip()
        n_out = int(new_tokens.shape[0])
        return Completion(
            text=text,
            model=self.model,
            usage=Usage(input_tokens=prompt_len, output_tokens=n_out),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            stop_reason="max_tokens" if n_out >= max_tokens else "end_turn",
        )

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
