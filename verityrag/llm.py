"""Pluggable LLM backends for answer synthesis.

LocalBackend (default): a small instruction-tuned model run on CPU with
HuggingFace transformers. Greedy decoding, short outputs — tuned for the
~10-minute demo budget, not for state-of-the-art quality.

OpenAICompatibleBackend: documented but NOT used in the demo; points at any
OpenAI-compatible chat-completions endpoint (e.g. Azure OpenAI) via env vars.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod

LOCAL_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """Return the model's continuation of ``prompt``."""


class LocalBackend(LLMBackend):
    def __init__(self, model_name: str = LOCAL_MODEL, num_threads: int | None = None):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        if num_threads:
            import torch

            torch.set_num_threads(num_threads)

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
        return self._tokenizer, self._model

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        import torch

        tokenizer, model = self._load()
        device = next(model.parameters()).device
        # Instruction-tuned models need their chat template to behave.
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAICompatibleBackend(LLMBackend):
    """Chat-completions backend for an OpenAI-compatible endpoint.

    NOT used in the demo. Configure with:
        VERITYRAG_OPENAI_BASE_URL  e.g. https://your-resource.openai.azure.com
        VERITYRAG_OPENAI_API_KEY   your key
        VERITYRAG_OPENAI_MODEL     deployment / model name
    """

    def __init__(self):
        self.base_url = os.environ.get("VERITYRAG_OPENAI_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("VERITYRAG_OPENAI_API_KEY", "")
        self.model = os.environ.get("VERITYRAG_OPENAI_MODEL", "")
        if not (self.base_url and self.api_key and self.model):
            raise RuntimeError(
                "OpenAICompatibleBackend needs VERITYRAG_OPENAI_BASE_URL, "
                "VERITYRAG_OPENAI_API_KEY and VERITYRAG_OPENAI_MODEL set."
            )

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()
