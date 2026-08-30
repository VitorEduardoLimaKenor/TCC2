"""Backends de geração.

Três implementações atrás da mesma interface:

* ``dummy`` — não carrega modelo. Serve para testar o pipeline inteiro sem GPU
  e para os testes automatizados.
* ``hf`` — transformers. Simples, funciona na GTX 1060 com um modelo pequeno.
  É o caminho de desenvolvimento.
* ``vllm`` — o caminho do experimento confirmatório. O ganho não é o modelo,
  é o *batching* contínuo: as ~3.000 gerações do teste saem em horas, não em
  um dia. E o cache de prefixo aproveita o fato de que todos os prompts de um
  mesmo banco compartilham a instrução e o DDL inteiro.

Decodificação gulosa, amostragem desabilitada e semente fixa em todos eles
(Seção 4.4). Ainda assim a igualdade bit a bit não é garantida — kernels de
atenção e composição de lote introduzem variação —, e é por isso que a Seção
4.6.3 prevê a verificação de determinismo sobre uma amostra.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

from ..config import GenerationConfig
from ..schemas import Generation


class Backend(Protocol):
    def generate(self, prompts: list[str]) -> list[Generation]: ...
    def count_tokens(self, text: str) -> int: ...
    @property
    def descriptor(self) -> dict: ...


class DummyBackend:
    """Devolve uma consulta determinística derivada do hash do prompt.

    Não serve para medir acurácia; serve para exercitar o caminho completo —
    seleção, montagem, execução no sandbox, log — sem GPU.
    """

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    def generate(self, prompts: list[str]) -> list[Generation]:
        out = []
        for prompt in prompts:
            digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
            text = f"```sql\nSELECT 1 AS dummy_{digest};\n```"
            out.append(
                Generation(
                    text=text,
                    prompt_tokens=self.count_tokens(prompt),
                    completion_tokens=self.count_tokens(text),
                    latency_ms=0.0,
                )
            )
        return out

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @property
    def descriptor(self) -> dict:
        return {"backend": "dummy", "model": "none"}


class HFBackend:
    def __init__(self, cfg: GenerationConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        torch.manual_seed(cfg.seed)

        kwargs: dict = {"torch_dtype": getattr(torch, cfg.dtype, torch.float16)}
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        if cfg.quantization == "bitsandbytes-4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, revision=cfg.revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **kwargs)
        self.model.eval()

    def _wrap(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompts: list[str]) -> list[Generation]:
        results: list[Generation] = []
        batch = self.cfg.batch_size
        for start in range(0, len(prompts), batch):
            chunk = [self._wrap(p) for p in prompts[start : start + batch]]
            encoded = self.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.max_context_tokens - self.cfg.max_new_tokens,
            ).to(self.model.device)

            began = time.perf_counter()
            with self.torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            elapsed = (time.perf_counter() - began) * 1000 / max(1, len(chunk))

            for i in range(len(chunk)):
                prompt_len = int(encoded["input_ids"][i].shape[0])
                completion = output[i][prompt_len:]
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
                results.append(
                    Generation(
                        text=text,
                        prompt_tokens=prompt_len,
                        completion_tokens=int(completion.shape[0]),
                        latency_ms=elapsed,
                    )
                )
        return results

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    @property
    def descriptor(self) -> dict:
        return {
            "backend": "hf",
            "model": self.cfg.model,
            "revision": self.cfg.revision,
            "quantization": self.cfg.quantization,
            "dtype": self.cfg.dtype,
        }


class VLLMBackend:
    def __init__(self, cfg: GenerationConfig):
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        kwargs: dict = {
            "model": cfg.model,
            "dtype": cfg.dtype,
            "max_model_len": cfg.max_context_tokens,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "enable_prefix_caching": cfg.enable_prefix_caching,
            "seed": cfg.seed,
        }
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        if cfg.quantization:
            kwargs["quantization"] = cfg.quantization

        self.llm = LLM(**kwargs)
        self.params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=cfg.max_new_tokens, seed=cfg.seed
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _wrap(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(self, prompts: list[str]) -> list[Generation]:
        wrapped = [self._wrap(p) for p in prompts]
        began = time.perf_counter()
        outputs = self.llm.generate(wrapped, self.params)
        elapsed = (time.perf_counter() - began) * 1000 / max(1, len(prompts))

        results: list[Generation] = []
        for output in outputs:
            completion = output.outputs[0]
            results.append(
                Generation(
                    text=completion.text,
                    prompt_tokens=len(output.prompt_token_ids),
                    completion_tokens=len(completion.token_ids),
                    latency_ms=elapsed,
                )
            )
        return results

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    @property
    def descriptor(self) -> dict:
        return {
            "backend": "vllm",
            "model": self.cfg.model,
            "revision": self.cfg.revision,
            "quantization": self.cfg.quantization,
            "dtype": self.cfg.dtype,
            "prefix_caching": self.cfg.enable_prefix_caching,
        }


def make_backend(cfg: GenerationConfig) -> Backend:
    if cfg.backend == "dummy":
        return DummyBackend(cfg)
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "vllm":
        return VLLMBackend(cfg)
    raise ValueError(f"backend desconhecido: {cfg.backend}")
