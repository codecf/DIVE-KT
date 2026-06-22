"""
DIVE-KT SPM Pipeline — vLLM Text Generator
"""
import json
import os
import sys
from typing import Any, Dict, List

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("错误: pip install vllm")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("错误: pip install transformers")
    sys.exit(1)

from config import LLMConfig, logger


class VLLMTextGenerator:

    def __init__(self, config: LLMConfig):
        self.config = config
        trust = os.environ.get("VLLM_TRUST_REMOTE_CODE", "1").strip() not in ("0", "false", "False")

        logger.info("Loading tokenizer: %s", config.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=trust)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        tp = int(os.environ.get("VLLM_TP", "1"))
        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.7"))
        max_len_env = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
        max_len = None if max_len_env <= 0 else max_len_env

        logger.info("Loading vLLM: tp=%d gpu_mem=%.2f max_len=%s", tp, gpu_mem, max_len)
        self.llm = LLM(
            model=config.model_name_or_path,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=trust,
            max_model_len=max_len,
            dtype="auto",
        )
        logger.info("vLLM engine ready.")

    def _chat_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def _strip_think(text: str) -> str:
        tag = "</think>"
        if tag in text:
            return text[text.rfind(tag) + len(tag):].strip()
        return text.strip()

    def generate_batch(self, prompts: List[str]) -> List[Dict[str, str]]:
        """批量生成，返回 [{prompt, chat_prompt, raw_text, text}, ...]"""
        chat_prompts = [self._chat_prompt(p) for p in prompts]
        sp = SamplingParams(max_tokens=self.config.max_new_tokens)
        outputs = self.llm.generate(chat_prompts, sp)

        results: List[Dict[str, str]] = []
        for p, cp, out in zip(prompts, chat_prompts, outputs):
            raw = out.outputs[0].text.strip()
            results.append({
                "prompt": p,
                "chat_prompt": cp,
                "raw_text": raw,
                "text": self._strip_think(raw),
            })
        return results

    def generate(self, prompt: str) -> Dict[str, str]:
        return self.generate_batch([prompt])[0]
