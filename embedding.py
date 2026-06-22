"""
DIVE-KT — LLM Embedding Encoder
"""
import gc
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class LLMEmbeddingEncoder:
    """
    Load a frozen LLM, batch-encode texts into L2-normalised embedding vectors,
    then release all GPU memory.

    Usage::

        encoder = LLMEmbeddingEncoder(config)
        embeddings = encoder.encode(["text 1", "text 2"])   # (N, D) on CPU
        encoder.release_memory()
    """

    def __init__(self, config):
        self.device = config.DEVICE
        model_path = config.LLM_MODEL_NAME

        print(f"[LLMEmbeddingEncoder] Loading: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(self.device).eval()

        # Use reduced precision when possible to save VRAM.
        if self.device == "cuda":
            if torch.cuda.is_bf16_supported():
                self.model.bfloat16()
            else:
                self.model.half()

    def encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """
        Encode a batch of texts into L2-normalised embedding vectors.

        Args:
            texts:      List of input strings.
            max_length: Tokeniser truncation length.

        Returns:
            Tensor of shape ``(len(texts), embed_dim)`` on CPU.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state          # (B, L, D)
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1    # last valid token idx
            batch_idx = torch.arange(hidden.size(0), device=self.device)
            last_token = hidden[batch_idx, seq_lengths]               # (B, D)
            embeddings = F.normalize(last_token, p=2, dim=1)

        return embeddings.cpu()

    def release_memory(self):
        """Explicitly free model weights and flush CUDA cache."""
        print("[LLMEmbeddingEncoder] Releasing memory.")
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
