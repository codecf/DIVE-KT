"""DIVE-KT — Global Configuration."""
import os
import sys
from dataclasses import dataclass, field
from enum import Enum

import torch
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Progress bar helper
# ──────────────────────────────────────────────────────────────────────────────

def build_progress_bar(*args, **kwargs):
    """Return a tqdm bar that auto-disables when not writing to a TTY."""
    return tqdm(*args, disable=not sys.stderr.isatty(), file=sys.stderr, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Global config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    DATA_DIR:       str = ""
    CKPT_DIR:       str = ""
    LLM_MODEL_NAME: str = (
        ""
        "/Qwen/Qwen3-Embedding-0___6B"
    )

    # ── Model dimensions ───────────────────────────────────────────────────────
    LLM_EMBED_DIM:    int   = 1024   # output dimension of the frozen LLM encoder
    HIDDEN_DIM:       int   = 256    # internal hidden size (d)
    BEHAVIOR_FEAT_DIM: int  = 6      # length of the behaviour feature vector

    # ── Regularisation ─────────────────────────────────────────────────────────
    DROPOUT:   float = 0.4   # dropout applied to Stage-1 / embedding modules
    DROPOUT_2: float = 0.1   # dropout applied to Stage-2 (VMR/VEC) modules

    # ── I²FRU: incremental influence factors ───────────────────────────────────
    I2FRU_ETA_P_POS: float = 0.15   # η for correct responses
    I2FRU_ETA_P_NEG: float = 0.03   # η for incorrect responses

    # ── Video Correlation Retrieval (VMR) ──────────────────────────────────────
    VIDEO_CORR_MAX_BUF: int   = 8   # max video memories kept per student (K)
    VIDEO_CORR_TEMP:    float = 0.25   # softmax temperature τ_v

    # ── Evidence delta-time encoding ───────────────────────────────────────────
    EVID_DT_BINS:      int   = 64    # number of Δt buckets
    EVID_DT_LOG_SCALE: float = 6.0   # log-scale multiplier for bucketing

    # ── Data loading ───────────────────────────────────────────────────────────
    BATCH_SIZE:         int  = 128
    EMBED_BATCH_SIZE:   int  = 32
    PROBLEM_SEQ_LEN:    int  = 200   # problems per sequence chunk
    MAX_SEQ_LEN:        int  = 400   # hard upper bound on total events per chunk
    MAX_VIDEOS_PER_SEQ: int  = 200   # max video events kept after stratified sampling
    USE_VIDEO:          bool = True

    # ── Optimiser ──────────────────────────────────────────────────────────────
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY:  float = 1e-5
    EPOCHS:        int   = 150       # default when not overridden per-stage

    # ── Auxiliary loss weights (set to 0 to disable) ───────────────────────────
    AUX_LAMBDA = 0.0
    CL_LAMBDA  = 0.0

    # ── Stage-specific defaults ────────────────────────────────────────────────
    STAGE1_TRAIN_EMBED: bool  = True
    STAGE2_TRAIN_EMBED: bool  = True
    STAGE3_TRAIN_EMBED: bool  = True
    STAGE3_EPOCHS:      int   = 10
    STAGE3_PATIENCE:    int   = 5
    STAGE3_LR_SCALE:    float = 0.3   # stage3_lr = stage2_lr * STAGE3_LR_SCALE

    # ── Reproducibility ────────────────────────────────────────────────────────
    SEED: int = 42

    # ── Device (auto-detected) ─────────────────────────────────────────────────
    DEVICE: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    ))

    def __post_init__(self):
        os.makedirs(self.CKPT_DIR, exist_ok=True)
        print(
            f"[Config] device={self.DEVICE}  hidden_dim={self.HIDDEN_DIM}  "
            f"llm_dim={self.LLM_EMBED_DIM}  video_buf={self.VIDEO_CORR_MAX_BUF}"
        )


cfg = Config()
