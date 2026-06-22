"""
DIVE-KT SPM Pipeline — Configuration (Final)
"""
import logging
import os
from dataclasses import dataclass

ROOT_DIR = "/DIVE-KT/spm"


@dataclass
class LLMConfig:
    model_name_or_path: str = "/DeepSeek-14B/"
    max_new_tokens: int = 4096

@dataclass
class SPMConfig:
    """SPM 模块配置"""
    max_linked_problems: int = 6


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("SPM")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "spm.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = setup_logger()

TRACE_DIR = os.path.join(ROOT_DIR, "trace")
os.makedirs(TRACE_DIR, exist_ok=True)
# TRACE_DIR = None
