"""
DIVE-KT SPM Pipeline — Utilities
"""
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

from config import logger


# ═══════════════════════════════════════════════════════════════
# ID 归一化
# ═══════════════════════════════════════════════════════════════

def norm_course_id(course_id: Any) -> Optional[int]:
    if course_id is None:
        return None
    s = str(course_id).removeprefix("C_")
    try:
        return int(s)
    except Exception:
        return None


def norm_video_id(video_id: Any) -> Optional[int]:
    if video_id is None:
        return None
    if isinstance(video_id, int):
        return video_id
    s = str(video_id).strip().removeprefix("V_")
    m = re.findall(r"\d+", s)
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


def safe_filename(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", str(s)).strip("_") or "item"


# ═══════════════════════════════════════════════════════════════
# JSONL 读写
# ═══════════════════════════════════════════════════════════════

def read_jsonl(filepath: str) -> List[Dict]:
    data: List[Dict] = []
    if not os.path.exists(filepath):
        logger.error(f"文件不存在: {filepath}")
        return data
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    logger.info(f"读取 {filepath} -> {len(data)} 行")
    return data


def write_jsonl(filepath: str, data: List[Dict]):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"写入 {filepath} ({len(data)} 行)")


def append_jsonl(filepath: str, data: List[Dict]):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


# ═══════════════════════════════════════════════════════════════
# LLM 输出 → JSON 提取（鲁棒解析）
# ═══════════════════════════════════════════════════════════════

def extract_json_from_text(text: str) -> Optional[Dict]:
    """从 LLM 输出中尽力提取一个 dict。"""
    if not text:
        return None

    s = text.strip()

    def _strip_fences(x: str) -> str:
        x = x.strip()
        x = re.sub(r"^```(?:json)?\s*", "", x)
        x = re.sub(r"\s*```\s*$", "", x)
        return x.strip()

    def _remove_trailing_commas(x: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", x)

    def _fix_illegal_escapes(x: str) -> str:
        y = re.sub(r"\\([^\"\\/bfnrtu])", r"\\\\\1", x)
        y = re.sub(r"\\([fb])(?=[a-zA-Z])", r"\\\\\1", y)
        return y

    def _quote_bare_keys(x: str) -> str:
        if not re.search(r"\{\s*[A-Za-z_]\w*\s*:", x):
            return x
        return re.sub(r"([\{,]\s*)([A-Za-z_]\w*)(\s*:)", r'\1"\2"\3', x)

    def _try_parse(x: str) -> Optional[Dict]:
        x = _strip_fences(x.strip())
        if not x:
            return None
        dec = json.JSONDecoder()
        try:
            obj, _ = dec.raw_decode(x)
            if isinstance(obj, dict):
                return obj
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                return obj[0]
        except Exception:
            pass
        return None

    def _extract_braced(x: str) -> List[str]:
        out = []
        for st in (m.start() for m in re.finditer(r"\{", x)):
            depth, in_str, esc = 0, False, False
            for i in range(st, len(x)):
                ch = x[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(x[st: i + 1])
                        break
        return out

    # 0) 解除整体字符串包裹
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        try:
            inner = json.loads(s)
            if isinstance(inner, str) and "{" in inner:
                s = inner.strip()
        except Exception:
            pass

    # 1) 直接解析
    d = _try_parse(s)
    if d is not None:
        return d

    # 2) 提取候选块 + 修复尝试
    blobs = _extract_braced(s)
    if not blobs:
        first, last = s.find("{"), s.rfind("}")
        if first != -1 and last > first:
            blobs = [s[first: last + 1]]

    for blob in blobs:
        for repair in (
            lambda x: x,
            _remove_trailing_commas,
            _fix_illegal_escapes,
            lambda x: _remove_trailing_commas(_fix_illegal_escapes(x)),
            _quote_bare_keys,
            lambda x: _remove_trailing_commas(_quote_bare_keys(x)),
        ):
            cand = _strip_fences(repair(blob))
            d = _try_parse(cand)
            if d is not None:
                return d
            try:
                obj = ast.literal_eval(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # logger.warning(f"JSON 解析失败 | 片段: {_strip_fences(s)[:300]}...")     
    logger.warning(f"JSON 解析失败 | 片段: {_strip_fences(s)}...")    
    return None
