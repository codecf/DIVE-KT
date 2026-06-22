"""
DIVE-KT SPM Pipeline — Processors (Hardened v2)
"""
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from config import SPMConfig, logger
from llm_generator import VLLMTextGenerator
from utils import (
    append_jsonl,
    extract_json_from_text,
    norm_course_id,
    norm_video_id,
    read_jsonl,
    safe_filename,
)


# ═══════════════════════════════════════════════════════════════
# 字幕预处理工具函数
# ═══════════════════════════════════════════════════════════════

_META_SEGMENT_STARTERS: tuple = (
    "那么", "那", "然后", "所以", "因此", "当然",
    "好 ", "好，", "好。", "嗯", "啊",
    "下面", "接下来", "首先", "其次", "最后", "另外",
    "我们来看", "我们来", "让我们来", "来看看", "来看一下",
    "上一讲", "上次", "上节课", "我们上次", "我们之前",
    "我们说过", "我们已经", "我们知道", "大家", "同学们",
    "现在我们", "那我们", "这里我们",
)

_META_STMT_RE: List[re.Pattern] = [
    re.compile(r"^(本视频|本节|本讲|本章|这个视频|这节课|本课|该视频)", re.IGNORECASE),
    re.compile(r"^(this video|this lecture|this section|this chapter)", re.IGNORECASE),
    re.compile(
        r"(介绍了|讲解了|讲述了|涵盖了|covers|introduces|explains|discusses)\s*"
        r"(以下|这些|如下|the following|some|various)",
        re.IGNORECASE,
    ),
    re.compile(r"^(主要内容|章节概述|课程目标|学习目标|内容概述)", re.IGNORECASE),
]

_VAGUE_CC_RE: List[re.Pattern] = [
    re.compile(
        r"(基础知识|基本概念|基本内容|概述|简介|导论|导言|"
        r"overview|introduction|basics|fundamentals|summary)$",
        re.IGNORECASE,
    ),
]


def _is_meta_segment(text: str, min_info_len: int = 6) -> bool:

    if len(text) >= min_info_len:
        return False
    return any(text.startswith(s) for s in _META_SEGMENT_STARTERS)


def _stratified_sample(segments: List[str], n_strata: int, max_total: int) -> List[str]:

    if len(segments) <= max_total:
        return segments

    per_stratum  = max(1, max_total // n_strata)
    stratum_size = max(1, len(segments) // n_strata)
    if per_stratum == 1:
        front_per = 0
        tail_per  = 1
    else:
        front_per = max(1, per_stratum // 3)
        tail_per  = per_stratum - front_per

    result: List[str] = []
    for i in range(n_strata):
        start   = i * stratum_size
        end     = (start + stratum_size) if i < n_strata - 1 else len(segments)
        stratum = segments[start:end]

        if len(stratum) <= per_stratum:
            # 区间本身不够长，直接全取
            result.extend(stratum)
        else:
            taken: List[str] = []
            if front_per > 0:
                taken.extend(stratum[:front_per])
            if tail_per > 0:
                taken.extend(stratum[-tail_per:])
            result.extend(taken)

    return result[:max_total]


def build_subtitle_text(
    item: Dict,
    max_segments: int = 80,
    n_strata: int = 5,
    min_seg_chars: int = 4,
) -> str:

    starts = item.get("start", [])
    ends   = item.get("end",   [])
    texts  = item.get("text",  [])

    if not texts:
        return ""

    # ── 步骤 1-2：基础过滤 ────────────────────────────────────
    cleaned: List[tuple] = []  # (start, end, text)
    for i, t in enumerate(texts):
        t = t.strip()
        if len(t) < min_seg_chars:
            continue
        if _is_meta_segment(t):
            continue
        s = starts[i] if i < len(starts) else 0.0
        e = ends[i]   if i < len(ends)   else s
        cleaned.append((s, e, t))

    if not cleaned:
        return ""

    # ── 步骤 3：合并相邻短碎片 ────────────────────────────────
    merged: List[str] = []
    buf_text: List[str] = [cleaned[0][2]]
    buf_end: float = cleaned[0][1]

    for s, e, t in cleaned[1:]:
        gap = s - buf_end
        if gap < 1.5:          # 间隔 < 1.5s → 同一连续话语，合并
            buf_text.append(t)
        else:
            merged.append("".join(buf_text))
            buf_text = [t]
        buf_end = e
    merged.append("".join(buf_text))

    # ── 步骤 4：相邻重复去除 ──────────────────────────────────
    deduped: List[str] = []
    prev = ""
    for seg in merged:
        if seg != prev:
            deduped.append(seg)
        prev = seg

    # ── 步骤 5：分段抽样（核心，防止头部偏置）────────────────
    sampled = _stratified_sample(deduped, n_strata=n_strata, max_total=max_segments)

    return " ".join(sampled)


# ═══════════════════════════════════════════════════════════════
# 处理统计
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProcessStats:
    total: int = 0
    success: int = 0
    fail_json_parse: int = 0
    fail_validation: int = 0
    fail_exception: int = 0

    def report(self, kind: str) -> str:
        lines = [
            f"══ {kind} SPM Statistics ══",
            f"  Total:             {self.total}",
            f"  Success:           {self.success} ({self._pct(self.success)})",
            f"  Fail (JSON parse): {self.fail_json_parse}",
            f"  Fail (validation): {self.fail_validation}",
            f"  Fail (exception):  {self.fail_exception}",
        ]

        return "\n".join(lines)

    def _pct(self, n: int) -> str:
        return f"{n / self.total * 100:.1f}%" if self.total else "N/A"


# ═══════════════════════════════════════════════════════════════
# 基础处理器
# ═══════════════════════════════════════════════════════════════

class BaseProcessor(ABC):
    def __init__(
        self,
        input_path: str,
        output_path: str,
        generator: VLLMTextGenerator,
        trace_dir: Optional[str] = None,
    ):
        self.input_path  = input_path
        self.output_path = output_path
        self.generator   = generator
        self.trace_dir   = trace_dir
        self.stats       = ProcessStats()

    @abstractmethod
    def build_prompt(self, item: Dict) -> str: ...

    @abstractmethod
    def parse_and_validate(self, item: Dict, llm_text: str) -> Optional[Dict]: ...

    @abstractmethod
    def kind_label(self) -> str: ...

    def _item_id(self, item: Dict, idx: int) -> str:
        for key in ("problem_id", "video_id", "id"):
            if key in item:
                return str(item[key])
        return str(idx)

    def _dump_trace(self, rid: str, idx: int, meta: Dict, error: str = ""):
        if not self.trace_dir:
            return
        try:
            kind = self.kind_label()
            d = os.path.join(self.trace_dir, kind)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{kind}_{safe_filename(rid)}__{idx:08d}.json")
            obj = {"kind": kind, "id": rid, "index": idx}
            obj.update(meta)
            if error:
                obj["error"] = error
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.warning(f"trace 写入失败 idx={idx}: {e}")

    def run(self, batch_size: int = 4, flush_every: int = 200):
        batch_size  = max(1, batch_size)
        flush_every = max(1, flush_every)

        raw_data = read_jsonl(self.input_path)
        self.stats.total = len(raw_data)
        logger.info(
            f"[{self.kind_label()}] {self.input_path} "
            f"({len(raw_data)} 条, batch={batch_size}, flush={flush_every})"
        )

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        open(self.output_path, "w").close()

        buf: List[Dict] = []

        for start in range(0, len(raw_data), batch_size):
            batch   = raw_data[start: start + batch_size]
            idxs    = list(range(start, start + len(batch)))
            prompts = [self.build_prompt(it) for it in batch]

            metas: List[Dict] = []
            try:
                metas = self.generator.generate_batch(prompts)
                for meta in metas:
                    meta["gen_failed"] = False
            except Exception as e:
                logger.error(f"Batch 调用失败 start={start}: {e}")
                for p in prompts:
                    try:
                        one = self.generator.generate(p)
                        one["gen_failed"] = False
                        metas.append(one)
                    except Exception as e2:
                        logger.error(f"Single 调用失败: {e2}")
                        metas.append({
                            "prompt": p, "chat_prompt": "",
                            "raw_text": "", "text": "", "gen_failed": True,
                        })
                        self.stats.fail_exception += 1

            for j, (item, idx) in enumerate(zip(batch, idxs)):
                rid  = self._item_id(item, idx)
                meta = metas[j]
                if meta.get("gen_failed", False):
                    self._dump_trace(rid, idx, meta, error="generation_failed")
                    self.stats.fail_exception += 1
                    logger.error(f"rid:{rid}, idx:{rid}, generation_failed")
                    continue

                self._dump_trace(rid, idx, meta)

                result = None
                try:
                    result = self.parse_and_validate(item, meta.get("text", ""))
                except Exception as e:
                    self.stats.fail_exception += 1
                    self._dump_trace(rid, idx, meta, error=str(e))
                    logger.error(f"rid:{rid}, idx:{rid}, error={str(e)}")
                    continue

                if result is not None:
                    self.stats.success += 1
                    buf.append(result)

            if len(buf) >= flush_every:
                append_jsonl(self.output_path, buf)
                buf.clear()

            done = min(start + batch_size, len(raw_data))
            if done % (batch_size * 50) < batch_size:
                logger.info(f"进度: {done}/{len(raw_data)}")

        if buf:
            append_jsonl(self.output_path, buf)
            buf.clear()

        logger.info(f"\n{self.stats.report(self.kind_label())}")
        logger.info(f"输出: {self.output_path}")


# ═══════════════════════════════════════════════════════════════
# 阶段 0: Subtitle Digest Preprocessor（独立预计算）
# ═══════════════════════════════════════════════════════════════

class SubtitleDigestPreprocessor(BaseProcessor):

    # ── Prompt：偏抽取，限制推断 ──────────────────────────────
    DIGEST_PROMPT = r"""You are an educational knowledge extractor. Your task is to distill factual knowledge from a lecture transcript.

STRICT RULES:
1. Include ONLY claims that are **explicitly stated** in the transcript. Do NOT infer, generalize, or add background knowledge.
2. Preserve the original technical terms, symbols, and formulas exactly as used.
3. Each output item must be a complete, self-contained factual claim: a definition, a rule, a property, or a worked example.
4. EXCLUDE:
   - Discourse markers ("Now let's look at...", "As we saw last time...", "Let's move on to...")
   - Motivational or organizational remarks ("This is important", "We will cover three topics")
   - Repetitive restatements of the same fact (keep only one version, the clearest)
   - Meta-descriptions ("This lecture introduces...", "In this section we discuss...")
5. Output format: a numbered list of declarative sentences, **up to 8 items**.
   - Prefer fewer, higher-density claims over more, peripheral ones.
   - If the transcript only supports 3–4 high-quality claims, output only those.
   - Do NOT pad with vague or redundant items to reach a minimum count.
6. Do NOT output JSON. Do NOT add a preamble or conclusion.

Lecture transcript (may be fragmentary):
{transcript_text}

Knowledge claims (numbered list, declarative sentences only):
"""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        generator: VLLMTextGenerator,
        subtitle_max_segments: int = 40,
        subtitle_n_strata: int = 5,
        digest_max_input_tokens: int = 3600,
        trace_dir: Optional[str] = None,
    ):
        super().__init__(input_path, output_path, generator, trace_dir)
        self.subtitle_max_segments = subtitle_max_segments
        self.subtitle_n_strata     = subtitle_n_strata
        self.digest_max_input_tokens = digest_max_input_tokens

    def kind_label(self) -> str:
        return "subtitle_digest"
    
    def _count_chat_tokens_for_digest_prompt(self, transcript_text: str) -> int:
        """
        统计 digest prompt 在套上 chat template 后的真实输入 token 数。
        """
        prompt = self.DIGEST_PROMPT.format(transcript_text=transcript_text)

        chat_prompt = self.generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        token_ids = self.generator.tokenizer(
            chat_prompt,
            add_special_tokens=False,
        )["input_ids"]

        return len(token_ids)


    def _truncate_transcript_to_token_budget(self, transcript_text: str) -> str:

        if not transcript_text:
            return transcript_text

        current = transcript_text
        min_chars = 400 
        shrink_ratio = 0.85
        max_rounds = 8

        for _ in range(max_rounds):
            n_tokens = self._count_chat_tokens_for_digest_prompt(current)
            if n_tokens <= self.digest_max_input_tokens:
                return current

            if len(current) <= min_chars:
                return current

            current = current[: max(min_chars, int(len(current) * shrink_ratio))]

        return current

    def build_prompt(self, item: Dict) -> str:
        transcript = build_subtitle_text(
            item,
            max_segments=self.subtitle_max_segments,
            n_strata=self.subtitle_n_strata,
        )

        transcript = self._truncate_transcript_to_token_budget(transcript)


        return self.DIGEST_PROMPT.format(transcript_text=transcript)

    def parse_and_validate(self, item: Dict, llm_text: str) -> Optional[Dict]:

        text = llm_text.strip()
        if not text or len(text) < 80:
            logger.info(f"Video {item.get('video_id')}: digest 过短，跳过")
            return None

        sentence_count = max(
            len(re.findall(r"^\d+[\.\)]\s+", text, re.MULTILINE)),
            text.count("。") + text.count(". "),
        )
        if sentence_count < 3:
            logger.info(f"Video {item.get('video_id')}: digest 句子数不足，跳过")
            return None

        for pat in _META_STMT_RE:
            if pat.search(text[:60]):
                logger.info(f"Video {item.get('video_id')}: digest 为元叙述，跳过")
                return None

        vid = norm_video_id(item.get("video_id"))
        return {
            "video_id": vid,
            "digest":   text,
        }

    def run(self, batch_size: int = 4, flush_every: int = 200):

        raw_data = read_jsonl(self.input_path)

        # 过滤：只处理有字幕的视频
        has_subtitle = [item for item in raw_data if item.get("text")]
        logger.info(
            f"[subtitle_digest] 共 {len(raw_data)} 个视频，"
            f"其中 {len(has_subtitle)} 个有字幕，开始蒸馏…"
        )

        # 临时替换 input 数据为过滤后的列表
        orig_path = self.input_path
        import tempfile, json as _json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for item in has_subtitle:
                tmp.write(_json.dumps(item, ensure_ascii=False) + "\n")
            tmp_path = tmp.name

        self.input_path = tmp_path
        try:
            super().run(batch_size=batch_size, flush_every=flush_every)
        finally:
            self.input_path = orig_path
            os.unlink(tmp_path)


def load_digest_map(digest_path: str) -> Dict[int, str]:

    if not digest_path or not os.path.exists(digest_path):
        logger.info("未找到 digest 缓存文件，VideoProcessor 将退化为无字幕模式。")
        return {}
    mapping: Dict[int, str] = {}
    for row in read_jsonl(digest_path):
        vid = row.get("video_id")
        dig = row.get("digest", "").strip()
        if vid is not None and dig:
            mapping[int(vid)] = dig
    logger.info(f"加载 digest 缓存: {len(mapping)} 个视频")
    return mapping


# ═══════════════════════════════════════════════════════════════
# 阶段 1: Problem Processor
# ═══════════════════════════════════════════════════════════════

class ProblemProcessor(BaseProcessor):

    PROMPT = r"""You are a knowledge-point annotator for an educational question bank. From the question below, extract:

1. **core_concept**: The specific knowledge point tested. Concise academic noun phrase. Derive from the correct answer combined with the stem, not from distractors.

2. **statement**: A declarative sentence fusing the stem and correct answer into a factual knowledge claim. Remove question phrasing ("which of the following", "in the blank"), generalize fictional scenarios, never output a question.

# Input
- Course: {course_name}
- Stem: {content}
- Options: {option}
- Correct answer: {answer}

# Output (valid JSON only)
{{"core_concept": "...", "statement": "..."}}
"""

    def kind_label(self) -> str:
        return "problem"

    def build_prompt(self, item: Dict) -> str:
        return self.PROMPT.format(
            course_name=str(item.get("course_name", "")),
            content=str(item.get("content", "")),
            option=json.dumps(item.get("option", {}), ensure_ascii=False),
            answer=str(item.get("answer", "")),
        )

    def parse_and_validate(self, item: Dict, llm_text: str) -> Optional[Dict]:
        parsed = extract_json_from_text(llm_text)
        if parsed is None:
            self.stats.fail_json_parse += 1
            return None

        cc = str(parsed.get("core_concept", "")).strip()
        st = str(parsed.get("statement", "")).strip()

        if len(cc) < 2 or len(st) < 4:
            self.stats.fail_validation += 1
            logger.info(f"Problem {item.get('problem_id')}: cc={cc!r} st={st!r}")
            return None

        return {
            "id":            item.get("problem_id"),
            "exercise_id":   item.get("exercise_id"),
            "course_id":     norm_course_id(item.get("course_id")),
            "context_id":    item.get("context_id", []),
            "core_concept":  cc,
            "statement":     st,
            "resource_text": f"{cc}. {st}",
        }


# ═══════════════════════════════════════════════════════════════
# 阶段 2: Video Processor（增强版）
# ═══════════════════════════════════════════════════════════════


class VideoProcessor(BaseProcessor):


    PROMPT_DIGEST_ONLY = r"""You are a knowledge annotator extracting fine-grained educational semantics from video content.

1. **core_concept**: The central knowledge point this video teaches. Concise academic noun phrase.
   - Derive from the knowledge digest content, NOT just the title.
   - Should cover the **main knowledge focus** of the video, not an overly narrow sub-technique.
   - Do NOT use vague labels like "introduction", "overview", "basics", or chapter-level titles.

2. **statements**: 3–6 atomic declarative knowledge claims this video covers.
   - Derive directly from the knowledge digest.
   - Each must be a self-contained fact, definition, rule, or worked example.
   - Usable to judge a quiz answer as correct or incorrect.
   - NO meta-descriptions. NO vague phrases.

# Input
- Course: {course_name}
- Video title: {video_titles}
- Chapter: {chapter}
- Knowledge digest (extracted from transcript): {knowledge_digest}

# Output (valid JSON only)
{{"core_concept": "...", "statements": ["...", "..."]}}
"""

    PROMPT_LINKED_ONLY = r"""You are a knowledge annotator aligning educational video content to quiz-question granularity. From the video metadata and related question knowledge below, extract:

1. **core_concept**: The specific knowledge point this video teaches. Concise academic noun phrase. If related question knowledge is available, align granularity and terminology with it.

2. **statements**: 3–6 atomic declarative knowledge claims this video likely covers. Each must be a self-contained fact or definition usable to judge a quiz answer as correct or incorrect. No meta-descriptions.

# Input
- Course: {course_name}
- Video title: {video_titles}
- Chapter: {chapter}
- Knowledge from related questions: {linked_knowledge}

# Output (valid JSON only)
{{"core_concept": "...", "statements": ["...", "..."]}}
"""

    PROMPT_METADATA_ONLY = r"""You are a knowledge annotator extracting fine-grained educational semantics from video metadata. From the video metadata below, extract:

1. **core_concept**: The most specific knowledge point this video likely teaches. Use a concise academic noun phrase and prefer fine-grained, quiz-relevant granularity when supported by the metadata.

2. **statements**: 3–6 atomic declarative knowledge claims this video likely covers. Each must be a self-contained fact or definition usable to judge a quiz answer as correct or incorrect. No meta-descriptions.

# Input
- Course: {course_name}
- Video title: {video_titles}
- Chapter: {chapter}

# Output (valid JSON only)
{{"core_concept": "...", "statements": ["...", "..."]}}
"""

    def __init__(
        self,
        input_path: str,
        output_path: str,
        generator: VLLMTextGenerator,
        course_file_path: str,
        problem_file_path: str = "",
        digest_file_path: str = "",         # ← 新增：预计算 digest 缓存路径
        video_data_only: bool = True,        # 保持向后兼容
        spm_config: Optional[SPMConfig] = None,
        trace_dir: Optional[str] = None,
    ):
        super().__init__(input_path, output_path, generator, trace_dir)
        self.cfg          = spm_config or SPMConfig()
        self.course_map   = self._load_courses(course_file_path)
        self.video2problems: Dict[int, List[Dict]] = {}
        if problem_file_path:
            self.video2problems = self._load_video_problem_map(problem_file_path)
        self.digest_map: Dict[int, str] = load_digest_map(digest_file_path)

        # video_data_only 仅作日志用；四路分发已取代这个 flag
        self.video_data_only = video_data_only
        logger.info(
            f"VideoProcessor: digest_map={len(self.digest_map)} 条  "
            f"video2problems={len(self.video2problems)} 条"
        )

    def kind_label(self) -> str:
        return "video"

    # ── 数据加载 ──────────────────────────────────────────────────

    @staticmethod
    def _load_courses(path: str) -> Dict:
        mapping = {}
        for c in read_jsonl(path):
            cid = norm_course_id(c.get("id"))
            if cid:
                mapping[cid] = c
        logger.info(f"课程: {len(mapping)}")
        return mapping

    @staticmethod
    def _load_video_problem_map(path: str) -> Dict[int, List[Dict]]:
        v2p: Dict[int, List[Dict]] = {}
        count = 0
        for p in read_jsonl(path):
            ctx = p.get("context_id")
            if not ctx:
                continue
            if not isinstance(ctx, list):
                ctx = [ctx]
            count += 1
            for vid in ctx:
                v = norm_video_id(vid)
                if v is not None:
                    v2p.setdefault(v, []).append(p)
        logger.info(f"video→problems: {len(v2p)} 个视频 (来自 {count} 条 problem)")
        return v2p

    def _compress_linked_knowledge(self, video_id: int) -> str:
        problems = self.video2problems.get(video_id, [])
        if not problems:
            return "None available"
        parts: List[str] = []
        seen: Set[str] = set()
        for p in problems[: self.cfg.max_linked_problems]:
            cc = str(p.get("core_concept", "")).strip()
            if not cc or cc in seen:
                continue
            seen.add(cc)
            st    = str(p.get("statement", "")).strip()
            entry = f"{cc} — {st}" if st else cc
            parts.append(entry)
        return "; ".join(parts) if parts else "None available"

    # ── Prompt 构建：四路分发 ─────────────────────────────────────

    def build_prompt(self, item: Dict) -> str:
        course_id   = norm_course_id(item.get("course_id"))
        course_info = self.course_map.get(course_id, {}) if course_id else {}
        course_name = course_info.get("name", "Unknown")
        titles      = item.get("title", []) or []
        title_str   = " > ".join(t for t in titles if t and t != "Video")
        chapter     = item.get("chapter", "")

        vid    = norm_video_id(item.get("video_id"))
        linked = self._compress_linked_knowledge(vid) if vid is not None else "None available"


        digest= self.digest_map.get(int(vid), "") if vid is not None else ""
        has_digest = bool(digest)

        base = dict(course_name=course_name, video_titles=title_str, chapter=chapter)


        if has_digest:
            return self.PROMPT_DIGEST_ONLY.format(**base, knowledge_digest=digest)
        else:
            return self.PROMPT_METADATA_ONLY.format(**base)

    # ── 解析 + 校验（增强版）────────────────────────────────────

    def parse_and_validate(self, item: Dict, llm_text: str) -> Optional[Dict]:
        parsed = extract_json_from_text(llm_text)
        if parsed is None:
            self.stats.fail_json_parse += 1
            return None

        cc    = str(parsed.get("core_concept", "")).strip()
        stmts = parsed.get("statements", [])
        if isinstance(stmts, str):
            stmts = [stmts]
        if not isinstance(stmts, list):
            stmts = []
        stmts = [str(s).strip() for s in stmts if str(s).strip()]

        # ── 基础长度检查 ──────────────────────────────────────────
        if len(cc) < 2:
            self.stats.fail_validation += 1
            logger.info(f"Video {item.get('video_id')}: core_concept 过短 {cc!r}")
            return None
        if not stmts:
            self.stats.fail_validation += 1
            logger.info(f"Video {item.get('video_id')}: statements 为空")
            return None

        # # ── 质量过滤 1: core_concept 宽泛检测 ────────────────────
        # for pat in _VAGUE_CC_RE:
        #     if pat.search(cc):
        #         self.stats.fail_validation += 1
        #         logger.info(f"Video {item.get('video_id')}: core_concept 过于宽泛 {cc!r}")
        #         return None

        # if len(cc) > 50:
        #     self.stats.fail_validation += 1
        #     logger.info(f"Video {item.get('video_id')}: core_concept 过长（疑为句子）{cc!r}")
        #     return None

        # ── 质量过滤 2: statements 元叙述检测 ────────────────────
        clean_stmts: List[str] = []
        for s in stmts:
            is_meta = any(pat.search(s) for pat in _META_STMT_RE)
            if is_meta:
                logger.info(f"Video {item.get('video_id')}: 过滤元叙述 statement: {s!r}")
                continue
            clean_stmts.append(s)

        if not clean_stmts:
            self.stats.fail_validation += 1
            logger.info(f"Video {item.get('video_id')}: 所有 statements 均为元叙述")
            return None

        # ── 质量过滤 3: statements 相似度去重 ────────────────────
        clean_stmts = _dedup_statements(clean_stmts, max_bigram_overlap=0.65)

        if not clean_stmts:
            self.stats.fail_validation += 1
            return None

        resource_text = cc + ". " + ". ".join(clean_stmts)
        vid = norm_video_id(item.get("video_id"))

        return {
            "id":                    item.get("video_id"),
            "course_id":             norm_course_id(item.get("course_id")),
            "duration":              round(float(item.get("duration", 0.0)), 3),
            "core_concept":          cc,
            "statements":            clean_stmts,
            "resource_text":         resource_text,
        }


# ═══════════════════════════════════════════════════════════════
# 输出质量工具
# ═══════════════════════════════════════════════════════════════
def _dedup_statements(stmts: List[str], max_bigram_overlap: float = 0.70) -> List[str]:
    def _normalize(s: str) -> str:
        return re.sub(r"[\s\.,;：，。；！？、]", "", s)

    def _bigram_set(s: str) -> set:
        s = _normalize(s)
        return {s[i: i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()

    kept: List[str] = []
    kept_norm: List[str] = []

    for s in stmts:
        s_norm = _normalize(s)

        is_dup = False
        for k_norm, k in zip(kept_norm, kept):
            # 先处理极短句：只做 exact / normalized exact match
            if len(s_norm) < 6 or len(k_norm) < 6:
                if s_norm == k_norm:
                    is_dup = True
                    break
                continue

            s_bg = _bigram_set(s)
            k_bg = _bigram_set(k)
            union = len(s_bg | k_bg)
            if union == 0:
                continue
            if len(s_bg & k_bg) / union > max_bigram_overlap:
                is_dup = True
                break

        if not is_dup:
            kept.append(s)
            kept_norm.append(s_norm)

    return kept
