"""
DIVE-KT — Data Processor (Metadata & Embedding Builder)
"""
import gc
import json
import os

import torch

from config import build_progress_bar
from embedding import LLMEmbeddingEncoder


class KTProcessor:
    """
    Builds all static lookup tables and embedding matrices needed by
    :class:`~data_loader.KTDataset`.

    Call :meth:`load_metadata` once; after that the processor is ready to be
    passed to dataset constructors.
    """

    def __init__(self, config):
        self.cfg = config
        self.embed_batch_size = config.EMBED_BATCH_SIZE

        # ── Integer index maps ─────────────────────────────────────────────────
        self.id2idx:       dict = {"<PAD>": 0}
        self.course2idx:   dict = {"<PAD>": 0}
        self.chapter2idx:  dict = {"<PAD>": 0, "<UNK>": 1}

        # ── Temporary per-item maps (freed after _build_matrices) ──────────────
        self._item2course:   dict = {0: 0}
        self._item2chapter:  dict = {0: 0}
        self._res2main_chapter: dict = {}   # resource_id → "chid_<main>" string

        # ── Video max durations (needed by data_loader for coverage calc) ───────
        self.max_duration: dict = {}

        # ── LLM encoder (released after batch encoding) ────────────────────────
        self._encoder = LLMEmbeddingEncoder(config)

        # ── Embedding storage (populated during load_metadata) ─────────────────
        self._embed_dict: dict = {0: torch.zeros(config.LLM_EMBED_DIM)}

        # ── Final output tensors (set by _build_matrices) ──────────────────────
        self.embed_matrix:       torch.Tensor | None = None
        self.item_course_lookup: torch.Tensor | None = None
        self.item_chapter_lookup: torch.Tensor | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def load_metadata(
        self,
        video_file: str,
        problem_file: str,
        course_file: str,
    ) -> None:
        """
        Load video / problem / course metadata, encode all resource texts,
        and build the lookup matrices.

        Args:
            video_file:   Path to the video metadata JSONL.
            problem_file: Path to the problem metadata JSONL.
            course_file:  Path to the course structure JSONL.
        """
        print("[KTProcessor] Loading metadata …")
        self._load_chapter_map(course_file)

        all_texts:   list[str] = []
        all_indices: list[int] = []

        self._load_video_metadata(video_file, all_texts, all_indices)
        self._load_problem_metadata(problem_file, all_texts, all_indices)

        self._batch_encode(all_texts, all_indices)
        self._build_matrices()

        print(
            f"[KTProcessor] Done. "
            f"items={len(self.id2idx)}  "
            f"courses={len(self.course2idx)}  "
            f"chapters={len(self.chapter2idx)}"
        )

    def stats(self) -> dict:
        return {
            "num_items":    len(self.id2idx),
            "num_courses":  len(self.course2idx),
            "num_chapters": len(self.chapter2idx),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Chapter / course helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_chapter(raw) -> tuple[str | None, str | None]:
        """
        Convert a dot-separated chapter string to ``(full, main_level)``.

        ``'1.2.3'  →  ('1.2.3', '1')``
        Returns ``(None, None)`` for empty / malformed input.
        """
        if raw is None or str(raw).strip() == "":
            return None, None
        parts = [p.strip() for p in str(raw).split(".") if p.strip()]
        if not parts:
            return None, None
        return ".".join(parts), parts[0]

    def _load_chapter_map(self, course_file: str) -> None:
        if not os.path.exists(course_file):
            print(f"[KTProcessor] WARNING: course file not found: {course_file}")
            return
        with open(course_file, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for res in data.get("resource", []):
                    rid = res.get("resource_id")
                    _, main = self._parse_chapter(res.get("chapter"))
                    if rid and main:
                        key = f"chid_{main}"
                        self._res2main_chapter[rid] = key
                        if key not in self.chapter2idx:
                            self.chapter2idx[key] = len(self.chapter2idx)

    def _chapter_idx(self, resource_key: str) -> int:
        ch = self._res2main_chapter.get(resource_key)
        if not ch:
            return self.chapter2idx["<UNK>"]
        return self.chapter2idx.get(ch, self.chapter2idx["<UNK>"])

    def _register_course(self, item_idx: int, data: dict) -> None:
        cid = f"cid_{data.get('course_id', -1)}"
        if cid not in self.course2idx:
            self.course2idx[cid] = len(self.course2idx)
        self._item2course[item_idx] = self.course2idx[cid]

    # ──────────────────────────────────────────────────────────────────────────
    # Per-modality metadata loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_video_metadata(
        self,
        video_file: str,
        all_texts: list,
        all_indices: list,
    ) -> None:
        if not os.path.exists(video_file):
            return
        with open(video_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in build_progress_bar(lines, desc="Video metadata"):
            data = json.loads(line)
            vid = f"vid_{data['id']}"
            if vid not in self.id2idx:
                self.id2idx[vid] = len(self.id2idx)
            item_idx = self.id2idx[vid]

            self.max_duration[vid] = data.get("duration", 100.0)
            self._register_course(item_idx, data)
            self._item2chapter[item_idx] = self._chapter_idx(f"V_{data['id']}")

            text = data.get("resource_text", "")
            if not text:
                # Fallback: reconstruct from raw fields.
                cc    = data.get("core_concept", "")
                stmts = data.get("statements", [])
                if isinstance(stmts, str):
                    stmts = [stmts]
                text = cc + ". " + ". ".join(stmts) if stmts else cc

            all_texts.append(text)
            all_indices.append(item_idx)

    def _load_problem_metadata(
        self,
        problem_file: str,
        all_texts: list,
        all_indices: list,
    ) -> None:
        if not os.path.exists(problem_file):
            return
        with open(problem_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in build_progress_bar(lines, desc="Problem metadata"):
            data = json.loads(line)
            pid = f"pid_{data['id']}"
            if pid not in self.id2idx:
                self.id2idx[pid] = len(self.id2idx)
            item_idx = self.id2idx[pid]

            self._register_course(item_idx, data)
            self._item2chapter[item_idx] = self._chapter_idx(
                data.get("exercise_id", "")
            )

            text = data.get("resource_text", "")
            if not text:
                cc = data.get("core_concept", "")
                st = data.get("statement", "")
                text = f"{cc}. {st}" if st else cc


            all_texts.append(text)
            all_indices.append(item_idx)

    # ──────────────────────────────────────────────────────────────────────────
    # LLM batch encoding
    # ──────────────────────────────────────────────────────────────────────────

    def _batch_encode(self, all_texts: list, all_indices: list) -> None:
        print(f"[KTProcessor] Encoding {len(all_texts)} items …")
        bs = self.embed_batch_size
        n_batches = (len(all_texts) + bs - 1) // bs
        for i in build_progress_bar(range(n_batches), desc="Encoding"):
            start, end = i * bs, min((i + 1) * bs, len(all_texts))
            embs = self._encoder.encode(all_texts[start:end])
            for j, idx in enumerate(all_indices[start:end]):
                self._embed_dict[idx] = embs[j]

        self._encoder.release_memory()
        self._encoder = None

    # ──────────────────────────────────────────────────────────────────────────
    # Matrix / lookup-table construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_matrices(self) -> None:
        n = len(self.id2idx)
        d = self.cfg.LLM_EMBED_DIM

        # Embedding matrix: (n_items, llm_dim)
        mat = torch.zeros(n, d)
        for idx, emb in self._embed_dict.items():
            if idx < n:
                mat[idx] = emb
        self.embed_matrix = mat

        # Course lookup: item_idx → course_idx
        course_lut = torch.zeros(n, dtype=torch.long)
        for i, c in self._item2course.items():
            if i < n:
                course_lut[i] = c
        self.item_course_lookup = course_lut

        # Chapter lookup: item_idx → chapter_idx
        chapter_lut = torch.zeros(n, dtype=torch.long)
        for i, ch in self._item2chapter.items():
            if i < n:
                chapter_lut[i] = ch
        self.item_chapter_lookup = chapter_lut

        # Free large temporary dictionaries.
        del self._embed_dict, self._item2course, self._item2chapter, self._res2main_chapter
        gc.collect()
