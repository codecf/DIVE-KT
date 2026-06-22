"""
DIVE-KT — Dataset & DataLoader
"""
import json
import math
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

from config import build_progress_bar
from data_processor import KTProcessor


class KTDataset(Dataset):
    """
    PyTorch Dataset for DIVE-KT.

    Each item is a left-padded sequence of length ``MAX_SEQ_LEN`` containing
    interleaved problem and video events for a single student.
    """

    _TS_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f")

    def __init__(self, interaction_path: str, processor: KTProcessor, config):
        self.processor = processor
        self.cfg       = config
        self.beh_dim   = getattr(config, "BEHAVIOR_FEAT_DIM", 6)

        self.embed_matrix         = processor.embed_matrix
        self.item_course_lookup   = processor.item_course_lookup
        self.item_chapter_lookup  = processor.item_chapter_lookup

        self.problem_seq_len = getattr(config, "PROBLEM_SEQ_LEN", 200)
        self.hard_limit      = getattr(config, "MAX_SEQ_LEN", 400)
        self.use_video       = getattr(config, "USE_VIDEO", True)
        self.max_videos      = getattr(config, "MAX_VIDEOS_PER_SEQ", 9999)

        self.data: list[dict] = []
        self._load(interaction_path)
        print(f"[KTDataset] {len(self.data)} sequences loaded from {interaction_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Timestamp parsing
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def _parse_timestamp(cls, ts_str: str) -> int:
        """Parse a datetime string to a Unix timestamp (int). Returns 0 on failure."""
        if not ts_str:
            return 0
        ts_str = ts_str.strip()
        for fmt in cls._TS_FORMATS:
            try:
                return int(datetime.strptime(ts_str, fmt).timestamp())
            except ValueError:
                pass
        return 0

    # ──────────────────────────────────────────────────────────────────────────
    # Loading & pre-processing
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        raw_sequences = []
        for line in build_progress_bar(lines, desc="Parsing interactions"):
            user   = json.loads(line)
            parsed = self._parse_events(user["interactions"])
            if not parsed:
                continue
            parsed.sort(key=lambda x: x["ts_sec"])

            prob_positions = [i for i, x in enumerate(parsed) if x["type"] == "problem"]
            if not prob_positions:
                continue

            # Chunk by PROBLEM_SEQ_LEN problems per sequence.
            for chunk_start in range(0, len(prob_positions), self.problem_seq_len):
                chunk_end  = min(chunk_start + self.problem_seq_len, len(prob_positions))
                last_glob  = prob_positions[chunk_end - 1]
                first_glob = 0 if chunk_start == 0 else prob_positions[chunk_start - 1] + 1
                mixed      = parsed[first_glob: last_glob + 1]

                sampled = self._sample_videos(mixed)
                raw_sequences.append({
                    "ids":    [x["idx"]    for x in sampled],
                    "feats":  [x["feat"]   for x in sampled],
                    "labels": [x["label"]  for x in sampled],
                    "masks":  [x["mask"]   for x in sampled],
                    "ts":     [x["ts_sec"] for x in sampled],
                    "length": len(sampled),
                })

        print(f"[KTDataset] hard_limit={self.hard_limit}")

        pad_feat = [0.0] * self.beh_dim
        for seq in build_progress_bar(raw_sequences, desc="Padding"):
            pad_len = self.hard_limit - seq["length"]
            if pad_len > 0:
                ids    = [0]        * pad_len + seq["ids"]
                feats  = [pad_feat] * pad_len + seq["feats"]
                labels = [0.0]      * pad_len + seq["labels"]
                masks  = [0.0]      * pad_len + seq["masks"]
                ts     = [0]        * pad_len + seq["ts"]
            else:
                ids, feats, labels, masks, ts = (
                    seq["ids"], seq["feats"], seq["labels"], seq["masks"], seq["ts"]
                )
            self.data.append(dict(ids=ids, feats=feats, labels=labels, masks=masks, ts=ts))

        del raw_sequences

    # ──────────────────────────────────────────────────────────────────────────
    # Event parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_events(self, interactions: list) -> list[dict]:
        out = []
        for itr in interactions:
            itype = itr["type"]
            if not self.use_video and itype != "problem":
                continue

            prefix = "pid" if itype == "problem" else "vid"
            iid    = f"{prefix}_{itr['item_id']}"
            if iid not in self.processor.id2idx:
                continue
            idx = self.processor.id2idx[iid]

            raw   = itr["raw_data"]
            feat  = [0.0] * self.beh_dim
            mask, label = 0.0, 0.0

            if itype == "problem":
                correct  = float(raw.get("is_correct", 0))
                feat[0]  = 1.0
                feat[1]  = correct
                label, mask = correct, 1.0
            else:
                video_beh = self._parse_video_behaviour(raw, iid)
                if video_beh is None:
                    continue
                feat[0] = 0.0
                feat[2], feat[3], feat[4], feat[5] = video_beh

            ts = self._parse_timestamp(itr.get("timestamp", ""))
            out.append(dict(idx=idx, feat=feat, label=label, mask=mask,
                            type=itype, ts_sec=ts))
        return out

    def _parse_video_behaviour(self, raw: dict, iid: str) -> tuple | None:
        """
        Extract ``(wall_time, seg_center, coverage, speed)`` from a raw video event.

        Returns ``None`` if the segment is invalid or too short to be informative
        (wall-time < 30 s).  Corresponds to paper's v_t = [w_t, π_ctr, ρ, ν].
        """
        start = float(raw.get("start_point", 0.0))
        end   = float(raw.get("end_point",   0.0))
        speed = float(raw.get("speed",        1.0))
        if speed <= 0:
            speed = 1.0

        if not all(math.isfinite(v) for v in (start, end, speed)):
            return None
        start, end = max(0.0, start), max(0.0, end)

        max_dur = float(self.processor.max_duration.get(iid, 1.0)) or 1.0
        start, end = min(start, max_dur), min(end, max_dur)
        if end <= start:
            return None

        content_seg = end - start
        wall_time   = content_seg / speed

        if wall_time < 30.0:                 # discard very short / noise views
            return None
        wall_time = min(wall_time, 7200.0)   # cap at 2 hours

        coverage   = min(1.0, content_seg / max_dur)
        seg_center = min(1.0, ((start + end) * 0.5) / max_dur)

        return wall_time, seg_center, coverage, speed

    # ──────────────────────────────────────────────────────────────────────────
    # Stratified video sampling
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_videos(self, mixed: list) -> list:
        """
        Accumulated stratified sampling: proportionally keep recent videos per
        problem segment within the global quota, then enforce the hard limit.
        """
        prob_idx = [i for i, x in enumerate(mixed) if x["type"] == "problem"]
        vid_idx  = [i for i, x in enumerate(mixed) if x["type"] != "problem"]
        n_probs  = len(prob_idx)
        n_vids   = len(vid_idx)

        quota = max(0, min(self.hard_limit - n_probs, self.max_videos))
        if n_vids == 0 or quota == 0:
            return [mixed[i] for i in prob_idx]

        keep_ratio = quota / n_vids
        balance    = 0.0
        final      = []
        cursor     = 0
        added      = 0

        for pi in prob_idx:
            seg     = mixed[cursor:pi]
            seg_len = len(seg)
            if seg_len > 0 and quota > 0:
                theoretical = seg_len * keep_ratio
                base        = int(theoretical)
                balance    += theoretical - base
                extra       = int(balance)
                balance    -= extra
                keep        = min(base + extra, seg_len, quota - added)
                if keep > 0:
                    final.extend(seg[-keep:])
                    added += keep
            final.append(mixed[pi])
            cursor = pi + 1

        # Hard-limit safety: prefer keeping the most recent video events.
        if len(final) > self.hard_limit:
            probs = [x for x in final if x["type"] == "problem"]
            vids  = [x for x in final if x["type"] != "problem"]
            slots = max(0, self.hard_limit - len(probs))
            final = sorted(probs + vids[-slots:], key=lambda x: x["ts_sec"])

        return final

    # ──────────────────────────────────────────────────────────────────────────
    # PyTorch Dataset interface
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        s   = self.data[idx]
        ids = torch.tensor(s["ids"], dtype=torch.long)
        return {
            "item_ids":     ids,
            "course_ids":   self.item_course_lookup[ids],
            "chapter_ids":  self.item_chapter_lookup[ids],
            "text_embeds":  self.embed_matrix[ids],
            "beh_features": torch.tensor(s["feats"],   dtype=torch.float32),
            "labels":       torch.tensor(s["labels"],  dtype=torch.float32),
            "masks":        torch.tensor(s["masks"],   dtype=torch.float32),
            "timestamps":   torch.tensor(s["ts"],      dtype=torch.long),
        }
