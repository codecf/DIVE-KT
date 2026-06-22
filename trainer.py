"""
DIVE-KT — Trainer (single-epoch train / eval loops)
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import build_progress_bar


class KTTrainer:
    """Wraps train/eval loops for Knowledge Tracing with masked BCE loss."""

    def __init__(self, model, config, optimizer, scheduler=None):
        self.model     = model
        self.cfg       = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = config.DEVICE

        self.aux_weight = float(getattr(config, "AUX_LAMBDA", 0.0))
        self.cl_weight  = float(getattr(config, "CL_LAMBDA",  0.0))
        self.loss_fn    = nn.BCEWithLogitsLoss(reduction="none")

    # ──────────────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────────────

    def _masked_bce(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masks:  torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss averaged over valid (masked) positions only."""
        raw = self.loss_fn(logits, labels.float())
        return (raw * masks).sum() / masks.sum().clamp_min(1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Epoch loops
    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self, loader) -> tuple:
        self.model.train()
        sums   = dict(loss=0.0, kt=0.0, aux=0.0, cl=0.0)
        y_true, y_pred = [], []

        pbar = build_progress_bar(loader, desc="Train", leave=False)
        for batch in pbar:
            ids  = batch["item_ids"].to(self.device)
            embs = batch["text_embeds"].to(self.device)
            beh  = batch["beh_features"].to(self.device)
            lbl  = batch["labels"].to(self.device)
            msk  = batch["masks"].to(self.device)
            ts   = batch["timestamps"].to(self.device)

            self.optimizer.zero_grad()
            logits, aux, cl = self.model(ids, embs, beh, ts)

            if torch.isnan(aux):
                aux = torch.zeros((), device=self.device)
            if torch.isnan(cl):
                cl  = torch.zeros((), device=self.device)

            kt_loss = self._masked_bce(logits, lbl, msk)
            loss    = kt_loss + self.aux_weight * aux + self.cl_weight * cl

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                valid = msk.bool()
                if valid.any():
                    y_true.extend(lbl[valid].cpu().numpy())
                    y_pred.extend(torch.sigmoid(logits)[valid].cpu().numpy())

            sums["loss"] += loss.item()
            sums["kt"]   += kt_loss.item()
            sums["aux"]  += aux.item()
            sums["cl"]   += cl.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return self._build_metrics(sums, len(loader), y_true, y_pred)

    @torch.no_grad()
    def eval_epoch(self, loader) -> tuple:
        self.model.eval()
        sums   = dict(loss=0.0, kt=0.0, aux=0.0, cl=0.0)
        y_true, y_pred = [], []

        pbar = build_progress_bar(loader, desc="Eval", leave=False)
        for batch in pbar:
            ids  = batch["item_ids"].to(self.device)
            embs = batch["text_embeds"].to(self.device)
            beh  = batch["beh_features"].to(self.device)
            lbl  = batch["labels"].to(self.device)
            msk  = batch["masks"].to(self.device)
            ts   = batch["timestamps"].to(self.device)

            logits, aux, cl = self.model(ids, embs, beh, ts)
            if torch.isnan(aux):
                aux = torch.zeros((), device=self.device)
            if torch.isnan(cl):
                cl  = torch.zeros((), device=self.device)

            kt_loss = self._masked_bce(logits, lbl, msk)
            loss    = kt_loss + self.aux_weight * aux + self.cl_weight * cl

            sums["loss"] += loss.item()
            sums["kt"]   += kt_loss.item()
            sums["aux"]  += aux.item()
            sums["cl"]   += cl.item()

            valid = msk.bool()
            if valid.any():
                y_true.extend(lbl[valid].cpu().numpy())
                y_pred.extend(torch.sigmoid(logits)[valid].cpu().numpy())

            pbar.set_postfix(loss=f"{sums['loss'] / (pbar.n + 1):.4f}")

        return self._build_metrics(sums, len(loader), y_true, y_pred)

    # ──────────────────────────────────────────────────────────────────────────
    # Metric aggregation
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rmse(y_true: list, y_pred: list) -> float:
        a = np.asarray(y_true, np.float32)
        b = np.asarray(y_pred, np.float32)
        return float(np.sqrt(np.mean((b - a) ** 2))) if a.size else 0.0

    def _build_metrics(
        self,
        sums:     dict,
        n_batches: int,
        y_true:   list,
        y_pred:   list,
    ) -> tuple:
        """
        Returns an 8-tuple:
            (loss, kt_loss, aux_loss, cl_loss, auc, accuracy, f1_neg, rmse)
        """
        n   = max(1, n_batches)
        avg = {k: v / n for k, v in sums.items()}

        if not y_true:
            return (avg["loss"], avg["kt"], avg["aux"], avg["cl"],
                    0.5, 0.0, 0.0, 0.0)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = 0.5

        y_bin = np.array(y_pred) >= 0.5
        acc   = accuracy_score(y_true, y_bin)
        f1n   = f1_score(y_true, y_bin, pos_label=0, zero_division=0)
        rmse  = self._rmse(y_true, y_pred)

        return (avg["loss"], avg["kt"], avg["aux"], avg["cl"],
                auc, acc, f1n, rmse)
