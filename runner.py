"""
DIVE-KT — Clean Stage Runner (s1_problem_s2_problem_video)
"""
import os
from dataclasses import dataclass
from typing import Callable

import torch

from trainer import KTTrainer


# ──────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StageConfig:
    """Hyper-parameters for one training stage."""
    epochs:      int
    patience:    int
    lr:          float
    wd:          float
    train_embed: bool
    ckpt_path:   str


@dataclass
class ScheduleConfig:
    """Two-stage schedule for the full DIVE-KT backbone."""
    stage1: StageConfig
    stage2: StageConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def pack_metrics(t: tuple) -> dict:
    keys = ("loss", "kt", "aux", "cl", "auc", "acc", "f1neg", "rmse")
    return {k: float(v) for k, v in zip(keys, t)}


def log_epoch(stage: str, ep: int, total: int, lr: float,
              tr: tuple, va: tuple) -> None:
    print(f"[{stage}] Epoch [{ep}/{total}]  lr={lr:.2e}")
    print(f"  Train  Loss={tr[0]:.4f}  KT={tr[1]:.4f}"
          f"  |  AUC={tr[4]:.4f}  ACC={tr[5]:.4f}  F1N={tr[6]:.4f}  RMSE={tr[7]:.4f}")
    print(f"  Valid  Loss={va[0]:.4f}  KT={va[1]:.4f}"
          f"  |  AUC={va[4]:.4f}  ACC={va[5]:.4f}  F1N={va[6]:.4f}  RMSE={va[7]:.4f}")


def _load_checkpoint(model, path: str, device: str) -> None:
    model.load_state_dict(torch.load(path, map_location=device), strict=True)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

class DiveKTRunner:
    """
    Runs the fixed s1_s2_joint schedule for DIVE-KT.

    Args:
        cfg:            Global Config instance.
        build_optim_fn: Callable ``(model, epochs, lr, wd) → (opt, sched)``.
    """

    def __init__(self, cfg, build_optim_fn: Callable):
        self.cfg         = cfg
        self.build_optim = build_optim_fn

    # ── Public entry points ───────────────────────────────────────────────────

    def run(
        self,
        model,
        train_loader,
        valid_loader,
        test_loader,
        schedule: ScheduleConfig,
    ) -> dict:
        """
        Two-stage schedule: problem -> problem+video joint fine-tuning.

        Returns dict with keys:
            stage1, stage2, final, final_source
        """
        loaders = (train_loader, valid_loader, test_loader)
        s1 = self._run_stage1(model, loaders, schedule.stage1)
        s2 = self._run_stage2(model, loaders, schedule.stage2,
                              resume_from=s1["ckpt_path"])
        return {
            "stage1":       s1,
            "stage2":       s2,
            "final":        s2["best_test"],
            "final_source": "stage2_problem_video",
        }

    def run_stage1_only(
        self,
        model,
        train_loader,
        valid_loader,
        test_loader,
        stage1_cfg: StageConfig,
    ) -> dict:
        """
        Stage-1 only (used by ablation variant A3: no video branch).

        Returns dict with keys:
            stage1, final, final_source
        """
        loaders = (train_loader, valid_loader, test_loader)
        s1 = self._run_stage1(model, loaders, stage1_cfg)
        return {
            "stage1":       s1,
            "stage2":       None,
            "final":        s1["best_test"],
            "final_source": "stage1",
        }

    # ── Stage helpers ─────────────────────────────────────────────────────────

    def _run_stage1(self, model, loaders, cfg: StageConfig) -> dict:
        model.configure_for_stage1(train_embed=cfg.train_embed)
        return self._run_stage("Stage-1", model, loaders, cfg)

    def _run_stage2(
        self,
        model,
        loaders,
        cfg: StageConfig,
        resume_from: str,
    ) -> dict:
        """Load Stage-1 checkpoint and enable problem+video joint fine-tuning."""
        _load_checkpoint(model, resume_from, self.cfg.DEVICE)
        model.configure_for_stage2_joint(train_embed=cfg.train_embed)
        return self._run_stage("Stage-2-Problem+Video", model, loaders, cfg)

    @staticmethod
    def _current_lr(opt, fallback: float) -> float:
        """Return the first parameter-group LR for logging."""
        if opt is None or not getattr(opt, "param_groups", None):
            return float(fallback)
        return float(opt.param_groups[0].get("lr", fallback))

    @staticmethod
    def _step_scheduler(sched, epoch: int, val_auc: float) -> None:
        """
        Step a scheduler returned by build_optim().

        The current baseline uses a regular LambdaLR scheduler
        (fixed warmup + epoch-independent exponential decay), so sched.step()
        is sufficient.  The dict branch is kept only for backward compatibility
        with older warmup + ReduceLROnPlateau experiments.
        """
        if sched is None:
            return

        if isinstance(sched, dict):
            warmup_epochs = int(sched.get("warmup_epochs", 0) or 0)
            warmup = sched.get("warmup")
            plateau = sched.get("plateau")

            if epoch <= warmup_epochs and warmup is not None:
                warmup.step()
            elif plateau is not None:
                plateau.step(float(val_auc))
            return

        # Fallback for standard epoch-based schedulers.
        sched.step()

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _run_stage(
        self,
        name:       str,
        model,
        loaders:    tuple,
        stage_cfg:  StageConfig,
    ) -> dict:
        """
        Single training stage with early stopping.

        Returns:
            best_epoch, best_valid, best_test, ckpt_path
        """
        train_loader, valid_loader, test_loader = loaders
        print(f"\n[{name}] params: {model.param_summary()}")

        opt, sched = self.build_optim(model, stage_cfg.epochs, stage_cfg.lr, stage_cfg.wd)

        trainer = KTTrainer(model, self.cfg, opt, None)

        best_auc   = -1.0
        best_epoch = 0
        best_valid: dict = {}
        wait       = 0

        for ep in range(1, stage_cfg.epochs + 1):
            tr = trainer.train_epoch(train_loader)
            va = trainer.eval_epoch(valid_loader)

            val_auc = va[4]
            cur_lr = self._current_lr(opt, fallback=stage_cfg.lr)
            log_epoch(name, ep, stage_cfg.epochs, cur_lr, tr, va)
            self._step_scheduler(sched, ep, val_auc)

            if val_auc > best_auc:
                best_auc, best_epoch, best_valid, wait = val_auc, ep, pack_metrics(va), 0
                torch.save(model.state_dict(), stage_cfg.ckpt_path)
                print(f"  >>> [{name}] Best valid AUC={best_auc:.4f}  saved → {stage_cfg.ckpt_path}")
            else:
                wait += 1
                print(f"  ... [{name}] No improvement ({wait}/{stage_cfg.patience})")
                if wait >= stage_cfg.patience:
                    print(f"  [{name}] Early stop at epoch {ep}")
                    break

        best_test = self.evaluate_checkpoint(
            model, stage_cfg.ckpt_path, test_loader, f"{name} Test"
        )
        return {
            "best_epoch": best_epoch,
            "best_valid": best_valid,
            "best_test":  best_test,
            "ckpt_path":  stage_cfg.ckpt_path,
        }

    # ── Checkpoint evaluation ─────────────────────────────────────────────────

    def evaluate_checkpoint(
        self,
        model,
        ckpt_path:  str,
        loader,
        label:      str = "Test",
    ) -> dict:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        _load_checkpoint(model, ckpt_path, self.cfg.DEVICE)
        model.eval()
        m = pack_metrics(KTTrainer(model, self.cfg, optimizer=None).eval_epoch(loader))
        print(f"[{label}]  AUC={m['auc']:.4f}  ACC={m['acc']:.4f}"
              f"  F1N={m['f1neg']:.4f}  RMSE={m['rmse']:.4f}")
        return m
