"""
DIVE-KT — Full Model Entry Point
"""
import argparse
import json
import os
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from config import Config, cfg
from data_loader import KTDataset
from data_processor import KTProcessor
from model import DIVEKT
from runner import DiveKTRunner, ScheduleConfig, StageConfig


MODE_NAME = "s1_problem_s2_problem_video"


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser factory
# ──────────────────────────────────────────────────────────────────────────────

def build_optim(model, epochs: int, lr: float, wd: float):
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )

    warmup_epochs = 8
    start_factor = 0.01
    gamma = 0.99
    min_lr = 1e-6
    min_factor = min_lr / max(lr, 1e-12)

    def lr_lambda(step: int) -> float:
        # LambdaLR is initialized with step=0.  The first training epoch will
        # therefore start at start_factor * lr.
        if step < warmup_epochs:
            progress = step / max(1, warmup_epochs)
            return start_factor + (1.0 - start_factor) * progress

        decay_steps = step - warmup_epochs
        return max(min_factor, gamma ** decay_steps)

    sched = LambdaLR(opt, lr_lambda=lr_lambda)
    return opt, sched

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_metadata_files(config: Config, dataset: str) -> tuple[str, str, str]:
    data_dir = os.path.join(config.DATA_DIR, f"mcx_{dataset}")
    vf = os.path.join(data_dir, f"{dataset}_video_ex_sub.json")
    pf = os.path.join(data_dir, f"{dataset}_problem.json")
    cf = os.path.join(data_dir, f"raw/{dataset}_course.json")
    for fp, name in [(vf, "Video metadata"), (pf, "Problem metadata"), (cf, "Course structure")]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"{name} not found: {fp}")
    return vf, pf, cf


def get_loaders(config: Config, dataset: str, fold: int) -> tuple:
    proc     = KTProcessor(config)
    data_dir = os.path.join(config.DATA_DIR, f"mcx_{dataset}/5folds")
    vf, pf, cf = resolve_metadata_files(config, dataset)
    proc.load_metadata(video_file=vf, problem_file=pf, course_file=cf)
    stats = proc.stats()
    print(f"Items={stats['num_items']}  Courses={stats['num_courses']}  Chapters={stats['num_chapters']}")

    for split in ("train", "valid", "test"):
        p = os.path.join(data_dir, f"{dataset}_{split}{fold}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fold file not found: {p}")

    kwargs   = dict(batch_size=config.BATCH_SIZE, num_workers=4)
    train_ld = DataLoader(KTDataset(os.path.join(data_dir, f"{dataset}_train{fold}.json"), proc, config), shuffle=True,  **kwargs)
    valid_ld = DataLoader(KTDataset(os.path.join(data_dir, f"{dataset}_valid{fold}.json"), proc, config), shuffle=False, **kwargs)
    test_ld  = DataLoader(KTDataset(os.path.join(data_dir, f"{dataset}_test{fold}.json"),  proc, config), shuffle=False, **kwargs)
    return train_ld, valid_ld, test_ld, stats


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def metric_mean_std(values: list) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def print_fold_line(label: str, metrics: dict) -> None:
    print(f">>> {label}  AUC={metrics['auc']:.4f}  ACC={metrics['acc']:.4f}  "
          f"F1N={metrics['f1neg']:.4f}  RMSE={metrics['rmse']:.4f}")


def print_summary(name: str, fold_results: list) -> dict | None:
    if not fold_results:
        print(f"[WARN] No results for {name}")
        return None
    metrics = {k: [r[k] for r in fold_results] for k in ("auc", "acc", "f1neg", "rmse")}
    summary = {k: dict(zip(("mean", "std"), metric_mean_std(v))) for k, v in metrics.items()}
    print("\n" + "=" * 72)
    print(f"{name} Test Summary  (n={len(fold_results)} folds)")
    print("=" * 72)
    for k, s in summary.items():
        print(f"{k.upper():<6}: {s['mean']:.4f} ± {s['std']:.4f}")
    return summary


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """Support plain state_dict and common wrapped checkpoint formats."""
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if obj and all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(
        "Unsupported checkpoint format. Expected a state_dict or a dict "
        "containing model_state_dict/state_dict/model."
    )


def load_model_checkpoint_if_exists(model: DIVEKT, path: str, device: str) -> bool:
    """Load a checkpoint if it exists, otherwise keep the current in-memory model."""
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found, continue with current weights: {path}")
        return False

    obj = torch.load(path, map_location=device)
    state = _extract_state_dict(obj)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print(f"[WARN] Missing keys when loading {path}: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading {path}: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    print(f"[CKPT LOADED] {path}")
    return True


def collect_stage_metrics(stage_accum: dict[str, list], results: dict[str, Any]) -> None:
    for key, value in results.items():
        if key == "final":
            continue
        if isinstance(value, dict) and "best_test" in value:
            stage_accum.setdefault(key, []).append(value["best_test"])
    if "final" in results:
        stage_accum.setdefault("final", []).append(results["final"])


def print_all_stage_lines(prefix: str, results: dict[str, Any]) -> None:
    for key, value in results.items():
        if key == "final":
            continue
        if isinstance(value, dict) and "best_test" in value:
            print_fold_line(f"{prefix} {key}", value["best_test"])
    print_fold_line(f"{prefix} FINAL", results["final"])


def make_checkpoint_eval_result(
    *,
    runner: DiveKTRunner,
    model: DIVEKT,
    ckpt_path: str,
    test_loader,
    label: str,
) -> dict[str, Any]:
    """Evaluate an existing checkpoint and wrap it like a runner stage result."""
    metrics = runner.evaluate_checkpoint(model, ckpt_path, test_loader, label)
    return {
        "best_epoch": None,
        "best_valid": {},
        "best_test": metrics,
        "ckpt_path": ckpt_path,
        "resumed_from_ckpt": True,
    }


def resume_full_fold_if_possible(
    *,
    args: argparse.Namespace,
    runner: DiveKTRunner,
    model: DIVEKT,
    schedule: ScheduleConfig,
    test_loader,
    fold: int,
) -> dict[str, Any] | None:
    """Skip a fold when its final Stage-2 checkpoint already exists."""
    final_ckpt = schedule.stage2.ckpt_path
    if not args.resume or not os.path.isfile(final_ckpt):
        return None

    print("\n" + "#" * 80)
    print(f"[RESUME SKIP] Fold {fold} final checkpoint found: {final_ckpt}")
    print("[RESUME SKIP] Re-evaluating saved checkpoint(s) on the test set.")
    print("#" * 80)

    stage1_result = None
    if os.path.isfile(schedule.stage1.ckpt_path):
        stage1_result = make_checkpoint_eval_result(
            runner=runner,
            model=model,
            ckpt_path=schedule.stage1.ckpt_path,
            test_loader=test_loader,
            label=f"Fold {fold} Stage-1 Test [RESUME]",
        )
    else:
        print(f"[RESUME WARN] Stage-1 checkpoint not found: {schedule.stage1.ckpt_path}")

    stage2_result = make_checkpoint_eval_result(
        runner=runner,
        model=model,
        ckpt_path=final_ckpt,
        test_loader=test_loader,
        label=f"Fold {fold} Stage-2 Test [RESUME]",
    )

    return {
        "stage1_problem": stage1_result,
        "stage2_problem_video": stage2_result,
        "final": stage2_result["best_test"],
        "final_source": "stage2_problem_video",
        "resumed_from_ckpt": True,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _add_bool_arg(p: argparse.ArgumentParser, name: str, default: bool) -> None:
    p.add_argument(f"--{name}",    dest=name, action="store_true")
    p.add_argument(f"--no_{name}", dest=name, action="store_false")
    p.set_defaults(**{name: default})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        f"DIVE-KT Full Model ({MODE_NAME})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run_id",  type=int, default=0)
    p.add_argument("--dataset", type=str, default="social", choices=["social", "seam", "humanities"])
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--train_set",     type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all_folds",     action="store_true")

    p.add_argument("--stage1_epochs", type=int, default=120)
    p.add_argument("--stage2_epochs", type=int, default=80)

    p.add_argument("--patience_1",    type=int, default=10)
    p.add_argument("--patience_2",    type=int, default=10)

    p.add_argument("--stage1_lr",     type=float, default=None)
    p.add_argument("--stage2_lr",     type=float, default=None)
    p.add_argument("--weight_decay",  type=float, default=None)

    _add_bool_arg(p, "stage1_train_embed", getattr(cfg, "STAGE1_TRAIN_EMBED", True))
    _add_bool_arg(p, "stage2_train_embed", getattr(cfg, "STAGE2_TRAIN_EMBED", True))
    _add_bool_arg(p, "resume", True)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print(f"\n>>> DIVE-KT ({MODE_NAME}) | dataset={args.dataset} | device={cfg.DEVICE}")

    ckpt_dir = os.path.join(cfg.CKPT_DIR, str(args.run_id))
    os.makedirs(ckpt_dir, exist_ok=True)

    runner = DiveKTRunner(cfg, build_optim)
    folds  = list(range(1, 6)) if args.all_folds else [args.train_set]

    stage_accum: dict[str, list] = {
        "stage1_problem": [],
        "stage2_problem_video": [],
        "final": [],
    }
    all_results: list[dict[str, Any]] = []

    wd        = args.weight_decay if args.weight_decay is not None else cfg.WEIGHT_DECAY
    stage1_lr = args.stage1_lr    if args.stage1_lr    is not None else cfg.LEARNING_RATE
    stage2_lr = args.stage2_lr    if args.stage2_lr    is not None else cfg.LEARNING_RATE

    for fold in folds:
        set_seed(args.seed)
        print("\n" + "#" * 80)
        print(f"Fold {fold}/{len(folds)}  |  Seed {args.seed}  |  Mode={MODE_NAME}")
        print("#" * 80)

        train_ld, valid_ld, test_ld, stats = get_loaders(cfg, args.dataset, fold)

        def ckpt(tag: str) -> str:
            return os.path.join(ckpt_dir, f"divekt_{args.dataset}_fold{fold}_{tag}.pth")

        schedule = ScheduleConfig(
            stage1=StageConfig(
                epochs=args.stage1_epochs, patience=args.patience_1,
                lr=stage1_lr, wd=wd,
                train_embed=args.stage1_train_embed, ckpt_path=ckpt("stage1"),
            ),
            stage2=StageConfig(
                epochs=args.stage2_epochs, patience=args.patience_2,
                lr=stage2_lr, wd=wd,
                train_embed=args.stage2_train_embed, ckpt_path=ckpt("stage2"),
            ),
        )
        model = DIVEKT(cfg, stats).to(cfg.DEVICE)

        results = resume_full_fold_if_possible(
            args=args,
            runner=runner,
            model=model,
            schedule=schedule,
            test_loader=test_ld,
            fold=fold,
        )

        if results is None:
            raw12 = runner.run(model, train_ld, valid_ld, test_ld, schedule)
            results = {
                "stage1_problem": raw12.get("stage1"),
                "stage2_problem_video": raw12.get("stage2"),
                "final": raw12["final"],
                "final_source": raw12.get("final_source", "stage2_problem_video"),
            }

        print_all_stage_lines(f"Fold {fold}", results)
        collect_stage_metrics(stage_accum, results)
        all_results.append({"fold": fold, "seed": args.seed, "results": results})

        del model
        torch.cuda.empty_cache()

    summaries = {}
    for sk, fr in stage_accum.items():
        if fr:
            summaries[sk] = print_summary(sk.replace("_", "-").title(), fr)

    fold_tag     = "5fold" if args.all_folds else f"fold{args.train_set}"
    summary_path = os.path.join(ckpt_dir, f"divekt_{args.dataset}_{fold_tag}_{MODE_NAME}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "mode": MODE_NAME,
                   "folds": folds, "summaries": summaries, "details": all_results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
