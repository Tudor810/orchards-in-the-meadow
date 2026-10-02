"""End-to-end L2A retrain: experts -> caches -> meta-classifier -> comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
LOG_DIR = ARTIFACTS / "l2a_pipeline_logs"
L1C_BACKUP = ARTIFACTS / "L1C_backup"
_LOCK_HANDLE = None


def acquire_pipeline_lock() -> None:
    """Hold an OS-level lock so only one orchestrator can write artifacts."""
    global _LOCK_HANDLE
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    handle = (ARTIFACTS / ".l2a_pipeline.lock").open("a+b")
    if os.fstat(handle.fileno()).st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise SystemExit("Another L2A pipeline instance is already running.") from exc
    _LOCK_HANDLE = handle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full L2A retrain + evaluation pipeline.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--skip-meta", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume L1 from best.pt.")
    p.add_argument("--start-at", choices=("l1", "l2a", "l2b", "l2c", "l3"), default="l1")
    p.add_argument("--orchard-multiplier", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--l3-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    return p.parse_args()


def run_step(name: str, cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\nSTEP: {name}\n{'=' * 72}", flush=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONUNBUFFERED": "1",
    }
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {name}\n# {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = proc.wait()
        if code != 0:
            raise SystemExit(f"Step failed ({name}) exit={code}; see {log_path}")


def _best_from_frontier(rows: list[dict], *, acc_min: float, orchard_min: float) -> dict | None:
    """Highest-accuracy row that clears both targets, or None."""
    feasible = [
        r for r in rows
        if r.get("accuracy") is not None
        and r["accuracy"] >= acc_min
        and r["orchard_recall"] >= orchard_min
    ]
    if not feasible:
        return None
    return max(feasible, key=lambda r: (r["orchard_recall"], r["accuracy"]))


def parse_frontier_results(json_path: Path) -> dict:
    """Summarize the frontier sweep JSON into the comparison payload."""
    out = {
        "accuracy": None,
        "orchard_recall": None,
        "orchard_precision": None,
        "wheat_recall": None,
        "source": str(json_path),
    }
    if not json_path.is_file():
        return out
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows") or []
        preferred = (
            payload.get("best_stretch")
            or payload.get("best_production")
            or payload.get("closest_stretch")
        )
        if preferred:
            out.update(
                accuracy=preferred.get("accuracy"),
                orchard_recall=preferred.get("orchard_recall"),
                orchard_precision=preferred.get("orchard_precision"),
                wheat_recall=preferred.get("wheat_recall"),
                selected_m=preferred.get("m"),
                selected_bias=preferred.get("b_orchard"),
                selected_meadow_bias=preferred.get("b_meadow"),
                selected_blend=preferred.get("blend"),
            )
            return out
    else:
        rows = payload
    if not rows:
        return out
    # Prefer the stretch target, then production, then best raw accuracy.
    best = (
        _best_from_frontier(rows, acc_min=81.0, orchard_min=60.0)
        or _best_from_frontier(rows, acc_min=80.69, orchard_min=35.0)
        or max(rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
    )
    out.update(
        accuracy=best["accuracy"],
        orchard_recall=best["orchard_recall"],
        orchard_precision=best.get("orchard_precision"),
        wheat_recall=best.get("wheat_recall"),
        selected_m=best.get("m"),
        selected_bias=best.get("b_orchard"),
        selected_meadow_bias=best.get("b_meadow"),
        selected_blend=best.get("blend"),
    )
    return out


def parse_l1c_baseline() -> dict:
    out = {
        "accuracy": 80.76,
        "orchard_recall": 35.80,
        "orchard_precision": None,
        "wheat_recall": 97.57,
        "source": "L2A migration plan baseline",
    }
    return out


def main() -> None:
    acquire_pipeline_lock()
    args = parse_args()
    py = args.python
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ep = str(args.epochs)
    bs = str(args.batch_size)

    l1_cmd = [
        py, "scripts/train_transformer_breizhcrops.py",
        "--from-scratch", "--no-freeze-backbone", "--use-doy",
        f"--epochs={ep}", f"--batch-size={bs}",
    ]
    if args.resume:
        l1_cmd.append("--resume-best")
    train_steps = [
        ("train_l1_transformer", l1_cmd),
        ("train_l2a_cereals", [py, "scripts/train_expert_cereals.py", "--from-scratch", f"--epochs={ep}", f"--batch-size={bs}"]),
        ("train_l2b_anomaly_gate", [py, "scripts/train_l2b_anomaly_gate.py", f"--epochs={ep}", f"--batch-size={bs}"]),
        ("train_l2c_meadow", [py, "scripts/train_l2c_meadow_expert.py", "--no-pretrained", f"--epochs={ep}", f"--batch-size={bs}"]),
        ("train_l3_protonet", [py, "scripts/train_l3_protonet.py", "--level=L2A", f"--epochs={args.l3_epochs}", "--checkpoint-dir=artifacts/checkpoints/level3_protonet"]),
    ]

    if not args.skip_train:
        start_indices = {"l1": 0, "l2a": 1, "l2b": 2, "l2c": 3, "l3": 4}
        for name, cmd in train_steps[start_indices[args.start_at]:]:
            run_step(name, cmd, LOG_DIR / f"{ts}_{name}.log")

    if not args.skip_cache:
        # build_meta_caches patches L3 distances (Mahalanobis) and appends 8-d metadata.
        run_step("build_meta_caches", [py, "scripts/build_meta_caches.py"], LOG_DIR / f"{ts}_build_meta_caches.log")

    frontier_log = ARTIFACTS / "sweep_compact_dual_bias.log"
    frontier_json = ARTIFACTS / "sweep_compact_dual_bias.json"
    if not args.skip_meta:
        run_step(
            "train_meta_classifier",
            [
                py,
                "scripts/train_meta_classifier.py",
                f"--orchard-multiplier={args.orchard_multiplier}",
                "--layout=compact",
            ],
            LOG_DIR / f"{ts}_train_meta_classifier.log",
        )
        run_step(
            "sweep_compact_dual_bias",
            [
                py, "scripts/sweep_compact_dual_bias.py",
                "--layout=compact",
                "--multipliers=1,2,3",
                "--output-log", str(frontier_log),
                "--output-json", str(frontier_json),
            ],
            LOG_DIR / f"{ts}_sweep_compact_dual_bias.log",
        )

    l2a = parse_frontier_results(frontier_json)
    l1c = parse_l1c_baseline()
    comparison = {
        "timestamp": ts,
        "level": "L2A",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "l1c_baseline": l1c,
        "l2a_results": l2a,
        "feature_layout": "compact 39-d (L3+cereal+msresnet+transformer+metadata) + dual orchard/meadow bias",
    }
    out_json = ARTIFACTS / "l2a_vs_l1c_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    report = "\n".join([
        "L1C vs L2A Comparison",
        "",
        f"L2A config: epochs={args.epochs}, batch_size={args.batch_size}, m={args.orchard_multiplier:g}",
        f"L1C baseline: {l1c}",
        f"L2A best (frontier sweep): {l2a}",
        "",
        f"Saved: {out_json.resolve()}",
    ])
    (ARTIFACTS / "l2a_vs_l1c_comparison.log").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
