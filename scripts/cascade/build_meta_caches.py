"""Build train + eval meta-classifier caches from production DMOE checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_L1 = Path("artifacts/checkpoints/transformer/best.pt")
DEFAULT_L2A = Path("artifacts/checkpoints/expert_cereals/best.pt")
DEFAULT_L2B = Path("artifacts/checkpoints/l2b_anomaly_gate/best.pt")
DEFAULT_L2C = Path("artifacts/checkpoints/l2c_meadow_expert/best.pt")
DEFAULT_L3 = Path("artifacts/checkpoints/level3_protonet/best.pt")
DEFAULT_TRAIN_CACHE = Path("artifacts/cached_logits_train.pt")
DEFAULT_EVAL_CACHE = Path("artifacts/cached_logits_eval.pt")
DEFAULT_L1C_TRAIN = Path("artifacts/L1C_backup/cached_logits_train.pt")
DEFAULT_L1C_EVAL = Path("artifacts/L1C_backup/cached_logits_eval.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract DMOE logits to train/eval caches.")
    p.add_argument("--l1-weights", type=Path, default=DEFAULT_L1)
    p.add_argument("--l2a-weights", type=Path, default=DEFAULT_L2A)
    p.add_argument("--l2b-weights", type=Path, default=DEFAULT_L2B)
    p.add_argument("--l2c-weights", type=Path, default=DEFAULT_L2C)
    p.add_argument("--l3-weights", type=Path, default=DEFAULT_L3)
    p.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    p.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL_CACHE)
    p.add_argument("--l1c-train-cache", type=Path, default=DEFAULT_L1C_TRAIN)
    p.add_argument("--l1c-eval-cache", type=Path, default=DEFAULT_L1C_EVAL)
    p.add_argument("--skip-l1c-global", action="store_true")
    p.add_argument("--train-regions", nargs="+", default=["frh01", "frh02", "frh03"])
    p.add_argument("--eval-regions", nargs="+", default=["frh04"])
    p.add_argument("--level", default="L2A")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--with-metadata", action="store_true", default=True)
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def extract_cache(args: argparse.Namespace, *, regions: list[str], cache_path: Path) -> None:
    cmd = [
        args.python,
        "scripts/cascade/sweep_dmoe_parameters.py",
        "--extract-only",
        "--skip-msresnet",
        f"--l1-weights={args.l1_weights}",
        f"--l2a-weights={args.l2a_weights}",
        f"--l2b-weights={args.l2b_weights}",
        f"--l2c-weights={args.l2c_weights}",
        f"--l3-weights={args.l3_weights}",
        f"--cache={cache_path}",
        f"--level={args.level}",
        f"--batch-size={args.batch_size}",
        f"--max-samples={args.max_samples}",
        "--train-regions", *args.train_regions,
        "--regions", *regions,
    ]
    run(cmd)


def main() -> None:
    args = parse_args()
    for path in (args.l1_weights, args.l2a_weights, args.l2b_weights, args.l2c_weights, args.l3_weights):
        if not path.is_file():
            raise SystemExit(f"Missing checkpoint: {path}")

    extract_cache(args, regions=args.train_regions, cache_path=args.train_cache)
    extract_cache(args, regions=args.eval_regions, cache_path=args.eval_cache)

    if not args.skip_l1c_global:
        for target, source in (
            (args.train_cache, args.l1c_train_cache),
            (args.eval_cache, args.l1c_eval_cache),
        ):
            if not source.is_file():
                raise SystemExit(f"Missing L1C source cache: {source}")
            run([
                args.python, "scripts/cascade/append_l1c_cached_global_logits.py",
                "--target-cache", str(target),
                "--source-cache", str(source),
                f"--max-samples={args.max_samples}",
            ])

    # Patch L3 with the locked-prototype Mahalanobis metric from the episodic
    # ProtoNet; this is the regime that recovered rare-class (orchard) recall.
    run([
        args.python, "scripts/cascade/patch_l3_distances.py",
        "--mode", "mahalanobis",
        "--l3-weights", str(args.l3_weights),
        "--level", args.level,
        "--train-regions", *args.train_regions,
        str(args.train_cache), str(args.eval_cache),
    ])

    if args.with_metadata:
        run([
            args.python, "scripts/cascade/append_parcel_metadata.py",
            str(args.train_cache), str(args.eval_cache),
            "--level", args.level, "--with-area", "--with-doy",
        ])

    print(f"\nDone. Train cache: {args.train_cache.resolve()}")
    print(f"      Eval cache:  {args.eval_cache.resolve()}")


if __name__ == "__main__":
    main()
