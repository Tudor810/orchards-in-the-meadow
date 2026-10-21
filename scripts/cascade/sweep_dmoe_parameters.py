"""Cached-logit hyperparameter sweep for the soft-routing DMOE pipeline.

The probabilistic DMOE decode has seven sensitive knobs:

    t1, t2b, t2c, t3   -- per-stage softmax temperatures
    tau_route          -- compute/routing gate threshold
    b_meadow, b_tree   -- cost-sensitive log-space decision biases

Full model inference per configuration is wasteful: the *networks* never
change, only the post-hoc composition of their outputs does. This script
therefore runs in two phases:

Phase 1 (extraction, run once)
    Streams the evaluation split through all four models *unconditionally*
    (no routing gates) and caches the raw, unscaled outputs to
    ``artifacts/cached_logits.pt``:

        l1_logits    (N, 4)   l2a_logits (N, 2)   l2b_logits (N, 2)
        l2c_logits   (N, 2)   l3_distances (N, 5) msresnet_logits (N, 9)
        labels       (N,)

    L3 prototypes + Mahalanobis inverse variances are locked from the train
    regions with ``--ref-k-shot`` (default 50) support samples per class.

Phase 2 (in-memory sweep)
    Re-implements ``DMOEPipeline.forward_probs`` + the cost-sensitive decode
    as a pure, fully vectorized function of the cached tensors and the six
    hyperparameters, then grid/random-searches thousands of configurations in
    O(N) each. Configurations beating BOTH the accuracy baseline and the
    orchard-recall floor are reported, and the Top-K are ranked by macro F1.

NOTE: tune on a calibration split (e.g. a held-out train region), not on the
final test region, or the selected operating point will be optimistically
biased.

Examples
--------
Extract + sweep::

    python scripts/cascade/sweep_dmoe_parameters.py \
        --l1-weights  artifacts/checkpoints/transformer/best.pt \
        --l2a-weights artifacts/checkpoints/expert_cereals/best.pt \
        --l2b-weights artifacts/checkpoints/l2b_anomaly_gate/best.pt \
        --l2c-weights artifacts/checkpoints/l2c_meadow_expert/best.pt \
        --l3-weights  artifacts/checkpoints/level3_protonet/best.pt

Re-sweep an existing cache with a bigger random search::

    python scripts/cascade/sweep_dmoe_parameters.py --sweep-only --num-trials 20000
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import breizhcrops as bzh

from agricultureai.data.breizhcrops_loaders import build_cache_eval_loader
from agricultureai.data.episodic_sampler import (
    NUM_LEVEL3_CLASSES,
    Level3FewShotDataset,
    build_reference_support_pool,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN
from agricultureai.models.proto_net import squared_mahalanobis_distance
from agricultureai.pipeline import (
    GLOBAL_ORCHARDS,
    MEADOW_GLOBAL_IDS,
    NUM_GLOBAL_CLASSES,
    TREE_GLOBAL_IDS,
    decode_predictions,
)

# Reuse the checkpoint loaders from the integration eval script (same dir).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from evaluate_dmoe_pipeline import (  # noqa: E402
    load_l1_generalist,
    load_l2a_cereal,
    load_l2b_anomaly_gate,
    load_l2c_meadow,
    load_l3_protonet,
)

PARAM_NAMES = ("t1", "t2b", "t2c", "t3", "tau_route", "b_meadow", "b_tree")



# --------------------------------------------------------------------------- #
# Phase 1: extraction                                                          #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def extract_cached_logits(
    *,
    l1: torch.nn.Module,
    l2a: torch.nn.Module,
    l2b: torch.nn.Module,
    l2c: torch.nn.Module,
    l3: torch.nn.Module,
    msresnet: torch.nn.Module | None,
    prototypes: torch.Tensor,
    inv_variances: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run all five branches on every sample; no routing gates applied."""
    l1_chunks, l2a_chunks, l2b_chunks, l2c_chunks, l3_chunks = [], [], [], [], []
    msresnet_chunks: list[torch.Tensor] = []
    label_chunks = []

    for x, doy, padding_mask, targets, _ in tqdm(loader, desc="Extracting raw branch outputs"):
        x = x.to(device, non_blocking=True)
        doy = doy.to(device, non_blocking=True)
        padding_mask = padding_mask.to(device, non_blocking=True)

        l1_chunks.append(l1(x, doy=doy, padding_mask=padding_mask).float().cpu())
        l2a_chunks.append(l2a(x).float().cpu())
        l2b_chunks.append(l2b(x, doy, padding_mask).float().cpu())
        l2c_chunks.append(l2c(x, doy, padding_mask).float().cpu())

        embeddings = l3.embed(x)
        distances = squared_mahalanobis_distance(embeddings, prototypes, inv_variances)
        l3_chunks.append(distances.float().cpu())
        if msresnet is not None:
            msresnet_chunks.append(msresnet(x).float().cpu())
        label_chunks.append(targets.long().cpu())

    out: Dict[str, torch.Tensor] = {
        "l1_logits": torch.cat(l1_chunks, dim=0),
        "l2a_logits": torch.cat(l2a_chunks, dim=0),
        "l2b_logits": torch.cat(l2b_chunks, dim=0),
        "l2c_logits": torch.cat(l2c_chunks, dim=0),
        "l3_distances": torch.cat(l3_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
    }
    if msresnet is not None:
        out["msresnet_logits"] = torch.cat(msresnet_chunks, dim=0)
    return out


def run_extraction(args, cache_path: Path) -> None:
    for flag in ("l1_weights", "l2a_weights", "l2b_weights", "l2c_weights", "l3_weights"):
        if not getattr(args, flag):
            raise SystemExit(
                f"--{flag.replace('_', '-')} is required for extraction "
                f"(or pass --sweep-only to reuse {cache_path})."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extraction device: {device}")
    channels = 13 if args.level == "L1C" else 10

    l1 = load_l1_generalist(Path(args.l1_weights).resolve(), device, max_seq_len=args.max_seq_len)
    l2a = load_l2a_cereal(Path(args.l2a_weights).resolve(), device, max_seq_len=args.max_seq_len)
    l2b = load_l2b_anomaly_gate(Path(args.l2b_weights).resolve(), device, max_seq_len=args.max_seq_len)
    l2c = load_l2c_meadow(Path(args.l2c_weights).resolve(), device, channels=channels)
    l3, _hidden = load_l3_protonet(Path(args.l3_weights).resolve(), device, channels=channels)

    msresnet = None
    if not args.skip_msresnet:
        print("Loading pretrained MSResNet (9-class global logits) ...")
        msresnet = bzh.models.pretrained("MSResNet", device=device)
        msresnet.eval()

    # --- Lock L3 prototypes + Mahalanobis inverse variances -----------------
    print(
        f"\nLocking L3 prototypes/variances from {args.train_regions} "
        f"(ref_k_shot={args.ref_k_shot}, shrinkage={args.shrinkage}) ..."
    )
    proto_dataset = Level3FewShotDataset(
        regions=args.train_regions,
        level=args.level,
        max_samples=int(args.max_samples),
        max_seq_len=args.max_seq_len,
    )
    support_x, support_y = build_reference_support_pool(
        proto_dataset, k_shot=args.ref_k_shot, n_way=NUM_LEVEL3_CLASSES, seed=args.seed
    )
    prototypes, inv_variances = l3.prototypes_and_inv_variances_from_support(
        support_x.to(device),
        support_y.to(device),
        NUM_LEVEL3_CLASSES,
        shrinkage=args.shrinkage,
    )
    print(f"Locked {prototypes.size(0)} prototypes of dim {prototypes.size(1)}.")

    loader = build_cache_eval_loader(
        regions=args.regions,
        level=args.level,
        max_samples=int(args.max_samples),
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    cached = extract_cached_logits(
        l1=l1, l2a=l2a, l2b=l2b, l2c=l2c, l3=l3, msresnet=msresnet,
        prototypes=prototypes, inv_variances=inv_variances,
        loader=loader, device=device,
    )
    payload = {
        **cached,
        "prototypes": prototypes.cpu(),
        "inv_variances": inv_variances.cpu(),
        "meta": {
            "regions": list(args.regions),
            "train_regions": list(args.train_regions),
            "level": args.level,
            "ref_k_shot": int(args.ref_k_shot),
            "shrinkage": float(args.shrinkage),
            "seed": int(args.seed),
            "created": datetime.now().isoformat(timespec="seconds"),
            "msresnet": "pretrained" if msresnet is not None else "skipped",
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    n = cached["labels"].numel()
    print(f"\nCached raw outputs for {n} samples -> {cache_path}")


# --------------------------------------------------------------------------- #
# Phase 2: vectorized decode + sweep                                           #
# --------------------------------------------------------------------------- #
def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    idx = labels * NUM_GLOBAL_CLASSES + preds                        # (N,)
    flat = torch.bincount(idx, minlength=NUM_GLOBAL_CLASSES**2)
    return flat.view(NUM_GLOBAL_CLASSES, NUM_GLOBAL_CLASSES)


def summarize_confusion(cm: torch.Tensor) -> Dict[str, float]:
    tp = cm.diag().float()
    support = cm.sum(dim=1).float()
    predicted = cm.sum(dim=0).float()

    recall = 100.0 * tp / support.clamp(min=1.0)                     # (9,)
    precision = 100.0 * tp / predicted.clamp(min=1.0)                # (9,)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-9)

    return {
        "overall_accuracy": float(100.0 * tp.sum() / support.sum()),
        "macro_f1": float(f1.mean()),
        "macro_recall": float(recall.mean()),
        "orchard_recall": float(recall[GLOBAL_ORCHARDS]),
        "orchard_precision": float(precision[GLOBAL_ORCHARDS]),
        "tree_macro_recall": float(recall[list(TREE_GLOBAL_IDS)].mean()),
        "meadow_macro_recall": float(recall[list(MEADOW_GLOBAL_IDS)].mean()),
    }


def random_configs(num_trials: int, seed: int) -> Iterable[Dict[str, float]]:
    rng = random.Random(seed)
    for _ in range(num_trials):
        yield {
            "t1": rng.uniform(0.5, 3.0),
            "t2b": rng.uniform(0.5, 3.0),
            "t2c": rng.uniform(0.5, 3.0),
            "t3": math.exp(rng.uniform(math.log(0.1), math.log(50.0))),
            "tau_route": rng.uniform(0.01, 0.15),
            "b_meadow": rng.uniform(-2.0, 2.0),
            "b_tree": rng.uniform(0.0, 4.0),
        }


def grid_configs() -> Iterable[Dict[str, float]]:
    t1_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    t2b_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    t2c_grid = [0.5, 1.0, 1.5, 2.0, 3.0]
    t3_grid = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]
    tau_grid = [0.01, 0.05, 0.10, 0.15]
    b_meadow_grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    b_tree_grid = [0.0, 1.0, 2.0, 3.0, 4.0]
    for t1 in t1_grid:
        for t2b in t2b_grid:
            for t2c in t2c_grid:
                for t3 in t3_grid:
                    for tau in tau_grid:
                        for bm in b_meadow_grid:
                            for bt in b_tree_grid:
                                yield {
                                    "t1": t1, "t2b": t2b, "t2c": t2c, "t3": t3,
                                    "tau_route": tau, "b_meadow": bm, "b_tree": bt,
                                }


def run_sweep(args, cache_path: Path) -> None:
    if not cache_path.is_file():
        raise SystemExit(f"Cache not found: {cache_path}. Run extraction first.")

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload.get("meta", {})
    print(f"Loaded cache: {cache_path}  meta={meta}")

    device = torch.device(
        args.sweep_device
        if args.sweep_device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    l1_logits = payload["l1_logits"].to(device)
    l2a_logits = payload["l2a_logits"].to(device)
    l2b_logits = payload["l2b_logits"].to(device)
    l2c_logits = payload["l2c_logits"].to(device)
    l3_distances = payload["l3_distances"].to(device)
    labels = payload["labels"].to(device)
    n = labels.numel()
    print(f"Sweeping on device={device} over {n} cached samples.")

    def evaluate(cfg: Dict[str, float]) -> Dict[str, float]:
        preds = decode_predictions(
            l1_logits, l2a_logits, l2b_logits, l2c_logits, l3_distances, **cfg
        )
        return summarize_confusion(confusion_matrix(preds, labels))

    # Reference row: the uncalibrated defaults.
    identity = {
        "t1": 1.0, "t2b": 1.0, "t2c": 1.0, "t3": 1.0,
        "tau_route": 0.05, "b_meadow": 0.0, "b_tree": 0.0,
    }
    base = evaluate(identity)
    print(
        f"\nIdentity config (t=1, tau=0.05, b=0): "
        f"acc={base['overall_accuracy']:.2f}%  macro_f1={base['macro_f1']:.2f}%  "
        f"orchard_recall={base['orchard_recall']:.2f}%"
    )

    if args.search == "grid":
        configs = list(grid_configs())
    else:
        configs = list(random_configs(args.num_trials, args.seed))
    print(
        f"\nSearch: {args.search}  configs={len(configs)}  "
        f"filter: acc > {args.min_accuracy:.2f}% AND "
        f"orchard_recall > {args.min_orchard_recall:.2f}%\n"
    )

    results: List[Dict[str, float]] = []
    num_passing = 0
    for cfg in tqdm(configs, desc="Sweeping configurations"):
        metrics = evaluate(cfg)
        passes = (
            metrics["overall_accuracy"] > args.min_accuracy
            and metrics["orchard_recall"] > args.min_orchard_recall
        )
        if passes:
            num_passing += 1
            tqdm.write(
                "PASS  "
                + "  ".join(f"{k}={cfg[k]:.3f}" for k in PARAM_NAMES)
                + f"  | acc={metrics['overall_accuracy']:.2f}%"
                f"  orchard_rec={metrics['orchard_recall']:.2f}%"
                f"  macro_f1={metrics['macro_f1']:.2f}%"
            )
        results.append({**cfg, **metrics, "passes_filter": passes})

    print(f"\n{num_passing}/{len(configs)} configurations passed the filter.")

    ranked_pool = [r for r in results if r["passes_filter"]] or results
    if not num_passing:
        print("No config passed both constraints; ranking ALL configs instead.")
    ranked = sorted(ranked_pool, key=lambda r: r["macro_f1"], reverse=True)

    top_k = min(args.top_k, len(ranked))
    print(f"\n================= Top {top_k} configurations (by macro F1) =================")
    for rank, row in enumerate(ranked[:top_k], start=1):
        params = "  ".join(f"{k}={row[k]:.4f}" for k in PARAM_NAMES)
        print(
            f"#{rank}  {params}\n"
            f"     acc={row['overall_accuracy']:.2f}%  macro_f1={row['macro_f1']:.2f}%  "
            f"orchard_recall={row['orchard_recall']:.2f}%  "
            f"orchard_precision={row['orchard_precision']:.2f}%  "
            f"tree_recall={row['tree_macro_recall']:.2f}%  "
            f"meadow_recall={row['meadow_macro_recall']:.2f}%"
        )

    _maybe_write_results(results, identity_row={**identity, **base}, output=args.output)


def _maybe_write_results(
    results: List[Dict[str, float]],
    *,
    identity_row: Dict[str, float],
    output: str,
) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping results export.")
        return

    out_path = output.strip()
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path("Results") / f"sweep_dmoe_parameters_{ts}.xlsx")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    df_pass = df[df["passes_filter"]]
    df_identity = pd.DataFrame([identity_row])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_results", index=False)
        df_pass.to_excel(writer, sheet_name="passing", index=False)
        df_identity.to_excel(writer, sheet_name="identity_baseline", index=False)
    print(f"\nWrote: {out_path}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cache raw DMOE branch outputs once, then sweep the seven soft-routing "
            "hyperparameters (t1, t2b, t2c, t3, tau_route, b_meadow, b_tree) in memory."
        ),
    )
    # Extraction inputs (required unless --sweep-only).
    parser.add_argument("--l1-weights", default="", help="Level-1 generalist best.pt")
    parser.add_argument("--l2a-weights", default="", help="Level-2a cereal best.pt")
    parser.add_argument("--l2b-weights", default="", help="Level-2b anomaly gate best.pt")
    parser.add_argument("--l2c-weights", default="", help="Level-2c meadow expert best.pt")
    parser.add_argument("--l3-weights", default="", help="Level-3 protonet best.pt")

    parser.add_argument("--train-regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["frh04"],
        help=(
            "Regions to cache/sweep on. Prefer a calibration split over the "
            "final test region to avoid tuning on the test set."
        ),
    )
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--ref-k-shot",
        type=int,
        default=50,
        help="Support samples per class to lock L3 prototypes/variances.",
    )
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)

    # Cache control.
    parser.add_argument("--cache", default="artifacts/cached_logits.pt")
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Skip extraction and reuse the existing cache file.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run extraction and save the cache without sweeping.",
    )
    parser.add_argument(
        "--skip-msresnet",
        action="store_true",
        help="Do not extract pretrained MSResNet 9-class logits into the cache.",
    )

    # Sweep configuration.
    parser.add_argument("--search", choices=["random", "grid"], default="random")
    parser.add_argument("--num-trials", type=int, default=5000)
    parser.add_argument("--min-accuracy", type=float, default=80.69)
    parser.add_argument("--min-orchard-recall", type=float, default=35.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sweep-device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--output", default="", help="Optional Excel results path.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cache_path = Path(args.cache).resolve()

    if args.sweep_only:
        print(f"--sweep-only: skipping extraction, reusing {cache_path}")
    else:
        run_extraction(args, cache_path)

    if args.extract_only:
        print("--extract-only: skipping sweep.")
        return

    run_sweep(args, cache_path)


if __name__ == "__main__":
    main()
