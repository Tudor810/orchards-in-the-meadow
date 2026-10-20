"""Train a 9-class MSResNet on all dates or on cloud-screened dates (pilot).

Tests whether cleaner input helps: the same architecture, seed and schedule are
trained twice, differing only in which dates reach the network:

    --dates all     every date, as the stock BreizhCrops transform does
    --dates clear   only dates passing the cloud mask in the series store
                    (``build_series_store.py``); a parcel without a clear date
                    falls back to all its dates

Each epoch draws a fresh sorted 45-step resample per parcel (with replacement
when shorter), like the stock transform. Training runs on frh01-frh03; frh04 is
only monitored, never used to pick an epoch (the last epoch is final).

Long runs fit in short jobs: training stops before ``--time-budget-min`` would
be exceeded and resumes from ``<out-dir>/latest.pt`` on the next call. Once all
epochs are done, ``--export`` writes TTA logits (mean over ``--draws`` views) as
``<out-dir>/logits_<split>.npz`` (parcel_ids, labels, features, columns), the
format ``train_meta_variants.py --extra`` reads. An L1C store
(``build_series_store.py --level L1C``, via ``--store-dir``) trains the L1C
control; its parcels differ slightly from the L2A caches, so export it with
``--align-cache`` to get the cache's row order.

Example::

    python scripts/train_series_classifier.py --dates clear --out-dir artifacts/checkpoints/msresnet_l2a_clear
    python scripts/train_series_classifier.py --dates clear --out-dir artifacts/checkpoints/msresnet_l2a_clear --export eval
    python scripts/train_series_classifier.py --dates all --store-dir artifacts/series_store_l1c \\
        --out-dir artifacts/checkpoints/msresnet_l1c_all --export eval --align-cache artifacts/cached_logits_eval.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from agricultureai.cache_alignment import align_logits_by_parcel_id
from agricultureai.data.breizhcrops_series import resample_indices
from agricultureai.models.breizhcrops_msresnet import BreizhCropsMSResNetExpert

NUM_CLASSES, SEQ_LEN, CHANNELS = 9, 45, 10
ORCHARD = 5
INDEX_NAMES = ("NDVI", "WDRVI", "SAVI", "MSAVI", "OSAVI", "EVI", "NDMI", "MSI", "NDRE", "IRECI",
               "NDI45", "MCARI", "GNDVI", "PPR", "PVR")


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num)
    ok = np.abs(den) > 1e-6
    np.divide(num, den, out=out, where=ok)
    return out


def vegetation_indices(x: np.ndarray) -> np.ndarray:
    """15 vegetation indices per date from the 10 L2A bands (B2..B8A, B11, B12; reflectance 0-1).

    Padded dates (all zeros) stay 0; values are clipped to [-10, 10].
    """
    b2, b3, b4, b5, b6, b7, b8, _b8a, b11, _b12 = (x[..., i] for i in range(10))
    disc = np.maximum((2 * b8 + 1) ** 2 - 8 * (b8 - b4), 0.0)
    idx = np.stack([
        _ratio(b8 - b4, b8 + b4),                                  # NDVI
        _ratio(0.1 * b8 - b3, 0.1 * b8 + b3),                      # WDRVI
        _ratio(1.5 * (b8 - b4), b8 + b4 + 0.5),                    # SAVI
        (2 * b8 + 1 - np.sqrt(disc)) / 2,                          # MSAVI
        _ratio(1.16 * (b8 - b4), b8 + b4 + 0.16),                  # OSAVI
        _ratio(2.5 * (b8 - b4), b8 + 6 * b4 - 7.5 * b2 + 1),       # EVI
        _ratio(b8 - b11, b8 + b11),                                # NDMI
        _ratio(b11, b8),                                           # MSI
        _ratio(b8 - b5, b8 + b5),                                  # NDRE
        _ratio((b7 - b4) * b6, b5),                                # IRECI
        _ratio(b5 - b4, b5 + b4),                                  # NDI45
        _ratio((b6 - b5 - 0.2 * (b6 - b3)) * b6, b5),              # MCARI
        _ratio(b8 - b3, b8 + b3),                                  # GNDVI
        _ratio(b3 - b2, b3 + b2),                                  # PPR
        _ratio(b3 - b4, b3 + b4),                                  # PVR
    ], axis=-1)
    return np.clip(idx, -10.0, 10.0).astype(np.float32)


class SeriesStore:
    """Ragged parcel series from one or more ``build_series_store.py`` files."""

    def __init__(self, paths: List[Path], dates: str, indices: bool = False) -> None:
        self.indices = indices
        parts = [np.load(p) for p in paths]
        self.labels = np.concatenate([p["labels"] for p in parts]).astype(np.int64)
        self.parcel_ids = np.concatenate([p["parcel_ids"] for p in parts])
        spectral, rows, counts, base = [], [], [], 0
        for part in parts:
            off, clear = part["offsets"].astype(np.int64), part["clear"]
            spectral.append(part["spectral"])
            n_all = np.diff(off)
            keep = np.ones(len(clear), dtype=bool)
            n_keep = n_all
            if dates == "clear":
                cum = np.concatenate([[0], np.cumsum(clear, dtype=np.int64)])
                n_clear = cum[off[1:]] - cum[off[:-1]]
                has_clear = n_clear > 0
                # clear rows of parcels with any clear date; every row of the others
                keep = clear | ~np.repeat(has_clear, n_all)
                n_keep = np.where(has_clear, n_clear, n_all)
            rows.append(np.flatnonzero(keep) + base)
            counts.append(n_keep)
            base += len(part["spectral"])
        self.spectral = np.concatenate(spectral)
        self.rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        self.n_dates = np.concatenate(counts) if counts else np.zeros(0, np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.n_dates)]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, idx: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        x = np.zeros((len(idx), SEQ_LEN, CHANNELS), dtype=np.float32)
        for b, i in enumerate(idx):
            n = self.n_dates[i]
            if n:
                sel = self.rows[self.offsets[i] + resample_indices(int(n), SEQ_LEN, method="random", rng=rng)]
                x[b] = self.spectral[sel]
        if self.indices:
            x = np.concatenate([x, vegetation_indices(x)], axis=-1)
        return torch.from_numpy(x)


def build_model(seed: int, indices: bool = False) -> torch.nn.Module:
    torch.manual_seed(seed)
    return BreizhCropsMSResNetExpert(
        num_classes=NUM_CLASSES, input_dim=CHANNELS + (len(INDEX_NAMES) if indices else 0),
        hidden_dims=32, pretrained=False,
    )


@torch.inference_mode()
def predict(model, store: SeriesStore, *, draws: int, seed: int, batch_size: int = 1024) -> np.ndarray:
    """Mean logits over ``draws`` seeded resamples per parcel."""
    model.eval()
    rng = np.random.default_rng(seed)
    out = np.zeros((len(store), NUM_CLASSES), dtype=np.float32)
    for start in range(0, len(store), batch_size):
        idx = np.arange(start, min(start + batch_size, len(store)))
        acc = torch.zeros(len(idx), NUM_CLASSES)
        for _ in range(draws):
            acc += model(store.batch(idx, rng)).float()
        out[idx] = (acc / draws).numpy()
    return out


@torch.no_grad()
def recalibrate_batchnorm(model, store: SeriesStore, *, n: int, seed: int, batch_size: int = 512) -> None:
    """Recompute BatchNorm running statistics as a plain average over ``n`` training parcels.

    Running averages taken while the weights still move fast lag behind the
    final weights (eval accuracy far below train-mode accuracy); one pass with
    the final weights fixes that.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average
    model.train()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(store), size=min(n, len(store)), replace=False)
    for start in range(0, len(idx), batch_size):
        model(store.batch(idx[start:start + batch_size], rng))
    for bn, momentum in zip(bns, momenta):
        bn.momentum = momentum
    model.eval()


def summarize(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    pred = logits.argmax(1)
    orch = labels == ORCHARD
    return {
        "acc": 100.0 * float((pred == labels).mean()),
        "orchard_recall": 100.0 * float((pred[orch] == ORCHARD).mean()) if orch.any() else float("nan"),
        "wheat_recall": 100.0 * float((pred[labels == 1] == 1).mean()),
    }


def class_weights(mode: str, labels: np.ndarray, *, orchard_weight: float = 10.0, cap: float = 10.0) -> Optional[np.ndarray]:
    """Per-class loss weights: none, orchards up-weighted, or capped square-root inverse frequency."""
    if mode == "none":
        return None
    w = np.ones(NUM_CLASSES)
    if mode == "orchard":
        w[ORCHARD] = orchard_weight
    elif mode == "sqrt_inv":
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        w = np.minimum(np.sqrt(counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))), cap)
    else:
        raise ValueError(mode)
    return w


def train(args) -> None:
    store = SeriesStore([args.store_dir / f"{r}.npz" for r in args.train_regions], args.dates, args.indices)
    monitor = SeriesStore([args.store_dir / f"{r}.npz" for r in args.eval_regions], args.dates, args.indices)
    print(f"dates={args.dates} train={len(store):,} (median dates {np.median(store.n_dates):.0f}) "
          f"eval={len(monitor):,}", flush=True)
    cw = class_weights(args.class_weights, store.labels, orchard_weight=args.orchard_weight, cap=args.weight_cap)
    weight = None if cw is None else torch.tensor(cw, dtype=torch.float32)
    if cw is not None:
        print(f"class weights ({args.class_weights}): {np.round(cw, 2).tolist()}", flush=True)

    model = build_model(args.seed, args.indices)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, args.epochs) / args.epochs))
    )
    rng = np.random.default_rng(args.seed)
    state_path = args.out_dir / "latest.pt"
    epoch, history = 0, []
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        epoch, history = state["epoch"], state["history"]
        if state["config"] != vars_for_config(args):
            raise SystemExit(f"{state_path} was trained with a different configuration: {state['config']}")
        print(f"Resumed after epoch {epoch}", flush=True)

    started, last_epoch_s = time.time(), None
    while epoch < args.epochs:
        if last_epoch_s and time.time() - started + last_epoch_s > 60 * args.time_budget_min:
            print(f"PAUSED after epoch {epoch} (time budget); rerun to continue.", flush=True)
            return
        t0 = time.time()
        model.train()
        order = rng.permutation(len(store))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            x, y = store.batch(idx, rng), torch.from_numpy(store.labels[idx])
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=weight)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        epoch += 1
        mon = summarize(predict(model, monitor, draws=1, seed=epoch), monitor.labels)
        history.append({"epoch": epoch, "loss": total / len(store), **mon, "seconds": time.time() - t0})
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "epoch": epoch,
                    "history": history, "config": vars_for_config(args)}, state_path)
        last_epoch_s = time.time() - t0
        print(f"epoch {epoch}/{args.epochs} loss={history[-1]['loss']:.4f} frh04 acc={mon['acc']:.2f} "
              f"orchard={mon['orchard_recall']:.2f} wheat={mon['wheat_recall']:.2f} ({last_epoch_s:.0f}s)", flush=True)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not state.get("bn_recalibrated"):
        recalibrate_batchnorm(model, store, n=args.bn_samples, seed=args.seed)
        mon = summarize(predict(model, monitor, draws=1, seed=0), monitor.labels)
        print(f"after BatchNorm recalibration: frh04 acc={mon['acc']:.2f} orchard={mon['orchard_recall']:.2f} "
              f"wheat={mon['wheat_recall']:.2f}", flush=True)
        state.update(model=model.state_dict(), bn_recalibrated=True, final_monitor=mon)
        torch.save(state, state_path)
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print("TRAINING DONE", flush=True)


def export(args) -> None:
    state = torch.load(args.out_dir / "latest.pt", map_location="cpu", weights_only=False)
    if state["epoch"] < args.epochs or not state.get("bn_recalibrated"):
        raise SystemExit(f"Only {state['epoch']}/{args.epochs} epochs trained; finish training first.")
    model = build_model(args.seed, args.indices)
    model.load_state_dict(state["model"])
    regions = args.train_regions if args.export == "train" else args.eval_regions
    store = SeriesStore([args.store_dir / f"{r}.npz" for r in regions], args.dates, args.indices)
    t0 = time.time()
    logits = predict(model, store, draws=args.draws, seed=args.seed)
    print(f"{args.export}: {summarize(logits, store.labels)} TTA{args.draws} ({time.time() - t0:.0f}s)", flush=True)
    parcel_ids, labels = store.parcel_ids, store.labels
    if args.align_cache:
        cache = torch.load(args.align_cache, map_location="cpu", weights_only=True)
        aligned, matched = align_logits_by_parcel_id(
            torch.from_numpy(logits), torch.from_numpy(store.labels), torch.from_numpy(store.parcel_ids),
            cache["parcel_ids"], cache["labels"],
        )
        logits = aligned.numpy()
        logits[~matched.numpy()] = np.nan  # parcel absent from this store: missing, not neutral
        parcel_ids, labels = cache["parcel_ids"].numpy(), cache["labels"].numpy()
        print(f"aligned to {args.align_cache}: {int(matched.sum()):,}/{len(matched):,} rows matched", flush=True)
    out = args.out_dir / f"logits_{args.export_tag or args.export}.npz"
    extra = {}
    if args.class_weights != "none":  # recorded so the training-weight bias can be removed later
        train_labels = SeriesStore([args.store_dir / f"{r}.npz" for r in args.train_regions], args.dates, args.indices).labels
        extra["class_weights"] = class_weights(args.class_weights, train_labels, orchard_weight=args.orchard_weight,
                                               cap=args.weight_cap)
    np.savez(out, parcel_ids=parcel_ids, labels=labels, features=logits,
             columns=np.array([f"logit_{c}" for c in range(NUM_CLASSES)]), **extra)
    print(f"-> {out}", flush=True)


def vars_for_config(args) -> dict:
    keys = ("dates", "train_regions", "epochs", "lr", "weight_decay", "batch_size", "seed")
    cfg = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in vars(args).items() if k in keys}
    if getattr(args, "indices", False):  # only when set, so earlier checkpoints keep their config
        cfg["indices"] = True
    if getattr(args, "class_weights", "none") != "none":
        cfg["class_weights"] = args.class_weights
        cfg["orchard_weight" if args.class_weights == "orchard" else "weight_cap"] = (
            args.orchard_weight if args.class_weights == "orchard" else args.weight_cap)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dates", choices=("all", "clear"), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--store-dir", type=Path, default=Path("artifacts/series_store"))
    p.add_argument("--train-regions", nargs="+", default=["frh01", "frh02", "frh03"])
    p.add_argument("--eval-regions", nargs="+", default=["frh04"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=6.27e-4, help="Pretrained BreizhCrops MSResNet setting.")
    p.add_argument("--weight-decay", type=float, default=4.75e-6)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget-min", type=float, default=24.0)
    p.add_argument("--export", choices=("train", "eval"), default=None)
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--bn-samples", type=int, default=60000, help="Parcels for the BatchNorm recalibration pass.")
    p.add_argument("--threads", type=int, default=0, help="Torch CPU threads (0 = default).")
    p.add_argument("--export-tag", default="", help="Write logits_<tag>.npz instead of logits_<split>.npz.")
    p.add_argument("--indices", action="store_true",
                   help="Append 15 vegetation indices per date to the 10 bands (25 input channels).")
    p.add_argument("--align-cache", type=Path, default=None,
                   help="Export rows in this cache's parcel order (NaN where the store lacks a parcel).")
    p.add_argument("--class-weights", choices=("none", "orchard", "sqrt_inv"), default="none",
                   help="Weighted cross-entropy: orchards x --orchard-weight, or sqrt inverse frequency capped at --weight-cap.")
    p.add_argument("--orchard-weight", type=float, default=10.0)
    p.add_argument("--weight-cap", type=float, default=10.0)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    export(args) if args.export else train(args)


if __name__ == "__main__":
    main()
