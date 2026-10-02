"""Replace L1 logits in existing caches with DOY-aware Transformer outputs."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import breizhcrops as bzh
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from agricultureai.data.breizhcrops_series import (  # noqa: E402
    BreizhCropsParcelDataset,
    pad_collate_breizhcrops_series,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN  # noqa: E402
from evaluate_dmoe_pipeline import load_l1_generalist  # noqa: E402


def update_cache(cache_path: Path, l1: torch.nn.Module, device: torch.device) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    regions = list(meta["regions"])
    level = str(meta.get("level", "L2A"))
    parts = []
    for region in regions:
        print(f"Loading {region} ...", flush=True)
        ds = bzh.BreizhCrops(region, level=level)
        parts.append(
            BreizhCropsParcelDataset(
                ds, list(range(len(ds))), max_seq_len=MAX_SEQ_LEN, train=False, seed=0
            )
        )
    loader = DataLoader(
        ConcatDataset(parts),
        batch_size=128,
        shuffle=False,
        collate_fn=partial(pad_collate_breizhcrops_series, max_seq_len=MAX_SEQ_LEN),
        pin_memory=torch.cuda.is_available(),
    )
    chunks = []
    with torch.inference_mode():
        for x, doy, padding_mask, _targets, _ in tqdm(
            loader, desc=f"L1 DOY logits {cache_path.name}"
        ):
            x = x.to(device)
            doy = doy.to(device)
            padding_mask = padding_mask.to(device)
            chunks.append(l1(x, doy=doy, padding_mask=padding_mask).float().cpu())
    logits = torch.cat(chunks, 0)
    if logits.size(0) != payload["labels"].numel():
        raise RuntimeError(
            f"Row mismatch: logits={logits.size(0)} labels={payload['labels'].numel()}"
        )
    payload["l1_logits"] = logits
    meta = dict(meta)
    meta["l1"] = "transformer_l2a_doy"
    meta["l1_weights"] = str(
        Path("artifacts/checkpoints/transformer_l2a_doy/best.pt").resolve()
    )
    payload["meta"] = meta
    torch.save(payload, cache_path)
    print(f"Updated l1_logits -> {cache_path} shape={tuple(logits.shape)}", flush=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = Path("artifacts/checkpoints/transformer_l2a_doy/best.pt").resolve()
    l1 = load_l1_generalist(weights, device, max_seq_len=MAX_SEQ_LEN)
    for name in ("cached_logits_train.pt", "cached_logits_eval.pt"):
        update_cache(Path("artifacts") / name, l1, device)
    print("L1_DOY_UPDATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
