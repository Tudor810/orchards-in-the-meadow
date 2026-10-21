"""Stage-4 integration eval: run the full DMOE pipeline on held-out frh04.

Loads the five trained checkpoints (L1 generalist, L2a cereal, L2b anomaly gate,
L2c meadow expert, L3 prototypical), locks the L3 prototype coordinates from the
training regions,
chains everything with ``DMOEPipeline``, streams ``frh04`` through it, and
reports a global 9-class confusion matrix plus the headline integration
metrics.

L2a and L3 were trained on the stock BreizhCrops transform (45-step resampling,
no zero padding). By default (``--expert-sampling tta``) they get 8 seeded
resampled views per parcel, averaged, and the L3 prototypes are locked from the
same kind of views; ``legacy`` feeds them the zero-padded series as before.

Example
-------
    python scripts/cascade/evaluate_dmoe_pipeline.py \
        --l1-weights   artifacts/checkpoints/transformer/best.pt \
        --l2a-weights  artifacts/checkpoints/expert_cereals/best.pt \
        --l2b-weights  artifacts/checkpoints/l2b_anomaly_gate/best.pt \
        --l2c-weights  artifacts/checkpoints/l2c_meadow_expert/best.pt \
        --l3-weights   artifacts/checkpoints/level3_protonet/best.pt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import breizhcrops as bzh

from agricultureai.data.breizhcrops_series import (
    BreizhCropsParcelDataset,
    BreizhCropsParcelViewsDataset,
    pad_collate_breizhcrops_series,
    pad_collate_with_views,
)
from agricultureai.data.episodic_sampler import (
    LEVEL3_RAW_IDS,
    NUM_LEVEL3_CLASSES,
    Level3FewShotDataset,
    build_reference_support_pool,
    build_resampled_support_pool,
)
from agricultureai.evaluation.confusion_metrics import summarize_confusion, safe_div
from agricultureai.evaluation.transformer import MAX_SEQ_LEN
from agricultureai.models.breizhcrops_gltae import BreizhCropsGLTAEExpert
from agricultureai.models.breizhcrops_ltae import BreizhCropsLTAEExpert
from agricultureai.models.breizhcrops_msresnet import BreizhCropsMSResNetExpert
from agricultureai.models.breizhcrops_transformer import BreizhCropsTransformerClassifier
from agricultureai.models.proto_net import PrototypicalNetwork
from agricultureai.pipeline import (
    GLOBAL_BARLEY,
    GLOBAL_NUTS,
    GLOBAL_ORCHARDS,
    GLOBAL_SUNFLOWER,
    GLOBAL_WHEAT,
    NUM_GLOBAL_CLASSES,
    DMOEPipeline,
)
from agricultureai.training.checkpoints import (
    architecture_config_from_checkpoint,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)

GLOBAL_ID_TO_NAME: Dict[int, str] = {
    0: "barley",
    1: "wheat",
    2: "rapeseed",
    3: "corn",
    4: "sunflower",
    5: "orchards",
    6: "nuts",
    7: "permanent meadows",
    8: "temporary meadows",
}

# True anomaly raw ids the rescuer is supposed to recover.
TRUE_ANOMALY_GLOBAL_IDS = (GLOBAL_SUNFLOWER, GLOBAL_ORCHARDS, GLOBAL_NUTS)  # 4, 5, 6
CEREAL_GLOBAL_IDS = (GLOBAL_BARLEY, GLOBAL_WHEAT)  # 0, 1


def _config_value(config: dict, key: str, fallback):
    value = config.get(key, fallback)
    return fallback if value is None else value


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
def load_l1_generalist(path: Path, device: torch.device, *, max_seq_len: int) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    model_type = str(_config_value(cfg, "model_type", "transformer")).lower()
    num_classes = int(_config_value(cfg, "num_classes", 4))
    print(f"L1 generalist: model_type={model_type} num_classes={num_classes}")

    if model_type != "transformer":
        raise ValueError(
            f"Unsupported L1 model_type={model_type!r}; expected 'transformer'."
        )
    input_dim = int(_config_value(cfg, "input_dim", cfg.get("channels", 13)))
    pretrained = bool(_config_value(cfg, "pretrained", input_dim == 13))
    use_doy = bool(_config_value(cfg, "use_doy", False))
    model = BreizhCropsTransformerClassifier(
        num_classes=num_classes,
        input_dim=input_dim,
        freeze_backbone=True,
        pretrained=pretrained,
        use_doy=use_doy,
        device=device,
    )
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    print(f"  input_dim={input_dim} pretrained={pretrained} use_doy={use_doy}")
    return model


def load_l2a_cereal(path: Path, device: torch.device, *, max_seq_len: int) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    seq_len = int(_config_value(cfg, "seq_len", max_seq_len))
    pretrained = bool(_config_value(cfg, "pretrained", True))
    model = BreizhCropsLTAEExpert(
        num_classes=2,
        sequencelength=seq_len,
        pretrained=pretrained,
        device=device,
    )
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    print(f"L2a cereal: LTAE num_classes=2 seq_len={seq_len} pretrained={pretrained}")
    return model


def load_l2b_anomaly_gate(path: Path, device: torch.device, *, max_seq_len: int) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    seq_len = int(_config_value(cfg, "seq_len", max_seq_len))
    num_classes = int(_config_value(cfg, "num_classes", 2))
    model = BreizhCropsGLTAEExpert(
        num_classes=num_classes,
        input_dim=int(_config_value(cfg, "input_dim", 13)),
        sequencelength=seq_len,
        d_model=int(_config_value(cfg, "d_model", 128)),
        n_heads=int(_config_value(cfg, "n_heads", 8)),
        n_layers=int(_config_value(cfg, "n_layers", 2)),
        dim_feedforward=int(_config_value(cfg, "dim_feedforward", 256)),
        conv_kernel_size=int(_config_value(cfg, "conv_kernel", 5)),
        dropout=float(_config_value(cfg, "dropout", 0.2)),
        device=device,
    )
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    print(f"L2b anomaly gate: gltae num_classes={num_classes} seq_len={seq_len}")
    return model


def load_l2c_meadow(path: Path, device: torch.device, *, channels: int) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    num_classes = int(_config_value(cfg, "num_classes", 2))
    use_doy = bool(_config_value(cfg, "use_doy", cfg.get("doy_positional_encoding", False)))
    model = BreizhCropsMSResNetExpert(
        num_classes=num_classes,
        input_dim=int(_config_value(cfg, "input_dim", channels)),
        hidden_dims=int(_config_value(cfg, "hidden_dims", 32)),
        pretrained=False,
        use_doy=use_doy,
        device=device,
    )
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    print(f"L2c meadow expert: msresnet num_classes={num_classes} use_doy={use_doy}")
    return model


def load_l3_protonet(
    path: Path, device: torch.device, *, channels: int
) -> Tuple[PrototypicalNetwork, int]:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    hidden_dim = int(_config_value(cfg, "hidden_dim", 64))
    encoder = str(_config_value(cfg, "encoder", "cnn")).lower()
    in_channels = int(_config_value(cfg, "in_channels", channels))
    model = PrototypicalNetwork(
        in_channels=in_channels, hidden_dim=hidden_dim, encoder=encoder, device=device
    )
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    print(f"L3 protonet: encoder={encoder} hidden_dim={hidden_dim} in_channels={in_channels}")
    return model, hidden_dim


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def report_global_metrics(confusion: torch.Tensor, l3_total: int):
    """Print the global confusion matrix and the headline integration metrics.

    Returns the ``(metrics, per_class)`` pair from ``summarize_confusion`` so callers
    can forward it to the Excel writer.
    """
    n = confusion.size(0)
    metrics, per_class = summarize_confusion(confusion)
    support = confusion.sum(dim=1)
    tp = confusion.diag()
    precision = per_class["precision"]
    recall = per_class["recall"]
    f1 = per_class["f1"]

    total = int(support.sum().item())
    correct = int(tp.sum().item())
    overall_acc = metrics["overall_accuracy"]
    macro_f1 = metrics["macro_f1"]

    print("\n================= DMOE Pipeline: Global Metrics =================")
    print(f"Overall accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"Global Macro F1-Score: {macro_f1:.2f}% (over {n} classes)\n")

    print("Per-class metrics (global raw ids):")
    for c in range(n):
        print(
            f"  - {c} ({GLOBAL_ID_TO_NAME[c]:<18}): "
            f"precision={precision[c]:6.2f}%  recall={recall[c]:6.2f}%  "
            f"f1={f1[c]:6.2f}%  support={int(support[c])}"
        )

    # 1. Cereal accuracy (Wheat vs Barley): accuracy over true cereal rows.
    cereal_rows = list(CEREAL_GLOBAL_IDS)
    cereal_support = int(confusion[cereal_rows, :].sum().item())
    cereal_correct = int(sum(int(confusion[c, c]) for c in cereal_rows))
    cereal_acc = 100.0 * safe_div(cereal_correct, cereal_support)

    # 2. True-anomaly recall (orchards/nuts/sunflowers), per-class + macro.
    anomaly_rows = list(TRUE_ANOMALY_GLOBAL_IDS)
    anomaly_recalls = {c: recall[c] for c in anomaly_rows}
    macro_anomaly_recall = safe_div(sum(anomaly_recalls.values()), len(anomaly_rows))

    print("\n----------------- Headline Integration Metrics ------------------")
    print(f"1. Global Macro F1-Score        : {macro_f1:.2f}%")
    print(
        f"2. Cereal accuracy (Wheat/Barley): {cereal_acc:.2f}% "
        f"({cereal_correct}/{cereal_support})"
    )
    print(f"3. True-anomaly recall (macro)  : {macro_anomaly_recall:.2f}%")
    for c in anomaly_rows:
        print(
            f"     - {GLOBAL_ID_TO_NAME[c]:<10}: recall={anomaly_recalls[c]:6.2f}%  "
            f"support={int(support[c])}"
        )
    print(f"4. Total fields processed by L3 : {l3_total}")

    print("\nGlobal 9x9 confusion matrix (rows=true, cols=pred):")
    header = "        " + "".join(f"{c:>6}" for c in range(n))
    print(header)
    for r in range(n):
        row = "".join(f"{int(confusion[r, c]):>6}" for c in range(n))
        print(f"  t{r}: {row}")

    return metrics, per_class


def maybe_write_excel(
    confusion: torch.Tensor,
    metrics: dict,
    per_class: dict,
    *,
    output: str,
    model_name: str,
) -> None:
    """Write the same workbook layout as scripts/cascade/evaluate_pretrained_models.py.

    Sheets:
        - ``summary``: one row per model with overall/macro/weighted metrics.
        - ``per_class_metrics``: one row per model with per-class support and
          accuracy/precision/recall/f1 columns.
        - ``cm_<model_name>``: the confusion matrix with ``{id} {name}`` labels.
    """
    if not output:
        return
    try:
        import pandas as pd
    except ImportError:
        print("pandas/openpyxl not installed; skipping Excel export.")
        return

    n = confusion.size(0)

    # per_class_metrics row (mirrors evaluate_pretrained_models column naming).
    row = {}
    for class_id in range(n):
        prefix = f"{class_id} {GLOBAL_ID_TO_NAME[class_id]}"
        row[f"{prefix} support"] = per_class["support"][class_id]
        row[f"{prefix} accuracy"] = per_class["accuracy"][class_id]
        row[f"{prefix} precision"] = per_class["precision"][class_id]
        row[f"{prefix} recall"] = per_class["recall"][class_id]
        row[f"{prefix} f1"] = per_class["f1"][class_id]

    df_summary = pd.DataFrame.from_dict({model_name: metrics}, orient="index")
    df_summary.index.name = "model"
    df_per_class = pd.DataFrame.from_dict({model_name: row}, orient="index")
    df_per_class.index.name = "model"

    cols = [f"{c} {GLOBAL_ID_TO_NAME[c]}" for c in range(n)]
    cm_df = pd.DataFrame(confusion.cpu().numpy(), index=cols, columns=cols)
    cm_df.index.name = "true\\pred"

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting Excel to: {output}")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary")
        df_per_class.to_excel(writer, sheet_name="per_class_metrics")
        cm_df.to_excel(writer, sheet_name=f"cm_{model_name}"[:31])
    print("Done.")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the end-to-end 3-Tier DMOE pipeline on frh04.",
    )
    parser.add_argument("--l1-weights", required=True, help="Level-1 generalist best.pt")
    parser.add_argument("--l2a-weights", required=True, help="Level-2a cereal best.pt")
    parser.add_argument("--l2b-weights", required=True, help="Level-2b anomaly gate best.pt")
    parser.add_argument("--l2c-weights", required=True, help="Level-2c meadow expert best.pt")
    parser.add_argument("--l3-weights", required=True, help="Level-3 protonet best.pt")

    parser.add_argument("--train-regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--val-regions", nargs="+", default=["frh04"])
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit to first N samples per region (0 = full region).",
    )
    parser.add_argument(
        "--anomaly-threshold",
        type=float,
        default=0.20,
        help="Route grassland fields to L3 when P(L2b class 2) > threshold.",
    )
    parser.add_argument(
        "--ref-k-shot",
        type=int,
        default=50,
        help="Clean reference examples per class used to lock L3 prototypes.",
    )
    parser.add_argument("--t1", type=float, default=1.0, help="L1 softmax temperature.")
    parser.add_argument("--t2b", type=float, default=1.0, help="L2b anomaly gate temperature.")
    parser.add_argument("--t2c", type=float, default=1.0, help="L2c meadow expert temperature.")
    parser.add_argument("--t3", type=float, default=1.0, help="L3 distance temperature.")
    parser.add_argument(
        "--tau-route",
        type=float,
        default=0.05,
        help="Routing gate: skip expert when subtree mass <= tau.",
    )
    parser.add_argument(
        "--b-meadow",
        type=float,
        default=0.0,
        help="Log-space bias added to meadow classes {7, 8} before argmax.",
    )
    parser.add_argument(
        "--b-tree",
        type=float,
        default=0.0,
        help="Log-space bias added to anomaly classes {4, 5, 6} before argmax.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expert-sampling",
        choices=("tta", "legacy"),
        default="tta",
        help="Input for L2a / L3 (trained on the stock BreizhCrops transform): mean over "
        "--draws seeded 45-step resamples (tta) or the zero-padded series (legacy).",
    )
    parser.add_argument("--draws", type=int, default=8, help="Resampled views per parcel for tta.")
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="",
        help="Optional Excel path (default: Results/dmoe_pipeline_<ts>.xlsx).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    channels = 13 if args.level == "L1C" else 10

    # --- Load the four models ----------------------------------------------
    l1 = load_l1_generalist(Path(args.l1_weights).resolve(), device, max_seq_len=args.max_seq_len)
    l2a = load_l2a_cereal(Path(args.l2a_weights).resolve(), device, max_seq_len=args.max_seq_len)
    l2b = load_l2b_anomaly_gate(Path(args.l2b_weights).resolve(), device, max_seq_len=args.max_seq_len)
    l2c = load_l2c_meadow(Path(args.l2c_weights).resolve(), device, channels=channels)
    l3, hidden_dim = load_l3_protonet(Path(args.l3_weights).resolve(), device, channels=channels)

    # --- Lock L3 prototypes from the training regions ----------------------
    print(f"\nLocking L3 prototypes from {args.train_regions} ...")
    proto_dataset = Level3FewShotDataset(
        regions=args.train_regions,
        level=args.level,
        max_samples=int(args.max_samples),
        max_seq_len=args.max_seq_len,
    )
    tta = args.expert_sampling == "tta"
    if tta:
        support_x, support_y = build_resampled_support_pool(
            proto_dataset,
            k_shot=args.ref_k_shot,
            method="tta",
            draws=args.draws,
            sampling_seed=args.sampling_seed,
            seq_len=args.max_seq_len,
            seed=args.seed,
        )
    else:
        support_x, support_y = build_reference_support_pool(
            proto_dataset, k_shot=args.ref_k_shot, n_way=NUM_LEVEL3_CLASSES, seed=args.seed
        )
    support_prototypes, support_inv_variances = l3.prototypes_and_inv_variances_from_support(
        support_x.to(device),
        support_y.to(device),
        NUM_LEVEL3_CLASSES,
        shrinkage=0.5,
    )
    print(
        f"Locked {support_prototypes.size(0)} prototypes of dim {support_prototypes.size(1)} "
        f"(L3 local ids -> global raw ids {LEVEL3_RAW_IDS})."
    )

    # --- Assemble the pipeline ---------------------------------------------
    print(
        "Decode params: "
        f"t1={args.t1:.4f} t2b={args.t2b:.4f} t2c={args.t2c:.4f} t3={args.t3:.4f} "
        f"tau={args.tau_route:.4f} b_meadow={args.b_meadow:.4f} b_tree={args.b_tree:.4f}"
    )
    pipeline = DMOEPipeline(
        l1_model=l1,
        l2a_cereal=l2a,
        l2b_anomaly_gate=l2b,
        l2c_meadow=l2c,
        l3_protonet=l3,
        support_prototypes=support_prototypes,
        support_inv_variances=support_inv_variances,
        anomaly_threshold=args.anomaly_threshold,
        t1=float(args.t1),
        t2b=float(args.t2b),
        t2c=float(args.t2c),
        t3=float(args.t3),
        tau_route=float(args.tau_route),
        b_meadow=float(args.b_meadow),
        b_tree=float(args.b_tree),
    ).to(device)
    pipeline.eval()

    # --- frh04 validation stream (raw ids 0-8 as ground truth) -------------
    val_parts = []
    for region in args.val_regions:
        print(f"Loading validation region={region} level={args.level} (all classes) ...")
        ds = bzh.BreizhCrops(region, level=args.level)
        indices = list(range(len(ds)))
        if int(args.max_samples) > 0:
            indices = indices[:min(int(args.max_samples), len(indices))]
        if tta:
            part = BreizhCropsParcelViewsDataset(
                ds,
                indices,
                method="tta",
                draws=args.draws,
                sampling_seed=args.sampling_seed,
                max_seq_len=args.max_seq_len,
            )
        else:
            part = BreizhCropsParcelDataset(
                ds, indices, max_seq_len=args.max_seq_len, train=False, seed=args.seed
            )
        val_parts.append(part)
    from torch.utils.data import ConcatDataset

    val_dataset = ConcatDataset(val_parts)
    collate = pad_collate_with_views if tta else pad_collate_breizhcrops_series
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, max_seq_len=args.max_seq_len),
        pin_memory=torch.cuda.is_available(),
    )

    # --- Run the pipeline ---------------------------------------------------
    print(f"L2a / L3 input: {args.expert_sampling}" + (f" ({args.draws} views)" if tta else ""))
    confusion = torch.zeros((NUM_GLOBAL_CLASSES, NUM_GLOBAL_CLASSES), dtype=torch.long)
    l3_total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="DMOE pipeline inference"):
            x, doy, padding_mask, targets, _ = batch[:5]
            views = batch[5].to(device, non_blocking=True) if tta else None
            x = x.to(device, non_blocking=True)
            doy = doy.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            targets = targets.to(device).long()  # global raw ids 0-8
            preds = pipeline(x, doy=doy, padding_mask=padding_mask, expert_views=views)
            l3_total += pipeline.last_l3_count
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[int(t), int(p)] += 1

    metrics, per_class = report_global_metrics(confusion, l3_total)

    output = args.output.strip()
    if not output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(Path("Results") / f"dmoe_pipeline_{'+'.join(args.val_regions)}_{ts}.xlsx")
    model_name = (
        f"DMOE_4tier_t1{args.t1:.2f}_t2b{args.t2b:.2f}_t2c{args.t2c:.2f}_"
        f"t3{args.t3:.2f}_tau{args.tau_route:.3f}_{args.expert_sampling}"
    )
    maybe_write_excel(
        confusion,
        metrics,
        per_class,
        output=output,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
