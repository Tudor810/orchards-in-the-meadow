import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import breizhcrops as bzh


MAX_SEQ_LEN = 45


def pad_collate(batch):
    xs, ys, ids = zip(*batch)
    padded_xs = []
    for x in xs:
        t, c = x.shape
        if t >= MAX_SEQ_LEN:
            padded_x = x[:MAX_SEQ_LEN, :]
        else:
            padding = torch.zeros(MAX_SEQ_LEN - t, c)
            padded_x = torch.cat([x, padding], dim=0)
        padded_xs.append(padded_x)
    return torch.stack(padded_xs), torch.tensor(ys).long(), ids


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def evaluate_metrics(model, loader, device, num_classes=9):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)  # rows=true, cols=pred

    with torch.inference_mode():
        for inputs, targets, _ in tqdm(loader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device).long()

            outputs = model(inputs)  # pretrained models return log-probabilities
            preds = outputs.argmax(dim=1)

            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[int(t), int(p)] += 1

    # Per-class counts
    support = confusion.sum(dim=1)  # true count per class
    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    accuracy_per_class = [
        100.0 * _safe_div(int(tp[c].item()), int(support[c].item())) for c in range(num_classes)
    ]
    precision_per_class = [
        100.0 * _safe_div(int(tp[c].item()), int((tp[c] + fp[c]).item())) for c in range(num_classes)
    ]
    recall_per_class = [
        100.0 * _safe_div(int(tp[c].item()), int((tp[c] + fn[c]).item())) for c in range(num_classes)
    ]
    f1_per_class = []
    for c in range(num_classes):
        p = precision_per_class[c] / 100.0
        r = recall_per_class[c] / 100.0
        f1_per_class.append(100.0 * _safe_div(2 * p * r, p + r))

    total = int(support.sum().item())
    correct = int(tp.sum().item())
    overall_acc = 100.0 * _safe_div(correct, total)

    # Macro / weighted averages (precision/recall/f1)
    macro_precision = sum(precision_per_class) / num_classes
    macro_recall = sum(recall_per_class) / num_classes
    macro_f1 = sum(f1_per_class) / num_classes

    weights = [int(s.item()) for s in support]
    weighted_precision = _safe_div(sum(precision_per_class[i] * weights[i] for i in range(num_classes)), sum(weights))
    weighted_recall = _safe_div(sum(recall_per_class[i] * weights[i] for i in range(num_classes)), sum(weights))
    weighted_f1 = _safe_div(sum(f1_per_class[i] * weights[i] for i in range(num_classes)), sum(weights))

    metrics = {
        "overall_accuracy": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    per_class = {
        "support": [int(s.item()) for s in support],
        "accuracy": accuracy_per_class,
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1": f1_per_class,
    }

    return metrics, per_class, confusion.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="frh04")
    parser.add_argument("--level", default="L1C", help="Use L1C for pretrained models (13 features).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    id_to_crop = {
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

    print(f"Loading dataset {args.region} level={args.level} ...")
    dataset = bzh.BreizhCrops(args.region, level=args.level)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # Only run requested models
    pretrained_args = {
        "MSResNet": "MSResNet",
        "Transformer": "Transformer",
    }

    per_class_rows = {}
    summary_rows = {}
    confusion_matrices = {}

    for display_name, pretrained_name in pretrained_args.items():
        print(f"\n=== {display_name} ===")
        model = bzh.models.pretrained(pretrained_name, device=device)
        metrics, per_class, confusion = evaluate_metrics(model, loader, device=device, num_classes=9)
        confusion_matrices[display_name] = confusion

        summary_rows[display_name] = metrics

        row = {}
        for class_id in range(9):
            prefix = f"{class_id} {id_to_crop[class_id]}"
            row[f"{prefix} support"] = per_class["support"][class_id]
            row[f"{prefix} accuracy"] = per_class["accuracy"][class_id]
            row[f"{prefix} precision"] = per_class["precision"][class_id]
            row[f"{prefix} recall"] = per_class["recall"][class_id]
            row[f"{prefix} f1"] = per_class["f1"][class_id]
        per_class_rows[display_name] = row

        print(
            "Summary: "
            f"acc={metrics['overall_accuracy']:.2f}% "
            f"macro_f1={metrics['macro_f1']:.2f}% "
            f"weighted_f1={metrics['weighted_f1']:.2f}%"
        )

    df_per_class = pd.DataFrame.from_dict(per_class_rows, orient="index")
    df_per_class.index.name = "model"

    df_summary = pd.DataFrame.from_dict(summary_rows, orient="index")
    df_summary.index.name = "model"

    out_path = args.output.strip()
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path("Results") / f"pretrained_metrics_{args.region}_{args.level}_{ts}.xlsx")

    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting Excel to: {out_path}")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary")
        df_per_class.to_excel(writer, sheet_name="per_class_metrics")

        # Confusion matrices: one sheet per model
        for model_name, cm in confusion_matrices.items():
            cols = [f"{i} {id_to_crop[i]}" for i in range(9)]
            cm_df = pd.DataFrame(cm.numpy(), index=cols, columns=cols)
            cm_df.index.name = "true\\pred"
            sheet = f"cm_{model_name}"
            # Excel sheet names max 31 chars
            cm_df.to_excel(writer, sheet_name=sheet[:31])

    print("Done.")


if __name__ == "__main__":
    main()

