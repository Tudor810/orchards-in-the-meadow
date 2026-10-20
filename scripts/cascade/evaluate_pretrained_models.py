import argparse

from datetime import datetime

from pathlib import Path



import pandas as pd

import torch

from torch.utils.data import DataLoader

from tqdm import tqdm



import breizhcrops as bzh



from agricultureai.data.breizhcrops_loaders import build_breizhcrops_region_eval_loader

from agricultureai.data.breizhcrops_super_classes import SUPER_CLASS_ID_TO_NAME

from agricultureai.evaluation.transformer import MAX_SEQ_LEN, pad_collate

from agricultureai.models.breizhcrops_transformer import BreizhCropsTransformerClassifier
from agricultureai.training.checkpoints import load_checkpoint, model_state_dict_from_checkpoint





def _safe_div(num, den):

    return float(num) / float(den) if den else 0.0





def evaluate_metrics(model, loader, device, num_classes=9):

    model.eval()

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)



    with torch.inference_mode():

        for inputs, targets, _ in tqdm(loader, desc="Evaluating", leave=False):

            inputs = inputs.to(device)

            targets = targets.to(device).long()

            outputs = model(inputs)

            preds = outputs.argmax(dim=1)



            for t, p in zip(targets.view(-1), preds.view(-1)):

                confusion[int(t), int(p)] += 1



    support = confusion.sum(dim=1)

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





def evaluate_metrics_filtered(model, loader, device, target_ids):

    """Like `evaluate_metrics`, but drops non-target rows and remaps labels to 0..C-1."""

    num_classes = len(target_ids)

    model.eval()

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)



    with torch.inference_mode():

        for inputs, targets, _ in tqdm(loader, desc="Evaluating fine-tuned Transformer", leave=False):

            inputs = inputs.to(device)

            targets = targets.to(device).long()



            mask = torch.zeros_like(targets, dtype=torch.bool)

            for target_id in target_ids:

                mask |= targets == target_id

            inputs = inputs[mask]

            targets = targets[mask]

            if targets.size(0) == 0:

                continue



            remapped_targets = torch.zeros_like(targets)

            for new_id, original_id in enumerate(target_ids):

                remapped_targets[targets == original_id] = new_id

            targets = remapped_targets



            outputs = model(inputs)

            preds = outputs.argmax(dim=1)



            for t, p in zip(targets.view(-1), preds.view(-1)):

                confusion[int(t), int(p)] += 1



    support = confusion.sum(dim=1)

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

    parser.add_argument("--level", default="L1C", help="Pretrained BreizhCrops models require L1C (13 features).")

    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--output", default="")

    parser.add_argument("--skip-baseline", action="store_true", help="Do not evaluate MSResNet / Transformer.")

    parser.add_argument(

        "--transformer-weights",

        default="",

        help="Fine-tuned Transformer state_dict (.pt). Uses same BreizhCrops split as train_transformer_breizhcrops.py.",

    )

    parser.add_argument(

        "--eval-regions",

        nargs="+",

        default=["frh01", "frh02", "frh03"],

        help="Regions for fine-tuned Transformer evaluation (must match training for comparable accuracy).",

    )

    parser.add_argument(

        "--target-ids",

        nargs="+",

        type=int,

        default=None,

        help="Super-class IDs kept for the loader (default: 0 1 2 3 like train_transformer_breizhcrops.py).",

    )

    parser.add_argument(

        "--max-samples",

        type=int,

        default=0,

        help="Subset first N samples per-region index table (0 = full, must match training).",

    )

    parser.add_argument(

        "--from-scratch",

        action="store_true",

        help="Construct Transformer without upstream pretrained init before loading checkpoint.",

    )

    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)

    parser.add_argument(

        "--full-region",

        action="store_true",

        help="Evaluate on all filtered samples in each --eval-regions (matches training val on frh04).",

    )

    args = parser.parse_args()

    target_ids = args.target_ids if args.target_ids is not None else list(range(4))

    finetuned_ckpt = (args.transformer_weights or "").strip()



    if args.skip_baseline and not finetuned_ckpt:

        raise SystemExit("Nothing to evaluate: pass --transformer-weights or remove --skip-baseline.")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")



    raw_id_to_crop = {

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

    id_to_crop = dict(SUPER_CLASS_ID_TO_NAME)



    per_class_rows = {}

    summary_rows = {}

    confusion_matrices = {}



    if not args.skip_baseline:

        print(f"Loading dataset {args.region} level={args.level} ...")

        dataset = bzh.BreizhCrops(args.region, level=args.level)

        loader = DataLoader(

            dataset,

            batch_size=args.batch_size,

            shuffle=False,

            collate_fn=lambda batch: pad_collate(batch, map_to_super_class=False),

            pin_memory=torch.cuda.is_available(),

        )



        pretrained_args = {

            "MSResNet": "MSResNet",

            "Transformer": "Transformer",

        }



        for display_name, pretrained_name in pretrained_args.items():

            print(f"\n=== {display_name} ===")

            model = bzh.models.pretrained(pretrained_name, device=device)

            metrics, per_class, confusion = evaluate_metrics(model, loader, device=device, num_classes=9)

            confusion_matrices[display_name] = confusion

            summary_rows[display_name] = metrics



            row = {}

            for class_id in range(9):

                prefix = f"{class_id} {raw_id_to_crop[class_id]}"

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



    if finetuned_ckpt:

        ckpt_path = Path(finetuned_ckpt).resolve()

        if not ckpt_path.is_file():

            raise SystemExit(f"Transformer weights not found: {ckpt_path}")



        display_name = f"Transformer_{ckpt_path.stem}"

        split_desc = "full region" if args.full_region else "80/20 test tail"

        print(f"\n=== {display_name} ({split_desc}) ===")

        print(f"Regions: {args.eval_regions}  target_ids: {target_ids}")



        eval_loader = build_breizhcrops_region_eval_loader(

            regions=args.eval_regions,

            level=args.level,

            target_ids=target_ids,

            max_samples=int(args.max_samples),

            batch_size=args.batch_size,

            max_seq_len=int(args.max_seq_len),

            full_region=bool(args.full_region),

        )



        num_classes = len(target_ids)

        model = BreizhCropsTransformerClassifier(

            num_classes=num_classes,

            freeze_backbone=True,

            pretrained=not args.from_scratch,

            device=device,

        )

        checkpoint = load_checkpoint(ckpt_path, map_location=device)

        model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)



        metrics, per_class, confusion = evaluate_metrics_filtered(

            model, eval_loader, device=device, target_ids=target_ids

        )

        confusion_matrices[display_name] = confusion

        summary_rows[display_name] = metrics



        row = {}

        for i, tid in enumerate(target_ids):

            prefix = f"{tid} {id_to_crop.get(tid, str(tid))}"

            row[f"{prefix} support"] = per_class["support"][i]

            row[f"{prefix} accuracy"] = per_class["accuracy"][i]

            row[f"{prefix} precision"] = per_class["precision"][i]

            row[f"{prefix} recall"] = per_class["recall"][i]

            row[f"{prefix} f1"] = per_class["f1"][i]

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

        tag_parts = []

        if not args.skip_baseline:

            tag_parts.append(f"baseline_{args.region}")

        if finetuned_ckpt:

            tag_parts.append(f"transformer_{Path(finetuned_ckpt).stem}")

        tag = "_".join(tag_parts) if tag_parts else "eval"

        out_path = str(Path("Results") / f"metrics_{tag}_{args.level}_{ts}.xlsx")



    Path(out_path).parent.mkdir(parents=True, exist_ok=True)



    print(f"\nWriting Excel to: {out_path}")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:

        df_summary.to_excel(writer, sheet_name="summary")

        df_per_class.to_excel(writer, sheet_name="per_class_metrics")



        for model_name, cm in confusion_matrices.items():

            if model_name.startswith("Transformer_"):

                cols = [f"{tid} {id_to_crop.get(tid, str(tid))}" for tid in target_ids]

            else:

                cols = [f"{i} {raw_id_to_crop[i]}" for i in range(9)]

            cm_df = pd.DataFrame(cm.numpy(), index=cols, columns=cols)

            cm_df.index.name = "true\\pred"

            cm_df.to_excel(writer, sheet_name=f"cm_{model_name}"[:31])



    print("Done.")





if __name__ == "__main__":

    main()


