"""

Evaluate each epoch_*.pt in a checkpoint directory on the same test split as

train_transformer_breizhcrops.py and report which achieved highest test accuracy.

"""

import argparse

import os

import re



import torch



from agricultureai.data.breizhcrops_loaders import build_breizhcrops_loaders

from agricultureai.evaluation.transformer import MAX_SEQ_LEN

from agricultureai.models.breizhcrops_transformer import BreizhCropsTransformerClassifier

from agricultureai.training.trainer import ModelTrainer





def sorted_epoch_checkpoints(checkpoint_dir: str):

    paths = []

    for name in os.listdir(checkpoint_dir):

        m = re.fullmatch(r"epoch_(\d+)\.pt", name, flags=re.IGNORECASE)

        if m:

            paths.append((int(m.group(1)), os.path.join(checkpoint_dir, name)))

    paths.sort(key=lambda x: x[0])

    return paths





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/transformer")

    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])

    parser.add_argument("--level", default="L1C")

    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)

    parser.add_argument(

        "--no-freeze-backbone",

        action="store_true",

        help="Must match the run that produced the checkpoints (default: frozen backbone).",

    )

    parser.add_argument(

        "--from-scratch",

        action="store_true",

        help="Must match the run that produced the checkpoints (default: pretrained init).",

    )

    parser.add_argument("--target-ids", nargs="+", type=int, default=[0, 1, 2, 3])

    parser.add_argument("--max-samples", type=int, default=0, help="Must match training (0 = full).")

    args = parser.parse_args()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.abspath(args.checkpoint_dir)

    epoch_paths = sorted_epoch_checkpoints(ckpt_dir)

    if not epoch_paths:

        raise SystemExit(f"No epoch_*.pt files under {ckpt_dir!r}")



    train_loader, test_loader = build_breizhcrops_loaders(

        train_regions=args.regions,

        val_regions=[],

        level=args.level,

        target_ids=args.target_ids,

        max_samples=int(args.max_samples),

        batch_size=args.batch_size,

        max_seq_len=args.max_seq_len,

        use_train_region_holdout=True,

    )



    num_classes = len(args.target_ids)

    model = BreizhCropsTransformerClassifier(

        num_classes=num_classes,

        freeze_backbone=not args.no_freeze_backbone,

        pretrained=not args.from_scratch,

        device=device,

    )

    trainer = ModelTrainer(

        model=model,

        train_loader=train_loader,

        test_loader=test_loader,

        criterion=torch.nn.CrossEntropyLoss(),

        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1.7e-4),

        device=device,

        target_ids=args.target_ids,

    )



    results = []

    print(f"Device: {device}\nEvaluating {len(epoch_paths)} checkpoints from {ckpt_dir} ...\n")

    for epoch_num, path in epoch_paths:

        state = torch.load(path, map_location=device)

        model.load_state_dict(state, strict=True)

        test_acc = trainer.evaluate()

        results.append((epoch_num, path, test_acc))

        print(f"  epoch {epoch_num:3d}  test_acc={test_acc:.4f}%  {path}")



    best_epoch, best_path, best_acc = max(results, key=lambda x: x[2])

    print()

    print(f"Best: epoch {best_epoch}  test_acc={best_acc:.4f}%")

    print(f"File: {best_path}")





if __name__ == "__main__":

    main()


