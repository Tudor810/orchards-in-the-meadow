import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import breizhcrops as bzh

from agricultureai.data.breizhcrops_super_classes import map_raw_to_super_class

MAX_SEQ_LEN = 45


def pad_collate(batch, map_to_super_class: bool = True, max_seq_len: int = MAX_SEQ_LEN):
    """Pads sequences to max_seq_len (T x C -> max_seq_len x C)."""
    xs, ys, ids = zip(*batch)
    padded_xs = []
    for x in xs:
        t, c = x.shape
        if t >= max_seq_len:
            padded_x = x[:max_seq_len, :]
        else:
            padding = torch.zeros(max_seq_len - t, c)
            padded_x = torch.cat([x, padding], dim=0)
        padded_xs.append(padded_x)
    if map_to_super_class:
        ys = [map_raw_to_super_class(y) for y in ys]
    return torch.stack(padded_xs), torch.tensor(ys).long(), ids


def evaluate_per_class(model, loader, device, id_to_crop, num_classes=4):
    model.eval()
    total = 0
    correct = 0

    correct_per_class = torch.zeros(num_classes, dtype=torch.long)
    total_per_class = torch.zeros(num_classes, dtype=torch.long)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets, _ = batch[:3]
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            for c in range(num_classes):
                mask = targets == c
                if mask.any():
                    total_per_class[c] += mask.sum().item()
                    correct_per_class[c] += (preds[mask] == c).sum().item()

            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[int(t), int(p)] += 1

            total += targets.numel()
            correct += (preds == targets).sum().item()

    overall_acc = 100.0 * correct / total if total else 0.0
    print(f"\nOverall accuracy: {overall_acc:.2f}% ({correct}/{total})\n")

    print("Per-class accuracy:")
    for c in range(num_classes):
        name = id_to_crop.get(c, str(c))
        denom = int(total_per_class[c].item())
        numer = int(correct_per_class[c].item())
        acc = 100.0 * numer / denom if denom else 0.0
        print(f"  - class {c:>2} ({name:<20}): {acc:6.2f}% ({numer}/{denom})")

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion.cpu().numpy())


def main(region: str = "frh04", level: str = "L1C", batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(torch.__version__)
    print("Loading datasets...")

    # Pretrained BreizhCrops models are trained with 13 input features (level="L1C").
    # Using level="L2A" yields 10 features and will crash in the first Linear layer.
    test_dataset = bzh.BreizhCrops(region, level=level)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: pad_collate(batch, map_to_super_class=False),
        pin_memory=torch.cuda.is_available(),
    )

    model = bzh.models.pretrained("Transformer", device=device)
    evaluate_per_class(model, test_loader, device=device, id_to_crop=id_to_crop, num_classes=9)


if __name__ == "__main__":
    main()

