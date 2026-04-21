import torch
import collections
from breizhcrops import BreizhCrops, models
from torch.utils.data import DataLoader

# 1. We need our padding function from earlier to keep matrix dimensions uniform
MAX_SEQ_LEN = 45


def pad_collate(batch):
    xs, ys, ids = zip(*batch)
    padded_xs = []
    for x in xs:
        T, C = x.shape
        if T >= MAX_SEQ_LEN:
            padded_x = x[:MAX_SEQ_LEN, :]
        else:
            padding = torch.zeros(MAX_SEQ_LEN - T, C)
            padded_x = torch.cat([x, padding], dim=0)
        padded_xs.append(padded_x)
    return torch.stack(padded_xs), torch.tensor(ys), ids


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    print("Loading dataset and pre-trained model...")
    dataset = BreizhCrops("frh04", level="L1C")

    model = models.pretrained("Transformer")
    model.to(device)

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    model.eval()

    id_to_crop = {
        0: "barley",
        1: "wheat",
        2: "rapeseed",
        3: "corn",
        4: "sunflower",
        5: "orchards",
        6: "nuts",
        7: "permanent meadows",
        8: "temporary meadows"
    }

    target_crops = ["sunflower", "nuts"]

    target_ids = [crop_id for crop_id, crop_name in id_to_crop.items() if crop_name in target_crops]

    class_correct = collections.defaultdict(int)
    class_total = collections.defaultdict(int)

    print(f"Starting tests targeted on: {target_crops}...")

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)

            for t, p in zip(targets, predicted):
                t_val = t.item()
                if t_val in target_ids:
                    class_total[t_val] += 1
                    if t_val == p.item():
                        class_correct[t_val] += 1

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}...")


    print(f"\n--- RESULTS FOR UNDERREPRESENTED CROPS ---")

    overall_target_correct = sum(class_correct.values())
    overall_target_total = sum(class_total.values())

    for t_id in target_ids:
        crop_name = id_to_crop[int(t_id)]
        total = class_total[t_id]
        correct = class_correct[t_id]

        if total > 0:
            acc = 100. * correct / total
            print(f"- {crop_name.capitalize()}: {acc:.2f}% ({correct}/{total} fields)")
        else:
            print(f"- {crop_name.capitalize()}: No fields found in this test set.")

    # Print Combined Results for the targeted subset
    if overall_target_total > 0:
        combined_acc = 100. * overall_target_correct / overall_target_total
        print(
            f"\nCombined Targeted Accuracy: {combined_acc:.2f}% ({overall_target_correct}/{overall_target_total} fields tested)")
    else:
        print("\nNo targeted crop fields were found during testing.")

if __name__ == '__main__':
    main()