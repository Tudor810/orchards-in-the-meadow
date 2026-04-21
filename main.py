import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import breizhcrops as bzh
from tqdm import tqdm
import os
import json

# Import from your other files
from models.temp_cnn import TempCNN
from model_trainer import ModelTrainer

MAX_SEQ_LEN = 45

def pad_collate(batch):
    """Only handles sequence padding. The Trainer handles label filtering."""
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
    print(f"Running on device: {device}")
    print(torch.__version__)
    print("Loading datasets...")
    train_dataset = bzh.BreizhCrops("frh01", level="L2A")
    test_dataset = bzh.BreizhCrops("frh04", level="L2A")


    id_to_crop = {
        0: "barley", 1: "wheat", 2: "rapeseed", 3: "corn",
        4: "sunflower", 5: "orchards", 6: "nuts",
        7: "permanent meadows", 8: "temporary meadows"
    }

    target_crops = ["sunflower", "nuts"]
    target_ids = [crop_id for crop_id, crop_name in id_to_crop.items() if crop_name in target_crops]

    # Define file names for saving the cached indices
    train_cache = "train_target_indices.json"
    test_cache = "test_target_indices.json"

    # Check if we already did this hard work in a previous run
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print("Loading cached indices... (0 seconds!)")
        with open(train_cache, "r") as f:
            train_indices = json.load(f)
        with open(test_cache, "r") as f:
            test_indices = json.load(f)
    else:
        print("Pre-filtering datasets... (This will take a minute, but we are caching it!)")

        train_indices = []
        # Wrap the range() with tqdm() to get a progress bar and ETA!
        for i in tqdm(range(len(train_dataset)), desc="Filtering Train Data"):
            _, y, _ = train_dataset[i]
            if int(y) in target_ids:
                train_indices.append(i)

        test_indices = []
        for i in tqdm(range(len(test_dataset)), desc="Filtering Test Data"):
            _, y, _ = test_dataset[i]
            if int(y) in target_ids:
                test_indices.append(i)

        # Save the indices so we never have to run this loop again
        with open(train_cache, "w") as f:
            json.dump(train_indices, f)
        with open(test_cache, "w") as f:
            json.dump(test_indices, f)
        print("Indices successfully cached for future runs.")

    # 2. Create the smaller Subsets
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        collate_fn=pad_collate,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        collate_fn=pad_collate,
        pin_memory=True
    )


    # Initialize model using the imported TempCNN class
    model = TempCNN(input_dim=10, num_classes=len(target_ids), sequence_length=MAX_SEQ_LEN).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    # Initialize trainer using the imported ModelTrainer class
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        target_ids=target_ids
    )

    trainer.fit(epochs=15, save_path="./models/tempCNN_minority_crops.pth")


if __name__ == '__main__':
    main()