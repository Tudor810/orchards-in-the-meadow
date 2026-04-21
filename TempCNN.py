import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import breizhcrops as bzh
import time


# ==========================================
# 1. TEMPCNN MODEL DEFINITION
# ==========================================
class TempCNN(nn.Module):
    def __init__(self, input_dim=10, num_classes=9, sequence_length=45):
        super(TempCNN, self).__init__()
        # Convolutions
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm1d(64)

        # Regularization & Classification
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * sequence_length, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # PyTorch Conv1d expects (Batch, Channels, Sequence_Length)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten for the dense layer
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 2. DATA HANDLING & PADDING
# ==========================================
MAX_SEQ_LEN = 45  # Force all fields to have exactly 45 time steps


def pad_collate(batch):
    """
    BreizhCrops time series have variable lengths.
    This function pads shorter sequences with zeros and truncates longer ones.
    """
    xs, ys, ids = zip(*batch)
    padded_xs = []

    for x in xs:
        T, C = x.shape
        if T >= MAX_SEQ_LEN:
            padded_x = x[:MAX_SEQ_LEN, :]  # Truncate
        else:
            padding = torch.zeros(MAX_SEQ_LEN - T, C)
            padded_x = torch.cat([x, padding], dim=0)  # Pad
        padded_xs.append(padded_x)

    return torch.stack(padded_xs), torch.tensor(ys), ids


# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
def main():
    # Use GPU if available for faster training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    print("Loading datasets (this may take a moment to download on first run)...")
    # Using L2A (10 bands). frh01 for train, frh04 for test.
    train_dataset = bzh.BreizhCrops("frh01", level="L2A")
    test_dataset = bzh.BreizhCrops("frh04", level="L2A")

    print("Finish loading!")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)

    # Initialize Model, Optimizer, and Loss Function
    model = TempCNN(input_dim=10, num_classes=len(train_dataset.classes), sequence_length=MAX_SEQ_LEN)
    model = model.to(device)

    # Matching the paper's optimization setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 5  # Set low for a quick test run. Increase to 20-50 for actual results.

    for epoch in range(EPOCHS):
        start_time = time.time()

        print(f"Starting epoch {epoch + 1}")
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, targets, _) in enumerate(train_loader):

            if batch_idx % 10 == 0:
                print(f"Processed batch: {batch_idx}")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

        train_acc = 100. * correct_train / total_train

        # --- EVALUATION PHASE ---
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_test += targets.size(0)
                correct_test += predicted.eq(targets).sum().item()

        test_acc = 100. * correct_test / total_test
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch + 1}/{EPOCHS}] "
              f"Time: {epoch_time:.0f}s | "
              f"Train Loss: {total_train_loss / len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")


    torch.save(model.state_dict(), "./models/tempCNN.pth")


if __name__ == '__main__':
    main()