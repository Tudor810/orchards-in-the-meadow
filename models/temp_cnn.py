import torch.nn as nn
import torch.nn.functional as F

class TempCNN(nn.Module):
    def __init__(self, input_dim=10, num_classes=2, sequence_length=45):
        super(TempCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm1d(64)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * sequence_length, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x