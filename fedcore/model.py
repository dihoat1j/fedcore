import torch
import torch.nn as nn
import torch.nn.functional as F

class FederatedModel(nn.Module):
    """
    A standard CNN architecture designed for federated learning tasks.
    Optimized for small-scale image classification (e.g., MNIST/CIFAR).
    """
    def __init__(self, input_channels=1, num_classes=10):
        super(FederatedModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def get_weights(self):
        return {k: v.cpu().clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)
