import torch
from torch.utils.data import DataLoader
import logging

class Client:
    """
    Represents a local node in the federated network.
    Handles local training and weight updates.
    """
    def __init__(self, client_id, dataset, device="cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.logger = logging.getLogger(f"Client-{client_id}")

    def train(self, model, epochs=1, lr=0.01, batch_size=32):
        model.to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self.logger.info(f"Finished training. Avg Loss: {total_loss/len(loader):.4f}")
        return model.get_weights(), len(self.dataset)
