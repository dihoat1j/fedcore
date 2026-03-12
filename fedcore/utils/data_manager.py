import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

def get_mnist_partitions(num_clients=5):
    """
    Partitions MNIST dataset into N non-overlapping subsets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    indices = torch.randperm(len(full_train)).tolist()
    partition_size = len(full_train) // num_clients
    
    partitions = []
    for i in range(num_clients):
        start = i * partition_size
        end = (i + 1) * partition_size
        partitions.append(Subset(full_train, indices[start:end]))
        
    return partitions
