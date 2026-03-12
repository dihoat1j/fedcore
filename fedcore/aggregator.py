import torch
import copy
from typing import List, Dict

class Aggregator:
    """
    Implements the Federated Averaging (FedAvg) algorithm.
    """
    @staticmethod
    def aggregate(client_weights: List[Dict[str, torch.Tensor]], client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        total_samples = sum(client_sizes)
        if not client_weights:
            return {}

        # Initialize global weights with zeros
        global_weights = copy.deepcopy(client_weights[0])
        for key in global_weights.keys():
            global_weights[key] = torch.zeros_like(global_weights[key])

        # Weighted average
        for i in range(len(client_weights)):
            weight = client_sizes[i] / total_samples
            for key in global_weights.keys():
                global_weights[key] += client_weights[i][key] * weight

        return global_weights
