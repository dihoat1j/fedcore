import unittest
import torch
from fedcore.model import FederatedModel

class TestModel(unittest.TestCase):
    def test_weight_get_set(self):
        model = FederatedModel()
        weights = model.get_weights()
        
        # Modify weights
        for k in weights:
            weights[k] = torch.zeros_like(weights[k])
            
        model.set_weights(weights)
        new_weights = model.get_weights()
        
        for k in new_weights:
            self.assertTrue(torch.all(new_weights[k] == 0))

if __name__ == '__main__':
    unittest.main()
