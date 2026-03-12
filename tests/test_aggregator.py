import unittest
import torch
from fedcore.aggregator import Aggregator

class TestAggregator(unittest.TestCase):
    def test_weighted_average(self):
        w1 = {"layer.weight": torch.tensor([1.0, 1.0])}
        w2 = {"layer.weight": torch.tensor([2.0, 2.0])}
        
        # Equal weights
        result = Aggregator.aggregate([w1, w2], [10, 10])
        self.assertTrue(torch.equal(result["layer.weight"], torch.tensor([1.5, 1.5])))
        
        # Unequal weights
        result = Aggregator.aggregate([w1, w2], [10, 30])
        # (1*10 + 2*30) / 40 = 70/40 = 1.75
        self.assertTrue(torch.equal(result["layer.weight"], torch.tensor([1.75, 1.75])))

if __name__ == '__main__':
    unittest.main()
