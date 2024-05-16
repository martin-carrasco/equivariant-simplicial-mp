import unittest
import torch
from torch_geometric.data import Data
from empsn_refactured import EnhancedMessagePassing, EMPSNLayer

class TestEnhancedMessagePassing(unittest.TestCase):
    def setUp(self):
        self.layer = EnhancedMessagePassing(in_channels=5, out_channels=10, aggr_func='sum')

    def test_initialization(self):
        self.assertIsInstance(self.layer, EnhancedMessagePassing)

    def test_forward(self):
        x_source = torch.randn(10, 5)
        x_target = torch.randn(10, 5)
        edge_attr = torch.randn(10, 5)
        index = (torch.arange(10), torch.arange(10))
        
        output = self.layer(x=(x_source, x_target), index=index, edge_attr=edge_attr)
        self.assertEqual(output.shape, (10, 10))  # Check output shape based on aggregation function and output channels

    def test_message_and_aggregate(self):
        x_source = torch.randn(10, 5)
        x_target = torch.randn(10, 5)
        message_output = self.layer.message(x_source, x_target)
        self.assertEqual(message_output.shape, (10, 10))  # Source + Target concatenated
        
        aggregated_output = self.layer.aggregate(message_output.unsqueeze(0))
        self.assertEqual(aggregated_output.shape, (1, 10))  # Aggregate along the new dimension

class TestEMPSNLayer(unittest.TestCase):
    def setUp(self):
        adjacencies = ['0_0', '0_1', '1_1', '1_2']
        self.layer = EMPSNLayer(adjacencies, max_dim=2, num_hidden=5)

    def test_initialization(self):
        self.assertIsInstance(self.layer, EMPSNLayer)
        self.assertEqual(len(self.layer.message_passing), 4)  # Should match number of adjacencies

    def test_forward(self):
        x = {
            '0': torch.randn(10, 5),
            '1': torch.randn(10, 5),
            '2': torch.randn(10, 5)
        }
        adj = {key: torch.randn(10, 5) for key in self.layer.message_passing.keys()}
        inv = {key: torch.randn(10, 5) for key in self.layer.message_passing.keys()}
        
        output = self.layer(x, adj, inv)
        self.assertTrue(all([output[dim].shape == (10, 5) for dim in x]))  # Check output shape for all dimensions

if __name__ == '__main__':
    unittest.main()
