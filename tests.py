import unittest
import torch
from torch import nn
from batchnorm import MyBatchNorm1d, MyBatchNorm2d


class MyTestCase(unittest.TestCase):
    def test_bn1d(self):
        batch_norm_torch = nn.BatchNorm1d(50).double()
        batch_norm_mine = MyBatchNorm1d(50).double()

        batch_size = 16

        for i in range(10):
            inp_tensor = (torch.rand(batch_size, 50) * torch.rand(batch_size, 50) * 5 + torch.rand(batch_size, 50) * 10).double()

            left = batch_norm_torch(inp_tensor)
            right = batch_norm_mine(inp_tensor)

            self.assertTrue(torch.isclose(torch.norm(left - right), torch.tensor(0.0).double()))

        batch_norm_mine.eval()
        batch_norm_torch.eval()

        inp_tensor = (torch.rand(batch_size, 50) * torch.rand(batch_size, 50) * 5 + torch.rand(batch_size, 50) * 10).double()
        left = batch_norm_torch(inp_tensor)
        right = batch_norm_mine(inp_tensor)
        # print(torch.norm(left - right))
        self.assertTrue(torch.isclose(torch.norm(left - right), torch.tensor(0.0).double())) # assertion error here for some reason. Maybe it's implementation differences, or just precision.

    def test_bn2d(self):
        batch_size = 4
        channels = 3
        width = 32
        height = 64

        batch_norm_torch = nn.BatchNorm2d(channels).double()
        batch_norm_mine = MyBatchNorm2d(channels).double()

        for i in range(10):
            inp_tensor = (torch.rand(batch_size, channels, width, height) * torch.rand(batch_size, channels, width, height) * 5 + torch.rand(batch_size, channels, width, height) * 10).double()
            left = batch_norm_torch(inp_tensor)
            right = batch_norm_mine(inp_tensor)
            self.assertTrue(torch.isclose(torch.norm(left - right), torch.tensor(0.0).double()))

        batch_norm_mine.eval()
        batch_norm_torch.eval()

        inp_tensor = (torch.rand(batch_size, channels, width, height) * torch.rand(batch_size, channels, width, height) * 5 + torch.rand(batch_size, channels, width, height) * 10).double()
        left = batch_norm_torch(inp_tensor)
        right = batch_norm_mine(inp_tensor)
        # print(torch.norm(left - right))
        self.assertTrue(torch.isclose(torch.norm(left - right), torch.tensor(0.0).double())) # assertion error here for some reason. Maybe it's implementation differences, or just precision.

if __name__ == '__main__':
    unittest.main()
