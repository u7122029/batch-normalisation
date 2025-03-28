import torch
from torch import nn

class MyBatchNorm1d(nn.Module):
    def __init__(self, features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.features = features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

        self.running_mean = torch.zeros(features)
        self.running_var = torch.ones(features)

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, correction=0)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                self.running_var  = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()

            normalised = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            normalised = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return normalised * self.weight + self.bias

class MyBatchNorm2d(nn.Module):
    def __init__(self, features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.features = features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

        self.running_mean = torch.zeros(features)
        self.running_var = torch.ones(features)

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), correction=0, keepdim=True)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()

            normalised = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            print(self.running_mean.shape, self.running_var.shape)
            normalised = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)

        out = self.weight.view(1,-1,1,1) * normalised + self.bias.view(1,-1,1,1)
        return out

