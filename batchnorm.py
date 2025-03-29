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
            if len(x.shape) == 2:
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, correction=0)
            elif len(x.shape) == 3:
                batch_mean = x.mean(dim=(0, 2))
                batch_var = x.var(dim=(0, 2), correction=0)
            else:
                raise Exception("Invalid dimension")

            with torch.no_grad():
                m = x.numel() / x.size(1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                self.running_var  = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze() * m / (m - 1)
            if len(x.shape) == 2:
                normalised = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            elif len(x.shape) == 3:
                normalised = (x - batch_mean.view(1, -1, 1)) / torch.sqrt(batch_var.view(1, -1, 1) + self.eps)
            else:
                raise Exception("You shouldn't be here!")
        else:
            if len(x.shape) == 2:
                normalised = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
            elif len(x.shape) == 3:
                normalised = (x - self.running_mean.view(1, -1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1) + self.eps)
            else:
                raise Exception("You shouldn't be here!")

        if len(x.shape) == 2:
            return normalised * self.weight + self.bias
        else:
            return normalised * self.weight.view(1,-1,1) + self.bias.view(1,-1,1)


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
                m = x.numel() / x.size(1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze() * m / (m - 1)

            normalised = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            normalised = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)

        out = self.weight.view(1,-1,1,1) * normalised + self.bias.view(1,-1,1,1)
        return out