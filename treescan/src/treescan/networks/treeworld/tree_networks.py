"""Defines a simple network for the gridworld environment"""
import torch
from collections import OrderedDict


class SimpleConvNetwork(torch.nn.Module):
    """A Conv/ReLU/Conv/ReLU/Pool/FC/FC network for logits or values"""

    def __init__(self, input_channels, output_width, 
                 hidden_channels1=16, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=32, kernel2=3, stride2=1, padding2=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 32):
        """Init a policy network for stochastic discrete actions"""
        super().__init__()

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    # Convolution
                    ("conv1", torch.nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels1, kernel_size=kernel1, stride=stride1, padding=padding1)),
                    ("act1", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(in_channels=hidden_channels1, out_channels=hidden_channels2, kernel_size=kernel2, stride=stride2, padding=padding2)),
                    ("act2", torch.nn.ReLU()),
                    
                    # Pool to fixed size
                    ("pool", torch.nn.AdaptiveAvgPool2d((poolheight,poolwidth))),

                    # Flatten
                    ("flatten", torch.nn.Flatten()),

                    # FC
                    ("fc1", torch.nn.Linear(hidden_channels2*poolheight*poolwidth,fc1_width)),
                    ("nonlinear_layer", torch.nn.ReLU()),
                    ("output", torch.nn.Linear(fc1_width,output_width))
                ]
            )
        )

    def forward(self,X):

        if X.dim() == 3:
            X = X.unsqueeze(0)

        # for name, layer in self.network.named_children():
        #     X = layer(X)
        #     print(f"{name}: {X.shape}")

        Y = self.network(X)
        return Y
    

class ResidualBlock(torch.nn.Module):
    """Implements a residual Conv layer"""
    def __init__(self,in_channels):
        """Init a residual conv layer"""
        super().__init__()
        stride=1
        kernel_size=3
        padding=1

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv3", torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                    ("act3", torch.nn.ReLU()),
                ]
            )
        )
    
    def forward(self,X):

        Y = X + self.network(X)
        return Y




class BetterConvNetwork(torch.nn.Module):
    """A Conv/ReLU/Conv/ReLU/Pool/FC/FC network for logits or values"""

    def __init__(self, input_channels, output_width, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128):
        """Init a policy network for stochastic discrete actions"""
        super().__init__()

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    # Convolution
                    ("conv1", torch.nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels1, kernel_size=kernel1, stride=stride1, padding=padding1)),
                    ("act1", torch.nn.ReLU()),

                    ("conv2", torch.nn.Conv2d(in_channels=hidden_channels1, out_channels=hidden_channels2, kernel_size=kernel2, stride=stride2, padding=padding2)),
                    ("act2", torch.nn.ReLU()),
                    ("res1", ResidualBlock(in_channels=hidden_channels2)),

                    ("conv3", torch.nn.Conv2d(in_channels=hidden_channels2, out_channels=hidden_channels3, kernel_size=kernel3, stride=stride3, padding=padding3)),
                    ("act3", torch.nn.ReLU()),
                    ("res2", ResidualBlock(in_channels=hidden_channels3)),

                    ("act4", torch.nn.ReLU()),
                    
                    # Pool to fixed size
                    ("pool", torch.nn.AdaptiveMaxPool2d((poolheight,poolwidth))),

                    # Flatten
                    ("flatten", torch.nn.Flatten()),

                    # FC
                    ("fc1", torch.nn.Linear(hidden_channels3*poolheight*poolwidth,fc1_width)),
                    ("nonlinear_layer", torch.nn.ReLU()),
                    ("output", torch.nn.Linear(fc1_width,output_width))
                ]
            )
        )

    def forward(self,X):

        if X.dim() == 3:
            X = X.unsqueeze(0)

        # for name, layer in self.network.named_children():
        #     X = layer(X)
        #     print(f"{name}: {X.shape}")

        Y = self.network(X)
        return Y
    






class ResidualBlock2(torch.nn.Module):
    """Implements a residual Conv layer"""
    def __init__(self,in_channels):
        """Init a residual conv layer"""
        super().__init__()
        stride=1
        kernel_size=3
        padding=1

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    ("conv1", torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                    ("act", torch.nn.ReLU()),
                    ("conv2", torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                ]
            )
        )

        self.act = torch.nn.ReLU()
    
    def forward(self,X):

        H = X + self.network(X)
        Y = self.act(H)
        return Y




class BetterConvNetwork2(torch.nn.Module):
    """A Conv/ReLU/Conv/ReLU/Pool/FC/FC network for logits or values"""

    def __init__(self, input_channels, output_width, 
                 hidden_channels1=64, kernel1=3, stride1=1, padding1=1,
                 hidden_channels2=96, kernel2=3, stride2=1, padding2=1,
                 hidden_channels3=128, kernel3=3, stride3=1, padding3=1,
                 poolwidth = 8, poolheight = 8,
                 fc1_width = 128):
        """Init a policy network for stochastic discrete actions"""
        super().__init__()

        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    # Convolution
                    ("conv1", torch.nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels1, kernel_size=kernel1, stride=stride1, padding=padding1)),
                    ("act1", torch.nn.ReLU()),
                    ("res1", ResidualBlock2(in_channels=hidden_channels1)),

                    ("conv2", torch.nn.Conv2d(in_channels=hidden_channels1, out_channels=hidden_channels2, kernel_size=kernel2, stride=stride2, padding=padding2)),
                    ("act2", torch.nn.ReLU()),
                    ("res2", ResidualBlock2(in_channels=hidden_channels2)),

                    ("conv3", torch.nn.Conv2d(in_channels=hidden_channels2, out_channels=hidden_channels3, kernel_size=kernel3, stride=stride3, padding=padding3)),
                    ("act3", torch.nn.ReLU()),
                    ("res3", ResidualBlock2(in_channels=hidden_channels3)),
                    
                    # Pool to fixed size
                    ("pool", torch.nn.AdaptiveAvgPool2d((poolheight,poolwidth))),

                    # Flatten
                    ("flatten", torch.nn.Flatten()),

                    # FC
                    ("fc1", torch.nn.Linear(hidden_channels3*poolheight*poolwidth,fc1_width)),
                    ("nonlinear_layer", torch.nn.ReLU()),
                    ("output", torch.nn.Linear(fc1_width,output_width))
                ]
            )
        )

    def forward(self,X):

        if X.dim() == 3:
            X = X.unsqueeze(0)

        # for name, layer in self.network.named_children():
        #     X = layer(X)
        #     print(f"{name}: {X.shape}")

        Y = self.network(X)
        return Y