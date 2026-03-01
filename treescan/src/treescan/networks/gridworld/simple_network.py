"""Defines a simple network for the gridworld environment"""
import torch
from collections import OrderedDict


class SimplePolicyNetwork(torch.nn.Module):
        """A policy network for stochatstic discrete actions for the gridworld env (sorry not generic)"""

        def __init__(self, input_width, embedded_width, hidden_width, output_width):
            """Init a policy network for stochastic discrete actions"""
            super().__init__()

            self.embedding = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("embedding_layer", torch.nn.Linear(int(input_width/2),embedded_width)),
                    ]
                )
            )

            self.backbone = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("input_layer", torch.nn.Linear(embedded_width*2,hidden_width)),
                        ("nonlinear_layer", torch.nn.ReLU()),
                        ("linear_layer", torch.nn.Linear(hidden_width,output_width))
                        # ("linearlayer_1", torch.nn.Linear(hidden_width,hidden_width)),
                        # ("nonlinearlayer_2", torch.nn.Tanh()),
                    ]
                )
            )

            self.head = torch.nn.Softmax(dim=-1)

        def forward(self,X):
            if X.dim() == 1:
                 X = X.unsqueeze(0)

            agent_pos = X[:,:int(X.shape[1]/2)]
            goal_pos  = X[:,int(X.shape[1]/2):]
            agent_embedded = self.embedding(agent_pos)
            goal_embedded = self.embedding(goal_pos)
            Y = torch.cat([agent_embedded,goal_embedded],dim=1)
            h = self.backbone(Y)
            y = self.head(h)
            return y
        



class SimpleValueNetwork(torch.nn.Module):
        """A policy network for stochatstic discrete actions for the gridworld env (sorry not generic)"""

        def __init__(self, input_width, embedded_width, hidden_width):
            """Init a policy network for stochastic discrete actions"""
            super().__init__()

            self.embedding = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("embedding_layer", torch.nn.Linear(int(input_width/2),embedded_width)),
                    ]
                )
            )

            self.backbone = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("input_layer", torch.nn.Linear(embedded_width*2,hidden_width)),
                        ("nonlinear_layer", torch.nn.ReLU()),
                        ("linear_layer", torch.nn.Linear(hidden_width,hidden_width))
                        # ("linearlayer_1", torch.nn.Linear(hidden_width,hidden_width)),
                        # ("nonlinearlayer_2", torch.nn.Tanh()),
                    ]
                )
            )

            self.head = torch.nn.Linear(hidden_width,1)

        def forward(self,X):
            if X.dim() == 1:
                X = X.unsqueeze(0)

            agent_pos = X[:,:int(X.shape[1]/2)]
            goal_pos  = X[:,int(X.shape[1]/2):]
            agent_embedded = self.embedding(agent_pos)
            goal_embedded = self.embedding(goal_pos)
            Y = torch.cat([agent_embedded,goal_embedded],dim=1)
            h = self.backbone(Y)
            y = self.head(h)
            return y