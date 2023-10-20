"""
Copyright 2021 Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List
import pyro.distributions as dist
import slingpy as sp
import torch
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample
from genedisco.active_learning_methods.batchbald_redux import consistent_mc_dropout


class MLP(torch.nn.Module, sp.ArgumentDictionary):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size * 2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = x.float()
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)
        output = self.fc3(relu2)
        return output


class BayesianMLP(consistent_mc_dropout.BayesianModule, sp.ArgumentDictionary):
    def __init__(self, input_size: int = 808, hidden_size: int = 32):
        super(BayesianMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc1_drop = consistent_mc_dropout.ConsistentMCDropout2d()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)

    def mc_forward_impl(self, x: torch.Tensor, return_embedding=False) -> List[torch.Tensor]:
        x = x.float()
        emb = self.fc1(x)
        x = F.relu(self.fc1_drop(emb))
        x = self.fc2(x)
        if not self.training:
            x = x[:, 0]  # TODO: maybe there is a better fix to comply with the expected dimension during evaluation
        if return_embedding:
            return [x, emb]
        else:
            return [x]


class FullBayesianMLP(PyroModule):
    def __init__(self, input_size=808, hidden_size=32):
        super(FullBayesianMLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define first linear layer with weight and bias priors
        self.fc1 = PyroModule[torch.nn.Linear](self.input_size, self.hidden_size)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([self.hidden_size, self.input_size]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 10.).expand([self.hidden_size]).to_event(1))

        # Define second linear layer with weight and bias priors
        self.fc2 = PyroModule[torch.nn.Linear](self.hidden_size, 1)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([1, self.hidden_size]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

    def forward(self, x, return_embedding=False):
        x = x.float()
        emb = self.fc1(x)
        x = F.relu(emb)
        x = self.fc2(x)
        if not self.training:
            x = x[:, 0]
        if return_embedding:
            return x, emb
        else:
            return x
