import torch
import torch.nn as nn
import numpy as np

from utils.nn import SelectiveOutput

################################################################
# Feed-froward NN to predict conditional probabilities         #
################################################################

class NeuralBN(nn.Module):
    """
    NN with arbitrary selection of evidence variables and complementary target variables
    """
    def __init__(self, vars, hidden_dims):
        """
        vars: list indicating number of classes per var, length is equal to number of vars, total sum is equal to input dim (n)
        hidden_dims: dimensions of hidden layers, not including input & output layer (both have dimension n)
        """
        super(NeuralBN, self).__init__()

        # generate class mask which groups class activations belonging to same var together (for easy normalisation using softmax)
        self.n = sum(vars)
        class_mask = []
        for i in range(len(vars)):
            for _ in range(vars[i]):
                class_mask.append(sum(vars[:i])*[0] + vars[i]*[1] + sum(vars[i+1:])*[0])
        self.class_mask = torch.tensor(class_mask, dtype=torch.int32)

        self.hidden_dims = hidden_dims

        # initialization vector, size of last hidden layer
        h0_init = 0.2 * torch.rand([self.hidden_dims[-1]], dtype=torch.float64) - 0.1  # random between -0.1 and 0.1
        self.h0 = nn.parameter.Parameter(data=h0_init, requires_grad=True)  # to initialize target probabilities

        # simple feed-forward NN, with adapted output layer 
        self.i2h = nn.Linear(self.n, self.hidden_dims[0], bias=False, dtype=torch.float64)
        self.h2h = nn.ModuleList([])
        for i in range(len(self.hidden_dims)-1):
            layer = nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1], bias=True, dtype=torch.float64)
            self.h2h.append(layer)
        self.activation = nn.ReLU()
        self.h2o = SelectiveOutput(self.hidden_dims[-1], self.n, self.class_mask, bias=True, dtype=torch.float64)

    def forward(self, inputs, mask):
        """
        inputs: (batch, n) one-hot vector encoding of observed values for each var
        mask: (batch, n) integer mask with 0 at positions of evidence vars and 1 at positions of target vars

        Returns: (batch, n) output probability at all target positions, normalised per var (inputs are copied at evidence positions)
        """

        # TODO: effective differences with original implementation when max_steps = 1
        # - ReLU activation instead of Tanh (Tanh not great for feedforward networks)
        # - x = self.i2h(x) instead of x = self.i2h(x) + self.h0 (does not make sense in a feedforward model)
        # don't expect these to lead to qualitative changes in output but needs to be checked (test with 1 actual hidden layer, so dims go 14, 50, 50, 14)
        
        # intialization vector is obtained by transforming a vector of size hidden_dims[-1] into dim n (normalized per var) using the hidden-to-output layer
        # internally replaces values of targets in input by values from initialization vector
        x = self.h2o(self.h0, inputs, mask)

        x = self.i2h(x) # original implementation: self.i2h(x) + self.h0
        for hid in self.h2h:
            x = self.activation(hid(x))
        x = self.h2o(x, inputs, mask) # includes internal softmax activation over targets 

        return x

    def nparams(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])