import torch
import torch.nn as nn
import numpy as np

from utils.nn import SelectiveOutput

################################################################
# Recurrent model to predict output probabilities              #
################################################################

class BayesianRNN(nn.Module):
    """
    RNN with constant input, which should relax into output state;
    arbitrary selection of input variables and complementary output variables should be possible
    """
    def __init__(self, vars, statesizes, outsizes=[]):
        """
        vars = list indicating number of classes per var, length is equal to number of vars, total sum is equal to input + output dims (n)
        statesizes = hidden dims (list of tuples indicating the dimensions of the hidden layer(s))
        outsizes = output layer dims (list of tuples indicating the dimensions of the output layer(s))
        """
        super(BayesianRNN, self).__init__()

        self.n = sum(vars)
        class_mask = []
        for i in range(len(vars)):
            for _ in range(vars[i]):
                class_mask.append(sum(vars[:i])*[0] + vars[i]*[1] + sum(vars[i+1:])*[0])
        self.class_mask = torch.tensor(class_mask, dtype=torch.int32)

        if statesizes[0][0] != statesizes[-1][1]:
            print("statesizes incompatible with recurrent model!")
        self.statesizes = statesizes
        if outsizes[-1][1] != self.n or outsizes[0][0] != statesizes[-1][1]:
            print("outsizes incompatible with input size or hidden state size!")
        self.outsizes = outsizes

        h0_init = 0.2 * torch.rand([self.statesizes[0][0]], dtype=torch.float64) - 0.1  #random between -0.1 and 0.1
        self.h0 = nn.parameter.Parameter(data=h0_init, requires_grad=True)  #trainable initial hidden state allows to initialize target probabilities

        #elements of simple Elman RNN (to be extended; more hidden layers, gating, etc...)
        self.i2h = nn.Linear(self.n, self.statesizes[0][0], bias=False, dtype=torch.float64)  # bias already in h2h
        self.h2h = nn.ModuleList([nn.Linear(h[0], h[1], bias=True, dtype=torch.float64) for h in self.statesizes])
        self.hidden_activation = nn.Tanh()
        self.out_activation = nn.ReLU()
        self.h2o = nn.ModuleList([nn.Linear(o[0], o[1], bias=True, dtype=torch.float64) for o in self.outsizes[:-1]])
        self.h2o.append(SelectiveOutput(self.outsizes[-1][0], self.outsizes[-1][1], bias=True, dtype=torch.float64))

        # self.eq_not_reached = 0 # debugging of stop condition
        # self.forw_called = 0 # debugging of stop condition
        # self.n_batches_noteq = 0 # debugging of stop condition
        # self.n_targets = 0 # debugging of stop condition

        #todo: maybe revise initializations: bias start at zero?

    def forward(self, inputs, mask, max_steps=10, epsilon=None):
        """
        inputs: (batch, input+output dims) with inputs, and NaN for outputs (variable per instance)
        mask: (batch, input+output dims) integer mask with 0 at input positions and 1 at outputs
        max_steps: maximal number of time steps
        epsilon: tolerance for difference in outputs between consecutive steps in order to dynamically decide convergence  
        Returns: tuple with
            tensor (batch, input+output dims) with output predictions at last time step (intermixed with fixed inputs)
            tensor (batch, input+output dims) with logits (and +Inf for positions of inputs)
        """
        #self.forw_called += 1 # debugging of stop condition

        # initialize next inputs through initial hidden state (final output layer)
        h = self.h0
        o = h
        for out_layer in self.h2o[:-1]:
            o = out_layer(o)
        x = self.h2o[-1](o, inputs, mask, self.class_mask)

        stop = False
        step = 0
        # forward pass, unfolded in time
        while not stop and step < max_steps:

            prev_x = x

            h = self.hidden_activation(self.i2h(x) + h)

            for hidden_layer in self.h2h:
                h = self.hidden_activation(hidden_layer(h))

            o = h
            for out_layer in self.h2o[:-1]:
                o = self.out_activation(out_layer(o)) # ReLu instead of tanh for forward layers! more powerful
            x = self.h2o[-1](o, inputs, mask, self.class_mask)

            if epsilon:
                # stop = torch.allclose(prev_x, x, atol=epsilon, rtol=1E-8)
                batch_close = torch.sum(torch.isclose(prev_x, x, atol=epsilon, rtol=1E-8), 1)
                stop = torch.sum(batch_close == self.n) >= 0.8*len(batch_close)

            step += 1

        # if step == max_steps: # debugging of stop condition
        #     self.eq_not_reached += 1
        #     stop_cond = torch.sum(torch.isclose(prev_x, x, atol=epsilon, rtol=1E-8), 1)
        #     batches_noteq = inputs[stop_cond != self.n]
        #     self.n_targets += torch.sum(torch.isnan(batches_noteq)) # count number of target vars
        #     self.n_batches_noteq += torch.sum(stop_cond != self.n) # count number of instances that did not reach equilibrium

        return x

    def nparams(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

if __name__ == "__main__":

    model = BayesianRNN([2, 3, 2], [(10, 10)], outsizes = [(10, 7)])
    print(model.class_mask)

    print(model.nparams())
    inputs = torch.tensor([[1.0, 0.0, float('nan'), float('nan'), float('nan'), 0.0, 1.0], [float('nan'), float('nan'), 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0, 0]], dtype=torch.int32)
    print(f'inputs {inputs}')
    print(f'mask {mask}')
    model.forward(inputs, mask, max_steps=5, epsilon=1E-3)