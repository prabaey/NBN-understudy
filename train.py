import numpy as np
import time

import torch
import itertools
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import NeuralBN
from utils.nn import SelectiveCELoss, WeightedMSELoss, WeightedMAE

class Train():

    def __init__(self, **kwargs):

        self.epochs = kwargs["epochs"]
        self.bs = kwargs["bs"]
        self.var = kwargs["var"]
        self.alpha = kwargs["alpha"]
        self.mapping = kwargs["mapping"]
        self.method = kwargs["method"]

        # load ground-truth probabilities (test set)
        test_data = kwargs["test_set"]
        self.test_dataloader = DataLoader(test_data, batch_size=self.bs, shuffle=False) # data loader for ground-truth probabilities
        self.n_test = len(test_data)
        print(f'Loaded ground-truth probability test dataset with size {self.n_test}')

        # load sample data (train set)
        sample_data = kwargs["train_data"]
        self.sample_dataloader = DataLoader(sample_data, batch_size=self.bs, shuffle=True, drop_last=True) # data loader for bayesian model samples
        self.n_train = len(sample_data)
        print(f'Loaded sample dataset with size {self.n_train}')

        # load independence assumptions
        self.ind_data = kwargs["IR_data"]
        if self.method == "COR":
            for var_mask in itertools.product([1, 0], repeat=len(self.var)):
                self.ind_data.match(torch.tensor(var_mask, dtype=torch.int32)) # prepare the lookup table (mask -> applicable IRs)
        self.reg_dataloader = DataLoader(self.ind_data, batch_size=kwargs["bs_reg"], shuffle=True, drop_last=True) # data loader for independence testing
        self.n_reg = len(self.ind_data)
        print(f'Loaded {self.n_reg} independence assumptions')

        #initialize model and optimizer
        self.model = NeuralBN(self.var, kwargs["h"])
        print(f'Created NeuralBN with {self.model.nparams()} parameters, hidden layer dims = {kwargs["h"]}')
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs["lr"])

        #tensorboard setup
        self.writer = None
        if kwargs["tb"]: 
            timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.writer = SummaryWriter("runs/"+kwargs["log_dir"]+"/"+timestr)
        print(f'Training NN + {self.method} with the following params: bs={self.bs}, bs_reg={kwargs["bs_reg"]}, epochs={self.epochs}, lr={kwargs["lr"]}, alpha={self.alpha}')

    def __call__(self):
        if self.method == "REG":

            return self.train_REG()
        elif self.method == "COR":
            return self.train_COR()

    def train_REG(self):

        reg_iterator = iter(self.reg_dataloader)
        for epoch in range(self.epochs):

            epoch_train_loss = 0
            epoch_reg_loss = 0
            epoch_test_mae = 0
            n_reg = 0

            # train 
            for input in self.sample_dataloader:

                # generate random mask
                var_mask = torch.randint(0, 2, tuple([self.bs, len(self.var)]))

                # convert mask according to multivar dimensions
                stretch_mask = []
                for j in range(var_mask.shape[1]):
                    stretch_mask.append(var_mask[:, j].repeat(self.var[j], 1).T)
                mask = torch.cat(stretch_mask, dim=1)

                # forward and backward pass
                self.model.train()
                pred = self.model(input, mask)
                train_loss = SelectiveCELoss(pred, mask, input)
                train_loss = torch.sum(train_loss)
                epoch_train_loss += train_loss

                # get a batch of independence samples
                try: 
                    in_class1, in_class2, mask, ind_mask  = next(reg_iterator)
                except StopIteration:
                    reg_iterator = iter(self.reg_dataloader)
                    in_class1, in_class2, mask, ind_mask  = next(reg_iterator)

                # keep track of how many reg samples we see during 1 epoch
                n_reg += in_class1.shape[0]

                # compare both queries
                pred_class1 = self.model(in_class1, mask)
                pred_class2 = self.model(in_class2, mask)

                # minimize difference between the two class predictions, ONLY for targets involved in the independence assumption
                reg_loss = WeightedMSELoss(pred_class1, pred_class2, ind_mask, self.model.class_mask)
                reg_loss = torch.sum(reg_loss)
                epoch_reg_loss += reg_loss

                # update parameters based on train loss and reg loss, with a regularization tuning parameter
                loss = train_loss + self.alpha*reg_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            # test
            for input, target, mask in self.test_dataloader:

                #model predictions
                self.model.eval()
                pred = self.model(input, mask)

                # test mae (based on ground-truth bayesian model probabilities)
                mae = WeightedMAE(pred, target, mask, self.model.class_mask)
                epoch_test_mae += torch.sum(mae)

            if self.writer:
                self.writer.add_scalar('Train', epoch_train_loss/self.n_train, epoch)
                self.writer.add_scalar('Reg', epoch_reg_loss/n_reg, epoch)
                self.writer.add_scalar('Test', epoch_test_mae/self.n_test, epoch)

        if self.writer:
            self.writer.flush() #ensure that all pending events have been written to disk

        return self.model, epoch_test_mae/self.n_test, epoch_reg_loss/n_reg

    def train_COR(self):

        # iterate through data
        for epoch in range(self.epochs):
            epoch_train_loss = 0
            epoch_reg_loss = 0
            epoch_test_mae = 0

            # train
            for input in self.sample_dataloader:

                self.model.train()
                loss = 0

                for i in input: # iterate over all instances in batch

                    # generate random mask
                    var_mask = torch.randint(0, 2, tuple([1, len(self.var)]))

                    # convert mask according to multivar dimensions
                    stretch_mask = []
                    for j in range(var_mask.shape[1]):
                        stretch_mask.append(var_mask[:, j].repeat(self.var[j], 1).T)
                    mask = torch.cat(stretch_mask, dim=1)

                    # get applicable IR
                    IR = self.ind_data.lookup(var_mask[0])

                    if len(IR) != 0: # apply IR-based sampling

                        # transform var names to index positions
                        x_pos = self.mapping[IR["x"]]
                        Y_pos = []
                        for y in IR["Y"]: 
                            Y_pos += self.mapping[y]

                        # corrupt input by uniformly sampling over the possible classes for var x 
                        i_corr = torch.clone(i)
                        i_corr[x_pos] = 0.0
                        random_pos = np.random.choice(x_pos)
                        i_corr[random_pos] = 1.0

                        # calculate loss for all targets EXCEPT those in Y, with original evidence set
                        i = torch.unsqueeze(i, dim=0)
                        pred = self.model(i, mask)
                        loss_mask = torch.clone(mask)
                        for y in Y_pos: 
                            loss_mask[:,y] = 0 # targets in Y don't contribute to this part of the loss
                        loss_i = SelectiveCELoss(pred, i, loss_mask).sum()

                        # calculate loss ONLY for targets in Y, with corrupted evidence set
                        i_corr = torch.unsqueeze(i_corr, dim=0)
                        pred = self.model(i_corr, mask)
                        loss_mask = torch.zeros(mask.shape)
                        for y in Y_pos: 
                            loss_mask[:,y] = 1 # only targets in Y contribute to this part of the loss
                        loss_corr = SelectiveCELoss(pred, i, loss_mask).sum()
                        
                        # add both contributions to total loss, divide by number of targets to weigh properly
                        loss += loss_i + loss_corr

                    else: # proceed as normal

                        i = torch.unsqueeze(i, dim=0)
                        pred = self.model(i, mask)
                        loss += SelectiveCELoss(pred, i, mask).sum()

                # update parameters based on total train loss
                epoch_train_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            # reg
            for in_class1, in_class2, mask, ind_mask in self.reg_dataloader:

                # compare both queries
                pred_class1 = self.model(in_class1, mask)
                pred_class2 = self.model(in_class2, mask)

                # calculate difference between pos/neg predictions, ONLY for targets involved in the independence assumption
                reg_loss = WeightedMSELoss(pred_class1, pred_class2, ind_mask, self.model.class_mask)
                reg_loss = torch.sum(reg_loss)
                epoch_reg_loss += reg_loss

            # test
            for input, target, mask in self.test_dataloader:

                #model predictions
                self.model.eval()
                pred = self.model(input, mask) 

                # test mae (based on ground-truth bayesian model probabilities)
                mae = WeightedMAE(pred, target, mask, self.model.class_mask)
                epoch_test_mae += torch.sum(mae)

            if self.writer:
                self.writer.add_scalar('Train', epoch_train_loss/self.n_train, epoch)
                self.writer.add_scalar('Reg', epoch_reg_loss/self.n_reg, epoch)
                self.writer.add_scalar('Test', epoch_test_mae/self.n_test, epoch)

        if self.writer:
            self.writer.flush() #ensure that all pending events have been written to disk

        return self.model, epoch_test_mae/self.n_test, epoch_reg_loss/self.n_reg