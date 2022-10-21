import sys
import numpy as np
import random

from torch.utils.data import random_split, DataLoader
import torch

from train import Train
from utils.dataset import TestQueryDatasetMultivar, IRLookupDataset, SampleDataset
from utils.nn import WeightedMAE
from BN_baseline import clean_samples, evaluate_BN, get_asia_model, learn_model

def evaluate_NN(model, test_set):

    dataloader = DataLoader(test_set, batch_size=64, shuffle=False) # data loader for test queries
    total_mae = 0

    # test
    for input, target, mask in dataloader:

        #model predictions
        model.eval()
        pred = model(input, mask)

        # test mae (based on ground-truth bayesian model probabilities)
        mae = WeightedMAE(pred, target, mask, model.class_mask)
        total_mae += torch.sum(mae)

    return total_mae/len(test_set)

def learn_NN(train_sample, sample_test_set, N, **kwargs):

    mae_total, sample_mae_total = [], []

    for s in kwargs["seed"]:

        # set all seeds to s
        torch.manual_seed(s)
        np.random.seed(s)
        torch.cuda.manual_seed(s)
        random.seed(s)

        # select N training instances from train sample set
        train_data, _ = random_split(train_sample, [N, len(train_sample)-N], generator=torch.Generator().manual_seed(s))

        # train model for seed s
        train = Train(**kwargs, train_data = train_data)
        model, total_mae, _ = train()

        # calculate sample MAE for test instances sampled randomly from ground truth distribution
        sample_mae = evaluate_NN(model, sample_test_set)

        # save model metrics 
        mae_total.append(total_mae.detach().item())
        sample_mae_total.append(sample_mae.detach().item())

    return mae_total, sample_mae_total

def learn_BN(seed, GT_model, N, train_sample, total_test_path, sample_test_path):
    
    mae_total, sample_mae_total = [], []

    for s in seed: 

        # set all seed to s
        np.random.seed(s)
        random.seed(s)

        # select N training instances from train sample set
        train_data, _ = random_split(train_sample, [N, len(train_sample)-N], generator=torch.Generator().manual_seed(s))
        
        # transform Datasets to pandas dataframe for easy use with pgmpy library
        train_set = clean_samples(train_data, train_sample.mapping_vars, train_sample.mapping_states)

        # learn Bayesian network from training data
        learned_model = learn_model(GT_model, train_set)

        # calculate total MAE and sample MAE
        total_mae = evaluate_BN(learned_model, total_test_path)
        sample_mae = evaluate_BN(learned_model, sample_test_path) 
    
        # save model metrics
        mae_total.append(total_mae)
        sample_mae_total.append(sample_mae)

    return mae_total, sample_mae_total

if __name__ == "__main__":

    # input config
    train_path = 'data/asia/train_samples.txt' # 10000 train samples extracted from ground-truth distribution
    test_sample_path = 'data/asia/sample_test_set.txt' # test queries according to 1000 test samples extracted from ground-truth distribution (calculation of sample MAE)
    test_total_path = 'data/asia/total_test_set.txt' # ground-truth target probabilities for all possible queries (calculation of total MAE)
    IR_path = 'data/asia/independencies.txt' # independence relations extracted from ground-truth asia network

    var_names = ["asia", "smoke", "bronc", "dysp", "lung", "tub", "xray"] # names of variables, fixed order
    var = [2, 2, 2, 2, 2, 2, 2] # cardinality of all variables
    mapping = {"asia":[0, 1], "smoke":[2, 3], "bronc":[4, 5], "dysp":[6, 7], "lung":[8, 9], "tub":[10, 11], "xray":[12, 13]} # where to each variable's classes
    
    # get ground truth asia model
    GT_model = get_asia_model() 

    # architecture config
    n = sum(var)
    h = [50, 50] # dimensions of hidden layers
    arch_config = {"h":h}

    # training config
    bs = 16  # batch size train
    bs_reg = 16 # batch size reg
    epochs = 500 # number of epochs to train for
    lr = 0.001 # learning rate
    seed = [17, 22, 58, 35, 37, 78, 36, 13, 85, 41] # seed for train set selection and model initialization
    tb = False # write runs to tensor_board when True. if so, provide log_dir
    log_dir = None
    train_config = {"bs":bs, "bs_reg":bs_reg, "epochs":epochs, "lr":lr, "seed":seed}

    # generate datasets
    train_sample = SampleDataset(train_path) 
    
    total_test_set = TestQueryDatasetMultivar(test_total_path)
    sample_test_set = TestQueryDatasetMultivar(test_sample_path)

    IR_data = IRLookupDataset(IR_path, mapping, var_names, n)

    # select sample sizes
    size = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    # write results to logfile
    logfile = "logfile.txt"

    with open(logfile, 'w') as f:

        # write configuration to file
        f.write(f"BN structure: asia \n")
        f.write("------------------- \n")
        f.write("\n")
        for param, value in arch_config.items():
            f.write(f"{param} = {value} \n")
        f.write("\n")
        for param, value in train_config.items():
            f.write(f"{param} = {value} \n")
        f.write("\n")

    res_mean = []
    res_std = []

    for N in size:

        res_mean_sample = {}
        res_std_sample = {}

        with open(logfile, 'a') as f:

            f.write("----------------- \n")
            f.write("\n")
            f.write(f"Dataset size {N}\n")
            f.write("\n")

        # BN baseline
        with open(logfile, 'a') as f:

            f.write(f"BN \n")

            total_mae, sample_mae = learn_BN(seed, GT_model, N, train_sample, test_total_path, test_sample_path)

            f.write(f"total MAE = {total_mae} \n")
            f.write(f"sample MAE = {sample_mae} \n")
            f.write("\n")

        # NN baseline
        # NN+REG
        method = "REG"
        for alpha in [0, 1, 10, 100]:

            with open(logfile, 'a') as f:

                f.write(f"NN+REG, alpha = {alpha} \n") # when alpha = 0, we are effectively running base NN without REG

                total_mae, sample_mae = learn_NN(train_sample, sample_test_set, N, test_set=total_test_set, IR_data=IR_data, var=var, n=n, mapping=mapping, 
                                                                var_names=var_names, h=h, bs=bs, bs_reg=bs_reg, epochs=epochs, lr=lr,
                                                                seed=seed, alpha=alpha, tb=tb, log_dir=log_dir, method=method)
                
                f.write(f"total MAE = {total_mae} \n")
                f.write(f"sample MAE = {sample_mae} \n")
                f.write("\n")

                res_mean_sample[f"{method} {alpha}"] = {"total": np.mean(total_mae), "sample":np.mean(sample_mae)}
                res_std_sample[f"{method} {alpha}"] = {"total": np.std(total_mae), "sample":np.std(sample_mae)}
        
        # NN+COR
        method = "COR"
        with open(logfile, 'a') as f:

            f.write(f"NN+COR \n")
            total_mae, sample_mae = learn_NN(train_sample, sample_test_set, N, test_set=total_test_set, IR_data=IR_data, var=var, n=n, mapping=mapping, 
                                                                var_names=var_names, h=h, bs=bs, bs_reg=bs_reg, epochs=epochs, lr=lr, 
                                                                seed=seed, alpha=alpha, tb=tb, log_dir=log_dir, method=method)
            
            f.write(f"total MAE = {total_mae} \n")
            f.write(f"sample MAE = {sample_mae} \n")
            f.write("\n")

            res_mean_sample[f"{method}"] = {"total": np.mean(total_mae), "sample":np.mean(sample_mae)}
            res_std_sample[f"{method}"] = {"total": np.std(total_mae), "sample":np.std(sample_mae)}

            res_mean.append(res_mean_sample)
            res_std.append(res_std_sample)

    with open(logfile, 'a') as f:
    
        f.write("----------------- \n")
        f.write("\n")
        f.write(f"mean results {res_mean} \n")
        f.write(f"std results {res_std} \n")