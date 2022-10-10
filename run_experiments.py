import sys
import train
from utils.dataset import BayesianDatasetMultivar, IRLookupDataset, SampleDataset, TestSampleDataset
from utils.nn import WeightedMAE
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np
import random

def sample_based_evaluation(model, test_sample):
    
    test_dataloader = DataLoader(test_sample, batch_size = 64, shuffle=True)

    model.eval()

    mae = 0
    for sample, target, mask in test_dataloader:
        pred = model(sample, mask)
        mae += torch.sum(WeightedMAE(pred, target, mask, model.class_mask))

    return mae/len(test_sample)

def run_sample_seeds(train_split, full_sample_set, N, **kwargs):

    mae_total, sample_mae_total, reg_total = [], [], []

    for s in kwargs["seed"]:

        # set all seeds to s
        torch.manual_seed(s)
        np.random.seed(s)
        torch.cuda.manual_seed(s)
        random.seed(s)

        # perform train/test split and create test sample dataset
        train_sample, test_sample = random_split(full_sample_set, [train_split, len(full_sample_set)-train_split], generator=torch.Generator().manual_seed(s))
        test_sample = TestSampleDataset(test_sample, kwargs["test_prob"], kwargs["var"])

        # split off N training instances from remaining data
        train_data, _ = random_split(train_sample, [N, len(train_sample)-N], generator=torch.Generator().manual_seed(s))

        # train model for seed s
        model, test_mae, reg_loss = train(**kwargs, train_data = train_data)

        # calculate sample MAE for test instances sampled randomly from ground truth distribution
        sample_mae = sample_based_evaluation(model, test_sample)

        # save model metrics 
        mae_total.append(test_mae.detach().item())
        sample_mae_total.append(sample_mae.detach().item())
        reg_total.append(reg_loss.detach().item())

    return mae_total, sample_mae_total, reg_total

if __name__ == "__main__":
    
    bn_name = sys.argv[1]

    # input config
    sample_data_path = 'asia/asia_samples.txt' # 11000 samples extracted from ground-truth distribution
    test_prob_path = 'asia/asia_ground_truth.txt' # ground-truth target probabilities for all possible queries (calculation of total MAE)
    IR_data_path = 'asia/asia_independencies.txt' # independence relations extracted from ground-truth asia network
    var_names = ["asia", "smoke", "bronc", "dysp", "lung", "tub", "xray"] # names of variables, fixed order
    var = [2, 2, 2, 2, 2, 2, 2] # cardinality of all variables
    mapping = {"asia":[0, 1], "smoke":[2, 3], "bronc":[4, 5], "dysp":[6, 7], "lung":[8, 9], "tub":[10, 11], "xray":[12, 13]} # where to each variable's classes
    train_split = 10000 # total number of train samples to split off (remaining 1000 are test samples)
    
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
    full_sample_set = SampleDataset("data/"+sample_data_path) 
    test_prob = BayesianDatasetMultivar("data/"+test_prob_path)
    IR_data = IRLookupDataset("data/"+IR_data_path, mapping, var_names, n)

    # select sample sizes
    size = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    # write results to logfile
    logfile = "logfile.txt"

    with open(logfile, 'w') as f:

        # write configuration to file
        f.write(f"bayesian model: {bn_name} \n")
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

        method = "REG"
        for alpha in [0, 1, 10, 100]:

            with open(logfile, 'a') as f:

                f.write(f"NN+REG, alpha = {alpha} \n") # when alpha = 0, we are effectively running base NN without REG

                total_mae, sample_mae, reg_loss = run_sample_seeds(train_split, full_sample_set, N, test_prob=test_prob, IR_data=IR_data, var=var, n=n, mapping=mapping, 
                                                                var_names=var_names, h=h, bs=bs, bs_reg=bs_reg, epochs=epochs, lr=lr,
                                                                seed=seed, alpha=alpha, tb=tb, log_dir=log_dir, method=method)
                
                f.write(f"total MAE = {total_mae} \n")
                f.write(f"sample MAE = {sample_mae} \n")
                f.write(f"reg loss = {reg_loss} \n")
                f.write("\n")

                res_mean_sample[f"{method} {alpha}"] = {"total": np.mean(total_mae), "sample":np.mean(sample_mae), "reg":np.mean(reg_loss)}
                res_std_sample[f"{method} {alpha}"] = {"total": np.std(total_mae), "sample":np.std(sample_mae), "reg":np.std(reg_loss)}
        
        method = "COR"

        with open(logfile, 'a') as f:

            f.write(f"run models with COR approach \n")
            total_mae, sample_mae, reg_loss = run_sample_seeds(train_split, full_sample_set, N, test_prob=test_prob, IR_data=IR_data, var=var, n=n, mapping=mapping, 
                                                                var_names=var_names, h=h, bs=bs, bs_reg=bs_reg, epochs=epochs, lr=lr, 
                                                                seed=seed, alpha=alpha, tb=tb, log_dir=log_dir, method=method)
            
            f.write(f"total MAE = {total_mae} \n")
            f.write(f"sample MAE = {sample_mae} \n")
            f.write(f"reg loss = {reg_loss} \n")
            f.write("\n")

            res_mean_sample[f"{method}"] = {"total": np.mean(total_mae), "sample":np.mean(sample_mae), "reg":np.mean(reg_loss)}
            res_std_sample[f"{method}"] = {"total": np.std(total_mae), "sample":np.std(sample_mae), "reg":np.std(reg_loss)}

            res_mean.append(res_mean_sample)
            res_std.append(res_std_sample)

    with open(logfile, 'a') as f:
    
        f.write("----------------- \n")
        f.write("\n")
        f.write(f"mean results {res_mean} \n")
        f.write(f"std results {res_std} \n")