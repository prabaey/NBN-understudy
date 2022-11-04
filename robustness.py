from utils.dataset import TestQueryDatasetMultivar, IRLookupDataset, SampleDataset
from BN_baseline import get_asia_model
from run_experiments import learn_BN, learn_NN 

if __name__ == "__main__":

    # input config
    train_path = 'data/asia/train_samples.txt' # 20000 train samples extracted from ground-truth distribution
    test_sample_path = 'data/asia/sample_test_set.txt' # test queries according to 1000 test samples extracted from ground-truth distribution (calculation of sample MAE)
    test_total_path = 'data/asia/total_test_set.txt' # ground-truth target probabilities for all possible queries (calculation of total MAE)
    
    var_names = ["asia", "smoke", "bronc", "dysp", "lung", "tub", "xray"] # names of variables, fixed order
    var = [2, 2, 2, 2, 2, 2, 2] # cardinality of all variables
    mapping = {"asia":[0, 1], "smoke":[2, 3], "bronc":[4, 5], "dysp":[6, 7], "lung":[8, 9], "tub":[10, 11], "xray":[12, 13]} # where to each variable's classes
    
    # get ground truth asia model
    GT_model = get_asia_model() 
    smoothing = True
    BN_config = {"BN K2 smoothing": smoothing}

    # architecture config
    n = sum(var)
    h = [50, 50] # dimensions of hidden layers
    arch_config = {"h":h}

    # training config
    bs = 16  # batch size train
    bs_reg = 16 # batch size reg
    epochs = 500 # number of epochs to train for
    lr = 0.001 # learning rate
    seed = [17, 22, 58, 35, 37, 78, 36, 13, 85, 41] # seed for train set selection and model initialization$
    tb = False # write runs to tensor_board when True. if so, provide log_dir
    log_dir = None
    train_config = {"bs":bs, "bs_reg":bs_reg, "epochs":epochs, "lr":lr, "seed":seed}

    # generate datasets
    train_sample = SampleDataset(train_path) 
    
    total_test_set = TestQueryDatasetMultivar(test_total_path)
    sample_test_set = TestQueryDatasetMultivar(test_sample_path)

    # sample size
    N = 100

    # write results to logfile
    logfile = "robustness.txt"

    with open(logfile, 'w') as f:

        # write configuration to file
        f.write(f"Test robustness of asia BN, {N} samples \n")
        f.write("------------------- \n")
        f.write("\n")
        for param, value in BN_config.items():
            f.write(f"{param} = {value} \n")
        f.write("\n")
        for param, value in arch_config.items():
            f.write(f"{param} = {value} \n")
        f.write("\n")
        for param, value in train_config.items():
            f.write(f"{param} = {value} \n")
        f.write("\n")

    results = []

    edges_rm = [("tub", "xray"), ("lung", "xray"), ("smoke", "lung"), ("tub", "dysp"), ("asia", "tub")]
    edges_add = [("smoke", "dysp"), ("tub", "bronc"), ("lung", "bronc"), ("asia", "dysp"), ("asia", "xray")]

    # REMOVE one edge at a time
    for i in range(5):

        res_partial = {}

        IR_path = "data/asia/remove_edges/ind_r"+str(i+1)+".txt"
        IR_data = IRLookupDataset(IR_path, mapping, var_names, n)

        with open(logfile, 'a') as f:

            f.write("----------------- \n")
            f.write("\n")
            f.write(f"REMOVE edges {edges_rm[i][0]} and {edges_rm[i][1]} (ID: {i+1})\n")
            f.write("\n")

        # BN baseline
        with open(logfile, 'a') as f:

            f.write(f"BN \n")

            model = GT_model.copy()
            model.remove_edge(edges_rm[i][0], edges_rm[i][1]) # remove edge from BN structure
            total_mae, sample_mae = learn_BN(seed, model, N, train_sample, test_total_path, test_sample_path, smoothing)

            f.write(f"total MAE = {total_mae} \n")
            f.write(f"sample MAE = {sample_mae} \n")
            f.write("\n")

            res_partial["BN"] = {"total": total_mae, "sample":sample_mae}

        # NN baseline
        # NN+REG
        method = "REG"
        for alpha in [1, 10, 100]:

            with open(logfile, 'a') as f:

                f.write(f"NN+REG, alpha = {alpha} \n") # when alpha = 0, we are effectively running base NN without REG

                total_mae, sample_mae = learn_NN(train_sample, sample_test_set, N, test_set=total_test_set, IR_data=IR_data, var=var, n=n, mapping=mapping, 
                                                                var_names=var_names, h=h, bs=bs, bs_reg=bs_reg, epochs=epochs, lr=lr,
                                                                seed=seed, alpha=alpha, tb=tb, log_dir=log_dir, method=method)
                
                f.write(f"total MAE = {total_mae} \n")
                f.write(f"sample MAE = {sample_mae} \n")
                f.write("\n")

                res_partial[f"{method} {alpha}"] = {"total":total_mae, "sample":sample_mae}
        
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

            res_partial[f"{method}"] = {"total": total_mae, "sample":sample_mae}

            results.append(res_partial)

    # ADD one edge at a time
    for i in range(5):

        res_partial = {}

        IR_path = "data/asia/add_edges/ind_a"+str(i+1)+".txt"
        IR_data = IRLookupDataset(IR_path, mapping, var_names, n)

        with open(logfile, 'a') as f:

            f.write("----------------- \n")
            f.write("\n")
            f.write(f"ADD edges {edges_add[i][0]} and {edges_add[i][1]} (ID: {i+1})\n")
            f.write("\n")

        # BN baseline
        with open(logfile, 'a') as f:

            f.write(f"BN \n")

            model = GT_model.copy()
            model.add_edge(edges_add[i][0], edges_add[i][1]) # add edge to BN structure
            total_mae, sample_mae = learn_BN(seed, model, N, train_sample, test_total_path, test_sample_path, smoothing)

            f.write(f"total MAE = {total_mae} \n")
            f.write(f"sample MAE = {sample_mae} \n")
            f.write("\n")

            res_partial["BN"] = {"total": total_mae, "sample":sample_mae}

        # NN baseline
        # NN+REG
        method = "REG"
        for alpha in [1, 10, 100]:

            with open(logfile, 'a') as f:

                f.write(f"NN+REG, alpha = {alpha} \n") # when alpha = 0, we are effectively running base NN without REG

                total_mae, sample_mae = learn_NN(train_sample, sample_test_set, N, test_set=total_test_set, IR_data=IR_data, var=var, n=n, mapping=mapping, 
                                                                var_names=var_names, h=h, bs=bs, bs_reg=bs_reg, epochs=epochs, lr=lr,
                                                                seed=seed, alpha=alpha, tb=tb, log_dir=log_dir, method=method)
                
                f.write(f"total MAE = {total_mae} \n")
                f.write(f"sample MAE = {sample_mae} \n")
                f.write("\n")

                res_partial[f"{method} {alpha}"] = {"total": total_mae, "sample":sample_mae}
        
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

            res_partial[f"{method}"] = {"total":total_mae, "sample":sample_mae}

            results.append(res_partial)

    with open(logfile, 'a') as f:
    
        f.write("----------------- \n")
        f.write("\n")
        f.write(f"mean results {results} \n")