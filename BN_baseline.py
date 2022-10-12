import pandas as pd
import numpy as np
import itertools

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.utils import get_example_model
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

def get_asia_model(): 

    """
    create the ground-truth asia model based on http://bnlearn.com/bnrepository/discrete-small.html#asia
    remove the "either" node and adapt CPTs accordingly
    """

    asia = get_example_model('asia')

    asia.remove_node("either")
    asia.add_edge(u="tub",v="dysp")
    asia.add_edge(u="tub",v="xray")
    asia.add_edge(u="lung",v="dysp")
    asia.add_edge(u="lung",v="xray")

    cpd_xray = TabularCPD(variable='xray', variable_card=2, 
                          values=[[0.98, 0.98, 0.98, 0.05], # xray "yes": [L=y,T=y],[L=y,T=n],[L=n,T=y],[L=n,T=n]
                                  [0.02, 0.02, 0.02, 0.95]], # xray "no": [id.]
                          evidence=['lung', 'tub'],
                          evidence_card=[2, 2],
                          state_names={'xray': ['yes', 'no'],
                                       'lung': ['yes', 'no'],
                                       'tub': ['yes', 'no']})

    asia.add_cpds(cpd_xray)

    cpd_dysp = TabularCPD(variable='dysp', variable_card=2, 
                          values=[[0.9, 0.9, 0.9, 0.8, 0.7, 0.7, 0.7, 0.1], # xray "yes": [B=y,L=y,T=y],[B=y,L=y,T=n],[B=y,L=n,T=y],[B=y,L=n,T=n],[B=n,L=y,T=y],[B=n,L=y,T=n],[B=n,L=n,T=y],[B=n,L=n,T=n]
                                  [0.1, 0.1, 0.1, 0.2, 0.3, 0.3, 0.3, 0.9]], # xray "no": [id.]
                          evidence=['bronc', 'lung', 'tub'],
                          evidence_card=[2, 2, 2],
                          state_names={'dysp': ['yes', 'no'],
                                       'bronc': ['yes', 'no'],
                                       'lung': ['yes', 'no'],
                                       'tub': ['yes', 'no']})

    asia.add_cpds(cpd_dysp)
    
    return asia

def clean_samples_asia(dataset):

    """
    dataset: SampleDataset object used to train NNs
    returns: pandas dataframe containing the same samples in the same order, but is friendlier to use with pgmpy library 
    note: function is written specifically for asia dataset
    """
    
    samples = pd.DataFrame(columns = ["asia yes","asia no","smoke yes","smoke no","bronc yes","bronc no","dysp yes","dysp no","lung yes","lung no","tub yes","tub no","xray yes","xray no"])

    for i in range(len(dataset)):
        samples = samples.append(pd.DataFrame(dataset[i].reshape(1,-1), columns=samples.columns), ignore_index=True)

    df = pd.DataFrame()

    df["asia"] = samples["asia yes"] == 1.0
    df["smoke"] = samples["smoke yes"] == 1.0
    df["bronc"] = samples["bronc yes"] == 1.0
    df["dysp"] = samples["dysp yes"] == 1.0
    df["lung"] = samples["lung yes"] == 1.0
    df["tub"] = samples["tub yes"] == 1.0
    df["xray"] = samples["xray yes"] == 1.0

    cleanset = df.replace(to_replace={True: 'yes', False: 'no'})
    
    return cleanset

def learn_model(GT_model, train_set):

    """
    GT_model: contains the ground-truth causal structure of the model (edge connectivity)
    train_set: data from which to estimate the CPTs of the learned BN
    returns: BN with structure identical to GT_model, but with CPTs learned from the data
    """
    
    learned_model = BayesianNetwork(ebunch=GT_model.edges())
    learned_model.fit(data = train_set, estimator = MaximumLikelihoodEstimator)
    
    return learned_model

def BN_total_MAE(orig, learned):

    """
    Generates all possible evidence combinations. Calculates the difference (MAE) between the ground-truth model's predictions and the learned 
    model's predictions for all targets. The sum of all these errors makes up the total MAE metric. 
    These are effectively the same queries that are present in the TestQueryDataset used to evaluate the total MAE of NN models.

    orig: ground-truth BN model 
    learned: BN model learned from the data 
    returns: total MAE as described above
    """
    
    all_vars = list(orig.nodes) # list of vars in the model

    # states in original model and learned model might be ordered differently, so build a mapping between them
    all_states_orig = orig.states # dictionary: key=var, value=list of state names
    all_states_learned = learned.states
    map_learned_to_orig = {}
    for var in all_vars:
        map_var = []
        for val in all_states_learned[var]: 
            i_orig = all_states_orig[var].index(val)
            map_var.append(i_orig)
        map_learned_to_orig[var] = np.array(map_var)

    infer_orig = VariableElimination(orig) # inference object
    infer_learned = VariableElimination(learned)

    n = len(all_vars)

    mae = 0
    n_queries = 0
    
    for r in range(1, n): # loop over r = number of vars included in evidence state in this iteration

        comb = itertools.combinations(all_vars, r) # get all sets of r vars (= evidence sets)

        for c in comb: 

            states = []
            for var in c: 
                states.append(all_states_orig[var])
            prod = itertools.product(*states) # get all possible assignments of states to the set of vars in c

            for p in prod:
                evidence = {}

                for i in range(len(p)): 
                    evidence[c[i]] = p[i] # create the evidence set

                for var in all_vars: # query probability for all vars which are not included in evidence
                    if var not in evidence: 

                        orig_pred = infer_orig.query([var], evidence=evidence, show_progress=False).values
                        try: 
                            inf_failed = False
                            learned_pred = infer_learned.query([var], evidence=evidence, show_progress=False).values
                        except: 
                            inf_failed = True
                        
                        if inf_failed or np.isnan(np.sum(learned_pred)): # inference fails when unknown state name is encountered (did not occur in training data) 
                                                                         # or division by zero is encountered in variable elimination algorithm (result will be nan)
                            infer_learned = VariableElimination(learned) # need to make new object or lib crashes
                            learned_pred = infer_learned.query([var], show_progress=False).values # use prior probability
                        
                        idx_map = map_learned_to_orig[var] # use mapping from original model state ordering to learned model state ordering
                        mae_var = np.sum(np.abs(orig_pred[idx_map] - learned_pred))/len(learned_pred) # WeightedMAE
                        mae += mae_var

                n_queries += 1
    
    infer_learned = VariableElimination(learned) # need to make new object or lib crashes

    # empty evidence set
    for var in all_vars:

        orig_pred = infer_orig.query([var], show_progress=False).values
        learned_pred = infer_learned.query([var], show_progress=False).values

        idx_map = map_learned_to_orig[var] # use mapping from original model state ordering to learned model state ordering
        mae += np.sum(np.abs(orig_pred[idx_map] - learned_pred))/len(learned_pred)

    n_queries += 1

    return mae/n_queries

def BN_sample_MAE(orig, learned, test_sample_data, test_set_clean, var_map): 

    """
    Generates all possible evidence combinations. Calculates the difference (MAE) between the ground-truth model's predictions and the learned 
    model's predictions for all targets. The sum of all these errors makes up the total MAE metric. 
    These are effectively the same queries that are present in the TestQueryDataset used to evaluate the total MAE of NN models.
    
    orig: ground-truth BN model 
    learned: BN model learned from the data 
    test_sample_data: TestSampleDataset as used to evaluate sample MAE of NNs. Used to extract the mask which decides on evidence/target vars for 
                      each sample. Ensures that we use the same sample test set as we do for evaluating the NN models.
    test_set_clean: Test set sampled from the ground-truth distribution. Samples (and ordering) are the same as in test_sample_data, but format is
                    friendly for use with pgmpy library. Sample values are used as evidence as decided by the mask.
    var_map: dictionary mapping var name to positional indices in mask
    returns: total MAE as described above
    """

    # states in original model and learned model might be ordered differently, so build a mapping between them
    all_states_orig = orig.states # dictionary: key=var, value=list of state names
    all_states_learned = learned.states
    map_learned_to_orig = {}
    for var in var_map.keys():
        pos = []
        for val in all_states_learned[var]: 
            i_orig = all_states_orig[var].index(val)
            pos.append(i_orig)
        map_learned_to_orig[var] = np.array(pos)

    mae = 0
    n_queries = 0
    
    infer_orig = VariableElimination(orig)
    infer_learned = VariableElimination(learned)

    for idx, item in enumerate(test_sample_data): # extract masks from the same test sample dataset used to evaluate the NNs

        _, _, mask = item

        # create evidence and target set
        evidence = {}
        target = []
        for var in var_map.keys():
            pos = var_map[var]
            if mask[pos[0]] == 0: # if any of the positions for var are 0, this var is evidence 
                evidence[var] = test_set_clean.iloc[idx][var] # get the value for this evidence var from the sample set
            else: 
                target.append(var)
                
        # infer values of all targets for given evidence set 
        for t in target:
            
            orig_pred = infer_orig.query([t], evidence=evidence, show_progress=False).values
            try: 
                inf_failed = False
                learned_pred = infer_learned.query([t], evidence=evidence, show_progress=False).values
            except: 
                inf_failed = True
                learned_pred = float("nan")
                
            if inf_failed or np.isnan(np.sum(learned_pred)): # inference fails when unknown state name is encountered (did not occur in training data) 
                                                             # or division by zero is encountered in variable elimination algorithm (result will be nan)
                infer_learned = VariableElimination(learned) # need to make new object or lib crashes
                learned_pred = infer_learned.query([t], show_progress=False).values # use prior probability
                
            idx_map = map_learned_to_orig[t] # use mapping from original model state ordering to learned model state ordering
            mae_var = np.sum(np.abs(orig_pred[idx_map] - learned_pred))/len(learned_pred) # WeightedMAE
            mae += mae_var
            
        n_queries += 1
            
    return mae/n_queries