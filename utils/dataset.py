from torch.utils.data import Dataset
import numpy as np

class TestQueryDataset(Dataset):
    """
    contains the ground-truth conditional probability for all possible evidence combinations (total MAE)
    see data/asia_samples.txt for expected format of file
    """
    def __init__(self, path):
        with open(path, 'r') as file:
            self.samples = []
            self.masks = []
            self.targets = []
            lines = file.readlines()
            self.mapping_vars = lines[0][:-1].split(",")
            self.mapping_states = lines[1][:-1].split(",")
            for line in lines[2:]: 
                vals = line[:-1].split(',')
                sample = []
                mask = []
                target = []
                for val in vals: 
                    conv = self.convert(val) # returns (mask, sample, target)
                    mask.append(conv[0])
                    sample.append(conv[1])
                    target.append(conv[2])
                self.masks.append(mask)
                self.samples.append(sample)
                self.targets.append(target)
            self.samples = np.array(self.samples) # values of evidence, targets are represented as nan 
            self.targets = np.array(self.targets) # ground-truth probability of targets (query vars), evidence is represented as nan
            self.masks = np.array(self.masks) # mask with 1 for positions which act as target variables
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], self.targets[idx], self.masks[idx])


class TestQueryDatasetBinary(TestQueryDataset):
    """
    can be used with bayesian networks where all variables are binary 
    """
    def convert(self, val):
        if val == "True" or val == "False":
            mask = 0
            sample = float(val == "True")
            target = float("nan")
        else:
            mask = 1
            sample = float("nan")
            target = float(val)
        return (mask, sample, target)


class TestQueryDatasetMultivar(TestQueryDataset):
    """
    general dataset to use with bayesian network where nodes can be multivariate (more than 2 classes)
    see data/asia_ground_truth.txt for expected format of file
    """
    def convert(self, val):
        if val.isdigit():
            mask = 0
            sample = int(val)
            target = float("nan")
        else:
            mask = 1
            sample = float("nan")
            target = float(val)
        return (mask, sample, target)


class SampleDataset(Dataset):
    """
    stores dataset containing one-hot representation of observed class values for all variables
    see data/asia_samples.txt for expected format of file
    """
    def __init__(self, path):
        with open(path, 'r') as file:
            lines = file.readlines()
            self.mapping_vars = lines[0][:-1].split(",")
            self.mapping_states = lines[1][:-1].split(",")
            self.samples = np.array([[float(e) for e in line[:-1].split(',')] for line in lines[2:]])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class TestSampleDataset(Dataset):
    """
    stores ground-truth target probabilities for randomly sampled evidence values reflecting the underlying distribution of the data (sample MAE)
    at initialization, receives: 
    - samples: SampleDataset object containing test samples which serve as the observed values for the evidence, following the underlying ground-truth distribution
    - ground-truth: TestQueryDataset object containing the target probabilities for all combinations of evidence, serves as a lookup
    - var: list indicating number of classes per var, length is equal to number of vars, total sum is equal to input dim (n)
    """
    def __init__(self, samples, ground_truth, var):

        self.samples, self.masks, self.targets = [], [], []

        for sample in samples:

            # generate random mask
            mask = np.random.randint(0, 2, size = len(var))
            while np.isclose(np.sum(mask), 0): # we will not be able to find zero-mask in ground-truth probability set 
                mask = np.random.randint(0, 2, size = len(var))
            stretch_mask = []
            for j in range(len(mask)):
                stretch_mask += var[j]*[mask[j]]
            mask = np.array(stretch_mask)

            # go through ground truth probabilities for all evidence combinations (test dataset) to get expected output probability
            masked_sample = np.where(mask == 0, sample, [float("nan")]*len(sample))
            for i, t, m in ground_truth:
                if np.array_equal(mask, m) and np.isclose(masked_sample, i, equal_nan=True).all():
                    target = t
                    break

            # store trio of sample, mask and corresponding target
            self.samples.append(sample)
            self.masks.append(mask)
            self.targets.append(target)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (self.samples[idx], self.targets[idx], self.masks[idx])

class IRLookupDataset(Dataset):
    """
    class providing functionality for implementing the REG and COR approaches
    - can provide contrastive examples to test whether a particular independence relation is respected by the model (REG)
    - can match any evidence mask with an independence relation which is applicable according to the evidence mask (COR)
      these matches are stored in a lookup table for efficient lookups during training
    """
    def __init__(self, path, var_map, idx_to_var, n):
        """
        path: path to file containing independence relations
        var_map: dictionary mapping var name to positional indices in model inputs
        idx_to_var: array specifying which var is encoded on which index position of the input vector
        n: total number of input positions (sum of class cardinality for every variable)
        """
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        self.var_map = var_map
        self.idx_to_var = np.array(idx_to_var)
        self.n = n 

        self.indep = {}
        self.iterable_IRs = []

        for line in lines:
            # line of the form (input ⟂ targets | evidence), targets and evidence possibly containing multiple var names
            line = line[1:-2]

            if len(line.split('|')) == 2:

                # extract names of evidence vars
                evidence = line.split('|')[1]
                evidence = evidence.replace(' ', '').split(',')

                # extract part before conditioning (containing input var and target vars)
                ind_vars = line.split('|')[0]
            
            else:
                # no evidence vars
                evidence = []
                ind_vars = line

            # extract name of input var
            input = ind_vars.split('⟂')[0].strip()

            # extract name of target var
            targets = ind_vars.split('⟂')[1]
            targets = targets.replace(' ', '').split(',')

            # build up dictionary of independence assumptions
            # key: conditioning set (vars after the conditioning bar)
            # value: set of vars that are independent given the conditioning set (vars before the conditioning bar)
            for target in targets: 
                if frozenset(evidence) in self.indep: 
                    vars = self.indep[frozenset(evidence)]
                    if frozenset([input, target]) not in vars: # ensures that each combination before the conditioning bar is only added once
                        vars.add(frozenset([input, target]))
                else: 
                    self.indep[frozenset(evidence)] = set([frozenset([input, target])])

            self.iterable_IRs.append({"input":input, "target":targets, "evidence":evidence}) # data structure which is more friendly for iterating over all possible independence relations
                                                                                             # __get_item__ function operates on this so we can easily build a dataloader for the regularization approach or loss calculation

        self.mask_to_indp = {} # dictionary containing a mapping between all possible masks and their matching IRs

    def match(self, mask): 
        """
        mask: 0 on positions for which the corresponding variable is part of the evidence set, 1 where it is part of the target set
        output: all independence relations which match the query. this matching between masks and IRs is also stored as a look-up table
        """

        # turn mask into set of evidence var names
        ev_idx = (mask == 0).nonzero().flatten()
        ev_vars = self.idx_to_var[ev_idx]
        if not isinstance(ev_vars, np.ndarray):
            ev_vars = np.array([ev_vars])

        # iteratively select one evidence var as x (one of two vars before conditioning bar)
        relevant_IRs = {}
        for x in ev_vars: 
            C = frozenset(ev_vars[ev_vars != x])
            inds = self.indep.get(C, set()) # get all pairs of vars which are independent given conditioning set C
            for ind in inds: # check which one of these have x before the conditioning bar 
                if x in ind: 
                    if C in relevant_IRs: 
                        relevant_IRs[C].add(ind) # once again, key of dict is conditioning set, value is sets of vars before conditioning bar
                    else: 
                        relevant_IRs[C] = set([ind])

        if len(relevant_IRs) == 0:
            self.mask_to_indp[tuple(mask.numpy().tolist())] = [] # IR could not be found
        
        # build up list of IRs that match this mask, structured as a dict of C, x and Y 
        IR_list = []
        for C, inds in relevant_IRs.items():
            ind = list(inds.pop())
            for var in ind: 
                if var in ev_vars: 
                    x = var # x = var which is part of evidence, to be corrupted later
                else: 
                    Y = set([var]) # Y = set of targets 
            for ind in inds: 
                for var in list(ind): 
                    if var != x: 
                        Y.add(var) # add all other vars which are in front of the conditioning bar to Y 
            IR_list.append({"C": C, "x": x, "Y": Y})

        # lookup table of all IRs which match the mask 
        # IRs are of the form x ⟂ Y | C, where E = C ∪ x, with Y possible containing multiple vars
        self.mask_to_indp[tuple(mask.numpy().tolist())] = IR_list

        return self.mask_to_indp[tuple(mask.numpy().tolist())]

    def lookup(self, mask):
        """
        get an IR which matches the mask. select one IR randomly if multiple IRs match the query. 

        mask: 0 on positions for which the corresponding variable is part of the evidence set, 1 where it is part of the target set
        output: an independence relation which matches the query, represented as a dict with keys C, x and Y (x ⟂ Y | C)
        """

        # convert tensor mask to tuple and retrieve list of matched IRs
        mask = tuple(mask.numpy().tolist())
        relevant_IRs = self.mask_to_indp[mask]

        # select random independence relation from the matched IRs
        if len(relevant_IRs) != 0:
            IR = np.random.choice(relevant_IRs)
            return IR
        else: 
            return {}
    
    def __len__(self):
        return len(self.iterable_IRs)
    
    def __getitem__(self, idx):
        """
        Returns masks and inputs which are easy to use for independence loss calculation

        in_class1: conditioning vars are sampled randomly, input var is random class
        in_class2: conditioning vars are sampled randomly (but equal to in_class1), input var is other random class
        mask: contains 0 for all evidence vars (conditioning set + input)
        ind_mask: contains 1 for all target vars which are relevant to the independence relation

        """

        # get independence sample at index idx 
        indep_dict = self.iterable_IRs[idx]

        # get names of input, target and evidence vars
        input = indep_dict["input"]
        targets = indep_dict["target"]
        evidence = indep_dict["evidence"]

        # create mask
        mask = np.array([1]*self.n)
        mask[self.var_map[input]] = 0
        for ev in evidence:
            mask[self.var_map[ev]] = 0

        # set input var and evidence vars to random class 
        in_class1 = np.array([0.0]*self.n)
        random_class1 = np.random.choice(self.var_map[input]) # select random class among the options for the input var
        in_class1[random_class1] = 1.0 # set input var to random class
        for ev in evidence: 
            random_ev = np.random.choice(self.var_map[ev])
            in_class1[random_ev] = 1.0 # set evidence randomly
        
        # set input var and evidence vars to random class (same as in prev. case for evidence!)
        in_class2 = in_class1.copy() # copy evidence from class1 tensor
        choices = self.var_map[input][:] # make list of remaining classes (all classes belonging to input var except for random_class1)
        choices.remove(random_class1)
        random_class2 = np.random.choice(choices) # select random class among the remaining options for the input var
        in_class2[random_class1] = 0.0 # set input var to random class (not the same as random_class1)
        in_class2[random_class2] = 1.0 
        
        # create mask which selects only the targets relevant in this independence assumption
        ind_mask = np.array([0]*self.n)
        for t in targets:
            ind_mask[self.var_map[t]] = 1

        return (in_class1, in_class2, mask, ind_mask)