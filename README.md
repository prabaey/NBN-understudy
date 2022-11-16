# Neural Bayesian Network Understudy

This repository contains all code needed to reproduce results presented in [Neural Bayesian Network Understudy](https://arxiv.org/abs/2211.08243). This paper was presented at the [CML4Impact](https://www.cml-4-impact.vanderschaar-lab.com/) workshop at NeurIPS 2022. When reusing any of the ideas presented in this repository and the accompanying paper, please cite our work as follows:

**Citation**: Paloma Rabaey, Cedric De Boom, and Thomas Demeester. Neural Bayesian Network Understudy, 2022. URL https://arxiv.org/abs/2211.08243.


The following class and helper files are included: 
- models.py: feed-forward neural network with adapted output layer, to deal with arbitrary selection of evidence and target variables
- train.py: class to train neural understudy, using either REG or COR approach for incorporating causal structure
- BN_baseline.py: functionality to learn and evaluate a Bayesian network which serves as a baseline for the neural understudy
- utils/dataset.py: contains several Dataset classes to accommodate training and test sets 
- utils/nn.py: contains adapted output layer for neural understudy, as well as adapted softmax, loss and error functions
- run_experiments.py: shows how to use the class files above to train and evaluate neural and bayesian networks with varying settings and sample sets
- robustness.py: conducts experiments related to robustness against miss-specification of the causal structure

The following data-related files are included:
- data/asia: contains all data files relating to the asia BN
  - independencies.txt: all independence relations extracted from the ground-truth Asia DAG
  - sample_test_set.txt: ground-truth conditional probabilities for 1000 random evidence masks
  - total_test_set.txt: ground-truth conditional probabilities for all possible combinations of evidence variables (2059 queries)
  - train_samples.txt: 20000 train samples obtained from the ground-truth Asia network
  - add_edges/ind_ax.txt: independence relations extracted from the Asia DAG to which one random edge was added (5 miss-specified DAGs, one edge added at a time)
  - remove_edges/ind_rx.txt: independence relations extracted from the Asia DAG from which one random edge was removed (5 miss-specified DAGs, one edge removed at a time)
- Data generation.ipynb: Python notebook illustrating how we generated all files in *data/asia*

Inquiries can be directed at paloma.rabaey@ugent.be.
