# TSK fuzzy system: pyTorch Version

This code is an implementation of the paper "Optimize TSK Fuzzy Systems for Classiﬁcation Problems: Mini-Batch Gradient Descent with Uniform Regularization and Batch Normalization"

This paper proposes a mini- batch gradient descent (MBGD) based algorithm to efficiently and effectively train TSK fuzzy classifiers. It integrates two novel techniques: 1) uniform regularization (UR), which forces the rules to have similar average contributions to the output, and hence to increase the generalization performance of the TSK model; and, 2) batch normalization (BN), which extends BN from deep neural networks to TSK fuzzy classifiers to expedite the convergence and improve the generalization performance.

# Run

```
./run.sh
```

# File structure

* **main_diff_data_split.py**        Sample code for running TSK-BN-UR
* **lib.models.py**                  Sample code for constructing TSK with BN and UR in pyTorch
* **lib.tuning_training.py**         Sample code for training TSK fuzzy systems

# Citing 
This paper is under review, we will publish the bib text for citing as soon as it's accpeted.
