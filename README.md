# Test-Time Adaptation in the Presence of Out-of-Distribution Samples

This is the code for my bachelor thesis.
- build singularity file from utils in singularity/


## tta_ood
- adapted from https://github.com/DequanWang/tent
- main file: cifar10.py

### robustbench
- adapted from https://github.com/RobustBench/robustbench
- embedded in tent
- used to load datasets


## Corrupted datasets
- adapted from https://github.com/hendrycks/robustness/tree/master/ImageNet-C/create_c 
- create corrupted datasets with create_c_datasets.py
- adapt dir paths before usage
