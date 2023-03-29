# PitfallsOfLabelPoisoningAttacksForGNNs

Source code for our ICLR2023 TrustML Workshop paper titled: [Pitfalls in Evaluating GNNs under Label Poisoning Attacks](https://openreview.net/forum?id=qGKj3AHlSXv)

# Requirements:
```
optuna==3.1.0
torch==1.11.0+cu113
tqdm==4.62.3
numpy==1.24.2
scipy==1.10.0
scikit-learn==0.24.0
torch_geometric==2.0.4
python==3.8.X
```

Unzip PitfallsOfLabelPoisoningAttacksData.zip to extract the dataset files and poisoned labels for LafAK attack. Do this before executing the code.

# Execute code using:
```
python run.py --attack {random, degree, lp, lafak, MG} --model {GCN, GAT, APPNP} --dataset {cora_ml, citeseer, pubmed}  --setting {small_val, large_val, cv}

Add the flag --hyp_param_tuning to enable hyper-parameters tuning
Add the flag --random_train_val to run on random splits as defined in the paper
```

# Example usage of above command:
```
1. To run LP attack on the cora_ml dataset and cv setting use:
    python run.py --attack lp --dataset cora_ml --setting cv

2. To run the previous setting with hyper-parameters tuning and random splits use:
    python run.py --attack lp --dataset cora_ml --setting cv --hyp_param_tuning --random_train_val

```

# Citation
Please cite our paper if you use this code in your work using:
```latex
@inproceedings{
lingam2023pitfalls,
title={Pitfalls in Evaluating {GNN}s under Label Poisoning Attacks},
author={Vijay Lingam and Mohammad Sadegh Akhondzadeh and Aleksandar Bojchevski},
booktitle={ICLR 2023 Workshop on Pitfalls of limited data and computation for Trustworthy ML},
year={2023},
url={https://openreview.net/forum?id=qGKj3AHlSXv}
}
```

