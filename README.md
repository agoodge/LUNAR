# LUNAR
## Official Implementation of ["LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks"]()
### Adam Goodge, Bryan Hooi, Ng See Kiong and Ng Wee Siong (AAAI2022)

## Files
- main.py
- variables.py : hyperparameters
- utils.py : functions for loading datasets, pre-processing, graph construction, negative-sampling
- LUNAR.py : GNN model and training procedure
- requirements.txt : packages for virtualenv

We provide the trained models for the HRSS dataset with neighbour count k = 100 with the "Mixed" negative sampling scheme


## Experiments
To replicate the results on the HRSS dataset with neighbour count k = 100 and "Mixed" negative sampling scheme:

```
python3 main.py --dataset HRSS --samples MIXED --k 100"
```

To train a new model:

```
python3 main.py --dataset HRSS --samples MIXED --k 100 --train_new_model"
```

## Citation
TBC

