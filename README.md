# LUNAR
### Official Implementation of ["LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks"]()

Adam Goodge, Bryan Hooi, Ng See Kiong and Ng Wee Siong (AAAI2022)

## Files
- main.py
- variables.py : hyperparameters
- utils.py : functions for loading datasets, pre-processing, graph construction, negative-sampling
- LUNAR.py : GNN model and training procedure
- requirements.txt : packages for virtualenv
- data.zip : files for the HRSS dataset
- saved_models.zip : pretrained LUNAR models for HRSS with neighbour count k = 100 and "Mixed" negative sampling

## Experiments

Firstly, unzip data.zip and saved_models.zip 

To replicate the results on the HRSS dataset with neighbour count k = 100 and "Mixed" negative sampling scheme

```
python3 main.py --dataset HRSS --samples MIXED --k 100"
```

To train a new model:

```
python3 main.py --dataset HRSS --samples MIXED --k 100 --train_new_model"
```

## Citation
TBC

