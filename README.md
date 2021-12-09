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

## Data
- MI-F/MI-V: https://www.kaggle.com/shasun/tool-wear-detection-in-cnc-mill
- OPTDIGITS/PENDIGITS/SATELLITE/SHUTTLE/THYROID: http://odds.cs.stonybrook.edu

## Experiments

Firstly, extract data.zip

To replicate the results on the HRSS dataset with neighbour count k = 100 and "Mixed" negative sampling scheme

- Extract saved_models.zip
- Run:
```
python3 main.py --dataset HRSS --samples MIXED --k 100
```

To train a new model:
 
- Run:

```
python3 main.py --dataset HRSS --samples MIXED --k 100 --train_new_model
```

## Citation
TBC

