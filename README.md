# LUNAR
### Official Implementation of ["LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks"](https://arxiv.org/pdf/2112.05355.pdf), Adam Goodge, Bryan Hooi, Ng See Kiong and Ng Wee Siong (AAAI2022)

Many well-established anomaly detection methods use the distance of a sample to those in its local neighbourhood: so-called `local outlier methods', such as LOF and DBSCAN. They are popular for their simple principles and strong performance on unstructured, feature-based data that is commonplace in many practical applications. However, they cannot learn to adapt for a particular set of data due to their lack of trainable parameters. In this paper, we begin by unifying local outlier methods by showing that they are particular cases of the more general message passing framework used in graph neural networks. This allows us to introduce learnability into local outlier methods, in the form of a neural network, for greater flexibility and expressivity: specifically, we propose LUNAR, a novel, graph neural network-based anomaly detection method. LUNAR learns to use information from the nearest neighbours of each node in a trainable way to find anomalies. We show that our method performs significantly better than existing local outlier methods, as well as state-of-the-art deep baselines. We also show that the performance of our method is much more robust to different settings of the local neighbourhood size.

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

