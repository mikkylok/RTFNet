# RTFVNet-pytorch
  
## Introduction

RTFVNet is a data-fusion network for video classification. It consists of two image encoders, a temporal model and a classifier.

## Dataset
R21 dataset for action recognition.

0: non-gesture, 1: smoking, and 2: eating

## Usage

```shell
conda create --name rtfvnet python=3.10 -y
conda activate rtfvnet

pip install .
```

### Training
```shell
python train_parallel.py
```

### Heatplot
```shell
python heatmap_plot.py
```