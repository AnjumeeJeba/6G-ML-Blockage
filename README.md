# 6G THz Blockage and micromobility detection with machine learning


### 1. Importing Libraries
 ```python
import arff
import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim
import torch.nn.functional as F
```
1. arff: Used to work with datasets in the ARFF format (common in machine learning). Likely, your THz micromobility dataset is in ARFF format.
2. torch: A deep learning library for creating and training neural networks.
3. copy: Allows deep copying of objects. This might be used later for duplicating models, data structures, etc.
4. numpy: Fundamental library for numerical computations.
5. pandas: For handling tabular data in DataFrames.
6. seaborn and matplotlib: Visualization libraries to create graphs and charts.
7. sklearn: Tools for machine learning, including splitting data into training and testing sets.
8. torch.nn, optim, and F: Submodules of PyTorch for defining models (```nn```), optimization (```optim```), and additional utility functions (```F```).


### 2. Matplotlib Inline Configuration
```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```
```%matplotlib inline:``` Ensures that matplotlib plots are displayed directly inside the Jupyter notebook.
```%config InlineBackend.figure_format = 'retina':``` Makes plots rendered in a higher resolution for better clarity.
