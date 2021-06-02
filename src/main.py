import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_name = 'digits'
X = pd.read_csv(f"data/{data_name}_train.data", sep=' ', header=None)
X = X.drop(columns=[5000])
Y = pd.read_csv(f"data/{data_name}_train.labels", sep=' ', header=None)
X_test = pd.read_csv(f"data/{data_name}_valid.data", sep=' ', header=None)
X_test = X_test.drop(columns=[5000])
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, stratify=Y, random_state=42)