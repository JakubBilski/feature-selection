import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score


data_name = 'digits'
X = pd.read_csv(f"data/{data_name}_train.data", sep=' ', header=None)
X = X.drop(columns=[5000])
X
Y = pd.read_csv(f"data/{data_name}_train.labels", sep=' ', header=None)
# X_test = pd.read_(f"data/{data_name}_valid.data")
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, stratify=Y, random_state=42)
Y_train = np.ravel(Y_train)
Y_val = np.ravel(Y_val)
#---------------------------------------
# Boruta

# Super tutorial:
# https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np  # initialize Boruta
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=0)
# forest = RandomForestRegressor(max_depth = 5, random_state = 42  # its sooooo slow
boruta = BorutaPy(
   estimator = forest,
   n_estimators = 'auto',
   max_iter = 10  # number of trials to perform
)  # fit Boruta (it accepts np.array, not pd.DataFrame)
boruta.fit(np.array(X_train), np.array(Y_train))  # print results
green_area = X_train.columns[boruta.support_].to_list()
blue_area = X_train.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)
X_tr = boruta.transform(np.array(X_train))
forest = RandomForestClassifier(random_state=0)
forest.fit(X_tr, Y_train)
X_tr = boruta.transform(np.array(X_val))
print(forest.score(X_tr, Y_val))  # 0.9733333333333334