from sklearn import set_config
set_config(display='diagram')  # nwm co to jest

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

data_name = 'digits'
X = pd.read_csv(f"data/{data_name}_train.data", sep=' ', header=None)
X = X.drop(columns=[5000])
Y = pd.read_csv(f"data/{data_name}_train.labels", sep=' ', header=None)
# X_test = pd.read_(f"data/{data_name}_valid.data")
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, stratify=Y, random_state=42)
Y_train = np.ravel(Y_train)
Y_val = np.ravel(Y_val)


from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, chi2
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

num_features = 1000

variance_filter = VarianceThreshold()
anova_filter = SelectKBest(chi2, k=1000)
clf = LinearSVC()
anova_svm = make_pipeline(variance_filter, anova_filter, clf)
anova_svm.fit(X_train, Y_train)

y_pred = anova_svm.predict(X_val)
print(f'Report: {classification_report(Y_val, y_pred)}')
print(f'Score: {balanced_accuracy_score(Y_val, y_pred)}')  # 0.9766
print(f'Features: {num_features}')  # 1000

