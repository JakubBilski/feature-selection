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



# -------------------------------------------------
# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline

num_features = 1000

pip = make_pipeline(
    SelectFromModel(RandomForestClassifier(random_state=0), max_features=num_features),
    RandomForestClassifier(random_state=0)
)
pip.fit(X_train, Y_train)

print(f'Score: {pip.score(X_val, Y_val)}')  # 0.9753
print(f'Features: {min(num_features, len(pip[-1].feature_importances_))}')  # 636

