import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data_name = 'digits'
X = pd.read_csv(f"data/{data_name}_train.data", sep=' ', header=None)
X = X.drop(columns=[5000])
X
Y = pd.read_csv(f"data/{data_name}_train.labels", sep=' ', header=None)
# X_test = pd.read_(f"data/{data_name}_valid.data")
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, stratify=Y, random_state=42)

# -------------------------------------------------
# Random forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=0)
Y_tmp = np.ravel(Y_train)
forest.fit(X_train, Y_tmp)
importances = forest.feature_importances_
importances
# print(sorted(importances))
print(forest.score(X_val, Y_val))

scores = []
percentiles = list(np.linspace(90,100,100))
tresholds = []
for treshold_d in percentiles:
    treshold = np.percentile(importances, treshold_d)
    tresholds.append(treshold)
    chosen_labels = [i for i, im in enumerate(importances) if im>treshold]
    if not any(chosen_labels):
        scores.append(0)
        continue
    chosen_forest = RandomForestClassifier(random_state=0)
    chosen_forest.fit(X_train[chosen_labels], Y_tmp)
    scores.append(chosen_forest.score(X_val[chosen_labels], Y_val))
fig, ax = plt.subplots()
plt.plot(tresholds, scores)
ax.set_title("score based on treshold")
fig.show()
fig, ax = plt.subplots()
plt.plot(percentiles, scores, '*')
ax.set_title("score based on treshold")
fig.show()

# std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# forest_importances = pd.Series(importances)

plotted_importances = [im for im in importances if im>0.001]

fig, ax = plt.subplots()
plt.plot(list(range(len(plotted_importances))), sorted(plotted_importances))
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
fig.show()


# permutation importance https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance
# it lasts too long...
from sklearn.inspection import permutation_importance

perm_importances = permutation_importance(forest, X_val, Y_val, n_repeats=10, random_state=42, n_jobs=2)

plotted_importances = perm_importances
# plotted_importances = [im for im in perm_importances if im>0.001]

fig, ax = plt.subplots()
plt.plot(list(range(len(plotted_importances))), sorted(plotted_importances))
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
fig.show()