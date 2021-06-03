import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, chi2
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score


def model_svm(num_features=100):
    variance_filter = VarianceThreshold()
    my_filter = SelectKBest(f_classif, k=num_features)
    clf = LinearSVC()
    return make_pipeline(variance_filter, my_filter, clf)

def model_pca(num_features=100):
    variance_filter = VarianceThreshold()
    my_filter = PCA(n_components=num_features)  # TODO check other params?
    # print(pca.explained_variance_ratio_)
    clf = LinearSVC()
    return make_pipeline(variance_filter, my_filter, clf)

def model_forest(num_features=100):
    variance_filter = VarianceThreshold()
    my_filter = SelectFromModel(RandomForestClassifier(random_state=0), max_features=num_features)
    clf = RandomForestClassifier(random_state=0)
    return make_pipeline(variance_filter, my_filter, clf)

def model_chi2(num_features=100):
    variance_filter = VarianceThreshold()
    my_filter = SelectKBest(chi2, k=num_features)
    clf = RandomForestClassifier(random_state=0)
    return make_pipeline(variance_filter, my_filter, clf)

def model_lda(num_features=100):  # doesn't work
    variance_filter = VarianceThreshold()
    my_filter = LinearDiscriminantAnalysis(n_components=num_features)  # XD, it can be max 1...
    clf = RandomForestClassifier(random_state=0)
    return make_pipeline(variance_filter, my_filter, clf)


data_name = 'digits'
X = pd.read_csv(f"data/{data_name}_train.data", sep=' ', header=None)
X = X.drop(columns=[5000])
Y = pd.read_csv(f"data/{data_name}_train.labels", sep=' ', header=None)
X_test = pd.read_csv(f"data/{data_name}_valid.data", sep=' ', header=None)
X_test = X_test.drop(columns=[5000])
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, stratify=Y, random_state=42)
Y_train = np.ravel(Y_train)
Y_val = np.ravel(Y_val)
N = 5000


fig, ax = plt.subplots()
for model_f in [model_svm, model_pca, model_forest, model_chi2]:
    percentiles = list(np.linspace(95,100,20))

    scores = []
    num_features = []
    for percentile in percentiles:
        num = max(round(N*(100-percentile)/100),1)
        model = model_f(num)

        model.fit(X_train, Y_train)
        y_pred = model.predict(X_val)
        scores.append(balanced_accuracy_score(Y_val, y_pred))
        num_features.append(num)

    plt.plot(num_features, scores, label=str(model_f))

ax.set_title("Accuracy based on number of features")
ax.set_xlabel("number of features")
ax.set_ylabel("accuracy")
ax.legend()
fig.show()