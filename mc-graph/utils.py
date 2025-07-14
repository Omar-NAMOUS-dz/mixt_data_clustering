import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def datagen(dataset):
    data = pd.read_csv(f'./data/{dataset}/{dataset}.csv')
    data = data[~data.isin(['?']).any(axis=1)]
    with open(f'./data/{dataset}/{dataset}_feats.txt','r') as f:
        cat_feats = f.readline().strip().split()
        num_feats = f.readline().strip().split()
        target_feat = f.readline().strip()

        cat = data[cat_feats]
        cat = pd.get_dummies(cat, columns=cat_feats, dtype=float).to_numpy()

        A = cat@cat.T
        X = data[num_feats].to_numpy(dtype=float)
        labels = data[target_feat].to_numpy().reshape(-1)
        n_classes = len(np.unique(labels))

        print(len(labels))
        print(X.shape)
        print(A.shape)

    return A, X, labels, n_classes

def preprocess_dataset(A, X, labels):
    labels = LabelEncoder().fit_transform(labels)
    X = StandardScaler().fit_transform(X)

    rowsum = np.array(A.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    A = r_mat_inv.dot(A)

    return A, X, labels

def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def cmat_to_psuedo_y_true_and_y_pred(cmat):
    y_true = []
    y_pred = []
    for true_class, row in enumerate(cmat):
        for pred_class, elm in enumerate(row):
            y_true.extend([true_class] * elm)
            y_pred.extend([pred_class] * elm)
    return y_true, y_pred

def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, **kwargs):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)