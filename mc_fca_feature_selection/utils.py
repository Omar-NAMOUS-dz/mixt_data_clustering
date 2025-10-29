import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from prince import MCA

def datagen(dataset):
    data = pd.read_csv(f'./data/{dataset}/{dataset}.csv')
    data = data[~data.isin(['?']).any(axis=1)]
    with open(f'./data/{dataset}/{dataset}_feats.txt','r') as f:
        cat_feats = f.readline().strip().split()

        num_feats = f.readline().strip().split()
        target_feat = f.readline().strip()

        labels = data[target_feat].to_numpy().reshape(-1)
        labels = LabelEncoder().fit_transform(labels)
        n_classes = len(np.unique(labels))

        num = data[num_feats].to_numpy(dtype=float)
        num = StandardScaler().fit_transform(num)

        m = sum([len(data[cat_feat].unique()) for cat_feat in cat_feats])
        d = len(cat_feats)
        mca = MCA(
                    n_components= m - d,
                    n_iter=3,
                    copy=True,
                    check_input=True,
                    engine='sklearn',
                    random_state=42
                )
        mca = mca.fit(data[cat_feats])
        cat = mca.transform(data[cat_feats])
        cat = StandardScaler().fit_transform(cat)

        X = np.concatenate((cat, num), axis=1)

        print(n_classes)

    return X, labels, n_classes


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