from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import math
import pandas as pd

def init_W(X, proj_dim):
    svd = TruncatedSVD(proj_dim).fit(X)
    W = svd.components_.T
    return W

def init_G_F(XW, n_classes):
    km = KMeans(n_classes).fit(XW)
    G = km.labels_
    F = km.cluster_centers_
    return G, F

def update_rule_W(X, F, G):
    _, U, V = tf.linalg.svd(tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
    W = U @ tf.transpose(V)
    return W

def update_rule_G(XW, F):
    centroids_expanded = F[:, None, ...]
    distances = tf.reduce_mean(tf.math.squared_difference(XW, centroids_expanded), 2)
    G = tf.math.argmin(distances, 0, output_type=tf.dtypes.int32)
    return G

def update_rule_F(XW, G, n_classes):
    F = tf.math.unsorted_segment_mean(XW, G, n_classes)
    return F


def train_loop(X, F, G, n_classes, max_iter, tolerance):

    losses = []
    prev_loss = None
    for _ in range(max_iter):
        W = update_rule_W(X, F, G)
        XW = X@W
        G = update_rule_G(XW, F)
        F = update_rule_F(XW, G, n_classes)
        loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
        losses.append(loss)
        #if prev_loss is not None and abs(prev_loss - loss) < tolerance:
        #    break
        prev_loss = loss

        
    
    return G, F, XW, losses


def mc(X, proj_dim, n_classes, max_iter=50, tolerance=0):
    W = init_W(X, proj_dim)
    XW = X@W
    G, F = init_G_F(XW, n_classes)
    G, F, XW, loss_history = train_loop(X, F, G, n_classes, max_iter, tolerance)
    return G, F, XW, loss_history




def similarity(data, v1, v2):
    d = data['data']
    n_classes = data['n_classes']
    
    if v1 in data['num_feats'] and v2 in data['num_feats']:
        data_v1 = d[v1]
        data_v2 = d[v2]
        new_d = pd.concat([data_v1, data_v2], axis=1)

        km = KMeans(n_classes).fit(data_v1.to_frame())
        labels = km.labels_
        centers = km.cluster_centers_

        total_var = data_v2.var()
        s = 0.0
        for c in range(n_classes):
            indices = np.where(labels == c)[0]
            cluster = data_v2.iloc[indices]
            s += (cluster.shape[0] / labels.shape[0]) * cluster.var()
        return s / total_var
    
    if v1 in data['cat_feats'] and v2 in data['cat_feats']:
        data_v1 = d[v1]
        data_v2 = d[v2]
        A = data_v1.unique()
        new_d = pd.concat([data_v1, data_v2], axis=1)
        
        s = 0.0
        for a in A:
            cluster = new_d[new_d[v1] == a]
            proportions = cluster[v2].value_counts(normalize=True)
            gini_impurity = 1 - (proportions ** 2).sum()
            s += (cluster.shape[0] / new_d.shape[0]) * gini_impurity
        return s
    
    if v1 in data['num_feats'] and v2 in data['cat_feats']:
        temp = v1
        v1 = v2
        v2 = temp
    
    if v1 in data['cat_feats'] and v2 in data['num_feats']:
        data_v1 = d[v1]
        data_v2 = d[v2]
        total_var = data_v2.var(ddof=0)
        A = data_v1.unique()
        new_d = pd.concat([data_v1, data_v2], axis=1)
        
        s = 0.0
        for a in A:
            cluster = new_d.loc[new_d[v1] == a]
            cluster = cluster[v2]
            s += (cluster.shape[0] / new_d.shape[0]) * cluster.var(ddof=0)
        #print("s = ", s)
        #print("total var = ", total_var)
        return s / total_var
    

def similarity_matrix(data):
    var = data['num_feats'] + data['cat_feats']
    sim = {k1: {k2 : 0.0 for k2 in var} for k1 in var}
    for v1 in var:
        for v2 in var:
            print(v1, v2)
            if v1 != v2 : sim[v1][v2] = similarity(data, v1, v2)
            else: sim[v1][v2] = 0.0

    print("SIMILARITY")
    print(var)
    for v1 in sim:
        for v2 in sim:
            try:
                print(f'{v1} {v2} : {sim[v1][v2]:.2f}')
                #pass
            except:
                print(f'{v1} {v2} : {sim[v1][v2]}')
                #pass
