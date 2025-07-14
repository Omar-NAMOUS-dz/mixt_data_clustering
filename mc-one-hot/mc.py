from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
import math

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
    for i in range(max_iter):
        W = update_rule_W(X, F, G)
        XW = X@W
        G = update_rule_G(XW, F)
        F = update_rule_F(XW, G, n_classes)
        loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))

        if prev_loss is not None and abs(prev_loss - loss) < tolerance:
            break

        losses.append(loss)
        prev_loss = loss
    
    return G, F, XW, losses


def mc(X, proj_dim, n_classes, max_iter=50, tolerance=0):
    W = init_W(X, proj_dim)
    XW = X@W
    G, F = init_G_F(XW, n_classes)
    G, F, XW, loss_history = train_loop(X, F, G, n_classes, max_iter, tolerance)
    return G, F, XW, loss_history