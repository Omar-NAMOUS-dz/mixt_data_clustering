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

def init_P(X):
    P = np.identity(X.shape[1], dtype=float)
    return P


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


def update_rule_P(XP, W_t, G, F, n_classes):

    num_samples = XP.shape[0]
    num_variables = XP.shape[1]

    W = W_t.numpy()
    intra_variance = np.zeros(num_variables)
    total_variance = np.zeros(num_variables)
    for v in range(num_variables):
        
        variable = XP[:, v].reshape(-1, 1)
        w = W[v, :].reshape(1, -1)
        proj_variable = variable@w
        center = np.mean(proj_variable, axis=0)
        centered_proj_variable = proj_variable - center
        squared_norms = np.sum(centered_proj_variable**2, axis=1)
        total_variance[v] = np.sum(squared_norms)/num_samples

        for l in range(n_classes):
            indices = np.where(G == l)
            cluster_proj_variable = proj_variable[indices]
            cluster_center = np.mean(cluster_proj_variable, axis=0)
            centered_cluster_proj_variable = cluster_proj_variable - cluster_center
            cluster_squared_norms = np.sum(centered_cluster_proj_variable**2, axis=1)
            cluster_variance = np.sum(cluster_squared_norms)/cluster_squared_norms.shape[0]
            intra_variance[v] += cluster_variance * (cluster_squared_norms.shape[0] / num_samples)

    #p = 1.0 - (intra_variance / total_variance)
    #p = 1.0 - intra_variance
    alpha = 5.0
    p = np.exp(-alpha * intra_variance)

    p = p * (num_variables / np.sum(p))

    print("p = \n", p)

    P = np.diag(p)
    return P


def train_loop(X, P, F, G, n_classes, max_iter, tolerance):

    losses = []
    prev_loss = None
    XP = X@P
    for _ in range(max_iter):
        W = update_rule_W(XP, F, G)
        XW = X@W
        G = update_rule_G(XW, F)
        F = update_rule_F(XW, G, n_classes)
        
        loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
        losses.append(loss)
        prev_loss = loss
        #if prev_loss is not None and abs(prev_loss - loss) < tolerance:
        #    break

        P = update_rule_P(X, W, G, F, n_classes)
        XP = X @ P
    return G, F, P, XW, losses


def mc(X, proj_dim, n_classes, max_iter=50, tolerance=0):
    P = init_P(X)
    W = init_W(X, proj_dim)
    XW = X@W
    G, F = init_G_F(XW, n_classes)
    G, F, P, XW, loss_history = train_loop(X, P, F, G, n_classes, max_iter, tolerance)
    return G, F, P, XW, loss_history