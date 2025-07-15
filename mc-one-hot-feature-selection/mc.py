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
    P = (1 / X.shape[1]) * np.identity(X.shape[1], dtype=float)
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
    W = W_t.numpy()
    intra_variance = np.zeros(XP.shape[1])
    total_variance = np.zeros(XP.shape[1])
    for v in range(XP.shape[1]):
        
        #print("cluster = ", l , " len = ", len(cluster), " var = ", np.var(cluster, axis=0))
        variable = XP[:, v].reshape(-1, 1)
        w = W[v, :].reshape(1, -1)
        proj_variable = variable@w
        center = np.mean(proj_variable, axis=0)
        centered_proj_variable = proj_variable - center
        squared_norms = np.sum(centered_proj_variable**2, axis=1)
        total_variance[v] = np.sum(squared_norms)/XP.shape[0]

        for l in range(n_classes):
            indices = np.where(G == l)
            cluster_proj_variable = proj_variable[indices]
            cluster_center = np.mean(cluster_proj_variable, axis=0)
            centered_cluster_proj_variable = cluster_proj_variable - cluster_center
            cluster_squared_norms = np.sum(centered_cluster_proj_variable**2, axis=1)
            cluster_variance = np.sum(cluster_squared_norms)/cluster_squared_norms.shape[0]
            intra_variance[v] += cluster_variance * (cluster_squared_norms.shape[0] / XP.shape[0])

    p = 1 - (intra_variance / total_variance)
    p = p / np.sum(p)
    P = np.diag(p)
    return P


def train_loop(X, P, F, G, n_classes, max_iter, tolerance):

    losses = []
    prev_loss = None
    for _ in range(max_iter):
        XP = X@P
        W = update_rule_W(XP, F, G)
        XPW = XP@W
        G = update_rule_G(XPW, F)
        F = update_rule_F(XPW, G, n_classes)
        P = update_rule_P(XP, W, G, F, n_classes)
        loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
        losses.append(loss)
        prev_loss = loss
        #if prev_loss is not None and abs(prev_loss - loss) < tolerance:
        #    break

        
    
    return G, F, P, XPW, losses


def mc(X, proj_dim, n_classes, max_iter=50, inner_iter=5, tolerance=0):
    P = init_P(X)
    
    print(P.shape[0])
    print(X.shape[0])

    XP = X@P
    W = init_W(XP, proj_dim)
    XPW = XP@W
    G, F = init_G_F(XPW, n_classes)
    G, F, P, XPW, loss_history = train_loop(X, P, F, G, n_classes, max_iter, tolerance)
    return G, F, P, XPW, loss_history