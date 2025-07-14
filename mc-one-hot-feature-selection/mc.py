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

def update_rule_P(X, G, P, n_classes):
    var = np.zeros(X.shape[1])
    diag_p = np.diag(P).copy()
    for l in range(n_classes):
        indices = np.where(G == l)
        cluster = X[indices]
        #print("cluster = ", l , " len = ", len(cluster), " var = ", np.var(cluster, axis=0))
        var += np.var(cluster, axis=0)

    var /= n_classes
    m = np.argmax(var)
    print("m = ", m)
    print("var[m] = ", var[m])
    diag_p[m] = 0.0
    #print(" diag_p = ", diag_p)
    P = np.diag(diag_p)
    return P

def constr_error(X, W):
    W2 = np.array(W)
    err = np.linalg.norm(X - X @ W2 @ W2.T)
    return err

def cluster_error(X, F, G, W):
    #print(X[:10, :10])
    err = np.linalg.norm(X@W - np.take(F, G, axis=0))
    return err

def train_loop(X, P, F, G, n_classes, max_iter, inner_iter, tolerance):

    losses = []
    prev_loss = None
    for _ in range(max_iter):
        X = X@P
        W = init_W(X, n_classes + 1)
        XW = X@W
        G, F = init_G_F(XW, n_classes)
        for _ in range(inner_iter):
            W = update_rule_W(X, F, G)
            XW = X@W
            G = update_rule_G(XW, F)
            F = update_rule_F(XW, G, n_classes)
            loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
            losses.append(loss)
            prev_loss = loss
            #if prev_loss is not None and abs(prev_loss - loss) < tolerance:
            #    break

        P = update_rule_P(X, G, P, n_classes)
        

        
    
    return G, F, P, XW, losses


def mc(X, proj_dim, n_classes, max_iter=50, inner_iter=5, tolerance=0):
    P = init_P(X)
    W = init_W(X, proj_dim)
    XW = X@W
    G, F = init_G_F(XW, n_classes)
    G, F, P, XW, loss_history = train_loop(X, P, F, G, n_classes, max_iter, inner_iter, tolerance)
    return G, F, P, XW, loss_history