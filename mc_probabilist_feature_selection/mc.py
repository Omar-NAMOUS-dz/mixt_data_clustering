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


def update_rule_W(X, P, F, G):
    _, U, V = tf.linalg.svd(P @ tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
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


def update_rule_P(XP, W_t, G, F, n_classes, p_history, mask, alpha):
    #print("XP = ", XP)

    num_samples = XP.shape[0]
    num_variables = XP.shape[1]

    W = W_t.numpy()
    intra_variance = np.zeros(num_variables)
    total_variance = np.zeros(num_variables)
    for v in range(num_variables):
        
        variable = XP[:, v].reshape(-1, 1)
        #if v == 0: print("1st var = ", variable)
        w = W[v, :].reshape(1, -1)
        proj_variable = variable@w
        #if v == 0: print("projected 1st var = ", proj_variable)
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

    #p = 1 - (intra_variance / total_variance)
    #p = 1.0 / intra_variance

    alpha += 1

    p_current = np.exp(intra_variance * -1 * alpha)

    p_history += p_current

    num_relevant_variables = mask.sum()

    p_normalized = p_history * mask

    p_normalized = p_normalized * (num_relevant_variables / np.sum(p_normalized))

    p = p_normalized

    treshold = 0.5
    mask = np.array([1.0 if p[i] > treshold else 0.0 for i in range(num_variables)])

    #print("intra_variance = \n", intra_variance[:10], "\n", intra_variance[-10:])
    #print("p_history = \n", p_history[:10], "\n", p_history[-10:])
    print("p = \n", p)
    P = np.diag(p)
    return P, p_history, mask, alpha


def train_loop(X, P, F, G, n_classes, max_iter, tolerance, inner_iter):

    losses = []
    prev_loss = None
    p_history = np.array([0.0 for _ in range(X.shape[1])])
    mask = np.array([1.0 for _ in range(X.shape[1])])
    alpha = 1.0

    for _ in range(max_iter):
        for _ in range(inner_iter):
            W = update_rule_W(X, P, F, G)
            XW = X@W
            G = update_rule_G(XW, F)
            F = update_rule_F(XW, G, n_classes)
            
            loss = tf.linalg.norm(X - tf.gather(F @ tf.transpose(W), G))
            losses.append(loss)
            prev_loss = loss
            #if prev_loss is not None and abs(prev_loss - loss) < tolerance:
            #    break

        P, p_history, mask, alpha = update_rule_P(X, W, G, F, n_classes, p_history, mask, alpha)
    
    return G, F, P, XW, losses


def mc(X, proj_dim, n_classes, max_iter=50, tolerance=0, inner_iter=5):
    P = init_P(X)
    
    print("P #rows: ", P.shape[0])
    print("X #rows: ", X.shape[0])

    W = init_W(X, proj_dim)
    XW = X@W
    G, F = init_G_F(XW, n_classes)
    G, F, P, XW, loss_history = train_loop(X, P, F, G, n_classes, max_iter, tolerance, inner_iter)
    return G, F, P, XW, loss_history