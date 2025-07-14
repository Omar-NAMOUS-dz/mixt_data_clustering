import tensorflow as tf
from utils import datagen, clustering_accuracy, clustering_f1_score
from time import time
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from mc import mc, similarity_matrix



flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'heart', 'Dataset to use (heart, adult, credit, derma, bands, synth1, synth2 or synth3).')
flags.DEFINE_integer('max_iter', 10, 'Number of iterations of the algorithm.')
flags.DEFINE_integer('max_var', 30, 'Maximum number of variables to consider (>= 2)')
flags.DEFINE_float('tol', 1e-7, 'Tolerance threshold of convergence.')
flags.DEFINE_integer('runs', 1, 'Number of runs.')

dataset = FLAGS.dataset
max_iter = FLAGS.max_iter
inner_iter = FLAGS.max_var
tolerance = FLAGS.tol
runs = FLAGS.runs

print('-----------------', dataset, '-----------------')
data = {}
data['data'], data['labels'], data['n_classes'],\
      data['cat_feats'], data['num_feats'], data['target_feat'] = datagen(dataset)

similarity_matrix(data)
'''
metrics = {}
metrics['acc'] = []
metrics['nmi'] = []
metrics['ari'] = []
metrics['f1'] = []
metrics['loss'] = []
metrics['time'] = []

for run in range(runs):
    t0 = time()

    Z, F, XW, losses = mc(X, n_classes=data['n_classes'], proj_dim=data['n_classes'] + 1,\
                           max_iter=max_iter, tolerance=tolerance)

    metrics['time'].append(time()-t0)
    metrics['acc'].append(clustering_accuracy(data['labels'], Z))
    metrics['nmi'].append(nmi(data['labels'], Z))
    metrics['ari'].append(ari(data['labels'], Z))
    metrics['f1'].append(clustering_f1_score(data['labels'], Z, average='macro'))
    metrics['loss'].append(losses[-1])

results = {
    'mean': {k:np.mean(v).round(4) for k,v in metrics.items()}, 
    'std': {k:np.std(v).round(4) for k,v in metrics.items()}
}
means = results['mean']
  
print(means['acc'], means['f1'], means['nmi'], means['ari'], sep='\n')
'''
