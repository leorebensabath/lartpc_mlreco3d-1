import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score as ari
from pathlib import Path
import argparse

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import process_config, train, inference
from mlreco.utils.metrics import *
from mlreco.trainval import trainval
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
from mlreco.main_funcs import cycle


def make_inference_cfg(train_cfg, gpu=1, iterations=128, snapshot=None):
    '''
    Generate inference configuration file given training config.
    '''

    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg, verbose=False)
    inference_cfg = cfg.copy()
    data_keys = inference_cfg['iotool']['dataset']['data_keys']
    
    # Change dataset to validation samples
    data_val = []
    for file_path in data_keys:
        data_val.append(file_path.replace('train', 'test'))
    inference_cfg['iotool']['dataset']['data_keys'] = data_val
    inference_cfg['trainval']['iterations'] = iterations
    
    # Change batch size to 1 since no need for batching during validation
    inference_cfg['iotool']['batch_size'] = 1
    inference_cfg['iotool'].pop('sampler', None)
    inference_cfg['iotool'].pop('minibatch_size', None)
    inference_cfg['trainval']['gpus'] = str(gpu)
    inference_cfg['trainval']["train"] = False
    
    # Analysis keys for clustering
    inference_cfg['model']["analysis_keys"] = {
        "segmentation": 0,
        "clustering": 1,
        "seediness": 2,
        "margins": 3
    }
    
    # Get latest model path if checkpoint not provided.
    model_path = inference_cfg['trainval']['model_path']
    if snapshot is None:
        checkpoints = [int(re.findall('snapshot-([0-9]+).ckpt', f)[0]) for f in os.listdir(
            re.sub(r'snapshot-([0-9]+).ckpt', '', model_path)) if 'snapshot' in f]
        latest_ckpt = max(checkpoints)
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(str(latest_ckpt)), model_path)
    else:
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(snapshot), model_path)
    inference_cfg['trainval']['model_path'] = model_path
    process_config(inference_cfg, verbose=False)
    return inference_cfg


def gaussian_kernel(centroid, sigma):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1, keepdims=False)
        probs = np.exp(-dists / (2.0 * sigma**2))
        return probs
    return f


def find_cluster_means(features, labels):
    '''
    For a given image, compute the centroids \mu_c for each
    cluster label in the embedding space.

    INPUTS:
        features (torch.Tensor) - the pixel embeddings, shape=(N, d) where
        N is the number of pixels and d is the embedding space dimension.

        labels (torch.Tensor) - ground-truth group labels, shape=(N, )

    OUTPUT:
        cluster_means (torch.Tensor) - (n_c, d) tensor where n_c is the number of
        distinct instances. Each row is a (1,d) vector corresponding to
        the coordinates of the i-th centroid.
    '''
    group_ids = sorted(np.unique(labels).astype(int))
    cluster_means = []
    #print(group_ids)
    for c in group_ids:
        index = labels.astype(int) == c
        mu_c = features[index].mean(0)
        cluster_means.append(mu_c)
    cluster_means = np.vstack(cluster_means)
    return group_ids, cluster_means


def cluster_remainder(embedding, semi_predictions):
    if sum(semi_predictions == -1) == 0 or sum(semi_predictions != -1) == 0:
        return semi_predictions
    group_ids, predicted_cmeans = find_cluster_means(
        embedding, semi_predictions)
    semi_predictions[semi_predictions == -1] = np.argmin(
        cdist(embedding[semi_predictions == -1], predicted_cmeans[1:]), axis=1)
    return semi_predictions


def fit_predict(embeddings, seediness, margins, threshold=0.9, cluster_all=False):
    pred_labels = -np.ones(embeddings.shape[0])
    seediness_copy = np.copy(seediness).squeeze(1)
    check_completion = 0
    c = 0
    while check_completion <= embeddings.shape[0]:
        i = np.argsort(seediness_copy)[::-1][0]
        seedScore = seediness[i]
        if seedScore < threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        f = gaussian_kernel(centroid, sigma)
        pValues = f(embeddings)
        cluster_index = pValues > 0.5
        pred_labels[cluster_index] = c
        c += 1
        seediness_copy[cluster_index] = 0
        check_completion += sum(cluster_index)
    if cluster_all:
        pred_labels = cluster_remainder(embeddings, pred_labels)
    return pred_labels


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg,
        gpu=kwargs['gpu'], iterations=kwargs['iterations'])
    loader = loader_factory(inference_cfg)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()

    output = []

    iterations = inference_cfg['trainval']['iterations']
    threshold = kwargs['threshold']

    for i in range(iterations):

        print("Iteration: %d" % i)

        data_blob, res = Trainer.forward(dataset)
        print(res)
        # segmentation = res['segmentation'][0]
        embedding = res['embeddings'][0]
        seediness = res['seediness'][0]
        margins = res['margins'][0]
        semantic_labels = data_blob['segment_label'][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][:, -1]
        coords = data_blob['input_data'][0][:, :3]
        #perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
        # embedding = embedding[perm]
        # coords = coords[perm]
        # print(coords)
        # print(data_blob['segment_label'][0])
        # print(data_blob['cluster_label'][0])
        index = data_blob['index'][0]

        acc_dict = {}

        for c in (np.unique(semantic_labels)):
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]
            seed_class = seediness[semantic_mask]
            margins_class = margins[semantic_mask]

            pred = fit_predict(embedding_class, seed_class, margins_class, 
                               threshold=threshold, cluster_all=True)

            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            nclusters = len(np.unique(clabels))

            row = (index, c, ari, purity, efficiency, fscore, nclusters, t)
            print(row)
            output.append(row)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'num_clusters', 'threshold'])
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg_path', '--config_path', help='config_path', type=str)
    parser.add_argument('-t', '--target', help='path to target directory', type=str)
    parser.add_argument('-n', '--name', help='name of output', type=str)
    parser.add_argument('-i', '--num_events', type=int)
    parser.add_argument('-gpu', '--gpu', type=str)

    args = parser.parse_args()
    args = vars(args)

    train_cfg = args['config_path']
    thresholds = np.linspace(0.5, 1.0, 20)
    for t in thresholds:
        start = time.time()
        output = main_loop(train_cfg, threshold=t, iterations=args['num_events'],
                           gpu=args['gpu'])
        end = time.time()
        print("Time = {}".format(end - start))
        name = '{}_{}.csv'.format(args['name'], t)
        target = os.path.join(args['target'], name)
        output.to_csv(target, index=False)