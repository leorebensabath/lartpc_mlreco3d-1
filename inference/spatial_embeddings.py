import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml
import time
from scipy.spatial.distance import cdist
from scipy.spatial.kdtree import KDTree
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import DBSCAN
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


def make_inference_cfg(train_cfg, gpu=1, snapshot=None, batch_size=1, model_path=None):

    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg)
    inference_cfg = cfg.copy()
    data_keys = inference_cfg['iotool']['dataset']['data_keys']

    # Change dataset to validation samples
    data_val = []
    for file_path in data_keys:
        data_val.append(file_path.replace('train', 'test'))
    inference_cfg['iotool']['dataset']['data_keys'] = data_val

    # Change batch size to 1 since no need for batching during validation
    inference_cfg['iotool']['batch_size'] = batch_size
    inference_cfg['iotool'].pop('sampler', None)
    inference_cfg['iotool'].pop('minibatch_size', None)
    inference_cfg['trainval']['gpus'] = str(gpu)
    inference_cfg['trainval']["train"] = False

    # Analysis keys for clustering
    inference_cfg['model']["analysis_keys"] = {
        "segmentation": 0,
        "clustering": 1,
    }

    # Get latest model path if checkpoint not provided.
    if model_path is None:
        model_path = inference_cfg['trainval']['model_path']
    else:
        inference_cfg['trainval']['model_path'] = model_path
    if snapshot is None:
        checkpoints = [int(re.findall('snapshot-([0-9]+).ckpt', f)[0]) for f in os.listdir(
            re.sub(r'snapshot-([0-9]+).ckpt', '', model_path)) if 'snapshot' in f]
        print(checkpoints)
        latest_ckpt = max(checkpoints)
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(str(latest_ckpt)), model_path)
    else:
        model_path = re.sub(r'snapshot-([0-9]+)', 'snapshot-{}'.format(snapshot), model_path)
    inference_cfg['trainval']['model_path'] = model_path
    process_config(inference_cfg)
    return inference_cfg


def gaussian_kernel(centroid, sigma):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1, keepdims=False)
        probs = np.exp(-dists / (2.0 * sigma**2))
        return probs
    return f


def ellipsoidal_kernel(centroid, sigma):
    def f(x):
        dists = np.power(x - centroid, 2) / (2.0 * sigma**2)
        probs = np.exp(-np.sum(-dists, axis=1, keepdims=False))
        return probs
    return f


def locally_extreme_points(coords, data, neighbourhood, lookfor = 'max', p_norm = 2.):
    '''
    Find local maxima of points in a pointcloud.  Ties result in both points passing through the filter.

    Not to be used for high-dimensional data.  It will be slow.

    coords: A shape (n_points, n_dims) array of point locations
    data: A shape (n_points, ) vector of point values
    neighbourhood: The (scalar) size of the neighbourhood in which to search.
    lookfor: Either 'max', or 'min', depending on whether you want local maxima or minima
    p_norm: The p-norm to use for measuring distance (e.g. 1=Manhattan, 2=Euclidian)

    returns
        filtered_coords: The coordinates of locally extreme points
        filtered_data: The values of these points
    '''
    assert coords.shape[0] == data.shape[0], 'You must have one coordinate per data point'
    extreme_fcn = {'min': np.min, 'max': np.max}[lookfor]
    kdtree = KDTree(coords)
    neighbours = kdtree.query_ball_tree(kdtree, r=neighbourhood, p = p_norm)
    i_am_extreme = [data[i]==extreme_fcn(data[n]) for i, n in enumerate(neighbours)]
    extrema, = np.nonzero(i_am_extreme)  # This line just saves time on indexing
    return coords[extrema], data[extrema], extrema


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


def fit_predict2(embeddings, seediness, margins, fitfunc,
                 s_threshold=0.0, p_threshold=0.5, cluster_all=False):
    pred_labels = -np.ones(embeddings.shape[0])
    probs = []
    spheres = []
    seediness_copy = np.copy(seediness)
    count = 0
    while count < seediness.shape[0]:
        i = np.argsort(seediness_copy)[::-1][0]
        seedScore = seediness[i]
        if seedScore < s_threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        spheres.append((centroid, sigma))
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.reshape(-1, 1))
        cluster_index = np.logical_and((pValues > p_threshold), (seediness_copy > 0))
        seediness_copy[cluster_index] = -1
        count += sum(cluster_index)
        # print(count)
    if len(probs) == 0:
        return pred_labels, spheres, 1
    probs = np.hstack(probs)
    pred_labels = np.argmax(probs, axis=1)
    pred_num_clusters = probs.shape[1]
    return pred_labels, spheres, pred_num_clusters


def fit_predict3(embeddings, seediness, margins, fitfunc, neighborhood=0.02, seed_threshold=0.5):
    centroids, seed_values, extrema_indices = locally_extreme_points(embeddings, seediness, neighbourhood=neighborhood)
    extrema_indices = extrema_indices[seed_values > seed_threshold]
    extreme_margins = margins[extrema_indices]
    probs = []
    for i in range(len(extrema_indices)):
        f = fitfunc(centroids[i], extreme_margins[i])
        p = f(embeddings)
        probs.append(p.reshape(-1, 1))
    if len(probs) < 1:
        return np.ones(embeddings.shape[0]), 1
    probs = np.hstack(probs)
    pred_labels = np.argmax(probs, axis=1)
    pred_num_clusters = probs.shape[1]
    return pred_labels, pred_num_clusters


def fit_predict_dbscan(embeddings, seediness, margins, fitfunc, seed_threshold=0.5, eps=0.001):
    index = seediness > seed_threshold
    center_candidates = embeddings[index]
    margin_candidates = margins[index]
    prediction = DBSCAN(eps=eps).fit_predict(center_candidates)
    centroids = []
    margins = []
    spheres = []
    for i in np.unique(prediction):
        if i > 0:
            cluster = prediction == i
            cluster_centroid = np.mean(center_candidates[cluster], axis=0)
            cluster_margin = np.mean(margin_candidates[cluster])
            centroids.append(cluster_centroid)
            margins.append(cluster_margin)
    if len(centroids) == 0 or len(margins) == 0:
        return np.ones(embeddings.shape[0]), [], 1
    centroids = np.vstack(centroids)
    margins = np.vstack(margins)
    probs = []
    for mu, sigma in zip(centroids, margins):
        spheres.append((mu, sigma))
        f = fitfunc(mu, sigma)
        p = f(embeddings)
        probs.append(p.reshape(-1, 1))
    probs = np.hstack(probs)
    pred_num_clusters = probs.shape[1]
    pred_labels = np.argmax(probs, axis=1)
    return pred_labels, spheres, pred_num_clusters


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg, gpu=kwargs['gpu'], batch_size=1,
                        model_path=kwargs['model_path'])
    start_index = kwargs.get('start_index', 0)
    end_index = kwargs.get('end_index', 20000)
    event_list = list(range(start_index, end_index))
    loader = loader_factory(inference_cfg, event_list=event_list)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()
    output = []

    inference_cfg['trainval']['iterations'] = len(event_list)
    iterations = inference_cfg['trainval']['iterations']
    s_threshold = kwargs['s_threshold']
    p_threshold = kwargs['p_threshold']
    # s_thresholds = {0: 0.85, 1: 0.80, 2: 0.80, 3: 0.70, 4: 0.8}
    # p_thresholds = {0: 0.4, 1: 0.18, 2: 0.48, 3: 0.23, 4: 0.5}

    for i in event_list:

        print("Iteration: %d" % i)

        data_blob, res = Trainer.forward(dataset)
        # segmentation = res['segmentation'][0]
        embedding = res['embeddings'][0]
        seediness = res['seediness'][0].reshape(-1, )
        margins = res['margins'][0].reshape(-1, )
        semantic_labels = data_blob['segment_label'][0][:, 4]
        cluster_labels = data_blob['input_data'][0][:, 4]
        print(cluster_labels)
        coords = data_blob['input_data'][0][:, :3]
        index = data_blob['index'][0]

        acc_dict = {}

        for c in (np.unique(semantic_labels)):
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]
            seed_class = seediness[semantic_mask]
            margins_class = margins[semantic_mask]
            print(index, c)
            # pred, spheres, pred_num_clusters = fit_predict2(embedding_class, seed_class, margins_class, gaussian_kernel,
            #                     s_threshold=s_threshold, p_threshold=p_threshold, cluster_all=True)
            # pred, pred_num_clusters = fit_predict3(embedding_class, seed_class, margins_class, gaussian_kernel,
            #                     seed_threshold=s_threshold, neighborhood=p_threshold)
            # pred, _, pred_num_clusters = fit_predict_dbscan(embedding_class, seed_class, margins_class,
            #     gaussian_kernel, seed_threshold=s_threshold, eps=p_threshold)
            print('Seed Threshold: {}, P Threshold: {}'.format(s_threshold, p_threshold))
            pred, spheres, pred_num_clusters = fit_predict2(embedding_class, seed_class,
                                                            margins_class, gaussian_kernel,
                                                            s_threshold=s_threshold,
                                                            p_threshold=p_threshold, cluster_all=True)

            purity, efficiency = purity_efficiency(pred, clabels)
            fscore = 2 * (purity * efficiency) / (purity + efficiency)
            ari = ARI(pred, clabels)
            print("ARI = ", ari)
            sbd = SBD(pred, clabels)
            nclusters = len(np.unique(clabels))
            _, true_centroids = find_cluster_means(coords_class, clabels)
            for j, cluster_id in enumerate(np.unique(clabels)):
                margin = np.mean(margins_class[clabels == cluster_id])
                true_size = np.std(np.linalg.norm(coords_class[clabels == cluster_id] - true_centroids[j], axis=1))
                row = (index, c, ari, purity, efficiency, fscore, sbd, \
                    nclusters, pred_num_clusters, margin, true_size)
                output.append(row)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'FScore', 'SBD', 'num_clusters',
                'pred_num_clusters', 'margin', 'true_size'])
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-test_cfg', '--test_config', help='config_path', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)
    args = parser.parse_args()
    args = vars(args)
    cfg = yaml.load(open(args['test_config'], 'r'), Loader=yaml.Loader)

    train_cfg = cfg['config_path']
    s_thresholds = np.linspace(0, 0.95, 20)
    p_thresholds = np.linspace(0.05, 0.95, 20)
    for p in p_thresholds:
        for t in s_thresholds:
            start = time.time()
            output = main_loop(train_cfg, s_threshold=t, p_threshold=p, **cfg)
            end = time.time()
            print("Time = {}".format(end - start))
            name = '{}_{}_{}.csv'.format(cfg['name'], p, t)
            if not os.path.exists(cfg['target']):
                os.mkdir(cfg['target'])
            target = os.path.join(cfg['target'], name)
            output.to_csv(target, index=False, mode='a', chunksize=50)
