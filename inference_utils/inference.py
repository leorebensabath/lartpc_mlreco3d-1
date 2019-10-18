import numpy as np
import pandas as pd
import sys
import os, re
import torch
import yaml

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

from sklearn.cluster import DBSCAN

def join_training_logs():

    pass


def make_inference_cfg(train_cfg, gpu=1, iterations=128, snapshot=None):
    '''
    Generate inference configuration file given training config.
    '''

    cfg = yaml.load(open(train_cfg, 'r'), Loader=yaml.Loader)
    process_config(cfg)
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
    process_config(inference_cfg)
    return inference_cfg


def main_loop(train_cfg, **kwargs):

    inference_cfg = make_inference_cfg(train_cfg,
        gpu=kwargs['gpu'], iterations=kwargs['iterations'])
    loader = loader_factory(inference_cfg)
    dataset = iter(cycle(loader))
    Trainer = trainval(inference_cfg)
    loaded_iteration = Trainer.initialize()

    output = []
    clusterer = DBSCAN(eps=kwargs['eps'], min_samples=kwargs['min_samples'])

    iterations = inference_cfg['trainval']['iterations']

    for i in range(iterations):

        print("Iteration: %d" % i)

        data_blob, res = Trainer.forward(dataset)
        # segmentation = res['segmentation'][0]
        embedding = res['cluster_features'][0]
        semantic_labels = data_blob['segment_label'][0][:, -1]
        cluster_labels = data_blob['cluster_label'][0][:, -1]
        coords = data_blob['input_data'][0][:, :3]
        index = data_blob['index'][0][0]

        acc_dict = {}

        for c in (np.unique(semantic_labels)):
            semantic_mask = semantic_labels == c
            clabels = cluster_labels[semantic_mask]
            embedding_class = embedding[semantic_mask]
            coords_class = coords[semantic_mask]

            pred = clusterer.fit_predict(embedding_class)

            purity, efficiency = purity_efficiency(pred, clabels)
            ari = ARI(pred, clabels)
            nclusters = len(np.unique(clabels))

            row = (index, c, ari, purity, efficiency, nclusters)
            output.append(row)

    output = pd.DataFrame(output, columns=['Index', 'Class', 'ARI',
                'Purity', 'Efficiency', 'num_clusters'])
    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg_path', '--config_path', help='config_path', type=str)
    parser.add_argument('-t', '--target', help='path to target directory', type=str)
    parser.add_argument('-n', '--name', help='name of output', type=str)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-gpu', '--gpu', type=str)
    #parser.add_argument('-ckpt', '--checkpoint_number', type=int)

    args = parser.parse_args()
    args = vars(args)

    train_cfg = args['config_path']

    eps_list = np.linspace(0.1, 3.0, 30)
    ms_list = [3, 5]

    for eps in eps_list:
        for ms in ms_list:
            dbscan_args = {'eps': eps, 'min_samples': ms,
                'gpu': args['gpu'], 'iterations': args['iterations']}
            print(eps, ms)
            res = main_loop(train_cfg, **dbscan_args)
            name = '{}_eps{}_ms{}.csv'.format(args['name'], eps, ms)
            target = os.path.join(args['target'], name)
            res.to_csv(target, index=False)