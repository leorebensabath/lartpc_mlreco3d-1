import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
from collections import defaultdict
import time


class UResNet(torch.nn.Module):
    """
    UResNet

    For semantic segmentation, using sparse convolutions from SCN, but not the
    ready-made UNet from SCN library. The option `ghost` allows to train at the
    same time for semantic segmentation between N classes (e.g. particle types)
    and ghost points masking.

    Can also be used in a chain, for example stacking PPN layers on top.

    Configuration
    -------------
    num_strides : int
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    num_classes : int
        Should be number of classes (+1 if we include ghost points directly)
    data_dim : int
        Dimension 2 or 3
    spatial_size : int
        Size of the cube containing the data, e.g. 192, 512 or 768px.
    ghost : bool, optional
        Whether to compute ghost mask separately or not. See SegmentationLoss
        for more details.
    reps : int, optional
        Convolution block repetition factor
    kernel_size : int, optional
        Kernel size for the SC (sparse convolutions for down/upsample).
    features: int, optional
        How many features are given to the network initially.

    Returns
    -------
    In order:
    - segmentation scores (N, 5)
    - feature map for PPN1
    - feature map for PPN2
    - if `ghost`, segmentation scores for deghosting (N, 2)
    """

    def __init__(self, cfg, name="uresnet_clustering"):
        super(UResNet, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg['modules'][name]

        # Whether to compute ghost mask separately or not
        self._ghost = self._model_config.get('ghost', False)
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        self.spatial_size = self._model_config.get('spatial_size', 512)
        num_classes = self._model_config.get('num_classes', 5)
        self._N1 = self._model_config.get('N1', 3)
        self._N2 = self._model_config.get('N2', 3)
        self._use_gpu = self._model_config.get('use_gpu', False)
        self._hypDim = self._model_config.get('hypDim', 16)
        self._seedDim = self._model_config.get('seedDim', 2)
        self._mode = self._model_config.get('mode', 0)

        if self._mode == 1:
            nInputFeatures = 4

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None
        leakiness = 0.20

        def block(m, a, b):  # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormLeakyReLU(a, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, a, b, 3, False))
                    .add(scn.BatchNormLeakyReLU(b, leakiness=leakiness))
                    .add(scn.SubmanifoldConvolution(self._dimension, b, b, 3, False)))
             ).add(scn.AddTable())

        self.input = scn.Sequential().add(
           scn.InputLayer(self._dimension, self.spatial_size, mode=3)).add(
           scn.SubmanifoldConvolution(self._dimension, nInputFeatures, m, 3, False)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        
        # Encoding
        self.bn = scn.BatchNormLeakyReLU(nPlanes[0], leakiness=leakiness)
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        module = scn.Sequential()
        for i in range(num_strides):
            module = scn.Sequential()
            for _ in range(reps):
                block(module, nPlanes[i], nPlanes[i])
            self.encoding_block.add(module)
            module2 = scn.Sequential()
            if i < num_strides-1:
                module2.add(
                    scn.BatchNormLeakyReLU(nPlanes[i], leakiness=leakiness)).add(
                    scn.Convolution(self._dimension, nPlanes[i], nPlanes[i+1],
                        downsample[0], downsample[1], False))
            self.encoding_conv.add(module2)

        # Decoder 1
        self.decoding_conv1, self.decoding_blocks1 = scn.Sequential(), scn.Sequential()
        
        for i in range(num_strides-2, -1, -1):
            module1 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
            self.decoding_conv1.add(module1)
            module2 = scn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i] * (2 if j == 0 else 1), nPlanes[i])
            self.decoding_blocks1.add(module2)

        # Decoder 2
        self.decoding_conv2, self.decoding_blocks2 = scn.Sequential(), scn.Sequential()
        
        for i in range(num_strides-2, -1, -1):
            module1 = scn.Sequential().add(
                scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                    downsample[0], downsample[1], False))
            self.decoding_conv2.add(module1)
            module2 = scn.Sequential()
            for j in range(reps):
                block(module2, nPlanes[i] * (2 if j == 0 else 1), nPlanes[i])
            self.decoding_blocks2.add(module2)

        # Segmentation 1x1 Convolutions
        self.seg_output = scn.Sequential().add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(self._dimension))
        self.seg_linear = torch.nn.Linear(m, num_classes)

        # Embedding 1x1 Convolutions
        self.embedding_convs = scn.Sequential()
        for _ in range(self._N1):
            block(self.embedding_convs, m, m)
        self.emb_output = scn.Sequential().add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(self._dimension))
        self.emb_linear = torch.nn.Linear(m, self._hypDim)

        # Seediness Map
        self.seediness_convs = scn.Sequential()
        for _ in range(self._N2):
            block(self.seediness_convs, m, m)
        self.seed_output = scn.Sequential().add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(self._dimension))
        self.seed_linear = torch.nn.Linear(m, self._seedDim)
        

    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        point_cloud = input[0]
        coords = point_cloud[:, 0:self._dimension+1].float()
        normalized_coords = (coords[:, 0:3] - self.spatial_size /2) / float(self.spatial_size)
        features = point_cloud[:, self._dimension+1:self._dimension+2].float()
        if self._mode == 1:
            features = torch.cat([normalized_coords, features], dim=1)
        #print(coords)
        x = self.input((coords, features))
        # Embeddings at each layer
        feature_maps = [x]
        
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)
            # Downsampled coordinates and feature map
        # U-ResNet decoding
        x1 = x
        
        for i, layer in enumerate(self.decoding_conv1):
            encoding_block = feature_maps[-i-2]
            x_non = layer(x1)
            x1 = self.concat([encoding_block, x_non])
            x1 = self.decoding_blocks1[i](x1)

        for i, layer in enumerate(self.decoding_conv2):
            encoding_block = feature_maps[-i-2]
            x_non = layer(x)
            x = self.concat([encoding_block, x_non])
            x = self.decoding_blocks2[i](x)
        
        # Note that the output tensors must be lexicographically sorted
        # to match the semantic/clustering labels. 
        #perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
        x_seg = self.seg_output(x1)
        x_seg = self.seg_linear(x_seg)  # Output of UResNet

        x_emb = self.embedding_convs(x1)
        x_emb = self.emb_output(x_emb)
        x_emb = self.emb_linear(x_emb)

        x_seed = self.seediness_convs(x)
        x_seed = self.seed_output(x_seed)
        x_seed = self.seed_linear(x_seed)
        seediness = torch.sigmoid(x_seed[:, 0]).view(-1, 1)
        x_seed = torch.cat([seediness, x_seed[:, 1:]], dim=1)

        return [[x_seg],
                [x_emb],
                [x_seed]]


class ClusteringLoss(torch.nn.Module):
    '''
    Implementation of the Clustering Loss Function in Pytorch.
    https://arxiv.org/pdf/1708.02551.pdf
    Note that there are many other implementations in Github, yet here
    we tailor it for use in conjuction with Sparse UResNet.
    '''

    def __init__(self, cfg, reduction='sum'):
        super(ClusteringLoss, self).__init__()
        self._cfg = cfg['modules']['discriminative_loss']
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce = torch.nn.BCELoss(reduction='mean')
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        self.delta_dist = self._cfg.get('delta_dist', 1.5)

        self._w = {}
        self._w['var'] = self._cfg.get('var_weight', 3.0)
        self._w['dist'] = self._cfg.get('dist_weight', 1.0)
        self._w['reg'] = self._cfg.get('reg_weight', 0.001)
        self._w['seg'] = self._cfg.get('seg_weight', 0.0)
        self._w['smoothing'] = self._cfg.get('smoothing_weight', 5.0)
        self._w['seed'] = self._cfg.get('seediness_weight', 5.0)

        self._threshold = self._cfg.get('p0', 0.5)

    def find_cluster_means(self, features, labels):
        '''
        For a given image, compute the centroids \mu_c for each
        cluster label in the embedding space.
        Inputs:
            features (torch.Tensor): the pixel embeddings, shape=(N, d) where
            N is the number of pixels and d is the embedding space dimension.
            labels (torch.Tensor): ground-truth group labels, shape=(N, )
        Returns:
            cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
            distinct instances. Each row is a (1,d) vector corresponding to
            the coordinates of the i-th centroid.
        '''
        n_clusters = labels.unique().size()
        clabels = labels.unique(sorted=True)
        cluster_means = []
        for c in clabels:
            mu_c = features[labels == c].mean(0)
            cluster_means.append(mu_c)
        cluster_means = torch.stack(cluster_means)
        return cluster_means


    def intra_cluster_loss(self, features, seediness, labels, cmeans):
        '''
        Implementation of variance loss in Discriminative Loss.
        Inputs:)
            features (torch.Tensor): pixel embedding, same as in find_cluster_means.
            labels (torch.Tensor): ground truth instance labels
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): constant used to specify delta_v in paper. Think of it
            as the size of each clusters in embedding space. 
        Returns:
            var_loss: (float) variance loss (see paper).
        '''
        var_loss, smoothness_loss = 0.0, 0.0
        cluster_labels = sorted(np.unique(labels.detach().cpu().numpy()))

        with torch.no_grad():
            seed_truth = torch.zeros((seediness.shape[0], )).cuda()
        for i, c in enumerate(cluster_labels):
            binary_mask = torch.zeros((features.shape[0], )).cuda()
            binary_mask[labels == c] = 1
            binary_mask[labels != c] = 0
            dist1 = torch.norm(features[labels == c] - cmeans[i], p=self._cfg['norm'], dim=1)
            dist2 = torch.norm(features[labels != c] - cmeans[i], p=self._cfg['norm'], dim=1)
            # print("dist1 = ", dist1)
            # print("dist2 = ", dist2)
            var_map = torch.exp(seediness[:, 1][labels == c])
            var = torch.mean(var_map)
            # print("Var = ", var)
            smoothness_loss += torch.mean((var_map - var)**2)
            dist1 = torch.exp(-dist1**2 / ( 2 * var))
            dist2 = torch.exp(-dist2**2 / ( 2 * var))
            # print("dist1 = ", dist1)
            # print("dist2 = ", dist2)
            prob_map = torch.zeros((features.shape[0], )).cuda()
            prob_map[labels == c] = dist1
            prob_map[labels != c] = dist2
            seed_truth[labels == c] = dist1.detach()
            # prob1 = prob1[prob1 < self._threshold]
            # prob2 = prob2[prob2 >= self._threshold]
            num_points = dist1.nelement() + dist2.nelement()
            # if num_points == 0:
            #     continue
            #loss = -(torch.sum(torch.log(prob1)) + torch.sum(torch.log(1.0 - prob2))) / float(num_points)
            loss = self.bce(prob_map, binary_mask)
            var_loss += loss

        var_loss /= float(len(cluster_labels))
        smoothness_loss /= float(len(cluster_labels))
        seed_loss = torch.mean((seed_truth - seediness[:, 0])**2)
        #time.sleep(10)
        return var_loss, smoothness_loss, seed_loss


    def inter_cluster_loss(self, cluster_means, margin=1.5):
        '''
        Implementation of distance loss in Discriminative Loss.
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
            margin (float/int): the magnitude of the margin delta_d in the paper.
            Think of it as the distance between each separate clusters in
            embedding space.
        Returns:
            dist_loss (float): computed cross-centroid distance loss (see paper).
            Factor of 2 is included for proper normalization.
        '''
        dist_loss = 0.0
        n_clusters = len(cluster_means)
        if n_clusters < 2:
            # Inter-cluster loss is zero if there only one instance exists for
            # a semantic label.
            return 0.0
        else:
            for i, c1 in enumerate(cluster_means):
                for j, c2 in enumerate(cluster_means):
                    if i != j:
                        dist = torch.norm(c1 - c2, p=self._cfg['norm'])
                        hinge = torch.clamp(2.0 * margin - dist, min=0)
                        dist_loss += torch.pow(hinge, 2)
                        
            dist_loss /= float((n_clusters - 1) * n_clusters)
            return dist_loss

    def regularization(self, cluster_means):
        '''
        Implementation of regularization loss in Discriminative Loss
        Inputs:
            cluster_means (torch.Tensor): output from find_cluster_means
        Returns:
            reg_loss (float): computed regularization loss (see paper).
        '''
        reg_loss = 0.0
        n_clusters, feature_dim = cluster_means.shape
        for i in range(n_clusters):
            reg_loss += torch.norm(cluster_means[i, :], p=self._cfg['norm'])
        reg_loss /= float(n_clusters)
        return reg_loss


    def combine(self, features, seediness, group_labels):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''
        delta_dist = self.delta_dist
        
        c_means = self.find_cluster_means(features, group_labels)
        dist_loss = self.inter_cluster_loss(c_means, margin=delta_dist)
        var_loss, smoothness_loss, seed_loss = self.intra_cluster_loss(features, 
            seediness, group_labels, c_means)
        reg_loss = self.regularization(c_means)

        loss = self._w['var'] * var_loss + self._w['dist'] * dist_loss + self._w['reg'] * reg_loss
        loss += self._w['smoothing'] * smoothness_loss
        loss += self._w['seed'] * seed_loss
        return [loss, float(self._w['var'] * var_loss), float(self._w['dist'] * dist_loss), 
                float(self._w['reg'] * reg_loss), float(self._w['seed'] * seed_loss),
                float(self._w['smoothing'] * smoothness_loss)]


    def combine_multiclass(self, features, seediness, semantic_labels, group_labels):
        '''
        Wrapper function for combining different components of the loss, 
        in particular when clustering must be done PER SEMANTIC CLASS. 

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud. 

        INPUTS: 
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class. 
            loss[i] = computed DLoss for semantic class <i>. 
            acc_segs (list): list of computed clustering accuracy for each semantic class. 
        '''
        loss_dict = {}
        num_seg = semantic_labels.unique().nelement()
        for sc in np.unique(semantic_labels.detach().cpu().numpy()):
            index = (semantic_labels == sc)
            num_clusters = len(group_labels[index].unique())
            loss_blob = self.combine(features[index], seediness[index], 
                group_labels[index])
            loss_dict['total_loss'] = loss_blob[0] / float(num_seg)
            loss_dict['var_loss'] = loss_blob[1] / float(num_seg)
            loss_dict['dist_loss'] = loss_blob[2] / float(num_seg)
            loss_dict['reg_loss'] = loss_blob[3] / float(num_seg)
            loss_dict['seed_loss'] = loss_blob[4] / float(num_seg)
            loss_dict['smoothness_loss'] = loss_blob[5] / float(num_seg)
        return loss_dict


    def compute_segmentation_loss(self, output_seg, slabels, batch_idx):
        '''
        Compute the semantic segmentation loss for the final output layer. 

        INPUTS:
            - output_seg (torch.Tensor): (N, num_classes) Tensor.
            - slabels (torch.Tensor): (N, 5) Tensor with semantic segmentation labels.
            - batch_idx (list): list of batch indices, ex. [0, 1, 2 ,..., 4]

        OUTPUT:
            - loss (torch.Tensor): scalar number (1x1 Tensor) corresponding to
                to calculated semantic segmentation loss over a given layer. 
        '''
        loss = 0.0
        acc = 0.0
        loss = torch.mean(self.cross_entropy(output_seg, slabels[:, 4].long()))
        with torch.no_grad():
            pred = torch.argmax(self.log_softmax(output_seg), dim=1)
            acc = (pred == slabels[:, 4].long()).sum().item() / float(pred.nelement())
        return loss, acc

    def fit_predict(self, embedding, seediness, semantic_labels, group_labels):
        '''
        Computes several clustering accuracy metrics along with
        predicted clustering labels.

        INPUTS:
            - embedding
            - seediness
            - semantic_labels (torch.Tensor): (N, ) Semantic Labels
            - group_labels (torch.Tensor): (N, ) Instance Labels
        NOTE: Both semantic and group labels must be lexsorted appropriately. 
        '''
        acc_dict = {}
        from sklearn.metrics import adjusted_rand_score
        # Generate predicted instance labels. 
        emb = embedding.detach().cpu()
        seed_map = seediness.detach().cpu()
        slabels = semantic_labels.detach().long()
        clabels = group_labels.detach().long()
        for sc in slabels.unique(sorted=True):
            pred, _ = self.fit_predict_subevent(emb[slabels == sc], 
                seed_map[slabels == sc])
            # Compute Accuracy Metrics
            ari = adjusted_rand_score(clabels[slabels == sc].cpu().numpy(), pred.detach().cpu().numpy())
            acc_dict[sc.item()] = ari
        acc_avg = sum([acc for acc in acc_dict.values()]) / len(acc_dict.keys())
        acc_dict['accuracy'] = acc_avg
        return acc_dict
                
    def fit_predict_subevent(self, embedding_group, seed_group):
        '''

        NOTE: Inputs Tensors must be detached!
        '''
        with torch.no_grad():
            prob_tensor = []
            seed_keep_track = seed_group.detach()
            while (seed_keep_track[:, 0] > 0.5).any():
                seed, loc = torch.max(seed_keep_track[:, 0], 0)
                centroid = embedding_group[loc.item()]
                log_var = seed_group[:, 1][loc.item()]
                dists = torch.norm(embedding_group - centroid,
                    p=self._cfg['norm'], dim=1)
                logits = - dists / (torch.exp(log_var))
                prob_tensor.append(logits.view(-1, 1))
                group = logits > np.log(self._threshold)
                if sum(group) < 5:
                    break
                seed_keep_track[:, 0][group] = -1.0
            if not prob_tensor:
                pred = torch.zeros((embedding_group.shape[0], ))
                prob_tensor = torch.zeros((embedding_group.shape[0], ))
            else:
                prob_tensor = torch.cat(prob_tensor, dim=1)
                pred = torch.argmax(prob_tensor, dim=1)
            return pred, prob_tensor
        

    def forward(self, out, semantic_labels, group_labels):
        '''
        Forward function for the Discriminative Loss Module.

        Inputs:
            out: output of UResNet; embedding-space coordinates.
            semantic_labels: ground-truth semantic labels
            group_labels: ground-truth instance labels
        Returns:
            (dict): A dictionary containing key-value pairs for
            loss, accuracy, etc. 
        '''
        # Decreasing Spatial Size Order
        #slabels = semantic_labels[0]
        #clabels = group_labels[0]
        loss_dict = defaultdict(list)
        acc_dict = defaultdict(list)
        acc_dict['accuracy'] = 0.0

        coords = semantic_labels[0][:, 0:4].detach().cpu().numpy()
        perm = np.lexsort(coords.T)
        semantic_labels = semantic_labels[0][perm].long()
        #print(semantic_labels)

        coords = group_labels[0][:, 0:4].detach().cpu().numpy()
        perm = np.lexsort(coords.T)
        group_labels = group_labels[0][perm].long()
        #print(group_labels)

        # Semantic Loss
        batch_idx = np.unique(coords[:, 3])
        loss_seg, acc_seg = self.compute_segmentation_loss(out[0][0], semantic_labels, batch_idx)
        loss_dict['loss_seg'] = [float(loss_seg) * 16.0]

        for bidx in batch_idx:
            # Get tensors for current batch. 
            index = semantic_labels[:, 3] == bidx
            embedding_batch = out[1][0][index]
            seediness_batch = out[2][0][index]
            slabels_batch = semantic_labels[index][:, 4]
            clabels_batch = group_labels[index][:, 4]
            # Compute Clustering Loss
            loss_dict_batch = self.combine_multiclass(embedding_batch, seediness_batch, slabels_batch, clabels_batch)
            for key, val in loss_dict_batch.items():
                loss_dict[key].append(val)
            # acc_dict_batch = self.fit_predict(embedding_batch, seediness_batch, slabels_batch, clabels_batch)
            # for key, val in acc_dict_batch.items():
            #     acc_dict[key].append(val)

        for key, val in loss_dict.items():
            loss_dict[key] = sum(val)
        loss_dict['total_loss'] += loss_seg * 16.0 * self._w['seg']
        # for key, val in acc_dict.items():
        #     acc_dict[key] = sum(val) / len(val) * 16.0

        #print(loss_dict)
        #print(acc_dict)

        res = {
            "loss": loss_dict['total_loss'],
            "var_loss": loss_dict['var_loss'],
            "dist_loss": loss_dict['dist_loss'],
            "reg_loss": loss_dict['reg_loss'],
            "seed_loss": loss_dict['seed_loss'],
            "smoothness_loss": loss_dict['smoothness_loss'],
            "seg_loss": float(loss_seg * len(batch_idx)),
            # "acc_0": acc_dict[0],
            # "acc_1": acc_dict[1],
            # "acc_2": acc_dict[2],
            # "acc_3": acc_dict[3],
            # "acc_4": acc_dict[4],
            # "acc_seg": acc_seg,
            "accuracy": acc_seg * 16.0
        }
        print(res)

        return res