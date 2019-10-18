import torch
import sys
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sparseconvnet as scn
from collections import defaultdict


class KDENet(torch.nn.Module):
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

    def __init__(self, cfg, name="clusternet_density"):
        super(KDENet, self).__init__()
        import sparseconvnet as scn
        self._model_config = cfg['modules'][name]

        # Whether to compute ghost mask separately or not
        self._dimension = self._model_config.get('data_dim', 3)
        reps = self._model_config.get('reps', 2)  # Conv block repetition factor
        kernel_size = self._model_config.get('kernel_size', 2)
        num_strides = self._model_config.get('num_strides', 5)
        m = self._model_config.get('filters', 16)  # Unet number of features
        nInputFeatures = self._model_config.get('features', 1)
        self.spatial_size = self._model_config.get('spatial_size', 512)
        num_classes = self._model_config.get('num_classes', 5)
        self._N = self._model_config.get('N', 0)
        self._simpleN = self._model_config.get('simple_conv', False)
        self._hypDim = self._model_config.get('hypDim', 16)

        self._add_coordinates = self._model_config.get('cluster_add_coords', False)
        self._density_estimate = self._model_config.get('density_estimate', False)
        self._N_density = self._model_config.get('N_density', 1)
        self._radii = self._model_config.get('radius', [1.0])

        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        downsample = [kernel_size, 2]  # [filter size, filter stride]
        self.last = None
        leakiness = 0.0

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

        # Decoding
        self.decoding_conv, self.decoding_blocks = scn.Sequential(), scn.Sequential()
        # Upsampling Transpose Convolutions for Embeddings
        self.embedding_upsampling = scn.Sequential()

        # Build separate decoders for segmentation/clustering
        if self._N > 0:
            for i in range(num_strides-2, -1, -1):
                module1 = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                    scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                        downsample[0], downsample[1], False))
                module3 = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                    scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                        downsample[0], downsample[1], False))
                self.decoding_conv.add(module1)
                self.embedding_upsampling.add(module3)
                module2 = scn.Sequential()
                for j in range(reps):
                    block(module2, nPlanes[i] * (3 if j == 0 else 1), nPlanes[i])
                self.decoding_blocks.add(module2)
        else:
            for i in range(num_strides-2, -1, -1):
                module1 = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(nPlanes[i+1], leakiness=leakiness)).add(
                    scn.Deconvolution(self._dimension, nPlanes[i+1], nPlanes[i],
                        downsample[0], downsample[1], False))
                self.decoding_conv.add(module1)
                module2 = scn.Sequential()
                for j in range(reps):
                    block(module2, nPlanes[i] * (2 if j == 0 else 1), nPlanes[i])
                self.decoding_blocks.add(module2)

        self.output = scn.Sequential().add(
           scn.BatchNormReLU(m + (3 if self._add_coordinates else 0))).add(
           scn.OutputLayer(self._dimension))

        # N Convolutions for Scaled Hypercoordinates
        if self._N > 0:
            self.nConvs = scn.Sequential()
            for fDim in reversed(nPlanes):
                module = scn.Sequential()
                for j in range(self._N):
                    if self._simpleN:
                        module.add(scn.SubmanifoldConvolution(self._dimension, fDim + (3 if j == 0 and self._add_coordinates else 0), fDim, 3, False))
                        module.add(scn.BatchNormLeakyReLU(fDim, leakiness=leakiness))
                    else:
                        block(module, fDim + (3 if j == 0 and self._add_coordinates else 0), fDim)
                self.nConvs.add(module)
        
        # Convolutions for Density Estimation Maps
        if self._density_estimate:
            self.nDensity = scn.Sequential()
            for fDim in reversed(nPlanes):
                module = scn.Sequential()
                for j in range(self._N_density):
                    if self._simpleN:
                        module.add(scn.SubmanifoldConvolution(self._dimension, fDim, fDim, 3, False))
                        module.add(scn.BatchNormLeakyReLU(fDim, leakiness=leakiness))
                    else:
                        block(module, fDim, fDim)
                module.add(scn.SubmanifoldConvolution(self._dimension, fDim, len(self._radii) * 2, 1, False))
                self.nDensity.add(module)

        # Last Linear Layers
        self.seg_linear = torch.nn.Linear(m + (3 if self._add_coordinates else 0), num_classes)
        self.emb_linear = torch.nn.Linear(m + (3 if self._add_coordinates else 0), self._hypDim)


    def forward(self, input):
        """
        point_cloud is a list of length minibatch size (assumes mbs = 1)
        point_cloud[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature
        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        """
        point_cloud = input[0]
        coords = point_cloud[:, 0:self._dimension+1].float()
        features = point_cloud[:, self._dimension+1:self._dimension+2].float()
        if self._add_coordinates:
            normalized_coords = (coords[:, 0:3] - self.spatial_size /2) / float(self.spatial_size)

        x = self.input((coords, features))

        # Embeddings at each layer
        feature_maps = [x]
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            feature_maps.append(x)
            x = self.encoding_conv[i](x)

        # Feature Maps for PPN and Clustering
        feature_clustering = []
        feature_density = []

        for i, layer in enumerate(self.decoding_conv):
            x_non = layer(x)    # Upsampled Via Convolution Before N Convolution
            if self._N > 0:
                if self._add_coordinates:
                    coords = x.get_spatial_locations()[:, :3].float()
                    if torch.cuda.is_available():
                        coords = coords.cuda()
                        normalized_coords = (coords[:, :3] - self.spatial_size /2) / float(self.spatial_size)
                    x.features = torch.cat([x.features, normalized_coords], dim=1)
                x_emb = self.nConvs[i](x)   # feature_clustering space after N Convolutions
                feature_clustering.append(x_emb)
                x_den = self.nDensity[i](x_emb)
                feature_density.append(x_den)
                x_emb = self.embedding_upsampling[i](x_emb)  # Transpose Convolution Upsampling
                encoding_block = feature_maps[-i-2] # Layers from Encoder
                x = self.concat([encoding_block, x_non, x_emb])
                x = self.decoding_blocks[i](x)
            else:
                feature_clustering.append(x)
                encoding_block = feature_maps[-i-2]
                x_non = layer(x)
                x = self.concat([encoding_block, x_non])
                x = self.decoding_blocks[i](x)

        # Last Feature Map
        if self._add_coordinates:
            coords = x.get_spatial_locations()[:, :3].float()
            if torch.cuda.is_available():
                coords = coords.cuda()
                normalized_coords = (coords[:, :3] - self.spatial_size /2) / float(self.spatial_size)
            x.features = torch.cat([x.features, normalized_coords], dim=1)
        x_emb = self.nConvs[-1](x)   # feature_clustering space after N Convolutions
        feature_clustering.append(x_emb)
        if self._density_estimate:
            x_den = self.nDensity[-1](x_emb)
        feature_density.append(x_den)
        seg_coords = x.get_spatial_locations().detach().cpu().numpy()
        perm = np.lexsort((seg_coords[:, 2], seg_coords[:, 1], seg_coords[:, 0], seg_coords[:, 3]))
        x = self.output(x)
        x = x[perm]
        x_seg = self.seg_linear(x)  # Output of UResNet
        # Reverse Sort <feature_clustering> for consistency with scaled labels
        feature_clustering = feature_clustering[::-1]
        feature_density = feature_density[::-1]
        # feature_clustering[0] = self.emb_linear(feature_clustering[0])

        res = {
            'segmentation': [x_seg],
            'cluster_feature': [feature_clustering],
            'density_feature': [feature_density]
        }

        return res


class ClusteringLoss(torch.nn.modules.loss._Loss):
    """

    """

    def __init__(self, cfg, reduction='sum'):
        super(ClusteringLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['modules']['clustering_loss']
        self._num_classes = self._cfg.get('num_classes', 5)
        self._depth = self._cfg.get('stride', 5)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.mseloss = torch.nn.MSELoss(reduction='mean')

        # Clustering Loss Parameters
        self._intra_weight = self._cfg.get('intra_weight', 1.0)
        self._inter_weight = self._cfg.get('inter_weight', 1.0)
        self._reg_weight = self._cfg.get('reg_weight', 0.001)
        self._dimension = self._cfg.get('data_dim', 3)
        self._intra_margin = self._cfg.get('intracluster_margin', 0.5)
        self._inter_margin = self._cfg.get('intercluster_margin', 3.0)
        self._norm = self._cfg.get('norm', 2)

        # Density Loss Parameters
        self._density_estimate = self._cfg.get('density_estimate', True)
        self._density_estimate_weight = self._cfg.get('density_estimate_weight', 0.01)
        self._density_weight = self._cfg.get('density_weight', 0.001)
        self._density_weightF = 1. # Relative Weight between friend/enemy
        self._density_weightE = 1.
        # Target Proportion of Friends
        self._targetF = self._cfg.get('target_friends', [15.0])
        # Target Proportion of Enemies
        self._targetE = self._cfg.get('target_enemies', [0.0])
        self._radii = self._cfg.get('radius', [1.0])
        assert len(self._radii) == len(self._targetE) == len(self._targetF)

        # Total Loss Weighting Parameters 
        self._real_distance_weight = self._cfg.get('real_distance_weight', 0.0)
        self._clustering_weight = self._cfg.get('clustering_weight', 1.0)
        self._density_weight = self._cfg.get('density_weight', 1.0)
        self._segmentation_weight = self._cfg.get('seg_weight', 1.0)

    def distances2(self, points):
        """
        Uses BLAS/LAPACK operations to efficiently compute pairwise distances.
        """
        M = points
        transpose  = M.permute([0, 2, 1])
        zeros = torch.zeros(1, 1, 1)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        inner_prod = torch.baddbmm(zeros, M, transpose, alpha=-2.0, beta=0.0)
        squared    = torch.sum(torch.mul(M, M), dim=-1, keepdim=True)
        squared_tranpose = squared.permute([0, 2, 1])
        inner_prod += squared
        inner_prod += squared_tranpose
        return inner_prod

    def similarity_matrix(self, mat, mat2=None):
        """
        Computes the similarity matrix for pairwise distance computations.

        INPUTS:
            - mat (torch.Tensor): N1 x d Tensor to compute pairwise distances
            - mat2 (torch.Tensor): N2 x d Tensor (optional)
        
        OUTPUT:
            A similarity matrix, giving all pairwise distances for a given 
            hyperspace embedding tensor. 

            If mat2 is given, then function computes N1 x N2 tensor where
            each matrix entry M_ij is the distance from row i of mat and 
            row j of mat2. 
        """
        if mat2 is None:
            r = torch.mm(mat, mat.t())
            diag = r.diag().unsqueeze(0)
            diag = diag.expand_as(r)
            return diag + diag.t() - 2.0 * r
        else:
            r = torch.mm(mat, mat2.t())
            diag1 = torch.mm(mat, mat.t()).diag().unsqueeze(1)
            diag2 = torch.mm(mat2, mat2.t()).diag().unsqueeze(0)
            diag1 = diag1.expand_as(r)
            diag2 = diag2.expand_as(r)
            return diag1 + diag2 - 2.0 * r


    def compute_segmentation_loss(self, output_seg, slabels):
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

    def intra_cluster_loss(self, features, labels, cluster_means, margin=1):
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
        var_loss = 0.0
        n_clusters = len(cluster_means)
        cluster_labels = labels.unique(sorted=True)
        for i, c in enumerate(cluster_labels):
            dists = torch.norm(features[labels == c] - cluster_means[i],
                               p=self._norm,
                               dim=1)
            hinge = torch.clamp(dists - margin, min=0)
            l = torch.mean(torch.pow(hinge, 2))
            var_loss += l
        var_loss /= n_clusters
        return var_loss

    def inter_cluster_loss(self, cluster_means, margin=2):
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
                        dist = torch.norm(c1 - c2, p=self._norm)
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
            reg_loss += torch.norm(cluster_means[i, :], p=self._norm)
        reg_loss /= float(n_clusters)
        return reg_loss


    def combine(self, features, labels):
        '''
        Wrapper function for combining different components of the loss function.
        Inputs:
            features (torch.Tensor): pixel embeddings
            labels (torch.Tensor): ground-truth instance labels
        Returns:
            loss: combined loss, in most cases over a given semantic class.
        '''

        cmeans = self.find_cluster_means(features, labels)
        loss_dist = self.inter_cluster_loss(cmeans, margin=self._inter_margin)
        loss_var = self.intra_cluster_loss(features,
                                           labels,
                                           cmeans,
                                           margin=self._intra_margin)
        loss_reg = self.regularization(cmeans)

        loss = self._intra_weight * loss_var + self._inter_weight * loss_dist + self._reg_weight * loss_reg
        loss_components = [float(self._intra_weight * loss_var),
                           float(self._inter_weight * loss_dist),
                           float(self._reg_weight * loss_reg)]

        return loss, loss_components, cmeans


    def coordinate_distance_loss(self, embedding, coords_class, 
                                 cluster_class, cmeans):
        '''
        Compute real distance loss
        '''
        real_means = self.find_cluster_means(coords_class.float().cuda(), cluster_class[:, -1]).float().cuda()
        predicted_labels = torch.argmin(self.similarity_matrix(embedding, cmeans), dim=1)
        rloss = torch.mean(torch.pow(torch.norm(
            real_means[predicted_labels] - coords_class.float().cuda(), dim=1), 2))
        return rloss

    def compute_density_loss(self, embedding_class, cluster_class):
        """

        """
        if self._density_estimate:
            dloss_map = torch.zeros((embedding_class.shape[0], 2 * len(self._radii))).cuda()
        loss = 0
        dlossF = defaultdict(list)
        dlossE = defaultdict(list)
        n = embedding_class.shape[1]
        dist = self.distances2(embedding_class[None,...][..., :3]).squeeze(0)
        cluster_ids = cluster_class.unique()
        for c in cluster_ids:
            index = cluster_class == c
            embedding_instance = embedding_class[index]
            for j, r in enumerate(self._radii):
                neighbors = dist[index, :] < r
                friends = neighbors[:, index].sum(dim=1).float()
                enemies = neighbors[:, ~index].sum(dim=1).float()
                dloss_map[index, 2 * j] = friends
                dloss_map[index, 2 * j + 1] = enemies
                lossF = torch.mean(torch.sqrt(torch.pow(torch.clamp(self._targetF[j] - friends, min=0), 2)))
                lossE = torch.mean(torch.sqrt(torch.pow(torch.clamp(enemies - self._targetE[j], min=0), 2)))
                loss += self._density_weightF * lossF
                loss += self._density_weightE * lossE
                dlossF[j].append(float(lossF))
                dlossE[j].append(float(lossE))
        #loss /= len(cluster_ids)
        for key, val in dlossF.items():
            dlossF[key] = sum(val) / len(cluster_ids) * len(self._radii)
        for key, val in dlossE.items():
            dlossE[key] = sum(val) / len(cluster_ids) * len(self._radii)
        loss /= len(cluster_ids) * len(self._radii)
        return loss, dlossF, dlossE, dloss_map


    def forward(self, result, segment_label, cluster_label):
        
        seg_loss, seg_acc = 0., 0.
        intracluster_loss = []
        intercluster_loss = []
        reg_loss = []
        real_distance_loss = []

        clustering_loss = []
        intracluster_loss_per_class = defaultdict(list)
        intercluster_loss_per_class = defaultdict(list)
        reg_loss_per_class = defaultdict(list)
        real_distance_loss_per_class = defaultdict(list)
        clustering_loss_per_class = defaultdict(list)

        density_loss = []
        density_lossF_estimate, density_lossF_target = defaultdict(list), defaultdict(list)
        density_lossE_estimate, density_lossE_target = defaultdict(list), defaultdict(list)
        density_estimate_loss_combined = []

        accuracy = []

        # Segmentation Loss
        seg_loss, acc_seg = self.compute_segmentation_loss(result['segmentation'][0], 
            segment_label[0][0])

        # Loop first over scaled feature maps
        for depth in range(self._depth):

            batch_ids = segment_label[0][depth][:, 3].detach().cpu().int().numpy()
            batch_ids = np.unique(batch_ids)
            batch_size = len(batch_ids)

            embedding = result['cluster_feature'][0][depth]
            density_depth = result['density_feature'][0][depth]
            clabels_depth = cluster_label[0][depth]
            slabels_depth = segment_label[0][depth]

            coords = embedding.get_spatial_locations()[:, :4]
            perm = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], coords[:, 3]))
            coords = coords[perm]
            feature_map = embedding.features[perm]
            if self._density_estimate:
                density_map = density_depth.features[perm]

            for bidx in batch_ids:
                batch_mask = coords[:, 3] == bidx
                hypercoordinates = feature_map[batch_mask]
                slabels_event = slabels_depth[batch_mask]
                clabels_event = clabels_depth[batch_mask]
                coords_event = coords[batch_mask]

                # Loop over semantic labels:
                semantic_classes = slabels_event[:, -1].unique()
                n_classes = len(semantic_classes)
                for class_ in semantic_classes:
                    class_mask = slabels_event[:, -1] == class_
                    # Clustering Loss
                    embedding_class = hypercoordinates[class_mask]
                    cluster_class = clabels_event[class_mask]
                    coords_class = coords_event[class_mask]
                    closs, loss_components, cmeans = self.combine(embedding_class, cluster_class[:, -1])
                    clustering_loss_per_class[class_].append(float(closs))
                    intracluster_loss_per_class[class_].append(loss_components[0])
                    intracluster_loss.append(loss_components[0])
                    intercluster_loss_per_class[class_].append(loss_components[1])
                    intercluster_loss.append(loss_components[1])
                    reg_loss_per_class[class_].append(loss_components[2])
                    reg_loss.append(loss_components[2])
                    # Real Distance Loss
                    real_dloss = self.coordinate_distance_loss(embedding_class,
                        coords_class, cluster_class, cmeans)
                    real_distance_loss_per_class[class_].append(float(real_dloss))
                    # Density Loss
                    dloss, dlossF, dlossE, dloss_map = self.compute_density_loss(embedding_class, cluster_class[:, -1])
                    for key, val in dlossF.items():
                        density_lossF_target[key].append(val)
                    for key, val in dlossE.items():
                        density_lossE_target[key].append(val)

                    if self._density_estimate:
                        density_estimate_loss = self.mseloss(density_map[batch_mask][class_mask],
                                                             dloss_map)
                        density_estimate_loss_combined.append(density_estimate_loss)
                    density_loss.append(dloss)
                    real_distance_loss.append(real_dloss)
                    clustering_loss.append(closs)

        density_loss = sum(density_loss) / len(density_loss)
        real_distance_loss = sum(real_distance_loss) / len(real_distance_loss)
        clustering_loss = sum(clustering_loss) / len(clustering_loss)

        total_loss = 0.0
        total_loss += self._segmentation_weight * seg_loss
        total_loss += self._clustering_weight * clustering_loss
        total_loss += self._density_weight * density_loss
        total_loss += self._real_distance_weight * real_distance_loss

        res = {
            'accuracy': float(acc_seg),
            'seg_loss': float(seg_loss),
            'seg_acc': float(acc_seg),
            'intracluster_loss': sum(intracluster_loss) / len(intracluster_loss),
            'intercluster_loss': sum(intercluster_loss) / len(intercluster_loss),
            'reg_loss': sum(reg_loss) / len(reg_loss),
            'real_distance_loss': float(real_distance_loss),
            'density_loss': float(density_loss),
            'clustering_loss': float(clustering_loss)
        }

        if self._density_estimate:
            density_estimate_loss_combined = sum(density_estimate_loss_combined) \
                / len(density_estimate_loss_combined)
            total_loss += density_estimate_loss_combined * self._density_estimate_weight
            res['density_estimate_loss'] = density_estimate_loss_combined * self._density_estimate_weight

        res['loss'] = total_loss

        print(res)

        return res