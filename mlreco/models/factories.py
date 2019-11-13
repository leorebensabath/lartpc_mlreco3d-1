def model_dict():

    from . import uresnet_ppn
    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    from . import chain_track_clustering
    from . import uresnet_ppn_chain
    from . import edge_gnn
    from . import full_edge_gnn
    from . import node_gnn
    from . import iter_edge_gnn
    from . import chain_gnn
    from . import cluster_edge_gnn
    from . import cluster_dir_gnn
    from . import uresnet_clustering

    from . import discriminative_loss
    from . import clustercnn_single
    from . import clustercnn_neighbors
    from . import clustercnn_stable
    from . import clustercnn_stack
    from . import clustercnn_distances
    from . import clustercnn_offset
    from . import clustercnn_se

    from . import clusternet


    # Make some models available (not all of them, e.g. PPN is not standalone)
    models = {
        # Regular UResNet + PPN
        "uresnet_ppn": (uresnet_ppn.PPNUResNet, uresnet_ppn.SegmentationLoss),
        # Adding point classification layer
        "uresnet_ppn_type": (uresnet_ppn_type.PPNUResNet, uresnet_ppn_type.SegmentationLoss),
        # Using SCN built-in UResNet
        "uresnet": (uresnet.UResNet, uresnet.SegmentationLoss),
        # Using our custom UResNet
        "uresnet_lonely": (uresnet_lonely.UResNet, uresnet_lonely.SegmentationLoss),
        # Chain test for track clustering (w/ DBSCAN)
        "chain_track_clustering": (chain_track_clustering.Chain, chain_track_clustering.ChainLoss),
        "uresnet_ppn_chain": (uresnet_ppn_chain.Chain, uresnet_ppn_chain.ChainLoss),
        # Clustering
        "uresnet_clustering": (uresnet_clustering.UResNet, uresnet_clustering.SegmentationLoss),
        # Edge Model
        "edge_model": (edge_gnn.EdgeModel, edge_gnn.EdgeChannelLoss),
        # Full Edge Model
        "full_edge_model": (full_edge_gnn.FullEdgeModel, full_edge_gnn.FullEdgeChannelLoss),
        # Full Node Model
        "node_model": (node_gnn.NodeModel, node_gnn.NodeChannelLoss),
        # MST edge model
        ##"mst_edge_model": (mst_gnn.MSTEdgeModel, mst_gnn.MSTEdgeChannelLoss),
        # Iterative Edge Model
        "iter_edge_model": (iter_edge_gnn.IterativeEdgeModel, iter_edge_gnn.IterEdgeChannelLoss),
        # full cluster model
        "clust_edge_model": (cluster_edge_gnn.EdgeModel, cluster_edge_gnn.EdgeChannelLoss),
        # direction model
        "clust_dir_model": (cluster_dir_gnn.EdgeModel, cluster_dir_gnn.EdgeChannelLoss),
        # ClusterUNet Single
        "clusterunet_single": (clustercnn_single.ClusterCNN, clustercnn_single.ClusteringLoss),
        # ClusterUnet Single Offset
        "clustercnn_offset": (discriminative_loss.UResNet, clustercnn_offset.ClusteringLoss),
        # Same as ClusterUNet Single, but coordinate concat is done in first input layer.
        "discriminative_loss": (discriminative_loss.UResNet, discriminative_loss.DiscriminativeLoss),
        # ClusterUNet Affinity (NOTE: Unstable training due to GPU memory issues)
        "clusterunet_affinity": (clustercnn_neighbors.ClusterCNN, clustercnn_neighbors.ClusteringLoss),
        # ClusterUNet Stable (Multiscale loss with attention weighting and voxel-centroid push loss)
        "clusterunet": (clustercnn_stable.ClusterCNN, clustercnn_stable.ClusteringLoss),
        # ClusterCNN Stack (Multiscale Enhanced Loss Stacked UResNet)
        "clusterStack": (clustercnn_stack.ClusterCNN, clustercnn_stack.ClusteringLoss),
        # ClusterCNN Distances (Multiscale Enhanced Loss Stacked UResNet with Distance Estimation)
        "clustercnn_distances": (clustercnn_distances.ClusterCNN, clustercnn_distances.ClusteringLoss),
        # Colossal ClusterNet Model to Wrap them all
        "clusternet": (clusternet.ClusterCNN, clusternet.ClusteringLoss),
        # Spatial Embeddings
        "spatial_embeddings": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss),
        # Spatial Embeddings 2
        "spatial_embeddings2": (clustercnn_se.ClusterCNN, clustercnn_se.ClusteringLoss2)
    }
    # "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss)
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
