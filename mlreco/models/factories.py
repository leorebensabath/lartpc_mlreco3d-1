def model_dict():

    from . import uresnet_ppn
    from . import uresnet_ppn_type
    from . import uresnet_lonely
    from . import uresnet
    from . import chain_track_clustering
    from . import uresnet_ppn_chain
    from . import edge_gnn
    from . import full_edge_gnn
    from . import iter_edge_gnn
    from . import chain_gnn
    #from . import mst_gnn
    from . import cluster_edge_gnn
    from . import cluster_dir_gnn
    from . import uresnet_clustering

    from . import discriminative_loss
    from . import discriminative_multiLayerLoss
    from . import uresnet_clustering_alt
    from . import uresnet_clustering_v2
    from . import uresnet_clustering_v2_decoder_separate
    from . import uresnet_clustering_density
    from . import uresnet_clustering_alt2

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
        # MST edge model
        #"mst_edge_model": (mst_gnn.MSTEdgeModel, mst_gnn.MSTEdgeChannelLoss),
        # Iterative Edge Model
        "iter_edge_model": (iter_edge_gnn.IterativeEdgeModel, iter_edge_gnn.IterEdgeChannelLoss),
        # full cluster model
        "clust_edge_model": (cluster_edge_gnn.EdgeModel, cluster_edge_gnn.EdgeChannelLoss),
        # direction model
        "clust_dir_model": (cluster_dir_gnn.EdgeModel, cluster_dir_gnn.EdgeChannelLoss),
        # Discriminative Loss
        "discriminative_loss": (discriminative_loss.UResNet, discriminative_loss.DiscriminativeLoss),
        # Discriminative MultiLayer Loss (Loss at Decoding Layers)
        "discriminative_multiLayerLoss": (discriminative_multiLayerLoss.UResNet, discriminative_multiLayerLoss.DiscriminativeLoss),
        # Uresnet Clustering
        "uresnet_clustering_alt": (uresnet_clustering_alt.UResNet, uresnet_clustering_alt.DiscriminativeLoss),
        # Uresnet Clustering2
        "uresnet_clustering_alt2": (uresnet_clustering_alt2.UResNet, uresnet_clustering_alt2.DiscriminativeLoss),
        # Uresnet Clustering V2
        "uresnet_clustering_v2": (uresnet_clustering_v2.UResNet, uresnet_clustering_v2.ClusteringLoss),
        # Uresnet Clustering V2, with Two Decoders
        "uresnet_clustering_v2_decoder_separate": (uresnet_clustering_v2_decoder_separate.UResNet, uresnet_clustering_v2_decoder_separate.ClusteringLoss),
        # Uresnet Clustering Density
        "uresnet_clustering_density": (uresnet_clustering_density.ClusterNet_Density, uresnet_clustering_density.DensityLoss)
    }
    # "chain_gnn": (chain_gnn.Chain, chain_gnn.ChainLoss)
    return models


def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
