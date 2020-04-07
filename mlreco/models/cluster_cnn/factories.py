from . import losses
from . import embeddings


def backbone_dict():
    """
    returns dictionary of clustering models
    """
    from mlreco.models.layers import uresnet
    from mlreco.models.layers import fpn

    models = {
        # Encoder-Decoder Type Backbone Architecture.
        "uresnet": uresnet.UResNet,
        "fpn": fpn.FPN
    }

    return models


def cluster_model_dict():
    '''
    Returns dictionary of implemented clustering layers.
    '''
    models = {
        "single": None,
        "multi": embeddings.ClusterEmbeddings,
        "multi_fpn": embeddings.ClusterEmbeddingsFPN,
        "multi_stack": embeddings.StackedEmbeddings
    }
    return models


def clustering_loss_dict():
    '''
    Returns dictionary of various clustering losses with enhancements.
    '''
    from .losses import multi_layers, single_layers, spatial_embeddings, spherical_embeddings
    loss = {
        'single': single_layers.DiscriminativeLoss,
        'multi': multi_layers.MultiScaleLoss,
        'multi-weighted': multi_layers.DistanceEstimationLoss3,
        'multi-repel': multi_layers.DistanceEstimationLoss2,
        'multi-distance': multi_layers.DistanceEstimationLoss,
        'se_bce': spatial_embeddings.MaskBCELoss2,
        'se_bce_ellipse': spatial_embeddings.MaskBCELossBivariate,
        'se_lovasz': spatial_embeddings.MaskLovaszHingeLoss,
        'se_lovasz_inter': spatial_embeddings.MaskLovaszInterLoss,
        'se_focal': spatial_embeddings.MaskFocalLoss,
        'se_multivariate': spatial_embeddings.MultiVariateLovasz,
        'se_multivariate_quat': spatial_embeddings.MultivariateQuaternion,
        'se_multi_entropy': spatial_embeddings.MultiVariateEntropy,
        'se_ce_dice': spatial_embeddings.CEDiceLoss,
        'se_ce_f1': spatial_embeddings.CEF1Loss,
        'se_ce_lovasz': spatial_embeddings.CELovaszLoss,
        'se_offset_lovasz': spatial_embeddings.OffsetLovasz,
        'se_multi_layer_1': spherical_embeddings.MaskEmbeddingLoss
        # 'se_multi_layer': spatial_embeddings.MultiLayerLovasz
    }
    return loss


def backbone_construct(name):
    models = backbone_dict()
    if not name in models:
        raise Exception("Unknown backbone architecture name provided")
    return models[name]


def cluster_model_construct(name):
    models = cluster_model_dict()
    if not name in models:
        raise Exception("Unknown clustering model name provided")
    return models[name]


def clustering_loss_construct(name):
    loss_fns = clustering_loss_dict()
    print(name)
    if not name in loss_fns:
        raise Exception("Unknown clustering loss function name provided")
    return loss_fns[name]
