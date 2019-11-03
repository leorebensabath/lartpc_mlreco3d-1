from . import loss
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
    losses = {
        'single': loss.DiscriminativeLoss,
        'multi': loss.MultiScaleLoss,
        'multi-weighted': loss.WeightedMultiLoss,
        'multi-repel': loss.MultiScaleLoss2,
        # NOTE: Not Yet Working 'multi-both': loss.EnhancedEmbeddingLoss,
        # NOTE: Not Yet Working 'multi-neighbors': loss.NeighborLoss,
        'multi-distance': loss.DistanceEstimationLoss
    }
    return losses

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
    if not name in loss_fns:
        raise Exception("Unknown clustering loss function name provided")
    return loss_fns[name]