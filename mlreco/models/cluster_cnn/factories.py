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
        "stack_multi": embeddings.StackedEmbeddings
    }

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