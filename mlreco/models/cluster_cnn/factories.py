from mlreco.models import uresnet_lonely
from mlreco.models import uresnet
from . import clustering_loss
from . import adaptis
from . import branchnet
from . import clusternet
from . import stacknet

def cluster_model_dict():
    """
    returns dictionary of clustering models
    """
    
    from . import clusternet
    
    models = {
        # UResNet Backbone with Single Layer Loss
        "clusternet" : clusternet.ClusterUNet
    }
    
    return models


def cluster_model_construct(name):
    models = cluster_model_dict()
    if not name in models:
        raise Exception("Unknown clustering model name provided")
    return models[name]