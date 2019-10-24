def cluster_model_dict():
    """
    returns dictionary of clustering models
    """
    
    from . import clusternet
    
    models = {
        "single" : clusternet.KDENet
    }
    
    return models


def cluster_model_construct(name):
    models = cluster_model_dict()
    if not name in models:
        raise Exception("Unknown clustering model name provided")
    return models[name]