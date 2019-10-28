import torch
import torch.nn as nn
import numpy as np
import sparseconvnet as scn


def add_normalized_coordinates(input):
    '''
    Utility Method for attaching normalized coordinates to
    sparse tensor features.

    INPUTS:
        - input (scn.SparseConvNetTensor): sparse tensor to
        attach normalized coordinates with range (-1, 1)

    RETURNS:
        - output (scn.SparseConvNetTensor): sparse tensor with 
        normalized coordinate concatenated to first three dimensions.
    '''
    output = scn.SparseConvNetTensor()
    with torch.no_grad():
        coords = input.get_spatial_locations().float()
        normalized_coords = (coords[:, :3] - input.spatial_size.float() / 2) \
            / (input.spatial_size.float() / 2)
        if torch.cuda.is_available():
            normalized_coords = normalized_coords.cuda()
        output.features = torch.cat([normalized_coords, input.features], dim=1)
    output.metadata = input.metadata
    output.spatial_size = input.spatial_size
    return output