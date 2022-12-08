import torch
import torch.nn as nn
import torch.nn.functional as F

from det3d.ops.pointnet2_batch.pointnet2_utils import three_nn, three_interpolate


def three_interpolate_wrap(new_coords, coords, features, batch_size, **kwargs):
    """
    new_coords: (M1+M2+..., 4)
    coords: (N1+N2+..., 4) 
    features: (N1+N2+..., c)

    new_features: (M1+M2+..., c)
    """
    # assert coords.shape[0] == features.shape[0], "coords.shape: {}, features.shape: {}".format(coords.shape, features.shape) 
    
    new_features_list = []
    for i in range(batch_size):
        bs_mask = (coords[:, 0] == i)
        new_bs_mask = (new_coords[:, 0] == i)

        # (N_i, 3) -> (B=1, N_i, 3)
        cur_xyz = coords[bs_mask][:, 1:4].unsqueeze(dim=0)
        # (M_i, 3) -> (B=1, M_i, 3)
        cur_new_xyz = new_coords[new_bs_mask][:, 1:4].unsqueeze(dim=0)

        # get 3nn idx
        cur_dist, cur_idx = three_nn(cur_new_xyz.contiguous(), cur_xyz.contiguous())
        cur_dist_recip = 1.0 / (cur_dist + 1e-8)
        cur_norm = torch.sum(cur_dist_recip, dim=2, keepdim=True)
        cur_weight = cur_dist_recip / cur_norm

        # get features
        # (B=1, N_i, C) -> (B=1, C, N_i)
        cur_features = features[bs_mask].contiguous().unsqueeze(dim=0).permute(0,2,1).contiguous()

        # 3NN interpolate
        cur_interpolated_feats = three_interpolate(cur_features, cur_idx, cur_weight).contiguous()
        
        # (B=1, C, N_i) -> (B=1, N_i, C) -> (N_i, C)
        cur_interpolated_feats = cur_interpolated_feats.permute(0,2,1).contiguous().squeeze(dim=0)
        new_features_list.append(cur_interpolated_feats)

    # cat for output
    new_features = torch.cat(new_features_list, dim=0)
    
    # check
    # assert new_features.shape[0] == new_coords.shape[0]
    # assert new_features.shape[1] == features.shape[1]

    return new_features

