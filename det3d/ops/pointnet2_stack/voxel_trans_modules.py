from timm.models.layers import drop
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import voxel_query_utils
from typing import List


class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        """
        src: with shape (n_samples, total_n_points, C) as (L, B, E)
        """

        src = self.norm1(src)
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        
        return src



class NeighborVoxelTransSAModuleMSG(nn.Module):
                 
    def __init__(self, *, query_ranges: List[List[int]], radii: List[float], 
        nsamples: List[int], mlps: List[List[int]], nheads=1, dropout=0., num_layers=2, use_xyz: bool = True, pool_method='max_pool'):
        """
        Args:
            query_ranges: list of int, list of neighbor ranges to group with
            nsamples: list of int, number of samples in each ball query
            mlps: list of list of int, spec of the pointnet before the global pooling for each scale
            use_xyz:
            pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(query_ranges) == len(nsamples) == len(mlps)
        
        self.groupers = nn.ModuleList()
        self.mlps_pos = nn.ModuleList()
        self.chunks = nn.ModuleList()
        self.mlps_out = nn.ModuleList()
        basic_encoder = TransformerEncoderLayerPreNorm

        for i in range(len(query_ranges)):
            max_range = query_ranges[i]
            nsample = nsamples[i]
            radius = radii[i]
            self.groupers.append(voxel_query_utils.VoxelQueryAndGrouping(max_range, radius, nsample))
            mlp_spec = mlps[i]
            nc_in = mlp_spec[0]
            nc_out = mlp_spec[-1]
            n_head = nheads
            
            # this as a pe
            cur_mlp_pos = nn.Sequential(
                nn.Conv2d(3, nc_in//2, kernel_size=1, bias=False),
                nn.BatchNorm2d(nc_in//2),
                nn.ReLU(),
                nn.Conv2d(nc_in//2, nc_in, kernel_size=1, bias=True),
            )

            # pre qkv proj
            # 这个MultiheadAttention里面会自己做qkv_proj的.
            chunk = nn.TransformerEncoder(
                encoder_layer=basic_encoder(d_model=nc_in, nhead=n_head, dim_feedforward=2*nc_in, dropout=dropout),
                num_layers=num_layers,
            )

            cur_mlp_out = nn.Sequential(
                nn.Conv1d(nc_in, nc_out, kernel_size=1, bias=True),
                # nn.BatchNorm1d(mlp_spec[2]),
                # nn.ReLU()
            )

            self.mlps_pos.append(cur_mlp_pos)
            self.chunks.append(chunk)
            self.mlps_out.append(cur_mlp_out)

        self.relu = nn.ReLU()
        self.pool_method = pool_method

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, \
                                        new_coords_init, down_ratio, features, voxel2point_indices):
        """
        :param xyz: (N1 + N2 ..., 3) tensor of the xyz coordinates of the features
        :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
        :param new_xyz: (M1 + M2 ..., 3)
        :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
        :param features: (N1 + N2 ..., C) tensor of the descriptors of the the features
        :param point_indices: (B, Z, Y, X) tensor of point indices
        :return:
            new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
            new_features: (M1 + M2 ..., \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        
        assert len(self.groupers) == 1 # just single-scale
        # change the order to [batch_idx, z, y, x]
        # new_coords = new_coords[:, [0, 3, 2, 1]].contiguous()
        new_coords_bs = new_coords_init[:, 0:1]
        new_coords_x = new_coords_init[:, 1:2] // down_ratio[0]
        new_coords_y = new_coords_init[:, 2:3] // down_ratio[1]
        new_coords_z = new_coords_init[:, 3:4] // down_ratio[2]
        new_coords = torch.cat([new_coords_bs, new_coords_z, new_coords_y, new_coords_x], dim=1).int().contiguous()

        new_features_list = []
        for k in range(len(self.groupers)):
            # features_in: (M1+M2, C)
            features_in = features
            # grouped_features: (M1+M2, C, nsample)
            # grouped_xyz: (M1+M2, 3, nsample)
            grouped_features, grouped_xyz, empty_ball_mask = self.groupers[k](
                new_coords, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features_in, voxel2point_indices
            )
            grouped_features[empty_ball_mask] = 0

            # grouped_features: (1, C, M1+M2, nsample)
            grouped_features = grouped_features.permute(1, 0, 2).unsqueeze(dim=0)
            
            # torch.cuda.synchronize()
            # print('==> grouped_xyz.shape', grouped_xyz.shape)
            # print('==> new_xyz[100, ...]: ', new_xyz[100, ...])
            # print('==> grouped_xyz[100, ...]', grouped_xyz[100, ...])
            # print('==> new_xyz[1800, ...]: ', new_xyz[1800, ...])
            # print('==> grouped_xyz[1800, ...]', grouped_xyz[1800, ...])
            # print('==> new_xyz[8800, ...]: ', new_xyz[8800, ...])
            # print('==> grouped_xyz[8800, ...]', grouped_xyz[8800, ...])
            
            # grouped_xyz: (M1+M2, 3, nsample)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(-1)
            grouped_xyz[empty_ball_mask] = 0
            # grouped_xyz: (1, 3, M1+M2, nsample)
            grouped_xyz = grouped_xyz.permute(1, 0, 2).unsqueeze(0)
            # grouped_xyz: (1, C, M1+M2, nsample)

            # print('==> grouped_features[0, :, 5, :]: ', grouped_features[0, :, 5, :])
            # print('==> grouped_features[0, :, 105, :]: ', grouped_features[0, :, 105, :])
            # print('==> grouped_xyz[0, :, 5, :]: ', grouped_xyz[0, :, 5, :])
            # print('==> grouped_xyz[0, :, 105, :]: ', grouped_xyz[0, :, 105, :])

            position_features = self.mlps_pos[k](grouped_xyz)
            input_features = grouped_features + position_features

            # perform transformer.
            # torch.nn.MultiheadAttention requires the (L, B, E) shape.
            # (1, C, M1+M2, nsample) -> (C, M1+M2, nsample) -> (nsample, M1+M2, C)
            input_features = input_features.squeeze(0).permute(2, 1, 0)

            # NOTE empty_ball_mask 是否需要用作attention的mask? 它描述的是整个ball是empty的情况
            # (nsample, M1+M2, C) -> (C, M1+M2, nsample) -> (1, C, M1+M2, nsample)
            transformed_features = self.chunks[k](input_features).permute(2,1,0).unsqueeze(0)


            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    transformed_features, kernel_size=[1, transformed_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    transformed_features, kernel_size=[1, transformed_features.size(3)]
                ).squeeze(dim=-1)  # (1, C, M1 + M2 ...)
            else:
                raise NotImplementedError
            
            # a fc after max_pooling 
            new_features = self.mlps_out[k](new_features)
            new_features = new_features.squeeze(dim=0).permute(1, 0)  # (M1 + M2 ..., C)
            new_features_list.append(new_features)
        
        # (M1 + M2 ..., C)
        new_features = torch.cat(new_features_list, dim=1)
        return new_features

