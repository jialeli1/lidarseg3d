import torch
from torch import nn
from torch.nn import functional as F

from ..registry import READERS


@READERS.register_module
class MeanVoxelFeatureExtractor(nn.Module):
    """
    MeanVFE (AvgVFE)
    """
    def __init__(
        self, num_input_features=4, name="MeanVoxelFeatureExtractor"
    ):
        super(MeanVoxelFeatureExtractor, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        assert self.num_input_features == features.shape[-1]

        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        return points_mean.contiguous()



@READERS.register_module
class ImprovedMeanVoxelFeatureExtractor(nn.Module):
    """
    MeanVFE with multiple descriptor, max, min, density, std
    """
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV72"
    ):
        super(VoxelFeatureExtractorV72, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        """
        features: [n_voxels, n_points_per_voxel, num_input_features])
        num_voxels: [n_voxels, ], num points per voxel
        """
        assert self.num_input_features == features.shape[-1]
        n_points_per_voxel = features.shape[1]

        points_mean = features.sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        # [n_voxels, n_points_per_voxel, num_input_features]) -> [n_voxels, n_points_per_voxel]
        point_mask = (torch.sum(features, dim=-1) != 0).float() #(torch.float32)

        points_xyz = features[:, :, :3]     # (x,y,z) in cartesian
        points_feats = features[:, :, 3:]   # other input features 


        # max operation, excluding the zero-paddings
        max_x = torch.max(points_xyz[:, :, 0] - (1-point_mask) * 1e5, dim=1)[0] # [n_voxels, 1]
        max_y = torch.max(points_xyz[:, :, 1] - (1-point_mask) * 1e5, dim=1)[0] # [n_voxels, 1]
        max_z = torch.max(points_xyz[:, :, 2] - (1-point_mask) * 1e5, dim=1)[0] # [n_voxels, 1]
        points_max = torch.cat([max_x[:, None], max_y[:, None], max_z[:, None]], dim=-1)


        # min operation, excluding the zero-paddings
        min_x = torch.min(points_xyz[:, :, 0] + (1-point_mask) * 1e5, dim=1)[0]
        min_y = torch.min(points_xyz[:, :, 1] + (1-point_mask) * 1e5, dim=1)[0]
        min_z = torch.min(points_xyz[:, :, 2] + (1-point_mask) * 1e5, dim=1)[0]
        points_min = torch.cat([min_x[:, None], min_y[:, None], min_z[:, None]], dim=-1)


        # density: num_voxels.float() / n_points_per_voxel
        # [n_voxels, n_points_per_voxel] -> [n_voxels, ]
        density = torch.sum(point_mask, dim=-1) / n_points_per_voxel 


        # std: mean(|| p_i - p_m ||)       
        # [n_voxels, n_points, 3] -> [n_voxels, n_points, ]
        norm = torch.norm( (points_xyz - points_mean[:, None, 0:3])*point_mask[:, :, None], p=2, dim=-1 )
        # [n_voxels, n_points, ] -> [n_voxels, ]
        std = torch.sum(norm, dim=1) / num_voxels.type_as(features) 



        # the descriptor makes the model more stable
        voxel_features = torch.cat([points_mean[:, 0:3], points_max, points_min, points_mean[:, 3:], density[:, None], std[:, None]], dim=-1)


        return voxel_features.contiguous()



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


@READERS.register_module
class TransformerVoxelFeatureExtractor(nn.Module):
    """
    TransVFE: Transformer-based Voxel Feature Extractor
    """
    def __init__(
        self, num_input_features=4, num_compressed_features=16, num_embed=64, num_head=4, num_layers=2, norm_cfg=None, name="TransformerVoxelFeatureExtractor"
    ):
        super(TransformerVoxelFeatureExtractor, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        
        # with some extra descriptor features
        num_descriptor_features = num_input_features + 3 + 3 + 1 + 1 # min_xyz, max_xyz, density, std

        # feature_conv
        self.feature_conv = nn.Sequential(
            nn.Conv1d(num_input_features + num_descriptor_features, num_embed, 1, bias=True),
        )

        # local transformer for feature embedding
        self.chunck = nn.TransformerEncoder(
            encoder_layer=TransformerEncoderLayerPreNorm(d_model=num_embed, nhead=num_head, dim_feedforward=num_embed*2, dropout=0,),
            num_layers=num_layers,
        )
        
        if num_compressed_features > 0:
            self.compress_layer = nn.Sequential(
                    nn.Linear(num_embed, num_compressed_features),
                    nn.ReLU())
            self.num_out_features = num_compressed_features
        else: 
            self.compress_layer = None
            self.num_out_features = num_embed


    def forward(self, features, num_voxels, coors=None):
        """
        features: [n_voxels, n_points_per_voxel, num_input_features])
        num_voxels: [n_voxels, ], num points per voxel
        """
        assert self.num_input_features == features.shape[-1]
        n_points_per_voxel = features.shape[1]

        points_mean = features.sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)

        # [n_voxels, n_points_per_voxel, num_input_features]) -> [n_voxels, n_points_per_voxel]
        point_mask = (torch.sum(features, dim=-1) != 0).float() #(torch.float32)


        points_xyz = features[:, :, :3]     # (x,y,z) in cartesian
        points_feats = features[:, :, 3:]   # other input features 



        # max operation, excluding the zero-paddings
        max_x = torch.max(points_xyz[:, :, 0] - (1-point_mask) * 1e5, dim=1)[0] # [n_voxels, 1]
        max_y = torch.max(points_xyz[:, :, 1] - (1-point_mask) * 1e5, dim=1)[0] # [n_voxels, 1]
        max_z = torch.max(points_xyz[:, :, 2] - (1-point_mask) * 1e5, dim=1)[0] # [n_voxels, 1]
        points_max = torch.cat([max_x[:, None], max_y[:, None], max_z[:, None]], dim=-1)

        # min operation, excluding the zero-paddings
        min_x = torch.min(points_xyz[:, :, 0] + (1-point_mask) * 1e5, dim=1)[0]
        min_y = torch.min(points_xyz[:, :, 1] + (1-point_mask) * 1e5, dim=1)[0]
        min_z = torch.min(points_xyz[:, :, 2] + (1-point_mask) * 1e5, dim=1)[0]
        points_min = torch.cat([min_x[:, None], min_y[:, None], min_z[:, None]], dim=-1)

        # density: num_voxels.float() / n_points_per_voxel
        # [n_voxels, n_points_per_voxel] -> [n_voxels, ]
        density = torch.sum(point_mask, dim=-1) / n_points_per_voxel 

        # std: mean(|| p_i - p_m ||)       
        # [n_voxels, n_points, 3] -> [n_voxels, n_points, ]
        norm = torch.norm( (points_xyz - points_mean[:, None, 0:3])*point_mask[:, :, None], p=2, dim=-1 )
        # [n_voxels, n_points, ] -> [n_voxels, ]
        std = torch.sum(norm, dim=1) / num_voxels.type_as(features) 

        # the descriptor makes the model more stable
        descriptor = torch.cat([points_mean[:, 0:3], points_max, points_min, points_mean[:, 3:], density[:, None], std[:, None]], dim=-1)


        # expand + concat
        descriptor = descriptor[:, None, :].expand(-1, n_points_per_voxel, -1)
        # [n_voxels, n_points_per_voxel, C] -> [n_voxels, C, n_points_per_voxel]
        point_features = torch.cat([features, descriptor], dim=-1).permute(0,2,1)
        

        # transformer chunk
        # [n_voxels, C_pp, n_points_per_voxel]
        point_features = self.feature_conv(point_features) 
        # [n_voxels, C_pp, n_points_per_voxel] -> [n_points_per_voxel, n_voxels, C_pp] as (L,B,E)
        point_features = point_features.permute(2,0,1)
        transformed_feats = self.chunck(point_features)
        # [n_points_per_voxel, n_voxels, C_pp] -> [n_voxels, C_pp, n_points_per_voxel]
        transformed_feats = transformed_feats.permute(1,2,0)

        # final max_pooling
        voxel_features = torch.max(transformed_feats, dim=2)[0]

        if self.compress_layer is not None:
            voxel_features = self.compress_layer(voxel_features)

        return voxel_features.contiguous()

