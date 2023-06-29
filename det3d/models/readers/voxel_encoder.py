import torch
from torch import nn
from torch.nn import functional as F
import torch_scatter # for dynamic VFE

from ..registry import READERS
from det3d.core.utils import common_utils



def cart2cylind(input_xyz):
    """
    input_xyz: torch.tensor 
    """
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.cat((rho.unsqueeze(-1), phi.unsqueeze(-1), input_xyz[:, 2:]), dim=1)


def cart2spherical(input_xyz):
    """
    input_xyz: torch.tensor 
    phi: 方位角
    pitch: 天顶角
    """
    assert input_xyz.shape[1] == 3
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2 + input_xyz[:, 2] ** 2)
    phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])

    # NOTE: 这里的pitch原点和常规规定有点不一样，这里是以xy平面为原点的
    r = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    pitch = torch.atan2(input_xyz[:, 2], r)

    return torch.cat([rho.unsqueeze(-1), phi.unsqueeze(-1), pitch.unsqueeze(-1)], dim=1)




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
        self, num_input_features=4, norm_cfg=None, name="ImprovedMeanVoxelFeatureExtractor"
    ):
        super(ImprovedMeanVoxelFeatureExtractor, self).__init__()
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




@READERS.register_module
class PolarNetDynamicVoxelFeatureExtractor(nn.Module):

    def __init__(self, grid_size, point_cloud_range, average_points, num_input_features, num_output_features, fea_compre=None, voxel_label_enc=None, **kwargs):
        super(PolarNetDynamicVoxelFeatureExtractor, self).__init__()
        """
        ref 
        """
        # 2：cyl_x,  cyl_y
        # 8: additional features
        fea_dim = num_input_features + 2+8
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, num_output_features)
        )


        self.pool_dim = num_output_features

        # point feature compression
        self.fea_compre = fea_compre
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                    nn.Linear(self.pool_dim, self.fea_compre),
                    nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim



        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points
        # compute the voxel_size by grid_size
        self.voxel_size = [
            (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0],
            (point_cloud_range[4] - point_cloud_range[1]) / grid_size[1],
            (point_cloud_range[5] - point_cloud_range[2]) / grid_size[2],
        ]
        print("==> in DynamicPolarNetVoxelFeatureExtractor, grid_size: ", grid_size)
        print("==> in DynamicPolarNetVoxelFeatureExtractor, point_cloud_range: ", point_cloud_range)
        print("==> in DynamicPolarNetVoxelFeatureExtractor, voxel_size: ", self.voxel_size)

        self.voxel_label_enc = voxel_label_enc # "major or compact"


    def voxelize(self, points, reverse_coors=True):
        """
        points: [N0 + N1 + ... Nb, 3], 
        """
        point_vcoor0 = torch.floor( (points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] ).int()
        point_vcoor1 = torch.floor( (points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1] ).int()
        point_vcoor2 = torch.floor( (points[:, 2] - self.point_cloud_range[2]) / self.voxel_size[2] ).int()
        

        # NOTE: to be improved
        point_vcoor0 = torch.clamp(point_vcoor0, 0, self.grid_size[0]-1)
        point_vcoor1 = torch.clamp(point_vcoor1, 0, self.grid_size[1]-1)
        point_vcoor2 = torch.clamp(point_vcoor2, 0, self.grid_size[2]-1)


        valid_flag0 = (0 <= point_vcoor0) & (point_vcoor0 < self.grid_size[0])
        valid_flag1 = (0 <= point_vcoor1) & (point_vcoor1 < self.grid_size[1])
        valid_flag2 = (0 <= point_vcoor2) & (point_vcoor2 < self.grid_size[2])
        valid_flag = valid_flag0 & valid_flag1 & valid_flag2

        point_vcoors = torch.stack([point_vcoor0, point_vcoor1, point_vcoor2], dim=-1)
        
        if reverse_coors:
            point_vcoors = point_vcoors[:, [2,1,0]]


        return point_vcoors, valid_flag


    def prepare_input_feature(self, point_features, point_vcoors, inv_idx):
        """
        point_features: [#points, C], C is [cyl_x, cyl_y, cyl_z, card_x, card_y, ...]
        point_vcoors: [#points, 4], 4 is [bs_idx, vz, vy, vx]
        inv_idx: [#point, ]

        point_init_features: [#point, C + 5 + 3]
        """
        # 前五维都归一化一下
        pc_mean = torch_scatter.scatter_mean(point_features[:, :5], inv_idx, dim=0)[inv_idx]
        nor_pc = point_features[:, :5] - pc_mean

        # 恢复出极坐标, 针对的是向下取整的量化方式
        # [#point, 3]
        point_voxel_centers = common_utils.get_voxel_centers(
            point_vcoors[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        # 前三维作差
        center_to_point = point_features[:, :3] - point_voxel_centers

        point_init_features = torch.cat((point_features, nor_pc, center_to_point), dim=1)

        return point_init_features


    def voxelize_labels(self, point_labels, point_vcoors):
        """
        point_labels: [#points, ]
        point_vcoors: [#points, 4], 4 is [bs_idx, vz, vy, vx]
        
        labels: [#voxel, ], major voted voxel label
        """
        lbxyz = torch.cat([point_labels.reshape(-1, 1), point_vcoors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        
        if self.voxel_label_enc == "major":
            label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
            labels = unq_lbxyz[:, 0][label_ind]
        elif self.voxel_label_enc == "compact":
            assert False

        return labels



    def forward(self, batch_dict):
        """
        points: [N0 + N1 + ... Nb, C], C is [batch_idx, x, y, z, ...]
        """
        points, batch_size = batch_dict["points"], batch_dict["batch_size"]
        point_sem_labels = batch_dict["point_sem_labels"]



        # points -> polar coordinates system
        # points_cyl: [N0 + N1 + ... Nb, 3], batch_idx excluded
        points_xyz = points[:, 1:4]
        points_cyl = cart2cylind(points_xyz)



        point_vcoors_cyl, valid_flag = self.voxelize(points_cyl[:, 0:3], reverse_coors=False)
        point_vcoors_cyl = point_vcoors_cyl[valid_flag]
        points_cyl = points_cyl[valid_flag]
        points_valid = points[valid_flag]
        bs_idx_cyl = points[valid_flag][:, 0]
        if point_sem_labels is not None:
            point_sem_labels = point_sem_labels[valid_flag]

        # 4: [bs_idx, cyc_vx, cyc_vy, cyc_vz]
        point_vcoors = torch.cat([bs_idx_cyl.unsqueeze(-1), point_vcoors_cyl], dim=1).type(torch.long)
        # NOTE: ignore the z coord just for polarnet
        point_vcoors_to_unique = point_vcoors.clone()
        point_vcoors_to_unique[:, -1] = int(self.grid_size[2] // 2)
        unq, unq_inv, unq_cnt = torch.unique(point_vcoors_to_unique, return_inverse=True, return_counts=True, dim=0)
        # voxel_coors: [#voxels, 4]
        # voxel_coors_inv: [#points, ]
        voxel_coors = unq

        voxel_coors_inv = unq_inv

        # prepare input feature: feature normalization
        # [cyl_x, cyl_y, cyl_z, cardx, cardy, xxx]
        points_features = torch.cat([points_cyl, points_valid[:, 1:3], points_valid[:, 4:]], dim=1)
        points_features = self.prepare_input_feature(points_features, point_vcoors, voxel_coors_inv)
        points_features = self.PPmodel(points_features)


        # scatter to voxel 
        if self.average_points:
            features = torch_scatter.scatter_mean(points_features, voxel_coors_inv, dim=0)
        else:
            features = torch_scatter.scatter_max(points_features, voxel_coors_inv, dim=0)[0]


        if self.fea_compre:
            features = self.fea_compression(features)


        # reformat the voxel_features shape to [B, C, grid_size[0], grid_size[1]] for polarnet
        bev_data_dim = [batch_size, self.grid_size[0], self.grid_size[1], features.shape[-1]]
        bev_data = torch.zeros(bev_data_dim, dtype=features.dtype).to(features.device)
        bev_data[voxel_coors[:,0],voxel_coors[:,1],voxel_coors[:,2],:] = features
        # [B, grid_size[0], grid_size[1], C] -> [B, C, grid_size[0], grid_size[1]]
        bev_data = bev_data.permute(0,3,1,2)


        batch_dict["voxel_features"] = bev_data
        # for dense to sparse mapping in head
        batch_dict["point_vcoors"] = point_vcoors
        batch_dict["input_shape"] = self.grid_size


        batch_dict["num_points_in_voxel"] = unq_cnt

        

        if self.voxel_label_enc is not None:
            # encode voxel label for training
            if point_sem_labels is not None:
                # sparse voxel label
                voxel_labels = self.voxelize_labels(point_sem_labels, point_vcoors)
                batch_dict["voxel_sem_labels"] = voxel_labels

                # dense voxel label comsumes more GPU memory
                # bev_labels_dim = [batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2]]
                # bev_labels = torch.zeros(bev_labels_dim, dtype=torch.float32).to(features.device)
                # bev_labels[point_vcoors[:,0],point_vcoors[:,1],point_vcoors[:,2],point_vcoors[:,3]]=point_sem_labels
                # batch_dict["bev_sem_labels"] = bev_labels

        return batch_dict





@READERS.register_module
class Cylinder3DDynamicVoxelFeatureExtractor(nn.Module):

    def __init__(self, grid_size, point_cloud_range, average_points, num_input_features, num_output_features, fea_compre=None, voxel_label_enc=None, **kwargs):
        super(Cylinder3DDynamicVoxelFeatureExtractor, self).__init__()
        """
        ref 
        """
        # 2：cyl_x,  cyl_y
        # 8: additional features
        fea_dim = num_input_features + 2+8
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, num_output_features)
        )


        self.pool_dim = num_output_features

        # point feature compression
        self.fea_compre = fea_compre
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                    nn.Linear(self.pool_dim, self.fea_compre),
                    nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim



        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points
        # compute the voxel_size by grid_size
        self.voxel_size = [
            (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0],
            (point_cloud_range[4] - point_cloud_range[1]) / grid_size[1],
            (point_cloud_range[5] - point_cloud_range[2]) / grid_size[2],
        ]
        print("==> in DynamicCylinder3DVoxelFeatureExtractor, grid_size: ", grid_size)
        print("==> in DynamicCylinder3DVoxelFeatureExtractor, point_cloud_range: ", point_cloud_range)
        print("==> in DynamicCylinder3DVoxelFeatureExtractor, voxel_size: ", self.voxel_size)

        # None: without voxel label encoding
        # "major": major vote like cylinder3d
        # "compact": TBD
        self.voxel_label_enc = voxel_label_enc # "major or compact"


    def voxelize(self, points, reverse_coors=True):
        """
        points: [N0 + N1 + ... Nb, 3], 
        """
        point_vcoor0 = torch.floor( (points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0] ).int()
        point_vcoor1 = torch.floor( (points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1] ).int()
        point_vcoor2 = torch.floor( (points[:, 2] - self.point_cloud_range[2]) / self.voxel_size[2] ).int()
        

        # NOTE: to be improved
        point_vcoor0 = torch.clamp(point_vcoor0, 0, self.grid_size[0]-1)
        point_vcoor1 = torch.clamp(point_vcoor1, 0, self.grid_size[1]-1)
        point_vcoor2 = torch.clamp(point_vcoor2, 0, self.grid_size[2]-1)


        valid_flag0 = (0 <= point_vcoor0) & (point_vcoor0 < self.grid_size[0])
        valid_flag1 = (0 <= point_vcoor1) & (point_vcoor1 < self.grid_size[1])
        valid_flag2 = (0 <= point_vcoor2) & (point_vcoor2 < self.grid_size[2])
        valid_flag = valid_flag0 & valid_flag1 & valid_flag2

        point_vcoors = torch.stack([point_vcoor0, point_vcoor1, point_vcoor2], dim=-1)
        
        if reverse_coors:
            point_vcoors = point_vcoors[:, [2,1,0]]


        return point_vcoors, valid_flag


    def prepare_input_feature(self, point_features, point_vcoors, inv_idx):
        """
        point_features: [#points, C], C is [cyl_x, cyl_y, cyl_z, card_x, card_y, ...]
        point_vcoors: [#points, 4], 4 is [bs_idx, vz, vy, vx]
        inv_idx: [#point, ]

        point_init_features: [#point, C + 5 + 3]
        """
        # 前五维都归一化一下
        pc_mean = torch_scatter.scatter_mean(point_features[:, :5], inv_idx, dim=0)[inv_idx]
        nor_pc = point_features[:, :5] - pc_mean

        # 恢复出极坐标, 针对的是向下取整的量化方式
        # [#point, 3]
        point_voxel_centers = common_utils.get_voxel_centers(
            point_vcoors[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )

        # 前三维作差
        center_to_point = point_features[:, :3] - point_voxel_centers

        point_init_features = torch.cat((point_features, nor_pc, center_to_point), dim=1)

        return point_init_features


    def voxelize_labels(self, point_labels, point_vcoors):
        """
        point_labels: [#points, ]
        point_vcoors: [#points, 4], 4 is [bs_idx, vz, vy, vx]
        
        labels: [#voxel, ], major voted voxel label
        """
        lbxyz = torch.cat([point_labels.reshape(-1, 1), point_vcoors], dim=-1)
        unq_lbxyz, count = torch.unique(lbxyz, return_counts=True, dim=0)
        inv_ind = torch.unique(unq_lbxyz[:, 1:], return_inverse=True, dim=0)[1]
        
        if self.voxel_label_enc == "major":
            label_ind = torch_scatter.scatter_max(count, inv_ind)[1]
            labels = unq_lbxyz[:, 0][label_ind]
        elif self.voxel_label_enc == "compact":
            assert False

        return labels





    def forward(self, batch_dict):
        """
        points: [N0 + N1 + ... Nb, C], C is [batch_idx, x, y, z, ...]
        """
        points, batch_size = batch_dict["points"], batch_dict["batch_size"]
        point_sem_labels = batch_dict["point_sem_labels"]


        # points -> polar coordinates system
        # points_cyl: [N0 + N1 + ... Nb, 3], batch_idx excluded
        points_xyz = points[:, 1:4]
        points_cyl = cart2cylind(points_xyz)


        # NOTE: inverse the coords for spconv & filter out the points outside
        point_vcoors_cyl, valid_flag = self.voxelize(points_cyl[:, 0:3])
        point_vcoors_cyl = point_vcoors_cyl[valid_flag]
        points_cyl = points_cyl[valid_flag]
        points_valid = points[valid_flag]
        bs_idx_cyl = points[valid_flag][:, 0]
        if point_sem_labels is not None:
            point_sem_labels = point_sem_labels[valid_flag]

        # 4: [bs_idx, cyc_vz, cyc_vy, cyc_vx]
        point_vcoors = torch.cat([bs_idx_cyl.unsqueeze(-1), point_vcoors_cyl], dim=1).type(torch.long)
        unq, unq_inv, unq_cnt = torch.unique(point_vcoors, return_inverse=True, return_counts=True, dim=0)
        # voxel_coors: [#voxels, 4]
        # voxel_coors_inv: [#points, ]
        voxel_coors = unq
        voxel_coors_inv = unq_inv


        # prepare input feature: feature normalization
        # [cyl_x, cyl_y, cyl_z, cardx, cardy, xxx]
        points_features = torch.cat([points_cyl, points_valid[:, 1:3], points_valid[:, 4:]], dim=1)
        points_features = self.prepare_input_feature(points_features, point_vcoors, voxel_coors_inv)
        points_features = self.PPmodel(points_features)


        # scatter to voxel 
        if self.average_points:
            features = torch_scatter.scatter_mean(points_features, voxel_coors_inv, dim=0)
        else:
            features = torch_scatter.scatter_max(points_features, voxel_coors_inv, dim=0)[0]

        if self.fea_compre:
            features = self.fea_compression(features)


        batch_dict["voxel_features"] = features
        # for dense to sparse mapping in head
        # [b, vz, vy, vx] -> [b, vx, vy, vz]
        batch_dict["point_vcoors"] = torch.stack([point_vcoors[:, 0], point_vcoors[:, 3], point_vcoors[:, 2], point_vcoors[:, 1]], dim=-1) 
        
        
        batch_dict["input_shape"] = self.grid_size
        batch_dict["voxel_coords"] = voxel_coors # [#voxel, 4], 4: [b, vz, vy, vx]

        batch_dict["num_points_in_voxel"] = unq_cnt

        

        if self.voxel_label_enc is not None:
            # encode voxel label for training
            if point_sem_labels is not None:
                # sparse voxel label
                voxel_labels = self.voxelize_labels(point_sem_labels, point_vcoors)
                batch_dict["voxel_sem_labels"] = voxel_labels

                # dense voxel label comsumes more GPU memory
                # bev_labels_dim = [batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2]]
                # bev_labels = torch.zeros(bev_labels_dim, dtype=torch.float32).to(features.device)
                # bev_labels[point_vcoors[:,0],point_vcoors[:,1],point_vcoors[:,2],point_vcoors[:,3]]=point_sem_labels
                # batch_dict["bev_sem_labels"] = bev_labels

        return batch_dict



