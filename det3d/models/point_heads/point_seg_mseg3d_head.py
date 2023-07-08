from inspect import stack
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from ..registry import POINT_HEADS
from det3d.core.utils.loss_utils import lovasz_softmax
from .point_utils import three_interpolate_wrap


from .context_module import LiDARSemanticFeatureAggregationModule, SemanticFeatureFusionModule



@POINT_HEADS.register_module
class PointSegMSeg3DHead(nn.Module):
    def __init__(self, class_agnostic, num_class, model_cfg, **kwargs):
        super().__init__()
        
        if class_agnostic:
            self.num_class = 1
        else:
            self.num_class = num_class
        
        norm_layer=partial(nn.BatchNorm1d, eps=1e-6)
        act_layer=nn.ReLU



        voxel_in_channels = model_cfg["VOXEL_IN_DIM"]
        self.dp_ratio = model_cfg["DP_RATIO"]
        # auxiliary segmentation head on voxel features
        self.voxel_cls_layers = self.make_convcls_head(
            fc_cfg=model_cfg["VOXEL_CLS_FC"],
            input_channels=voxel_in_channels,
            output_channels=self.num_class,
            dp_ratio=self.dp_ratio,
        )



        # GF-Phase: geometry-based feature fusion phase
        voxel_align_channels = model_cfg["VOXEL_ALIGN_DIM"]
        self.gffm_lidar = nn.Sequential(
            nn.Linear(voxel_in_channels, voxel_align_channels),
            norm_layer(voxel_align_channels),
            act_layer(),
        ) 

        image_in_channels = model_cfg["IMAGE_IN_DIM"]
        image_align_channels = model_cfg["IMAGE_ALIGN_DIM"]
        self.gffm_camera = nn.Sequential(
            nn.Linear(image_in_channels, image_align_channels),
            norm_layer(image_align_channels),
            act_layer(),
        ) 

        fused_channels = model_cfg["GEO_FUSED_DIM"]
        self.gffm_lc = nn.Sequential(
            nn.Linear(voxel_align_channels + image_align_channels, fused_channels),
            nn.BatchNorm1d(fused_channels),
            act_layer(),
        ) 





        # cross-modal feature completion
        self.lidar_camera_mimic_layer = self.make_convcls_head(
            fc_cfg=model_cfg["MIMIC_FC"],
            input_channels=voxel_align_channels,
            output_channels=image_align_channels,
            dp_ratio=0,
        )




        # SF-Phase: semantic-based feature fusion phase
        SFPhase_CFG = model_cfg["SFPhase_CFG"]
        self.lidar_sfam = LiDARSemanticFeatureAggregationModule()
        self.sffm = SemanticFeatureFusionModule(
            d_input_point=fused_channels, 
            d_input_embeddings1=image_in_channels, 
            d_input_embeddings2=voxel_in_channels, 
            embeddings_proj_kernel_size=SFPhase_CFG["embeddings_proj_kernel_size"], 
            d_model=SFPhase_CFG["d_model"], 
            nhead=SFPhase_CFG["n_head"], 
            num_decoder_layers=SFPhase_CFG["n_layer"], 
            dim_feedforward=SFPhase_CFG["n_ffn"],
            dropout=SFPhase_CFG["drop_ratio"],
            activation=SFPhase_CFG["activation"], 
            normalize_before=SFPhase_CFG["pre_norm"],
        )





        # final output head for point-wise segmentation
        sem_fused_channels = self.sffm.d_model
        self.out_cls_layers = nn.Linear(sem_fused_channels, num_class)




        # build loss
        self.forward_ret_dict = {}
        self.ignored_label = model_cfg["IGNORED_LABEL"]
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=self.ignored_label)
        self.lovasz_softmax_func = lovasz_softmax
        self.mimic_loss_func = nn.MSELoss() 
        self.tasks = ["out"]


    def make_convcls_head(self, fc_cfg, input_channels, output_channels, dp_ratio=0):
        fc_layers = []
        c_in = input_channels
        if dp_ratio > 0:
            fc_layers.append(nn.Dropout(dp_ratio))
            
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]

        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)


    def get_loss(self, point_loss_dict=None):
        """
        batch-wise loss on out_logits (point_logits) and voxel_logits
        """
        point_loss = 0
        if point_loss_dict is None:
            point_loss_dict = {}


        # point-to-voxel loss in MSeg3D
        # voxel head loss
        voxel_ce_loss = self.cross_entropy_func( 
            self.forward_ret_dict["voxel_logits"], 
            self.forward_ret_dict["voxel_sem_labels"].long(), 
        )
        # NOTE: please change the ignored label (0) for other value
        voxel_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["voxel_logits"], dim=-1),
            self.forward_ret_dict["voxel_sem_labels"].long(),
            ignore=self.ignored_label, 
        )
        voxel_loss = voxel_ce_loss + voxel_lvsz_loss
        point_loss += voxel_loss
        point_loss_dict["voxel_ce_loss"] = voxel_ce_loss.detach()
        point_loss_dict["voxel_lovasz_loss"] = voxel_lvsz_loss.detach()


        # point loss in MSeg3D
        # out head loss
        out_ce_loss = self.cross_entropy_func(
            self.forward_ret_dict["out_logits"],
            self.forward_ret_dict["point_sem_labels"].long(),
        )
        out_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["out_logits"], dim=-1),
            self.forward_ret_dict["point_sem_labels"].long(),
            ignore=self.ignored_label, 
        )
        out_loss = out_ce_loss + out_lvsz_loss
        point_loss += out_loss
        point_loss_dict["out_ce_loss"] = out_ce_loss.detach()
        point_loss_dict["out_lovasz_loss"] = out_lvsz_loss.detach()




        # pixel-to-point loss in MSeg3D
        # mimic loss for feature completion
        point_features_pcamera = self.forward_ret_dict["point_features_pcamera"]
        point_features_camera = self.forward_ret_dict["point_features_camera"]
        assert self.forward_ret_dict["point_features_camera"].requires_grad == False
        out_mimic_loss = self.mimic_loss_func(
            self.forward_ret_dict["point_features_pcamera"],
            self.forward_ret_dict["point_features_camera"],
        )
        point_loss += out_mimic_loss
        point_loss_dict["out_mimic_loss"] = out_mimic_loss.detach()


        return point_loss, point_loss_dict



    def get_points_image_feature(self, input_img_feature, points_cuv, batch_idx):
        """
        img_feature: (batch, num_cam, num_chs, h, w)
        points_cuv: (n0+n1+...n_{b-1}, 4), 4:[valid, cam_id, hei_coord, wid_coord]

        point_features_camera: (n0+n1+...n_{b-1}, num_chs)
        """
        # (batch, num_cam, num_chs, h, w) -> (batch, num_chs, num_cam, h, w)
        img_feature = input_img_feature.transpose(2,1)

        batch_size, num_chs, num_cams, h, w = img_feature.shape
        point_feature_camera_list = []
        for i in range(batch_size):
            # (1, num_chs, num_cam, h, w)
            cur_img_feat = img_feature[i].unsqueeze(0)

            cur_batch_mask = (batch_idx == i)
            cur_points_cuv = points_cuv[cur_batch_mask]
            # (ni, 4) -> (1, 1, 1, ni, 4)
            cur_points_cuv = cur_points_cuv.reshape(1, 1, 1, cur_points_cuv.shape[0], cur_points_cuv.shape[-1])
            

            # NOTE: grid_sample
            # input: (B, num_chs, num_cam, h, w); grid: (ni, 3), 3 should be [w_coord, h_coord, cam_id]
            # set align_corners as True to avoid interpolation between cameras
            # cur_points_feature_camera: (1, num_chs, 1, 1, ni)
            cur_points_feature_camera = F.grid_sample(cur_img_feat, cur_points_cuv[..., (3,2,1)], mode='bilinear', padding_mode='zeros', align_corners=True)

            # (1, num_chs, 1, 1, ni) -> (1*num_chs, 1, 1, ni) -> (1*num_chs, 1*1*ni) -> (ni, num_chs)
            cur_points_feature_camera = cur_points_feature_camera.flatten(0, 1).flatten(1, 3).transpose(1, 0)
            point_feature_camera_list.append(cur_points_feature_camera)
        
        # (n0+n1+...n_{b-1}, num_chs)
        point_features_camera = torch.cat(point_feature_camera_list, dim=0)
        assert point_features_camera.shape[0] == points_cuv.shape[0]

        return point_features_camera



    def forward(self, batch_dict, return_loss=True, **kwargs):
        """
        Args:
            batch_dict:
                batch_size:
                conv_point_coords: (Nc1 + Nc2 + Nc3 + ..., 4), [bs_idx, x, y,z]
                points: (Np1 + Np2 + Np3 + ..., 4), [bs_idx, x, y,z]

        Returns:
            batch_dict:
                voxel_logits: (Nc1 + Nc2 + Nc3 + ..., n_cls)
                out_logits: (Np1 + Np2 + Np3 + ..., n_cls)
        """
        batch_size = batch_dict["batch_size"]


        # voxel features from the spconv backbone
        voxel_features = batch_dict["conv_point_features"]
        voxel_logits = self.voxel_cls_layers(voxel_features)
        self.forward_ret_dict.update({
            "voxel_logits": voxel_logits,
        })
        if return_loss:
            self.forward_ret_dict.update({
                "voxel_sem_labels": batch_dict["voxel_sem_labels"],
                "point_sem_labels": batch_dict["point_sem_labels"],
                "batch_size": batch_size,
            })
        



        # voxel features -> point lidar features
        voxel_coords = batch_dict["conv_point_coords"] 
        point_coords = batch_dict["points"]
        # voxel features -> point lidar features 
        point_features_lidar_0 = three_interpolate_wrap(
            new_coords=point_coords, 
            coords=voxel_coords, 
            features=voxel_features, 
            batch_size=batch_size)        
        point_features_lidar = self.gffm_lidar(point_features_lidar_0)




        # image feature maps -> point camera features
        image_features = batch_dict["image_features"]
        points_cuv = batch_dict["points_cuv"]
        valid_mask = (points_cuv[:, 0] == 1)
        # point_features_camera_0 shape: [#all_points, C_img]
        point_features_camera_0 = self.get_points_image_feature(
            input_img_feature=image_features, 
            points_cuv=points_cuv[valid_mask],          # mask out the points outside 
            batch_idx=point_coords[:, 0][valid_mask],   # mask out the points outside
        )
        # point_features_camera.shape: [#points_inside, C_img]
        point_features_camera = self.gffm_camera(point_features_camera_0)




        # cross-modal feature completion
        # predict the pseudo camera features
        # point_features_pcamera.shape: [#points_inside, C_img]
        point_features_pcamera = self.lidar_camera_mimic_layer(point_features_lidar[valid_mask])
        assert point_features_camera.shape[0] == point_features_pcamera.shape[0]
        if return_loss:
            # NOTE: compute the feature completion loss on points inside only
            self.forward_ret_dict.update({
                "point_features_pcamera": point_features_pcamera,
                "point_features_camera": point_features_camera.detach(),
            })
        # prepare for feature completion
        point_features_camera_pad0 = torch.zeros(
            (valid_mask.shape[0], point_features_camera.shape[1]), 
            dtype=point_features_camera.dtype, 
            device=point_features_camera.device
        )
        point_features_camera_pad0[valid_mask] = point_features_camera
        point_features_pcamera_pad0 = torch.zeros(
            (valid_mask.shape[0], point_features_pcamera.shape[1]), 
            dtype=point_features_pcamera.dtype, 
            device=point_features_pcamera.device
        )
        point_features_pcamera_pad0[valid_mask] = point_features_pcamera

        # feature completion
        assert point_features_camera_pad0.shape[0] == point_features_pcamera_pad0.shape[0]
        # point-wise completed camera features
        point_features_ccamera = torch.where(
            valid_mask.unsqueeze(-1).expand_as(point_features_camera_pad0), 
            point_features_camera_pad0, 
            point_features_pcamera_pad0, 
        )





        # GFFM in GF-Phase
        point_features_lc = torch.cat([point_features_lidar, point_features_ccamera], dim=1)
        point_features_geo_fused = self.gffm_lc(point_features_lc)





        # SF-Phase
        # camera_semantic_embeddings: [batch, C_img, num_cls, 1]
        camera_semantic_embeddings = batch_dict["camera_semantic_embeddings"]
        # lidar_semantic_embeddings: [batch, C_voxel, num_cls, 1]
        lidar_semantic_embeddings = self.lidar_sfam(
            feats=voxel_features, 
            probs=voxel_logits, 
            batch_idx=voxel_coords[:, 0], 
            batch_size=batch_size,
        )
        # sffm
        point_features_sem_fused = self.sffm(
            input_point_features=point_features_geo_fused, 
            input_sem_embeddings1=camera_semantic_embeddings, 
            input_sem_embeddings2=lidar_semantic_embeddings, 
            batch_idx=point_coords[:, 0], 
            batch_size=batch_size,
        )




        
        out_logits = self.out_cls_layers(point_features_sem_fused)

        batch_dict["out_logits"] = out_logits
        self.forward_ret_dict["out_logits"] = out_logits

        return batch_dict


    @torch.no_grad()
    def predict(self, example, test_cfg=None, **kwargs):
        """
        Decode the logits (N1+N2+..., n_cls) to labels (N1+N2+..., 1) 
        example: 
            batch_size:
            metadata:
            points:


        return: 
            ret_list: a list of batch_size items, each item is a dict denots a frame
        """
        batch_size = len(example["num_voxels"])

        tta_flag = test_cfg.get('tta_flag', False)
        stack_points = example["points"][:, 0:4]

        ret_list = []
        if tta_flag:
            merge_type = test_cfg.get('merge_type', "ArithmeticMean")
            num_tta_tranforms = test_cfg.get('num_tta_tranforms', 4)
            if "metadata" not in example or len(example["metadata"]) == 0:
                # [None, None, ...]
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                meta_list = meta_list[:num_tta_tranforms*int(batch_size):num_tta_tranforms]
                
                stack_pred_logits = self.forward_ret_dict["out_logits"]
                stack_pred_logits = torch.softmax(stack_pred_logits, dim=-1)


                # split the stacked data as single ones
                # [N1,N1,N1,N1,..., N2,N2,N2,N2,..., NB,NB,NB,NB,...]
                single_pc_list = []
                single_logits_list = []
                for i in range(batch_size):
                    bs_mask = stack_points[:, 0] == i
                    single_pc = stack_points[bs_mask]
                    single_logits = stack_pred_logits[bs_mask]
                    single_pc_list.append(single_pc)
                    single_logits_list.append(single_logits)

                # merge the predictions from input variants
                merged_pc_list = []
                merged_pred_sem_labels_list = []
                merged_num_point_list = []
                for i in range(0, batch_size, num_tta_tranforms):
                    merged_pc_list.append(single_pc_list[i])
                    merged_num_point_list.append(single_pc_list[i].shape[0])
                    if merge_type == "ArithmeticMean":
                        merged_logits_list = single_logits_list[i: i+num_tta_tranforms]
                        # (num_tta_tranforms, N, C)
                        merged_logits = torch.stack(merged_logits_list, dim=0)
                        # (N, C)
                        merged_logits = torch.mean(merged_logits, dim=0)    
                    else: 
                        raise NotImplementedError
                    merged_pred_sem_labels = torch.argmax(merged_logits, dim=1)
                    merged_pred_sem_labels_list.append(merged_pred_sem_labels)

                left_ind = 0
                for i in range(int(batch_size/num_tta_tranforms)):
                    ret = {}
                    ret["metadata"] = meta_list[i]
                    ret["pred_point_sem_labels"] = merged_pred_sem_labels_list[i]
                    
                    if "point_sem_labels" in example: 
                        # not used
                        right_ind = sum(merged_num_point_list[:i+1])
                        ret["point_sem_labels"] = example["point_sem_labels"][left_ind:right_ind]
                        left_ind = right_ind

                    ret_list.append(ret)

        else: 
            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]

                stack_pred_logits = self.forward_ret_dict["out_logits"]                
                stack_pred_sem_labels = torch.argmax(stack_pred_logits, dim=1)
                
                for i in range(batch_size):
                    # reformat data to frame-wise
                    ret = {}
                    ret["metadata"] = meta_list[i]

                    cur_bs_mask = (stack_points[:, 0] == i)
                    ret["pred_point_sem_labels"] = stack_pred_sem_labels[cur_bs_mask]

                    if "point_sem_labels" in example:
                        # not used
                        ret["point_sem_labels"] = example["point_sem_labels"][cur_bs_mask]
                        # ret["voxel_sem_labels"] = example["voxel_sem_labels"][cur_bs_mask]

                    ret_list.append(ret)

        return ret_list



