from inspect import stack
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from ..registry import POINT_HEADS
from det3d.core.utils.loss_utils import lovasz_softmax
from .point_utils import three_interpolate_wrap



@POINT_HEADS.register_module
class PointSegBatchlossHead(nn.Module):
    """
    point segmentation head with batch-wise loss
    """
    def __init__(self, class_agnostic, num_class, model_cfg, **kwargs):
        super().__init__()
        
        if class_agnostic:
            self.num_class = 1
        else:
            self.num_class = num_class
        
        norm_layer=partial(nn.BatchNorm1d, eps=1e-6)
        act_layer=nn.ReLU

        conv_in_channels = model_cfg["CONV_IN_DIM"]

        # conv head 
        self.conv_cls_layers = self.make_convcls_head(
            fc_cfg=model_cfg["CONV_CLS_FC"],
            input_channels=conv_in_channels,
            output_channels=self.num_class
        )


        out_conv_channels = model_cfg["CONV_ALIGN_DIM"]
        self.conv_align_layers = nn.Sequential(
            nn.Linear(conv_in_channels, out_conv_channels),
            norm_layer(out_conv_channels),
            act_layer(),
        ) 

        # out head, same structure with conv head 
        out_channels = out_conv_channels
        self.out_cls_layers = self.make_convcls_head(
            fc_cfg=model_cfg["OUT_CLS_FC"],
            input_channels=out_channels,
            output_channels=self.num_class
        )

        self.forward_ret_dict = {}
        # build loss
        self.ignored_label = model_cfg["IGNORED_LABEL"]
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=self.ignored_label)
        self.lovasz_softmax_func = lovasz_softmax

        self.tasks = ["out"]


    def make_convcls_head(self, fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
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
        batch-wise loss on out_logits and conv_logits
        """
        point_loss = 0
        if point_loss_dict is None:
            point_loss_dict = {}

        # conv head loss
        conv_ce_loss = self.cross_entropy_func( 
            self.forward_ret_dict["conv_logits"], 
            self.forward_ret_dict["voxel_sem_labels"].long(), 
        )
        # NOTE: please change the ignored label (0) for other value
        conv_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["conv_logits"], dim=-1),
            self.forward_ret_dict["voxel_sem_labels"].long(),
            ignore=self.ignored_label, 
        )
        conv_loss = conv_ce_loss + conv_lvsz_loss
        point_loss += conv_loss
        point_loss_dict["conv_ce_loss"] = conv_ce_loss.detach()
        point_loss_dict["conv_lovasz_loss"] = conv_lvsz_loss.detach()


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


        return point_loss, point_loss_dict


    def forward(self, batch_dict, return_loss=True, **kwargs):
        """
        Input:
            batch_dict:
                batch_size:
                conv_point_coords: (Nc1 + Nc2 + Nc3 + ..., 4), [bs_idx, x, y,z]
                points: (Np1 + Np2 + Np3 + ..., 4), [bs_idx, x, y,z]

        Return:
            batch_dict:
                conv_logits: (Nc1 + Nc2 + Nc3 + ..., n_cls)
                out_logits: (Np1 + Np2 + Np3 + ..., n_cls)
        """
        batch_size = batch_dict["batch_size"]
        conv_point_features = batch_dict["conv_point_features"]

        conv_logits = self.conv_cls_layers(conv_point_features)

        self.forward_ret_dict.update({
            "conv_logits": conv_logits,
        })

        if return_loss:
            self.forward_ret_dict.update({
                "voxel_sem_labels": batch_dict["voxel_sem_labels"],
                "point_sem_labels": batch_dict["point_sem_labels"],
            })


        # 3NN: voxel -> point 
        conv_point_coords = batch_dict["conv_point_coords"] 
        out_point_coords = batch_dict["points"]
        out_point_conv_features = three_interpolate_wrap(
            new_coords=out_point_coords, 
            coords=conv_point_coords, 
            features=conv_point_features, 
            batch_size=batch_size
        )
        out_point_features = self.conv_align_layers(out_point_conv_features)


        out_logits = self.out_cls_layers(out_point_features)
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


