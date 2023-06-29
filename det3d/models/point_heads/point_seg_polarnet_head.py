import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import POINT_HEADS
from det3d.core.utils.loss_utils import lovasz_softmax


@POINT_HEADS.register_module
class PointSegPolarNetHead(nn.Module):
    """
    """
    def __init__(self, class_agnostic, num_class, model_cfg, **kwargs):
        super().__init__()
        
        if class_agnostic:
            self.num_class = 1
        else:
            self.num_class = num_class
        

        self.forward_ret_dict = {}
        # build loss
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=model_cfg["IGNORED_LABEL"])
        self.lovasz_softmax_func = lovasz_softmax

        self.tasks = ["out"]


    def get_loss(self, point_loss_dict=None):
        """
        """
        point_loss = 0
        if point_loss_dict is None:
            point_loss_dict = {}
            

        # out head loss
        out_ce_loss = self.cross_entropy_func(
            self.forward_ret_dict["out_logits"],
            self.forward_ret_dict["point_sem_labels"].long(),
        )
        out_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["out_logits"], dim=-1),
            self.forward_ret_dict["point_sem_labels"].long(),
            ignore=0, 
        )
        out_loss = out_ce_loss + out_lvsz_loss
        point_loss += out_loss
        point_loss_dict["out_ce_loss"] = out_ce_loss.detach()
        point_loss_dict["out_lvsz_loss"] = out_lvsz_loss.detach()


        return point_loss, point_loss_dict


    def forward(self, batch_dict, return_loss=True, **kwargs):
        """
        直接拿BEV logits去计算loss太占用显存了, 480*360*32=5,529,600, 远大于点数
        所以尝试先序列化，再计算损失
        这样能减少几乎一半的训练显存占用
        Args:
            batch_dict:
                batch_size:
                voxel_features:
                voxel_coords:
                voxel_coors_inv:

        Returns:
            batch_dict:
                conv_logits: (Nc1 + Nc2 + Nc3 + ..., n_cls)
                out_logits: (Np1 + Np2 + Np3 + ..., n_cls)
                aux_logits_list: 
        """
        batch_size = batch_dict["batch_size"]


        # bev_data: [B, num_cls, grid_size[0], grid_size[1], grid_size[2]]
        # [B, C, L, H, W]
        bev_data = batch_dict["voxel_features"] # 对polarNet来说已经是logits了


        # batch_dict["bev_logits"] = bev_data
        assert bev_data.shape[1] == self.num_class

        # bev map -> point sequence
        # bev_data: [B, num_cls, grid_size[0], grid_size[1], grid_size[2]] -> [B, grid_size[0], grid_size[1], grid_size[2], num_cls]
        bev_data = bev_data.permute(0,2,3,4,1)
        grid_size = batch_dict["input_shape"]
        point_vcoors = batch_dict["point_vcoors"] 
        point_logits = bev_data[point_vcoors[:,0],point_vcoors[:,1],point_vcoors[:,2], point_vcoors[:, 3], :]


        self.forward_ret_dict.update({
            "out_logits": point_logits,
        })


        if return_loss:
            self.forward_ret_dict.update({
                # "voxel_sem_labels": batch_dict["voxel_sem_labels"],
                # bev_sem_labels: [B, grid_size[0], grid_size[1], grid_size[2]]
                # "bev_sem_labels": batch_dict["bev_sem_labels"],
                "point_sem_labels": batch_dict["point_sem_labels"],
            })


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


