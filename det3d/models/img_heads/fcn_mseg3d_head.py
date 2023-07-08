# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import IMG_HEADS
from .decode_head import BaseDecodeHead
from det3d.ops.mmseg_ops import resize
import torch.nn.functional as F
from .sc_conv import SCBottleneck

from det3d.core.utils.loss_utils import lovasz_softmax




class CameraSemanticFeatureAggregationModule(nn.Module):
    """
    Aggregate the semantic embeddings across all the multi-camera images
    """
    def __init__(self):
        super(CameraSemanticFeatureAggregationModule, self).__init__()

    def forward(self, _feats, _probs, batch_size):
        """
        _feats: [batch_size*num_cams, c, h, w]
        _probs: [batch_size*num_cams, num_cls, h, w]
        
        semantic_embeddings: [batch_size, c, num_cls, 1]
        """
        _, num_classes, height, width = _probs.size()
        channels = _feats.size(1)

        # [batch_size, num_cams, c, h, w] -> [batch_size, c, num_cams, h, w]
        probs = _probs.view(batch_size, -1, num_classes, height, width).permute(0,2,1,3,4).contiguous()
        feats = _feats.view(batch_size, -1, channels, height, width).permute(0,2,1,3,4).contiguous()
        # print("==> probs.shape: ", probs.shape)

        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, channels]
        feats = feats.permute(0, 2, 1)
        # [batch_size, num_classes, height*width]
        probs = F.softmax(probs, dim=2)
        # [batch_size, num_classes, channels]
        semantic_embeddings = torch.matmul(probs, feats)

        # [batch_size, channels, num_classes, 1]
        semantic_embeddings = semantic_embeddings.permute(0, 2, 1).contiguous().unsqueeze(3)
        
        return semantic_embeddings


@IMG_HEADS.register_module
class FCNMSeg3DHead(BaseDecodeHead):
    """
    Based on Fully Convolution Networks for Semantic Segmentation.
    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    
    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 ignore_index=0, 
                 loss_weight=1.0,
                 lovasz_loss_weight=-1.0,
                 use_sc_conv=False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNMSeg3DHead, self).__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            if use_sc_conv:
                convs.append(
                    SCBottleneck(
                        inplanes=self.channels, 
                        planes=self.channels, 
                        stride=1, 
                        downsample=None,
                        cardinality=1, 
                        bottleneck_width=32,
                        avd=False, 
                        dilation=1, 
                        is_first=False, 
                    )
                )
            else:
                convs.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg
                    )
                )
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        # NOTE: 暂时使用默认参数
        self.camera_sfam = CameraSemanticFeatureAggregationModule()

        # dict for loss func
        self.forward_ret_dict = {}
        # build ce loss
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.loss_weight = loss_weight


        self.lovasz_loss_weight = lovasz_loss_weight
        if self.lovasz_loss_weight > 0:
            self.lovasz_loss_func = lovasz_softmax


    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, batch_dict, return_loss=True, **kwargs):
        """Forward function."""
        inputs = batch_dict["inputs"] # a list
        
        feature = self._forward_feature(inputs)
        output = self.cls_seg(feature)

        # gather camera semantic embeddings
        camera_semantic_embeddings = self.camera_sfam(feature, output, batch_dict["batch_size"])


        self.forward_ret_dict.update({
            "image_logits": output
        })

        if return_loss:
            self.forward_ret_dict.update({
                "image_sem_labels": batch_dict["images_sem_labels"],
            })   


        batch_dict["image_logits"] = output
        batch_dict["image_features"] = feature

        batch_dict["camera_semantic_embeddings"] = camera_semantic_embeddings

        return batch_dict


    def get_loss(self, image_loss_dict=None):
        """Get loss """
        image_loss = 0
        
        if image_loss_dict is None:
            image_loss_dict = {}

        # [B*ncams, C, H, W]
        image_logits = self.forward_ret_dict["image_logits"]
        image_sem_labels = self.forward_ret_dict["image_sem_labels"]

        image_logits = resize(
            input=image_logits,
            size=image_sem_labels.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        # nn.CrossEntropyLoss()
        # Input: (N, C) or (N, C, d1, d2, ...)
        # Target: (N) or (N, d1, d2, ...)
        image_sem_targets = image_sem_labels.squeeze(1).long()

        # compute the ce loss, based on point-wise sparse supervision
        # point-to-pixel loss in MSeg3D
        pixel_ce_loss = self.loss_weight * self.cross_entropy_func(image_logits, image_sem_targets)

        image_loss += pixel_ce_loss

        if self.lovasz_loss_weight > 0:
            lovasz_loss = self.lovasz_loss_weight * self.lovasz_loss_func(
                torch.softmax(image_logits, dim=1), 
                image_sem_targets
            )
            image_loss += lovasz_loss


        # NOTE: add for log, without gradient
        image_loss_dict["image_ce_loss"] = pixel_ce_loss.detach()        
        if self.lovasz_loss_weight > 0:
            image_loss_dict["image_lvsz_loss"] = lovasz_loss.detach()


        return image_loss, image_loss_dict
