# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import IMG_HEADS
from .decode_head import BaseDecodeHead
from det3d.ops.mmseg_ops import resize


@IMG_HEADS.register_module
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

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
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(FCNHead, self).__init__(**kwargs)
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
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
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

        # dict for loss func
        self.forward_ret_dict = {}
        # build ce loss
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.loss_weight = loss_weight

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

        self.forward_ret_dict.update({
            "image_logits": output
        })

        if return_loss:
            self.forward_ret_dict.update({
                "image_sem_labels": batch_dict["images_sem_labels"],
            })   


        batch_dict["image_logits"] = output
        batch_dict["image_feature"] = feature

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


        pixel_ce_loss = self.loss_weight * self.cross_entropy_func(image_logits, image_sem_targets)

        image_loss += pixel_ce_loss

        image_loss_dict["image_ce_loss"] = pixel_ce_loss.detach()

        return image_loss, image_loss_dict
