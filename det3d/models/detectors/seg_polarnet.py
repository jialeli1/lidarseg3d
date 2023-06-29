from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint

@DETECTORS.register_module
class SegPolarNet(SingleStageDetector):
    """
    polarnet/cylinder3d
    """
    def __init__(
        self,
        reader,    
        backbone,
        point_head,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        super(SegPolarNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained=None
        )

        # build the point head
        self.point_head = builder.build_point_head(point_head)
        
        # load the singlestage with point head here
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))

    def extract_feat(self):
        assert False
        

    def forward(self, example, return_loss=True, **kwargs):
        """
        example: a dict including
            'metadata', 'points', 'voxels', 'shape', 'num_points', 'num_voxels', 'coordinates', 'gt_boxes_and_cls', 'hm', 'anno_box', 'ind', 'mask', 'cat'

        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        batch_size = len(num_voxels)
        

        # construct a batch_dict like pv-rcnn
        data = dict(
            batch_size=batch_size,
            points=example["points"],
            point_sem_labels=example.get("point_sem_labels", None)
        )
        # VFE1 & backbone1
        # the reader in polarnet is dynamicVFE
        data = self.reader(data)
        # data["voxel_features"] = self.backbone(data["voxel_features"])
        data = self.backbone(data)

        data = self.point_head(batch_dict=data, return_loss=return_loss)


        if return_loss:
            # compute the point_head loss.
            seg_loss_dict = {}
            point_loss, point_loss_dict = self.point_head.get_loss()

            # this item for Optimizer, formating as loss per task
            opt_loss = [point_loss]
            seg_loss_dict["loss"] = opt_loss

            # reformat for text logger
            for k, v in point_loss_dict.items():
                repeat_list = [v for i in range(len(opt_loss))]
                seg_loss_dict[k] = repeat_list

            return seg_loss_dict
        else:

            return self.point_head.predict(example=example, test_cfg=self.test_cfg)
