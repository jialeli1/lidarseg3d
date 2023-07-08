from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class SegMSeg3DNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        img_backbone,
        img_head,
        point_head,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        super(SegMSeg3DNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained=None
        )
        
        self.img_backbone = builder.build_img_backbone(img_backbone)
        self.img_head = builder.build_img_head(img_head)
        self.point_head = builder.build_point_head(point_head)
        


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
        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        batch_size = len(num_voxels)
        # ensure that the points just including [bs_idx, x, y, z]
        points = example["points"][:, 0:4]





        # camera branch
        # images: [batch, num_cams, num_ch, h, w] like [1, 5, 3, 640, 960]
        images = example["images"]         
        num_cams, hi, wi = images.shape[1], images.shape[3], images.shape[4] 
        # images: (batch, num_cams=5, 3, h, w) -> (batch*num_cams=5, 3, h, w)
        images = images.view(-1, 3, hi, wi)
        
        # img_backbone_return: (batch*num_cams=5, c, ho, wo)
        img_backbone_return = self.img_backbone(images)
        img_data = dict(
            inputs=img_backbone_return,
            batch_size=batch_size,
        )
        if return_loss:
            # (batch, num_cams=5, h, w) -> (batch*num_cams=5, h, w) -> (batch*num_cams=5, 1, h, w)
            images_sem_labels = example["images_sem_labels"]
            images_sem_labels = images_sem_labels.view(-1, hi, wi).unsqueeze(1)
            img_data["images_sem_labels"] = images_sem_labels
        img_data = self.img_head(batch_dict=img_data, return_loss=return_loss)
        # get image_features from the img_head
        image_features = img_data["image_features"]
        _, num_chs, ho, wo = image_features.shape
        image_features = image_features.view(batch_size, num_cams, num_chs, ho, wo)





        # lidar branch
        # construct a batch_dict like pv-rcnn
        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            voxel_coords=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            points=points,
        )

        # VFE
        input_features = self.reader(data["features"], data["num_voxels"], data["voxel_coords"])
        data["voxel_features"] = input_features
        
        # backbone
        data = self.backbone(data)

        # prepare labels for training
        if return_loss:
            data["voxel_sem_labels"] = example["voxel_sem_labels"]
            data["point_sem_labels"] = example["point_sem_labels"]





        # fusion and segmentation in point head
        data["points_cuv"] = example["points_cuv"]
        data["image_features"] = image_features
        data["camera_semantic_embeddings"] = img_data.get("camera_semantic_embeddings", None)
        data["metadata"] = example.get("metadata", None)

        data = self.point_head(batch_dict=data, return_loss=return_loss)


        if return_loss:
            seg_loss_dict = {}
            point_loss, point_loss_dict = self.point_head.get_loss()

            # compute the img head loss
            img_loss, point_loss_dict = self.img_head.get_loss(point_loss_dict)


            # this item for Optimizer, formating as loss per task
            total_loss = point_loss + img_loss
            opt_loss = [total_loss]
            seg_loss_dict["loss"] = opt_loss

            # reformat for text logger
            for k, v in point_loss_dict.items():
                repeat_list = [v for i in range(len(opt_loss))]
                seg_loss_dict[k] = repeat_list

            return seg_loss_dict

        else:
            return self.point_head.predict(example=example, test_cfg=self.test_cfg)
