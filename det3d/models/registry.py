from det3d.utils import Registry

READERS = Registry("reader")
BACKBONES = Registry("backbone")

IMG_BACKBONES = Registry("img_backbone")
IMG_HEADS = Registry("img_head")


NECKS = Registry("neck")
HEADS = Registry("head")
LOSSES = Registry("loss")
DETECTORS = Registry("detector")
SECOND_STAGE = Registry("second_stage")
ROI_HEAD = Registry("roi_head")
POINT_HEADS = Registry("point_head")