from .nuscenes import NuScenesDataset
from .nuscenes import SemanticNuscDataset

from .waymo import WaymoDataset
from .waymo import SemanticWaymoDataset

from .semantickitti import SemanticKITTIDataset



dataset_factory = {
    "NUSC": NuScenesDataset,
    "WAYMO": WaymoDataset,
    "SemanticKITTIDataset": SemanticKITTIDataset,
    "SemanticNuscDataset": SemanticNuscDataset,
    "SEMANTICWAYMO": SemanticWaymoDataset,
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
