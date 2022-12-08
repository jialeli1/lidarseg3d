import numpy as np
import pickle

from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

_kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]


# from Cylinder3D
labels = { 
  0: 'noise',
  1: 'animal',
  2: 'human.pedestrian.adult',
  3: 'human.pedestrian.child',
  4: 'human.pedestrian.construction_worker',
  5: 'human.pedestrian.personal_mobility',
  6: 'human.pedestrian.police_officer',
  7: 'human.pedestrian.stroller',
  8: 'human.pedestrian.wheelchair',
  9: 'movable_object.barrier',
  10: 'movable_object.debris',
  11: 'movable_object.pushable_pullable',
  12: 'movable_object.trafficcone',
  13: 'static_object.bicycle_rack',
  14: 'vehicle.bicycle',
  15: 'vehicle.bus.bendy',
  16: 'vehicle.bus.rigid',
  17: 'vehicle.car',
  18: 'vehicle.construction',
  19: 'vehicle.emergency.ambulance',
  20: 'vehicle.emergency.police',
  21: 'vehicle.motorcycle',
  22: 'vehicle.trailer',
  23: 'vehicle.truck',
  24: 'flat.driveable_surface',
  25: 'flat.other',
  26: 'flat.sidewalk',
  27: 'flat.terrain',
  28: 'static.manmade',
  29: 'static.other',
  30: 'static.vegetation',
  31: 'vehicle.ego',
}

labels_16 = {
  0: 'noise',
  1: 'barrier',
  2: 'bicycle',
  3: 'bus',
  4: 'car',
  5: 'construction_vehicle',
  6: 'motorcycle',
  7: 'pedestrian',
  8: 'traffic_cone',
  9: 'trailer',
  10: 'truck',
  11: 'driveable_surface',
  12: 'other_flat',
  13: 'sidewalk',
  14: 'terrain',
  15: 'manmade',
  16: 'vegetation'
  }
  
learning_map = {
  1: 0,
  5: 0,
  7: 0,
  8: 0,
  10: 0,
  11: 0,
  13: 0,
  19: 0,
  20: 0,
  0: 0,
  29: 0,
  31: 0,
  9: 1,
  14: 2,
  15: 3,
  16: 3,
  17: 4,
  18: 5,
  21: 6,
  2: 7,
  3: 7,
  4: 7,
  6: 7,
  12: 8,
  22: 9,
  23: 10,
  24: 11,
  25: 12,
  26: 13,
  27: 14,
  28: 15,
  30: 16
  }



classname_to_color = {  # RGB.
    "noise": (0, 0, 0),  # Black.
    "animal": (70, 130, 180),  # Steelblue
    "human.pedestrian.adult": (0, 0, 230),  # Blue
    "human.pedestrian.child": (135, 206, 235),  # Skyblue,
    "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
    "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
    "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
    "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
    "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
    "movable_object.barrier": (112, 128, 144),  # Slategrey
    "movable_object.debris": (210, 105, 30),  # Chocolate
    "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
    "movable_object.trafficcone": (47, 79, 79),  # Darkslategrey
    "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
    "vehicle.bicycle": (220, 20, 60),  # Crimson
    "vehicle.bus.bendy": (255, 127, 80),  # Coral
    "vehicle.bus.rigid": (255, 69, 0),  # Orangered
    "vehicle.car": (255, 158, 0),  # Orange
    "vehicle.construction": (233, 150, 70),  # Darksalmon
    "vehicle.emergency.ambulance": (255, 83, 0),
    "vehicle.emergency.police": (255, 215, 0),  # Gold
    "vehicle.motorcycle": (255, 61, 99),  # Red
    "vehicle.trailer": (255, 140, 0),  # Darkorange
    "vehicle.truck": (255, 99, 71),  # Tomato
    "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
    "flat.other": (175, 0, 75),
    "flat.sidewalk": (75, 0, 75),
    "flat.terrain": (112, 180, 60),
    "static.manmade": (222, 184, 135),  # Burlywood
    "static.other": (255, 228, 196),  # Bisque
    "static.vegetation": (0, 175, 0),  # Green
    "vehicle.ego": (255, 240, 245),
}


labels_16_color_map = {# RGB.
  0: (0, 0, 0), # Black.
  1: (112, 128, 144),  # Slategrey
  2: (220, 20, 60),  # Crimson
  3: (255, 69, 0), # Orangered
  4: (255, 158, 0),  # Orange, 'car',
  5: (233, 150, 70),  # Darksalmon, 'construction_vehicle',
  6: (255, 61, 99),  # Red, 'motorcycle',
  7: (0, 0, 230),  # Blue, 'pedestrian',
  8: (47, 79, 79),  # Darkslategrey, 'traffic_cone',
  9: (255, 140, 0),  # Darkorange, 'trailer',
  10: (255, 99, 71),  # Tomato, 'truck',
  11: (0, 207, 191),  # nuTonomy green, 'driveable_surface',
  12: (175, 0, 75), # 'other_flat',
  13: (75, 0, 75), # 'sidewalk',
  14: (112, 180, 60), # 'terrain',
  15: (222, 184, 135),  # Burlywood, 'manmade',
  16: (0, 175, 0),  # Green, 'vegetation'
}