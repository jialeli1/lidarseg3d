from cgi import test
import os.path as osp
from tracemalloc import start
from matplotlib.pyplot import annotate
import numpy as np
import pickle
import random

from pathlib import Path
from functools import reduce
from typing import Tuple, List
import os 
import json 
from tqdm import tqdm
import argparse

from tqdm import tqdm
try:
    import tensorflow as tf
    tf.enable_eager_execution()
except:
    print("No Tensorflow")

from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

import zlib
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_metrics_pb2
from waymo_open_dataset.protos import segmentation_submission_pb2



semantic_labels = {
    0: 'TYPE_UNDEFINED',
    1: 'TYPE_CAR',
    2: 'TYPE_TRUCK',
    3: 'TYPE_BUS',
    4: 'TYPE_OTHER_VEHICLE', # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    5: 'TYPE_MOTORCYCLIST',
    6: 'TYPE_BICYCLIST',
    7: 'TYPE_PEDESTRIAN',
    8: 'TYPE_SIGN',
    9: 'TYPE_TRAFFIC_LIGHT',
    10: 'TYPE_POLE', # Lamp post, traffic sign pole etc.
    11: 'TYPE_CONSTRUCTION_CONE', # Construction cone/pole.
    12: 'TYPE_BICYCLE',
    13: 'TYPE_MOTORCYCLE',
    14: 'TYPE_BUILDING',
    15: 'TYPE_VEGETATION', # Bushes, tree branches, tall grasses, flowers etc.
    16: 'TYPE_TREE_TRUNK',
    17: 'TYPE_CURB', # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
    18: 'TYPE_ROAD', # Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    19: 'TYPE_LANE_MARKER', # Marking on the road that’s specifically for defining lanes such as single/double white/yellow lines.
    20: 'TYPE_OTHER_GROUND', # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    21: 'TYPE_WALKABLE', # Most horizontal surface that’s not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    22: 'TYPE_SIDEWALK', # Nicely paved walkable surface when pedestrians most likely to walk on.
}

color_map = { #rgb value
    0: [32, 119, 181],
    1: [228, 119, 194],
    2: [43, 160, 43],
    3: [220, 220, 141],
    4: [197, 176, 213],
    5: [209, 255, 6],
    6: [248, 182, 210],
    7: [152,224, 137],
    8: [29, 190, 208],
    9: [21, 255, 92],
    10: [174, 199, 232],
    11: [172,127,127],
    12: [215, 39, 40],
    13: [12, 116, 255],
    14: [140, 86, 74],
    15: [255, 127, 25],
    16: [200, 200, 200],
    17: [255, 152, 149],
    18: [158, 218, 229],
    19: [196, 156, 148],
    20: [255, 187, 120],
    21: [188, 190, 33],
    22: [148, 103, 189],
}


TOP_LIDAR_ROW_NUM = 64
TOP_LIDAR_COL_NUM = 2650

CAT_NAME_TO_ID = {
    'VEHICLE': 1,
    'PEDESTRIAN': 2,
    'SIGN': 3,
    'CYCLIST': 4,
}
TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

# ignore sign class 
LABEL_TO_TYPE = {0: 1, 1:2, 2:4}

import uuid 

class UUIDGeneration():
    def __init__(self):
        self.mapping = {}
    def get_uuid(self,seed):
        if seed not in self.mapping:
            self.mapping[seed] = uuid.uuid4().hex 
        return self.mapping[seed]
uuid_gen = UUIDGeneration()


def compress_array(array: np.ndarray, is_int32: bool = False):
    """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

    Args:
        array: A numpy array.
        is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

    Returns:
        The compressed bytes.
    """
    if is_int32:
        m = open_dataset.MatrixInt32()
    else:
        m = open_dataset.MatrixFloat()
    m.shape.dims.extend(list(array.shape))
    m.data.extend(array.reshape([-1]).tolist())
    return zlib.compress(m.SerializeToString())

def decompress_array(array_compressed: bytes, is_int32: bool = False):
    """Decompress bytes (of serialized MatrixFloat/Int32) to a numpy array.

    Args:
        array_compressed: bytes.
        is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

    Returns:
        The decompressed numpy array.
    """
    decompressed = zlib.decompress(array_compressed)
    if is_int32:
        m = open_dataset.MatrixInt32()
        dtype = np.int32
    else:
        m = open_dataset.MatrixFloat()
        dtype = np.float32
        m.ParseFromString(decompressed)
    return np.array(m.data, dtype=dtype).reshape(m.shape.dims)


def get_range_image_point_indexing(range_images, ri_index=0):
  """Get the indices of the valid points (of the TOP lidar) in the range image.

  The order of the points match those from convert_range_image_to_point_cloud
  and convert_range_image_to_point_cloud_labels.

  Args:
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    points_indexing_top: (N, 2) col and row indices of the points in the
      TOP lidar.
  """
  points_indexing_top = None
  xgrid, ygrid = np.meshgrid(range(TOP_LIDAR_COL_NUM), range(TOP_LIDAR_ROW_NUM))
  col_row_inds_top = np.stack([xgrid, ygrid], axis=-1)
  range_image = range_images[open_dataset.LaserName.TOP][ri_index]
  range_image_tensor = tf.reshape(
      tf.convert_to_tensor(range_image.data), range_image.shape.dims)
  range_image_mask = range_image_tensor[..., 0] > 0
  points_indexing_top = col_row_inds_top[np.where(range_image_mask)]
  return points_indexing_top



def dummy_semseg_for_one_frame(frame, dummy_class=14):
    """Assign all valid points to a single dummy class.

    Args:
        frame: An Open Dataset Frame proto.
        dummy_class: The class to assign to. Default is 14 (building).

    Returns:
        segmentation_frame: A SegmentationFrame proto.
    """
    (range_images, camera_projections, segmentation_labels,
    range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)
    
    # Get the col, row indices of the valid points.
    points_indexing_top = get_range_image_point_indexing(range_images, ri_index=0)
    points_indexing_top_ri2 = get_range_image_point_indexing(
        range_images, ri_index=1)

    # Assign the dummy class to all valid points (in the range image)
    range_image_pred = np.zeros(
        (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
    range_image_pred[points_indexing_top[:, 1], points_indexing_top[:, 0], 1] = dummy_class
    range_image_pred_ri2 = np.zeros(
        (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
    range_image_pred_ri2[points_indexing_top_ri2[:, 1], points_indexing_top_ri2[:, 0], 1] = dummy_class

    # Construct the SegmentationFrame proto.
    segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
    segmentation_frame.context_name = frame.context.name
    segmentation_frame.frame_timestamp_micros = frame.timestamp_micros
    laser_semseg = open_dataset.Laser()
    laser_semseg.name = open_dataset.LaserName.TOP
    laser_semseg.ri_return1.segmentation_label_compressed = compress_array(
        range_image_pred, is_int32=True)
    laser_semseg.ri_return2.segmentation_label_compressed = compress_array(
        range_image_pred_ri2, is_int32=True)
    segmentation_frame.segmentation_labels.append(laser_semseg)

    return segmentation_frame


def compress_semseg_for_one_frame(frame, pred_point_semcls_top_ri1, pred_point_semcls_top_ri2):
    """Assign all valid points to a single dummy class.

    Args:
        frame: An Open Dataset Frame proto.
        pred_point_semcls_top_ri1: [num_points_of_top_lidar_ri1, ]
        pred_point_semcls_top_ri2: [num_points_of_top_lidar_ri2, ]

    Returns:
        segmentation_frame: A SegmentationFrame proto.
    """
    (range_images, camera_projections, segmentation_labels,
    range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)
    
    # Get the col, row indices of the valid points.
    points_indexing_top = get_range_image_point_indexing(range_images, ri_index=0)
    points_indexing_top_ri2 = get_range_image_point_indexing(
        range_images, ri_index=1)

    # Assign the pred class to all valid points (in the range image)
    range_image_pred = np.zeros((TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
    range_image_pred[points_indexing_top[:, 1], points_indexing_top[:, 0], 1] = pred_point_semcls_top_ri1
    range_image_pred_ri2 = np.zeros((TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
    range_image_pred_ri2[points_indexing_top_ri2[:, 1], points_indexing_top_ri2[:, 0], 1] = pred_point_semcls_top_ri2

    # Construct the SegmentationFrame proto.
    segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
    segmentation_frame.context_name = frame.context.name
    segmentation_frame.frame_timestamp_micros = frame.timestamp_micros
    laser_semseg = open_dataset.Laser()
    laser_semseg.name = open_dataset.LaserName.TOP
    laser_semseg.ri_return1.segmentation_label_compressed = compress_array(
        range_image_pred, is_int32=True)
    laser_semseg.ri_return2.segmentation_label_compressed = compress_array(
        range_image_pred_ri2, is_int32=True)
    segmentation_frame.segmentation_labels.append(laser_semseg)

    return segmentation_frame



def _create_pd_segmentation(detections, infos, result_path, test_set):
    """
    Creates a prediction objects file.
    detections: segmentations

    """
    if test_set:
        root_path = "data/SemanticWaymo/tfrecord_testing"
    else:
        root_path = "data/SemanticWaymo/tfrecord_validation"


    segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()
    for token, segmentation in tqdm(detections.items()):
        """
        segmentation: a dict
            pred_point_sem_labels: cpu torch tensor
        """
        info = infos[token]
        anno_obj = get_obj(info['anno_path'])
        lidar_example_obj = get_obj(info['path'])


        pred_point_sem_labels = segmentation["pred_point_sem_labels"].numpy().astype(np.int32)
        # split the pred_point_sem_labels into pred_point_semcls_top_ri1 and pred_point_semcls_top_ri2
        num_points_of_top_lidar_ri1 = lidar_example_obj["lidars"]["num_points_of_top_lidar"]["ri_return1"]
        num_points_of_top_lidar_ri2 = lidar_example_obj["lidars"]["num_points_of_top_lidar"]["ri_return2"]
        pred_point_semcls_top_ri1 = pred_point_sem_labels[:num_points_of_top_lidar_ri1]
        pred_point_semcls_top_ri2 = pred_point_sem_labels[num_points_of_top_lidar_ri1: num_points_of_top_lidar_ri1+num_points_of_top_lidar_ri2]



        context_name = anno_obj['scene_name']
        frame_timestamp_micros = int(anno_obj['frame_name'].split("_")[-1])
        tfrecord_local_filename = "segment-{}_with_camera_labels.tfrecord".format(context_name)

        # frame_name= anno_obj['frame_name']
        # frame_id = anno_obj['frame_id']
        # print("==> context_name: {}, frame_name: {}, frame_id: {}".format(context_name, frame_name, frame_id))
        # ==> context_name: 10444454289801298640_4360_000_4380_000, frame_name: 10444454289801298640_4360_000_4380_000_location_sf_Day_1557162926373707, frame_id: 115

        tfrecord_filename = os.path.join(root_path, tfrecord_local_filename)
        dataset = tf.data.TFRecordDataset(tfrecord_filename, compression_type='')
        
        for data in dataset:
            cur_frame = open_dataset.Frame()
            cur_frame.ParseFromString(bytearray(data.numpy()))
            cur_context_name = cur_frame.context.name
            cur_timestamp = cur_frame.timestamp_micros
            if cur_context_name == context_name and cur_timestamp == frame_timestamp_micros:
                # print("==> found the frame.")
                break


        if not test_set:
            assert cur_frame.lasers[0].ri_return1.segmentation_label_compressed

        # segmentation_frame = dummy_semseg_for_one_frame(frame)
        segmentation_frame = compress_semseg_for_one_frame(cur_frame, pred_point_semcls_top_ri1, pred_point_semcls_top_ri2)

        segmentation_frame_list.frames.append(segmentation_frame)


    print('Total number of frames in _create_pd_segmentation: ', len(segmentation_frame_list.frames))
    


    submission = segmentation_submission_pb2.SemanticSegmentationSubmission()
    submission.account_name = 'yourAccount@gmail.com'
    submission.unique_method_name = 'xxx' # 25 characters are allowed.
    submission.affiliation = 'Anonymous submission'
    submission.authors.append('Anonymous authors')
    submission.description = "A 3D semantic segmentaion method."
    submission.method_link = 'NA'
    submission.sensor_type = 1
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(segmentation_frame_list)


    output_filename = os.path.join(result_path, 'semseg_pred_submission.bin')
    f = open(output_filename, 'wb')
    f.write(submission.SerializeToString())
    f.close()
    print("==> Submit the generated file in {} to waymo online server for evaluation.".format(output_filename))





def _create_pd_detection(detections, infos, result_path, tracking=False):
    """Creates a prediction objects file."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2

    objects = metrics_pb2.Objects()

    for token, detection in tqdm(detections.items()):
        info = infos[token]
        obj = get_obj(info['anno_path'])

        box3d = detection["box3d_lidar"].detach().cpu().numpy()
        scores = detection["scores"].detach().cpu().numpy()
        labels = detection["label_preds"].detach().cpu().numpy()

        # transform back to Waymo coordinate
        # x,y,z,w,l,h,r2
        # x,y,z,l,w,h,r1
        # r2 = -pi/2 - r1  
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]

        if tracking:
            tracking_ids = detection['tracking_ids']

        for i in range(box3d.shape[0]):
            det  = box3d[i]
            score = scores[i]

            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label] 

            if tracking:
                o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

            objects.objects.append(o)

    # Write objects to a file.
    if tracking:
        path = os.path.join(result_path, 'tracking_pred.bin')
    else:
        path = os.path.join(result_path, 'detection_pred.bin')

    print("results saved to {}".format(path))
    f = open(path, 'wb')
    f.write(objects.SerializeToString())
    f.close()

def _create_gt_detection(infos, tracking=True):
    """Creates a gt prediction object file for local evaluation."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    
    objects = metrics_pb2.Objects()

    for idx in tqdm(range(len(infos))): 
        info = infos[idx]

        obj = get_obj(info['path'])
        annos = obj['objects']
        num_points_in_gt = np.array([ann['num_points'] for ann in annos])
        box3d = np.array([ann['box'] for ann in annos])

        if len(box3d) == 0:
            continue 

        names = np.array([TYPE_LIST[ann['label']] for ann in annos])

        box3d = box3d[:, [0, 1, 2, 3, 4, 5, -1]]

        for i in range(box3d.shape[0]):
            if num_points_in_gt[i] == 0:
                continue 
            if names[i] == 'UNKNOWN':
                continue 

            det  = box3d[i]
            score = 1.0
            label = names[i]

            o = metrics_pb2.Object()
            o.context_name = obj['scene_name']
            o.frame_timestamp_micros = int(obj['frame_name'].split("_")[-1])

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = CAT_NAME_TO_ID[label]
            o.object.num_lidar_points_in_box = num_points_in_gt[i]
            o.object.id = annos[i]['name']

            objects.objects.append(o)
        
    # Write objects to a file.
    f = open(os.path.join(args.result_path, 'gt_preds.bin'), 'wb')
    f.write(objects.SerializeToString())
    f.close()

def veh_pos_to_transform(veh_pos):
    "convert vehicle pose to two transformation matrix"
    rotation = veh_pos[:3, :3] 
    tran = veh_pos[:3, 3]

    global_from_car = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=False
    )

    car_from_global = transform_matrix(
        tran, Quaternion(matrix=rotation), inverse=True
    )

    return global_from_car, car_from_global

def _fill_infos(root_path, frames, split='train', nsweeps=1):
    # load all train infos
    infos = []
    camera_ids = ['1', '2', '3', '4', '5'] # cam id

    if split == 'test':
        # testing_set_frame_file = '/path/3d_semseg_test_set_frames.txt'
        testing_set_frame_file = os.path.join(root_path, '3d_semseg_test_set_frames.txt')
        context_name_timestamp_tuples = [x.rstrip().split(',') for x in (open(testing_set_frame_file, 'r').readlines())]
        print("==> len(context_name_timestamp_tuples): ", len(context_name_timestamp_tuples))
        # 2982
        # print("==> context_name_timestamp_tuples[0]: ", context_name_timestamp_tuples[0])
    
    # count = 0
    for frame_name in tqdm(frames):  # global id
        # count += 1
        # if count > 100:
        #     # stop for debug 
        #     break

        lidar_path = os.path.join(root_path, split, 'lidar', frame_name)
        ref_path = os.path.join(root_path, split, 'annos', frame_name)

        ref_obj = get_obj(ref_path)
        ref_time = 1e-6 * int(ref_obj['frame_name'].split("_")[-1])

        ref_pose = np.reshape(ref_obj['veh_to_global'], [4, 4])
        _, ref_from_global = veh_pos_to_transform(ref_pose)

        # indication for seg annotated frames
        seg_annotated = ref_obj["seg_labels"]["points_seglabel"].shape[0] > 0


        # set seg_annotated as False for test set
        if split == 'test':
            context_name = ref_obj['scene_name']
            # frame_timestamp_micros = int(ref_obj['frame_name'].split("_")[-1])
            frame_timestamp_micros = ref_obj['frame_name'].split("_")[-1]
            # print("==> context_name: {}, frame_timestamp_micros: {}, str(): {}".format(context_name, frame_timestamp_micros, str(frame_timestamp_micros)))
            # ==> context_name: 3522804493060229409_3400_000_3420_000, frame_timestamp_micros: 1557855897472206, str(): 1557855897472206
            if [context_name, frame_timestamp_micros] in context_name_timestamp_tuples:
                # print("==> set seg_annotated as True for context_name: {}, frame_timestamp_micros: {} on TEST set.".format(context_name, frame_timestamp_micros))
                # ==> set seg_annotated as True for context_name: 3522804493060229409_3400_000_3420_000, frame_timestamp_micros: 1557855897472206 on TEST set.
                seg_annotated = True


        # frame_name should be 'seq_{}_frame_{}.pkl'.format(idx, frame_id)
        sequence_id = int(frame_name.split("_")[1])
        frame_id = int(frame_name.split("_")[3][:-4]) # remove postfix ".pkl"

        # image_file = os.path.join(CAM_PATH, 'seq_{}_frame_{}_cam_{}.png'.format(idx, frame_id, camera_id))
        cam_paths = {}
        for camera_id in camera_ids:
            frame_name_of_cam = 'seq_{}_frame_{}_cam_{}.png'.format(sequence_id, frame_id, camera_id)
            cam_paths[camera_id] = os.path.join(root_path, split, 'cam', frame_name_of_cam)


        info = {
            "path": lidar_path,
            "anno_path": ref_path, 
            "cam_paths": cam_paths, 
            "token": frame_name,
            "timestamp": ref_time,
            "sweeps": [],
            "seg_annotated": seg_annotated, 
        }


        prev_id = frame_id
        sweeps = [] 
        while len(sweeps) < nsweeps - 1:
            if prev_id <= 0:
                if len(sweeps) == 0:
                    sweep = {
                        "path": lidar_path,
                        "anno_path": ref_path, 
                        "token": frame_name,
                        "transform_matrix": None,
                        "time_lag": 0,
                        "seg_annotated": seg_annotated, 
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                prev_id = prev_id - 1
                # global identifier  

                curr_name = 'seq_{}_frame_{}.pkl'.format(sequence_id, prev_id)
                curr_lidar_path = os.path.join(root_path, split, 'lidar', curr_name)
                curr_label_path = os.path.join(root_path, split, 'annos', curr_name)
                
                curr_obj = get_obj(curr_label_path)
                curr_pose = np.reshape(curr_obj['veh_to_global'], [4, 4])
                global_from_car, _ = veh_pos_to_transform(curr_pose) 
                
                tm = reduce(
                    np.dot,
                    [ref_from_global, global_from_car],
                )

                curr_time = int(curr_obj['frame_name'].split("_")[-1])
                time_lag = ref_time - 1e-6 * curr_time

                curr_seg_annotated = curr_obj["seg_labels"]["points_seglabel"].shape[0] > 0

                sweep = {
                    "path": curr_lidar_path,
                    "anno_path": curr_label_path, 
                    "transform_matrix": tm,
                    "time_lag": time_lag,
                    "seg_annotated": curr_seg_annotated, 
                }
                sweeps.append(sweep)

        info["sweeps"] = sweeps

        if split != 'test':
            # read boxes 
            TYPE_LIST = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST']
            annos = ref_obj['objects']
            num_points_in_gt = np.array([ann['num_points'] for ann in annos])
            gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 9)
            
            if len(gt_boxes) != 0:
                # transform from Waymo to KITTI coordinate 
                # Waymo: x, y, z, length, width, height, rotation from positive x axis clockwisely
                # KITTI: x, y, z, width, length, height, rotation from negative y axis counterclockwisely 
                gt_boxes[:, -1] = -np.pi / 2 - gt_boxes[:, -1]
                gt_boxes[:, [3, 4]] = gt_boxes[:, [4, 3]]

            gt_names = np.array([TYPE_LIST[ann['label']] for ann in annos])
            mask_not_zero = (num_points_in_gt > 0).reshape(-1)    

            # filter boxes without lidar points 
            info['gt_boxes'] = gt_boxes[mask_not_zero, :].astype(np.float32)
            info['gt_names'] = gt_names[mask_not_zero].astype(str)

        infos.append(info)
    return infos


def sort_frame(frames):
    indices = [] 

    for f in frames:
        seq_id = int(f.split("_")[1])
        frame_id= int(f.split("_")[3][:-4])

        idx = seq_id * 1000 + frame_id
        indices.append(idx)

    rank = list(np.argsort(np.array(indices)))

    frames = [frames[r] for r in rank]
    return frames

def get_available_frames(root, split):
    dir_path = os.path.join(root, split, 'lidar')
    available_frames = list(os.listdir(dir_path))

    sorted_frames = sort_frame(available_frames)

    print(split, " split ", "exist frame num:", len(available_frames))
    return sorted_frames


def create_waymo_infos(root_path, split='train', nsweeps=1):
    frames = get_available_frames(root_path, split)

    waymo_infos = _fill_infos(
        root_path, frames, split, nsweeps
    )

    print(
        f"sample: {len(waymo_infos)}"
    )
    with open(
        os.path.join(root_path, "infos_"+split+"_{:02d}sweeps_segdet_filter_zero_gt.pkl".format(nsweeps)), "wb"
    ) as f:
        pickle.dump(waymo_infos, f)




def parse_args():
    parser = argparse.ArgumentParser(description="Waymo 3D Extractor")
    parser.add_argument("--path", type=str, default="data/Waymo/tfrecord_training")
    parser.add_argument("--info_path", type=str)
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--gt", action='store_true' )
    parser.add_argument("--tracking", action='store_true')
    args = parser.parse_args()
    return args


def reorganize_info(infos):
    new_info = {}

    for info in infos:
        token = info['token']
        new_info[token] = info

    return new_info 

if __name__ == "__main__":
    args = parse_args()

    with open(args.info_path, 'rb') as f:
        infos = pickle.load(f)
    
    if args.gt:
        _create_gt_detection(infos, tracking=args.tracking)
        exit() 

    infos = reorganize_info(infos)
    with open(args.path, 'rb') as f:
        preds = pickle.load(f)
    _create_pd_detection(preds, infos, args.result_path, tracking=args.tracking)
