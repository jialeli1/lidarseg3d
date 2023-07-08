from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pickle import NONE
from turtle import shape

import zlib
import numpy as np

import tensorflow.compat.v2 as tf
from pyquaternion import Quaternion

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
tf.enable_v2_behavior()



def decode_images(frame, frame_id):
  """
  Decodes imgs like mmdet3d, Waymo2KITTI.save_image
  """
  encoded_images = sorted(frame.images, key=lambda i:i.name)


  image_list = []
  camera_list = []
  for encoded_image in encoded_images:
    decoded_image = tf.image.decode_jpeg(encoded_image.image).numpy()
    image_list.append(decoded_image)
    camera_list.append(encoded_image.name)
    # print("==> decoded_image.shape: {}".format(decoded_image.shape) )
    # (1280, 1920, 3) or (886, 1920, 3)


  # images_dict = {
  #   'cameras': camera_list,
  #   'images': image_list, 
  # }

  return image_list, camera_list


def decode_frame_points_and_seglabels(frame, frame_id, return_camera_proj=False):
  """Decodes native waymo Frame proto to tf.Examples."""

  lidars, seg_labels = extract_points_and_seglabels(
                            frame.lasers,
                            frame.context.laser_calibrations,
                            frame.pose,
                            return_camera_proj)

  frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
      scene_name=frame.context.name,
      location=frame.context.stats.location,
      time_of_day=frame.context.stats.time_of_day,
      timestamp=frame.timestamp_micros)

  example_data = {
      'scene_name': frame.context.name,
      'frame_name': frame_name,
      'frame_id': frame_id,
      'lidars': lidars,
  }

  veh_to_global = np.array(frame.pose.transform)
  annos = {
    'scene_name': frame.context.name,
    'frame_name': frame_name,
    'frame_id': frame_id,
    'veh_to_global': veh_to_global,  
    'seg_labels': seg_labels,
  }

  return example_data, annos



def extract_points_and_seglabels_from_range_image(laser, calibration, frame_pose, return_camera_proj=False):
  """
  Decode points & seg labels for each lidar from range_image.
  laser: 
  """
  if laser.name != calibration.name:
    raise ValueError('Laser and calibration do not match')
  if laser.name == dataset_pb2.LaserName.TOP:
    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame_pose.transform), [4, 4]))
    range_image_top_pose = dataset_pb2.MatrixFloat.FromString(
        zlib.decompress(laser.ri_return1.range_image_pose_compressed))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[...,
                                                                         2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[...,
                                                                          3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    frame_pose = tf.expand_dims(frame_pose, axis=0)
    pixel_pose = tf.expand_dims(range_image_top_pose_tensor, axis=0)
  else:
    pixel_pose = None
    frame_pose = None

  # range_image  
  range_image_first_return = zlib.decompress(
      laser.ri_return1.range_image_compressed)
  range_image_second_return = zlib.decompress(
      laser.ri_return2.range_image_compressed)
  range_image_returns = [range_image_first_return, range_image_second_return]


  # ---------------------seg labels------------------------------
  # Check if the current frame has a segmentation_label
  if len(laser.ri_return1.segmentation_label_compressed) > 0:
    segmentation_label_first_return = zlib.decompress(
        laser.ri_return1.segmentation_label_compressed)
  else:
    segmentation_label_first_return = None  
  if len(laser.ri_return2.segmentation_label_compressed) > 0:
    segmentation_label_second_return = zlib.decompress(
        laser.ri_return2.segmentation_label_compressed)
  else:
    segmentation_label_second_return = None  
  segmentation_label_returns = [segmentation_label_first_return, segmentation_label_second_return]
  # ---------------------------------------------------


  # ---------------------L-to-C------------------------------
  if return_camera_proj:
    # camera_projection_first_return = tf.io.decode_compressed(laser.ri_return1.camera_projection_compressed, 'ZLIB').numpy()
    # camera_projection_second_return = tf.io.decode_compressed(laser.ri_return2.camera_projection_compressed, 'ZLIB').numpy()
    camera_projection_first_return = zlib.decompress(laser.ri_return1.camera_projection_compressed)
    camera_projection_second_return = zlib.decompress(laser.ri_return2.camera_projection_compressed)

    camera_projection_returns = [camera_projection_first_return, camera_projection_second_return]
    # cp = dataset_pb2.MatrixInt32()
    # cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))

    # store [idx of cam, idx of width (col), idx of height (row),] for each point
    points_cp_list = [] 
  else:
    camera_projection_returns = [None, None]
  # ---------------------------------------------------


  points_list = []
  points_seglabel_list = []
  for range_image_str, segmentation_label_str, camera_projection_str in zip(range_image_returns, segmentation_label_returns, camera_projection_returns):
    range_image = dataset_pb2.MatrixFloat.FromString(range_image_str)
    # label_img
    if not calibration.beam_inclinations:
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([
              calibration.beam_inclination_min, calibration.beam_inclination_max
          ]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(calibration.beam_inclinations)
    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(calibration.extrinsic.transform), [4, 4])
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = (
        range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose,
            frame_pose=frame_pose))
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(
        tf.concat([range_image_cartesian, range_image_tensor[..., 1:4]],
                  axis=-1),
        tf.where(range_image_mask))

    # first_return and second_return
    points_list.append(points_tensor.numpy())


    if segmentation_label_str is not None:
      # str -> int32  
      segmentation_label = dataset_pb2.MatrixInt32.FromString(segmentation_label_str)
      segmentation_label_tensor = tf.reshape(tf.convert_to_tensor(segmentation_label.data), segmentation_label.shape.dims) # [64, 2650, 2]
      # gather_nd一下
      points_seglabel_tensor = tf.gather_nd(segmentation_label_tensor, tf.where(range_image_mask))
      assert points_seglabel_tensor.shape[0] == points_tensor.shape[0]
      assert points_seglabel_tensor.shape[1] == 2 # [instance_labels, semantic_labels]
      points_seglabel_numpy = points_seglabel_tensor.numpy()
    else:
      # set as undefined or just empty
      # unlabeled_points_num = tf.reduce_sum(tf.cast(range_image_mask, dtype=tf.int32))
      # NOTE: TYPE_UNDEFINED is 0 
      # points_seglabel_tensor = tf.zeros([unlabeled_points_num, 2], dtype=tf.int32)
      points_seglabel_numpy = np.zeros(shape=(0, 2), dtype=np.int32)

    points_seglabel_list.append(points_seglabel_numpy)

    # ---------------------L-to-C------------------------------
    if return_camera_proj:
      # cp = dataset_pb2.MatrixInt32()
      # cp.ParseFromString(bytearray(camera_projection_str)) 
      cp = dataset_pb2.MatrixInt32.FromString(camera_projection_str)
      # range_image_mask.shape: [64, 2650] or [200, 600]
      # cp_tensor.shape: [64, 2650, 6] or [200, 600, 6]
      cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims) 
      points_cp_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
      assert points_cp_tensor.shape[0] == points_tensor.shape[0]
      points_cp_numpy = points_cp_tensor.numpy() # [points_tensor.shape[0], 6]
      # keep only [idx of cam, idx of width (col), idx of height (row)] for now
      # NOTE: The cam id will have a value of 0, in order to mark those points that cannot be projected on any camera


      points_cp_list.append(points_cp_numpy[:, :3])
    else:
      points_cp_list = None
    # ---------------------------------------------------

  return points_list, points_seglabel_list, points_cp_list



def extract_points_and_seglabels(lasers, laser_calibrations, frame_pose, return_camera_proj=False):
  """Extract point clouds from ALL_LIDAR and seglabels from the annotated TOP_LIDAR."""
  sort_lambda = lambda x: x.name
  lasers_with_calibration = zip(
      sorted(lasers, key=sort_lambda),
      sorted(laser_calibrations, key=sort_lambda))
  
  points_xyz = []
  points_feature = []
  points_nlz = []
  points_seglabel = []
  if return_camera_proj:
    points_cp = []


  for laser, calibration in lasers_with_calibration:
    # points_list of [first_return, second_return]
    # points_seglabel_list of [first_return, second_return], 
    points_list, points_seglabel_list, points_cp_list = extract_points_and_seglabels_from_range_image(laser, calibration, frame_pose, return_camera_proj)
    if laser.name == dataset_pb2.LaserName.TOP:
      # record the point number of first_return and second_return
      num_points_of_top_lidar = {
        'ri_return1':points_list[0].shape[0],
        'ri_return2':points_list[1].shape[0]
      }
    points = np.concatenate(points_list, axis=0)
    points_xyz.extend(points[..., :3].astype(np.float32))
    points_feature.extend(points[..., 3:5].astype(np.float32))
    points_nlz.extend(points[..., 5].astype(np.float32))
    
    seglabel = np.concatenate(points_seglabel_list, axis=0)
    points_seglabel.extend(seglabel.astype(np.int32))
    if laser.name == dataset_pb2.LaserName.TOP and seglabel.shape[0] > 0:
      assert points.shape[0] == seglabel.shape[0], 'for TOP lidar, points: {} and seglabel: {} do not match'.format(points.shape, seglabel.shape)

    if return_camera_proj:
      cp = np.concatenate(points_cp_list, axis=0)
      points_cp.extend(cp.astype(np.int32)) # NOTE: check type

  lidar_dict = {
      'points_xyz': np.asarray(points_xyz),
      'points_feature': np.asarray(points_feature),
      'num_points_of_top_lidar': num_points_of_top_lidar,
  }
  if return_camera_proj:
    lidar_dict['points_cp'] = np.asarray(points_cp)

  seglabel_dict = {
      'points_seglabel': np.asarray(points_seglabel).reshape(-1, 2),
  }


  return lidar_dict, seglabel_dict