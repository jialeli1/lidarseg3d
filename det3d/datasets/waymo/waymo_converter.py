"""Tool to convert Waymo Open Dataset to pickle files.
    Adapted from https://github.com/WangYueFt/pillar-od
    # Copyright (c) Massachusetts Institute of Technology and its affiliates.
    # Licensed under MIT License
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, argparse, tqdm, pickle, os 

import waymo_decoder 
import semanticwaymo_decoder 


import tensorflow.compat.v2 as tf
from waymo_open_dataset import dataset_pb2

from multiprocessing import Pool 
import cv2

tf.enable_v2_behavior()

fnames = None 
LIDAR_PATH = None
ANNO_PATH = None 
CAM_PATH = None 
DECODE_CAM_PROJ = True

def convert(idx):
    global fnames
    fname = fnames[idx]
    dataset = tf.data.TFRecordDataset(fname, compression_type='')
    for frame_id, data in enumerate(dataset):

        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # decoded_frame = waymo_decoder.decode_frame(frame, frame_id)

        # decode seg annotation with data
        # decoded_frame, decoded_annos_seg = semanticwaymo_decoder.decode_frame_points_and_seglabels(frame, frame_id)
        decoded_frame, decoded_annos_seg = semanticwaymo_decoder.decode_frame_points_and_seglabels(frame, frame_id, return_camera_proj=DECODE_CAM_PROJ)

        # decode det annotation
        decoded_annos_det = waymo_decoder.decode_annos(frame, frame_id)

        # decode multicam imgs
        decoded_images, decoded_cameras = semanticwaymo_decoder.decode_images(frame, frame_id)

        # combine the det & seg annos
        decoded_annos = {}
        decoded_annos.update(decoded_annos_det)
        decoded_annos["seg_labels"] = decoded_annos_seg["seg_labels"]


        with open(os.path.join(LIDAR_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_frame, f)
        
        with open(os.path.join(ANNO_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(decoded_annos, f)

        # add: dump the cam images
        for image, camera_id in zip(decoded_images, decoded_cameras):
            # rgb mode to bgr mode for cv2.imwrite
            # print("==> camera_id: {}, image.shape: {}".format(camera_id, image.shape))
            image_file = os.path.join(CAM_PATH, 'seq_{}_frame_{}_cam_{}.png'.format(idx, frame_id, camera_id))
            cv2.imwrite(image_file, image[:, :, [2,1,0]])


def main(args):
    global fnames 
    fnames = list(glob.glob(args.record_path))

    print("Number of files {}".format(len(fnames)))

    # with Pool(128) as p: # change according to your cpu
    with Pool(16) as p: # change according to your cpu
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--record_path', type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    LIDAR_PATH = os.path.join(args.root_path, 'lidar')
    ANNO_PATH = os.path.join(args.root_path, 'annos')
    CAM_PATH = os.path.join(args.root_path, 'cam')

    if not os.path.isdir(LIDAR_PATH):
        os.mkdir(LIDAR_PATH)

    if not os.path.isdir(ANNO_PATH):
        os.mkdir(ANNO_PATH)
    
    if not os.path.isdir(CAM_PATH):
        os.mkdir(CAM_PATH)

    main(args)
