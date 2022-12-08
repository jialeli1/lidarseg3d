import numpy as np
import copy
from track_utils import greedy_assignment
import copy 
import importlib
import sys 

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}



class PubTracker(object):
  def __init__(self,  hungarian=False, max_age=0):
    self.hungarian = hungarian
    self.max_age = max_age

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

    self.reset()
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    if len(results) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      for det in results:
        # filter out classes not evaluated for tracking 
        if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
          continue 

        det['ct'] = np.array(det['translation'][:2])
        # 这个tracking是匀速运动模型假设下的位移: delta_x = v * delta_t
        # 可以理解为将当前帧检测到的物体往后退至已有轨迹的最近时刻, 方向为objs -> tracks
        det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
        det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
        temp.append(det)

      results = temp

    # results 存放的时候是当前帧的所有检测到的物体, 包括合法的与不合法的
    # N是数目
    N = len(results)
    # self.tracks 存放的是已经匹配上/跟踪上/找到的轨迹
    # M 是已经匹配上/跟踪上/找到的轨迹数目
    M = len(self.tracks)

    # N X 2, 2 denotes [x, y] in the BEV 
    # 这里对当前帧的物体进行运动补偿/位置更新/中心点坐标更新
    if 'tracking' in results[0]:
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32)
       for det in results], np.float32)
    else:
      dets = np.array(
        [det['ct'] for det in results], np.float32) 

    # 这里是获取已有轨迹和当前帧物体的类别
    item_cat = np.array([item['label_preds'] for item in results], np.int32) # N
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

    # 匹配距离的阈值, 是经验估计的数值
    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

    # M x 2, 2 denotes [x, y] in the BEV
    # 获取已有轨迹的位置/中心点
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    # 只要不是第一帧, len(tracks)总是大于0的
    # 根据已有轨迹去寻找当前帧中matching cost最小的物体, 方向为objs -> tracks
    if len(tracks) > 0:  # NOT FIRST FRAME
      # 计算pair-wise的cost matrix, shaped as N x M
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # absolute distance in meter

      # 根据经验阈值做一个合法性检查, invalid是一个 bool matrix, also shaped as N x M
      invalid = ((dist > max_diff.reshape(N, 1)) + (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      # 将invalid的那些pair所对应的cost设置为无穷大(1e18), 
      # 而valid的那些pair的cost不改变.
      dist = dist  + invalid * 1e18
      if self.hungarian:
        dist[dist > 1e18] = 1e18
        # 匈牙利匹配算法进行匹配
        matched_indices = linear_assignment(copy.deepcopy(dist))
      else:
        # 或者贪心算法进行匹配
        matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:  # first few frame
      # 对第一帧进行一些初始化
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    # 获取没有匹配上的objs, new born 处理
    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]
    # 获取没有匹配上的已有轨迹, death 处理
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    # 根据matches中的pair index去获取已有轨迹的id, 并作为跟踪上/匹配上的id, 并输出
    # 因为是当前帧, 所以这些物体的age都是1
    # active会决定该轨迹是否会被当前步step输出, active == 0的轨迹不输出
    # 在已有轨迹self.tracks中会有一些non-active的, 比如某些中间帧没有被匹配上, 但又没有达到max_age的这些轨迹
    # 而对于non-active的这部分轨迹, 他们仍然是有可能再被匹配上的.
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    # 因为是当前帧, 所以这些未匹配上物体的age也都是1
    for i in unmatched_dets:
      track = results[i]
      # NOTE: 这里id_count是拿来做什么的呢？
      self.id_count += 1
      track['tracking_id'] = self.id_count
      track['age'] = 1
      track['active'] =  1
      ret.append(track)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.max_age:
        track['age'] += 1
        # 关掉active标值, 不在当前step输出, 但是会保留起来用于之后step里面可能会被匹配上的轨迹
        track['active'] = 0
        ct = track['ct']

        # movement in the last second
        # 运动模型补偿的恢复步骤, 不影响当前帧的物体位置, 方向是tracks -> objs
        # NOTE: 不过为什么会是这个条件呢: 'tracking' in track
        # 是因为前面输入的时候只对部分关心的类别进行了运动补偿, 所以这里也只操作这部分关心的类别
        if 'tracking' in track:
            offset = track['tracking'] * -1 # move forward 
            track['ct'] = ct + offset 
        ret.append(track)

    # 保留当前步step的结果, 作为下一步step的已有轨迹
    self.tracks = ret
    return ret
