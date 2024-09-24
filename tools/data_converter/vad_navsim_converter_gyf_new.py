import os
import math
import copy
import argparse
from os import path as osp
from collections import OrderedDict
from typing import List, Tuple, Union

import mmcv
import numpy as np
from pyquaternion import Quaternion
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import Box
# from shapely.geometry import MultiPoint, box
# from mmdet3d.datasets import NuScenesDataset
# from nuscenes.utils.geometry_utils import view_points
# from mmdet3d.core.bbox.box_np_ops import points_cam2img
from nuscenes.utils.geometry_utils import transform_matrix

import hydra
from hydra.utils import instantiate
from pathlib import Path
import os
import pickle
import lzma
from tqdm import *
import numpy.typing as npt
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from nuplan.common.actor_state.state_representation import StateSE2
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
    SE2Index,
    StateIndex,
)
from navsim.planning.metric_caching.metric_cache import MetricCache

ego_width, ego_length = 1.1485 * 2.0, 4.049 + 1.127

# nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
#                   'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
#                   'barrier')

# nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
#                   'pedestrian.moving', 'pedestrian.standing',
#                   'pedestrian.sitting_lying_down', 'vehicle.moving',
#                   'vehicle.parked', 'vehicle.stopped', 'None')

# def quart_to_rpy(qua):
#     x, y, z, w = qua
#     roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
#     pitch = math.asin(2 * (w * y - x * z))
#     yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
#     return roll, pitch, yaw

# def locate_message(utimes, utime):
#     i = np.searchsorted(utimes, utime)
#     if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
#         i -= 1
#     return i

# def cameras_to_dict(cameras):
#     camera_dict = {}
    
#     camera_dict["cam_f0"] = cameras.cam_f0
#     camera_dict["cam_l0"] = cameras.cam_l0
#     camera_dict["cam_l1"] = cameras.cam_l1
#     camera_dict["cam_l2"] = cameras.cam_l2
#     camera_dict["cam_r0"] = cameras.cam_r0
#     camera_dict["cam_r1"] = cameras.cam_r1
#     camera_dict["cam_r2"] = cameras.cam_r2
#     camera_dict["cam_b0"] = cameras.cam_b0
    
#     return camera_dict


def get_ego_status(raw_data):
    '''get ego pose in global frame'''
    ego_translation = raw_data["ego2global_translation"]
    ego_quaternion = Quaternion(*raw_data["ego2global_rotation"])
    global_ego_pose = np.array(
        [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
        dtype=np.float64,
    )
    ego_dynamic_state = raw_data["ego_dynamic_state"]
    return {
        "ego_pose": global_ego_pose,
        "ego_velocity": np.array(ego_dynamic_state[:2], dtype=np.float32),
        "ego_acceleration": np.array(ego_dynamic_state[2:], dtype=np.float32),
        "driving_command": raw_data["driving_command"],
        "in_global_frame": True,
    }


def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def convert_relative_to_absolute_array(
    origin: npt.NDArray[np.float64], pose_local: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an array from relative to global coordinates.
    :param origin: origin pose of relative coords system
    :param pose_local: array of states with (x,y,θ) in last dim
    :return: coords array in relative coordinates
    """

    theta = -origin[2]
    origin_array = np.expand_dims(origin, axis=0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    points_global = np.zeros_like(pose_local, dtype=np.float64)
    
    # rotate first
    points_global[..., :2] = pose_local[..., :2] @ R
    points_global[..., 2] = pose_local[..., 2]
    # translate next
    points_global = points_global + origin_array
    points_global[:, 2] = normalize_angle(points_global[:, 2])

    return points_global

def convert_absolute_to_relative_array(
    origin: npt.NDArray[np.float64], pose_global: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param pose_global: array of states with (x,y,θ) in last dim
    :return: coords array in relative coordinates
    """

    theta = -origin[2]
    origin_array = np.expand_dims(origin, axis=0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = pose_global - origin_array
    points_rel[..., :2] = points_rel[..., :2] @ R.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])

    return points_rel

def navsim_data_prep(logs_path,
                    sensor_path,
                    info_prefix,
                    dataset_type,
                    out_dir,
                    max_sweeps=10):
        
    available_vers = ['trainval', 'mini']
    assert dataset_type in available_vers
    
    import yaml
    with open(f'/data/zyp/Projects/VAD_open/VAD-main/tools/data_converter/{dataset_type}_split.yaml', 'r') as f:
        splits = yaml.safe_load(f)

    if dataset_type == 'trainval':    
        train_logs = splits["train_logs"]
        val_logs = splits["val_logs"]
    elif dataset_type == 'mini':
        train_logs = splits["train_logs"]
        val_logs =  splits["val_logs"]
   
    print('{} set split: train logs {}, val logs {}'.format(
        dataset_type, len(train_logs), len(val_logs)))
    
    # initializae scene loader
    hydra.initialize(config_path="../navsim/planning/script/config/common/scene_filter")
    scene_filter: SceneFilter = instantiate(hydra.compose(config_name="all_scenes"))
    openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

    scene_loader = SceneLoader(
        openscene_data_root / f"navsim_logs/{dataset_type}",
        openscene_data_root / f"sensor_blobs/{dataset_type}",
        scene_filter,
        sensor_config=SensorConfig.build_all_sensors(),
    )
    
    # metric_cache_path = Path(os.getenv("NAVSIM_EXP_ROOT"))
    # metric_cache_loader = MetricCacheLoader(metric_cache_path / "metric_cache")
    
    # save train pkl
    train_openscene_infos, train_key_tokens = fill_navsim_trainval_infos(scene_loader, logs_path, sensor_path, max_sweeps=max_sweeps)
    metadata = dict(dataset_type = dataset_type, key_tokens = train_key_tokens)
    data = dict(infos = train_openscene_infos, metadata = metadata)
    info_path = osp.join(out_dir, 'gyf_{}_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    
    # save val pkl
    val_openscene_infos, val_key_tokens = fill_navsim_trainval_infos(scene_loader, logs_path, sensor_path, val_logs, max_sweeps=max_sweeps)
    metadata = dict(dataset_type = dataset_type, key_tokens = val_key_tokens)
    data = dict(infos = val_openscene_infos, metadata = metadata)
    info_val_path = osp.join(out_dir, 'gyf_{}_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)
    
    print('train sample: {}, val sample: {}'.format(
        len(train_openscene_infos), len(val_openscene_infos)))
    
    

def fill_navsim_trainval_infos(scene_loader,
                                logs_path,
                                sensor_path,
                                logs_name=None, 
                                test=False,
                                max_sweeps=10,
                                fut_ts = 8,
                                his_ts = 4):
    # class_names = [
    #             'vehicle', 'place_holder1', 'place_holder2', 'place_holder3', 
    #             'czone_sign', 'bicycle', 'generic_object', 'pedestrian', 'traffic_cone', 'barrier'
    #             ]
    
    # cat2idx = {}
    # for idx, dic in enumerate(nusc.category):
    #     cat2idx[dic['name']] = idx
    
    # print("cat2idx", cat2idx)
    
    # get a dict of {log_name: [token1, token2, ...]}
    log_token_dict = scene_loader.get_tokens_list_per_log()
    # we only use tokens that record in the log_token_dict, which results 1 token per (small) scene
    key_tokens = []
    for log, tokens in log_token_dict.items():
        if logs_name:
            key_tokens += tokens if log in logs_name else []
        else:
            key_tokens += tokens
    
    # extract all log files data
    raw_data_dict = {} # {token: token_raw_data}
    for train_log in tqdm(log_token_dict.keys(), desc="Loading logs"):
        train_log = train_log + ".pkl"
        train_log = f"{logs_path}/{train_log}"
        with open(train_log, 'rb') as f:
            raw_data_list = pickle.load(f) # a list of raw data in this log
        for raw_data in raw_data_list:
            frame_token = raw_data['token']
            raw_data_dict[frame_token] = raw_data
    
    # find BIG scenes
    ## find the first frame of each BIG scene
    first_tokens = []
    for raw_data in raw_data_dict.values():
        if raw_data['sample_prev'] not in raw_data_dict.keys():
            first_tokens.append(raw_data['token'])
    
    ## form the BIG scene
    big_scene_dict = {}
    big_scene_index = 0
    big_scene_tokens = []
    for i in range(len(first_tokens)):
        # add first token of the BIG scene
        token = first_tokens[i]
        big_scene_tokens.append(token)
        # add following tokens
        while raw_data_dict[token]['sample_next'] in raw_data_dict.keys():
            token = raw_data_dict[token]['sample_next']
            big_scene_tokens.append(token)
        # add to big_scene_dict
        big_scene_dict['big_scene_{}'.format(big_scene_index)] = big_scene_tokens
        big_scene_index += 1
        big_scene_tokens = []
    
    # get infos from tokens to form the pkl file
    infos = []
    # for big_scene, big_scene_tokens_list in tqdm(big_scene_dict.items()):
    for big_scene, big_scene_tokens_list in mmcv.track_iter_progress(big_scene_dict.items()):
        # NOTE: to only convert a part of the data
        # big_scene_tokens_list = big_scene_tokens_list[-10:]
        for big_scene_frame_idx, big_scene_token in enumerate(tqdm(big_scene_tokens_list, desc='Processing big scene: {}'.format(big_scene))): 
            
            raw_data = raw_data_dict[big_scene_token]
            lidar_path = f"{sensor_path}/{raw_data['lidar_path']}"
            assert os.path.exists(lidar_path), "lidar_path does not exist"
            
            can_bus = raw_data['can_bus']
            lidar2ego_translation = raw_data['lidar2ego_translation']
            lidar2ego_rotation = raw_data['lidar2ego_rotation']
            ego2global_translation = raw_data['ego2global_translation']
            ego2global_rotation = raw_data['ego2global_rotation']
            sample_prev = raw_data['sample_prev'] # frame token in prev 1 frame
            sample_next = raw_data['sample_next'] # frame token in next 1 frame
            fut_valid_flag = True if (len(big_scene_tokens_list) - big_scene_frame_idx - 1) >= fut_ts else False
            
            # fill info with parameters:
            info = {
                    'lidar_path': lidar_path,
                    'token': big_scene_token, # it is FRAME token
                    'prev': sample_prev,
                    'next': sample_next,
                    'can_bus': can_bus,
                    'frame_idx': raw_data['frame_idx'],
                    'big_scene_frame_idx': big_scene_frame_idx, # NOTE: custom key
                    'sweeps': [],
                    'cams': {},
                    'scene_token': raw_data['scene_token'],
                    'big_scene_token': big_scene, # NOTE: custom key
                    'lidar2ego_translation': lidar2ego_translation,
                    'lidar2ego_rotation': lidar2ego_rotation,
                    'ego2global_translation': ego2global_translation,
                    'ego2global_rotation': ego2global_rotation,
                    'timestamp': raw_data['timestamp'],
                    'fut_valid_flag': fut_valid_flag,
                    'map_location': raw_data['map_location'],
                    }

            # file info with labels and inputs: 
            annotations = raw_data['anns']
            
            # get ego status
            ego_status = get_ego_status(raw_data)
            ego_pose = ego_status['ego_pose']
            
            # get other object's info, NOTE: all info from annotations are in ego frame
            gt_boxes = annotations['gt_boxes'] # (num, 7), (x, y, z, x_size, y_size, z_size, yaw)
            gt_names = annotations['gt_names'] # (num,)
            gt_velocity = annotations['gt_velocity_3d'][:, :2] # (num, 2), (vx, vy)
            object_tokens = annotations['track_tokens'] # list, len = num, objects' unique token
            num_object = gt_boxes.shape[0]
            
            ## other object's future traj (offset format)
            gt_fut_trajs = np.zeros((num_object, fut_ts, 2))
            gt_fut_masks = np.zeros((num_object, fut_ts))
            gt_fut_yaw = np.zeros((num_object, fut_ts))
            # gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2)
            gt_boxes_yaw = -gt_boxes[:,6] # TODO: remove np.pi / 2; why use nagative? -- it should be the same as gt_boxes[:, 6]
            gt_fut_goal = np.zeros((num_object))
            
            ## other object's lcf feat (x, y, yaw, vx, vy, width, length, height, type), TODO: lcf means Local Coordinate Frame?
            agent_lcf_feat = np.zeros((num_object, 9))
            for i in range(num_object):
                ## get object lcf feature
                agent_lcf_feat[i, 0:2] = gt_boxes[i][:2] # x, y
                agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
                agent_lcf_feat[i, 3:5] = gt_velocity[i] # in ego frame
                agent_lcf_feat[i, 5:8] = gt_boxes[i][3:6] # width, length, height
                agent_lcf_feat[i, 8] = -1 # TODO: this is a very specific type, like {'human.pedestrian.adult': 0}
                
                ## get object future traj
                object_token = object_tokens[i]
                ## get current GlOBAL position of this object
                object_position_in_ego_frame = np.append(gt_boxes[i][:2], np.array(gt_boxes[i][6]))
                object_position_in_global_frame = convert_relative_to_absolute_array(ego_pose, 
                                                                                    object_position_in_ego_frame)
                ## get the GlOBAL future trajectory of this object
                object_fut_traj_in_global_frame = object_position_in_global_frame
                for t in range(1, fut_ts + 1):
                    frame_idx_t = big_scene_frame_idx + t # current frame index in big scene
                    ### get frame data at index t 
                    if frame_idx_t > len(big_scene_tokens_list) - 1:
                        break
                    raw_data_t = raw_data_dict[big_scene_tokens_list[frame_idx_t]]
                    
                    ### temperal infos, with _t
                    annotations_t = raw_data_t['anns']
                    object_tokens_t = annotations_t['track_tokens']
                    if object_token in object_tokens_t: # this object is not exist in this frame
                        
                        ego_pose_t = get_ego_status(raw_data_t)['ego_pose'] # global frame

                        object_index_t = object_tokens_t.index(object_token)
                        gt_boxes_t = annotations_t['gt_boxes']
                        
                        ### get global position of this object
                        object_position_in_ego_frame = np.append(gt_boxes_t[object_index_t][:2], np.array(gt_boxes_t[object_index_t][6]))
                        object_position_in_global_frame = convert_relative_to_absolute_array(ego_pose_t, 
                                                                                            object_position_in_ego_frame)
                        object_fut_traj_in_global_frame = np.concatenate([object_fut_traj_in_global_frame, 
                                                                            object_position_in_global_frame], axis=0)
                        
                        ### updata mask
                        gt_fut_masks[i, t-1] = 1
                    else:
                        break
                        
                ## convert to ego frame
                object_fut_traj_in_ego_frame = convert_absolute_to_relative_array(ego_pose,
                                                                                    object_fut_traj_in_global_frame)
                ## convert to offset format
                delta_object_fut_traj_in_ego_frame = np.array(object_fut_traj_in_ego_frame[1:] - object_fut_traj_in_ego_frame[:-1])
                valid_points_num = delta_object_fut_traj_in_ego_frame.shape[0]
                gt_fut_trajs[i][:valid_points_num] = delta_object_fut_traj_in_ego_frame[:, :2]
                gt_fut_yaw[i][:valid_points_num] = delta_object_fut_traj_in_ego_frame[:, 2]
                
                assert gt_fut_masks[i].sum() == valid_points_num
                
                ## get object goal
                gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
                coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
                if coord_diff.max() < 1.0: # static
                    gt_fut_goal[i] = 9
                else:
                    box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                    gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class
            

            # get ego infos:
            # get ego history traj (offset format)
            ego_gt_his_trajs_in_global_frame = np.zeros((his_ts+1, 3))
            ego_gt_his_trajs_diff = np.zeros((his_ts+1, 3))
            ## get ego history GLOBAL traj
            frame_idx_t = big_scene_frame_idx # this is the current frame idx (in big scene)
            for t in range(his_ts, -1, -1):
                if frame_idx_t >= 0:
                    raw_data_t = raw_data_dict[big_scene_tokens_list[frame_idx_t]]
                    ### get ego GLOBAl position in t
                    ego_gt_his_trajs_in_global_frame[t] = get_ego_status(raw_data_t)['ego_pose']
                    ### get ego GLOBAl position diff in t
                    if frame_idx_t < len(big_scene_tokens_list) - 2:
                        raw_data_t_next = raw_data_dict[big_scene_tokens_list[frame_idx_t + 1]]
                        ego_pose_t_next = get_ego_status(raw_data_t_next)['ego_pose']
                        ego_gt_his_trajs_diff[t] = ego_pose_t_next - ego_gt_his_trajs_in_global_frame[t]
                    ### go to prev frame
                    frame_idx_t -= 1
                else:
                    ego_gt_his_trajs_in_global_frame[t] = ego_gt_his_trajs_in_global_frame[t+1] - ego_gt_his_trajs_diff[t+1]
                    ego_gt_his_trajs_diff[t] = ego_gt_his_trajs_diff[t+1]
            
            ## convert to ego frame
            ego_gt_his_trajs_in_ego_frame = convert_absolute_to_relative_array(ego_pose,
                                                                               ego_gt_his_trajs_in_global_frame)
            ## convert to offset format
            delta_ego_gt_his_trajs_in_ego_frame = np.array(ego_gt_his_trajs_in_ego_frame[1:] - ego_gt_his_trajs_in_ego_frame[:-1])
            
            # get ego future traj (offset format)
            ego_gt_fut_trajs_in_global_frame = np.zeros((fut_ts + 1, 3))
            ego_gt_fut_masks = np.zeros((fut_ts + 1))
            ## get ego future GLOBAL traj
            frame_idx_t = big_scene_frame_idx # this is the current frame idx (in big scene)
            raw_data_t = raw_data_dict[big_scene_tokens_list[frame_idx_t]]
            for t in range(fut_ts + 1):
                ### get ego GLOBAl position in t
                ego_status_t = get_ego_status(raw_data_t)
                ego_gt_fut_trajs_in_global_frame[t] = ego_status_t['ego_pose']
                ego_gt_fut_masks[t] = 1
                
                ### go to next frame
                frame_idx_t += 1
                if frame_idx_t < len(big_scene_tokens_list) - 1:
                    raw_data_t = raw_data_dict[big_scene_tokens_list[frame_idx_t]]
                else:
                    ego_gt_fut_trajs_in_global_frame[t+1:] = ego_gt_fut_trajs_in_global_frame[t]
                    break
            ## convert to ego frame
            ego_gt_fut_trajs_in_ego_frame = convert_absolute_to_relative_array(ego_pose,
                                                                                ego_gt_fut_trajs_in_global_frame)
            ## convert to offset format
            delta_ego_gt_fut_trajs_in_ego_frame = np.array(ego_gt_fut_trajs_in_ego_frame[1:] - ego_gt_fut_trajs_in_ego_frame[:-1])
            
            # get ego future command
            driving_command = ego_status['driving_command']
            
            # get ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw speed
            # TODO: all in global frame? what is the use of pi/2? we remove it
            # seems like vx and vy are velocity in Lidar frame? according to the code in nuscenes_converter
            ego_yaw = ego_pose[2]
            if big_scene_frame_idx - 1 >= 0:
                ego_pose_prev = get_ego_status(raw_data_dict[big_scene_tokens_list[big_scene_frame_idx - 1]])['ego_pose']
                ego_yaw_prev = ego_pose_prev[2]
                ego_w = (ego_yaw - ego_yaw_prev) / 0.5 # 0.5 is the dt
                ego_v = np.linalg.norm(ego_pose[:2] - ego_pose_prev[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw), ego_v * math.sin(ego_yaw)
            else:
                ego_pose_next = get_ego_status(raw_data_dict[big_scene_tokens_list[big_scene_frame_idx + 1]])['ego_pose']
                ego_yaw_next = ego_pose_next[2]
                ego_w = (ego_yaw_next - ego_yaw) / 0.5
                ego_v = np.linalg.norm(ego_pose_next[:2] - ego_pose[:2]) / 0.5
                ego_vx, ego_vy = ego_v * math.cos(ego_yaw), ego_v * math.sin(ego_yaw)
            
            # TODO: how to get steering/Kappa? in can_bus?                
            Kappa = 0
            
            ego_lcf_feat = np.zeros(9)
            ego_lcf_feat[:2] = np.array([ego_vx, ego_vy])
            ego_lcf_feat[2:4] = ego_status['ego_acceleration']
            ego_lcf_feat[4] = ego_w
            ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
            ego_lcf_feat[7] = ego_status['ego_velocity'][0] # TODO: longitudinal velocity, is it vx? m/s
            ego_lcf_feat[8] = Kappa
        
        
            # get cameras input
            camera_data = raw_data['cams']
            
            e2g_t = ego2global_translation
            e2g_r = ego2global_rotation
            l2e_t = lidar2ego_translation
            l2e_r = lidar2ego_rotation
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            
            # obtain 8 image's information per frame
            for camera_type, camera_info in camera_data.items():
                # get sensor2ego_translation and sensor2ego_rotation, which are omitted in nuplan
                fx_l2e_t_s_1 = camera_info['sensor2lidar_translation'] + e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                fx_l2e_t_s = (fx_l2e_t_s_1 @ l2e_r_mat.T @ e2g_r_mat.T - e2g_t) @ np.linalg.inv(e2g_r_mat).T
                fx_l2e_t_s = fx_l2e_t_s.tolist()
                sensor2ego_translation = fx_l2e_t_s
                
                fx_l2e_r_s_mat = np.linalg.inv(e2g_r_mat) @ e2g_r_mat @ l2e_r_mat @ camera_info['sensor2lidar_rotation']
                fx_l2e_r_s = Quaternion(matrix=fx_l2e_r_s_mat)
                fx_l2e_r_s = fx_l2e_r_s.elements.tolist()
                sensor2ego_rotation = fx_l2e_r_s
                
                assert sum([sensor2ego_rotation[i] - Quaternion(matrix=camera_info['sensor2lidar_rotation'])[i] for i in range(len(sensor2ego_rotation))]) < 1e-6
                assert (np.array(sensor2ego_translation) - camera_info['sensor2lidar_translation']).sum() < 1e-6
                
                # NOTE: seems like sensor2ego_translation = camera_info['sensor2lidar_translation'], 
                # and sensor2ego_rotation = camera_info['sensor2lidar_rotation']
                want_info = {
                            'data_path': f"{sensor_path}/{camera_info['data_path']}",
                            'type': camera_type,
                            'sample_data_token': token, # TODO: nuplan does not has unique token for its sensors data
                            'sensor2ego_translation': sensor2ego_translation,
                            'sensor2ego_rotation': sensor2ego_rotation,
                            'ego2global_translation': ego2global_translation,
                            'ego2global_rotation': ego2global_rotation,
                            'sensor2lidar_translation': camera_info['sensor2lidar_translation'],
                            'sensor2lidar_rotation': camera_info['sensor2lidar_rotation'],
                            'timestamp': raw_data['timestamp'], # TODO: nuplan does not has unique token for its sensors data
                            'cam_intrinsic': camera_info['cam_intrinsic'],
                            }
                info['cams'].update({camera_type: want_info})
                        
            # TODO: lidar input, seems no use in original VAD
            # obtain sweeps for a single key-frame
            lidar_path = raw_data['lidar_path'] 
            sweeps = []
            while len(sweeps) < max_sweeps:
                # if not sd_rec['prev'] == '':
                #     sweep = {
                #             'data_path': f"{sensor_path}/{lidar_path}",
                #             'type': 'lidar',
                #             'sample_data_token': token, # TODO: nuplan does not has unique token for its sensors data
                #             'sensor2ego_translation': ,
                #             'sensor2ego_rotation': ,
                #             'ego2global_translation': ego2global_translation,
                #             'ego2global_rotation': ego2global_rotation,
                #             'sensor2lidar_translation': ,
                #             'sensor2lidar_rotation': ,
                #             'timestamp': start_frame_data.timestamp, # TODO: nuplan does not has unique token for its sensors data
                #             }
                #     sweeps.append(sweep)
                #     sd_rec = nusc.get('sample_data', sd_rec['prev'])
                # else:
                #     break
                break
            
            info['sweeps'] = sweeps
            
            # TODO: nuplan does not has anno['num_lidar_pts'] and anno['num_radar_pts'] data
            # num_lidar_pts = lidar_data.lidar_pc.shape[1]
            fake_num_lidar_pts = []
            fake_num_radar_pts = []
            for i in range(num_object):
                fake_num_lidar_pts.append(i)
                fake_num_radar_pts.append(i)
            valid_flag = np.array([(fake_num_lidar_pts[i] + fake_num_lidar_pts[i]) > 0
                                    for i in range(num_object)],
                                    dtype=bool).reshape(-1)
                    
            # TODO: no cooresponding data in nuplan
            info['num_lidar_pts'] = np.array([fake_num_lidar_pts[i] for i in range(num_object)])
            info['num_radar_pts'] = np.array([fake_num_lidar_pts[i] for i in range(num_object)]) 
            
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names
            info['gt_velocity'] = gt_velocity.reshape(-1, 2)
            info['valid_flag'] = valid_flag
            info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
            info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
            info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
            info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
            info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
            info['gt_ego_his_trajs'] = delta_ego_gt_his_trajs_in_ego_frame[:, :2].astype(np.float32)
            info['gt_ego_fut_trajs'] = delta_ego_gt_fut_trajs_in_ego_frame[:, :2].astype(np.float32)
            info['gt_ego_fut_masks'] = ego_gt_fut_masks[1:].astype(np.float32)
            info['gt_ego_fut_cmd'] = driving_command.astype(np.float32)
            info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)
            
            # additionally add
            info['log_name'] = raw_data['log_name']
            
            infos.append(info)
    
    return infos, key_tokens


parser = argparse.ArgumentParser(description='Data converter arg parser')

parser.add_argument(
    '--logs_path',
    type=str,
    default='/data/zyp/Projects/VAD_open/VAD-main/data/openscene-v1.1/navsim_logs',
    help='specify the root path of dataset')

parser.add_argument(
    '--sensor_path',
    type=str,
    default='/data/zyp/Projects/VAD_open/VAD-main/data/openscene-v1.1/sensor_blobs',
    help='specify the root path of dataset')

parser.add_argument(
    '--dataset_type',
    type=str,
    default='mini',
    required=False,
    help='specify the dataset version, e.g. mini, trainval')

parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')

parser.add_argument(
    '--out_dir',
    type=str,
    default='/data/zyp/Projects/VAD_open/VAD-main/tools/data_converter',
    required=False,
    help='name of info pkl')

parser.add_argument('--extra-tag', type=str, default='navsim')

parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')

args = parser.parse_args()

if __name__ == '__main__':

    logs_path = f"{args.logs_path}/{args.dataset_type}"
    sensor_path = f"{args.sensor_path}/{args.dataset_type}"
    info_prefix =  f"{args.extra_tag}_{args.dataset_type}"
    
    if args.dataset_type == 'trainval':
        
        navsim_data_prep(
            logs_path = logs_path,
            sensor_path = sensor_path,
            info_prefix = info_prefix,
            dataset_type = args.dataset_type,
            out_dir = args.out_dir,
            max_sweeps = args.max_sweeps)

    elif args.dataset_type == 'mini':
        
        navsim_data_prep(
            logs_path = logs_path,
            sensor_path = sensor_path,
            info_prefix = info_prefix,
            dataset_type = args.dataset_type,
            out_dir = args.out_dir,
            max_sweeps = args.max_sweeps)

# python tools/data_converter/vad_navsim_converter.py --dataset_type mini