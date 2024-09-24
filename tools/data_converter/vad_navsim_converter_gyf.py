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
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import SE2Index
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array, normalize_angle
)
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

def convert_relative_to_absolute_se2_array(
    origin: StateSE2, state_se2_array: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Converts an StateSE2 array from global to relative coordinates.
    :param origin: origin pose of relative coords system
    :param state_se2_array: array of SE2 states with (x,y,θ) in last dim
    :return: SE2 coords array in relative coordinates
    """
    assert len(SE2Index) == state_se2_array.shape[-1]

    theta = -origin.heading
    origin_array = np.array([[origin.x, origin.y, origin.heading]], dtype=np.float64)

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    points_global = np.zeros_like(state_se2_array, dtype=np.float64)
    # rotate first
    points_global[..., :2] = state_se2_array[..., :2] @ R
    points_global[..., 2] = state_se2_array[..., 2]
    
    # translate next
    points_global = points_global + origin_array
    points_global[:, 2] = normalize_angle(points_global[:, 2])

    return points_global

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
    
    metric_cache_path = Path(os.getenv("NAVSIM_EXP_ROOT"))
    metric_cache_loader = MetricCacheLoader(metric_cache_path / "metric_cache")
    
    train_openscene_infos = fill_navsim_trainval_infos(scene_loader, metric_cache_loader, logs_path, sensor_path, train_logs, max_sweeps=max_sweeps)
    val_openscene_infos = fill_navsim_trainval_infos(scene_loader, metric_cache_loader, logs_path, sensor_path, val_logs, max_sweeps=max_sweeps)

    print('train sample: {}, val sample: {}'.format(
        len(train_openscene_infos), len(val_openscene_infos)))
    
    # save train pkl
    metadata = dict(dataset_type = dataset_type)
    data = dict(infos = train_openscene_infos, metadata = metadata)
    
    info_path = osp.join(out_dir,
                            '{}_infos_temporal_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    
    # save val pkl
    data['infos'] = val_openscene_infos
    info_val_path = osp.join(out_dir,
                                '{}_infos_temporal_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)

def fill_navsim_trainval_infos(scene_loader,
                               metric_cache_loader,
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
    # we only use tokens that record in the log_token_dict, which results 1 token per scene
    frame_tokens = []
    for log, tokens in log_token_dict.items():
        if logs_name:
            frame_tokens += tokens if log in logs_name else []
        else:
            frame_tokens += tokens
    
    # get infos from tokens to form the pkl file
    infos = []
    for token in tqdm(frame_tokens):
        # ready the scene and metric_cache
        scene = scene_loader.get_scene_from_token(token)
        # we only need metric_cache to get start steering
        metric_cache_path = metric_cache_loader.metric_cache_paths[token]
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        
        # get start frame data
        start_frame_idx = scene.scene_metadata.num_history_frames - 1
        start_frame_data = scene.frames[start_frame_idx] # data in this frame token
        start_ego_status = start_frame_data.ego_status # pose(3=x,y,yaw), velocity(2), acceleration(2), driving_command(4). all in global frame
        start_annotations = start_frame_data.annotations
        
        # fill info with parameters:
        raw_data = scene_loader.scene_frames_dicts[token][start_frame_idx]
        lidar_path = f"{sensor_path}/{raw_data['lidar_path']}"
        assert os.path.exists(lidar_path), "lidar_path does not exist"
        
        can_bus = raw_data['can_bus']
        lidar2ego_translation = raw_data['lidar2ego_translation']
        lidar2ego_rotation = raw_data['lidar2ego_rotation']
        ego2global_translation = raw_data['ego2global_translation']
        ego2global_rotation = raw_data['ego2global_rotation']
        sample_prev = raw_data['sample_prev'] # frame token in prev 1 frame
        sample_next = raw_data['sample_next'] # frame token in next 1 frame
        fut_valid_flag = True if fut_ts <= scene.scene_metadata.num_future_frames else False
        
        info = {
                'lidar_path': lidar_path,
                'token': token, # it is FRAME token
                'prev': sample_prev,
                'next': sample_next,
                'can_bus': can_bus,
                'frame_idx': start_frame_idx,
                'sweeps': [],
                'cams': {},
                'scene_token': scene.scene_metadata.scene_token,
                'lidar2ego_translation': lidar2ego_translation,
                'lidar2ego_rotation': lidar2ego_rotation,
                'ego2global_translation': ego2global_translation,
                'ego2global_rotation': ego2global_rotation,
                'timestamp': start_frame_data.timestamp,
                'fut_valid_flag': fut_valid_flag,
                'map_location': scene.scene_metadata.map_name,
                }
        
        # file info with labels and inputs: 
        
        # get other object's info, NOTE: all info from annotations are in ego frame
        start_gt_boxes = start_annotations.boxes # (num, 7), (x, y, z, x_size, y_size, z_size, yaw)
        gt_names = start_annotations.names # (num,)
        gt_velocity = start_annotations.velocity_3d[:, :2] # (num, 2), (vx, vy)
            
        # get other object's future traj (offset format)
        start_track_tokens = start_annotations.track_tokens # (num,)
        
        num_object = start_gt_boxes.shape[0]
        gt_fut_trajs = np.zeros((num_object, fut_ts, 2))
        gt_fut_masks = np.zeros((num_object, fut_ts))
        gt_fut_yaw = np.zeros((num_object, fut_ts))
        gt_boxes_yaw = -(start_gt_boxes[:,6] + np.pi / 2)
        gt_fut_goal = np.zeros((num_object))
        # TODO: lcf means Local Coordinate Frame?
        # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        agent_lcf_feat = np.zeros((num_object, 9))
        
        start_ego_pose = np.array([start_ego_status.ego_pose[0], start_ego_status.ego_pose[1], start_ego_status.ego_pose[2]])
        for i in range(num_object):
            # get object lcf feature
            agent_lcf_feat[i, 0:2] = start_gt_boxes[i][:2] # x, y
            agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
            agent_lcf_feat[i, 3:5] = gt_velocity[i]
            agent_lcf_feat[i, 5:8] = start_gt_boxes[i][3:6] # width,length,height
            agent_lcf_feat[i, 8] = -1 # TODO: this is a very specific type, like {'human.pedestrian.adult': 0}
            
            # get object future traj
            track_token = start_track_tokens[i]
            # get global position of this object
            object_position_in_ego_frame = np.append(start_gt_boxes[i][:2], np.array(start_gt_boxes[i][6]))
            object_position_in_global_frame = convert_relative_to_absolute_se2_array(StateSE2(*start_ego_pose), 
                                                                                       object_position_in_ego_frame)
            # we need to get the future trajectory of this object, in GlOBAL frame
            object_future_traj_in_global_frame = object_position_in_global_frame
            for t in range(1, fut_ts + 1):
                frame_idx = start_frame_idx + t
                frame_data_t = scene.frames[frame_idx]
                # temperal infos
                annotations_t = frame_data_t.annotations
                track_tokens_t = annotations_t.track_tokens
                if track_token in track_tokens_t:
                    ego_status_t = frame_data_t.ego_status
                    ego_pose_t = np.array([ego_status_t.ego_pose[0], ego_status_t.ego_pose[1], ego_status_t.ego_pose[2]])
                    
                    index_t = track_tokens_t.index(track_token)
                    gt_boxes_t = annotations_t.boxes
                    
                    # get global position of this object
                    object_position_in_ego_frame = np.append(gt_boxes_t[index_t][:2], np.array(gt_boxes_t[index_t][6]))
                    object_position_in_global_frame = convert_relative_to_absolute_se2_array(StateSE2(*ego_pose_t), 
                                                                                            object_position_in_ego_frame)
                    object_future_traj_in_global_frame = np.concatenate([object_future_traj_in_global_frame, 
                                                                         object_position_in_global_frame], axis=0)
                    
                    # updata mask
                    gt_fut_masks[i, t-1] = 1
                else:
                    break
                    
            # convert to start ego frame
            object_future_traj_in_ego_frame = convert_absolute_to_relative_se2_array(StateSE2(*start_ego_pose),
                                                                                     np.array(object_future_traj_in_global_frame))
            # offset format
            delta_object_future_traj_in_ego_frame = np.array(object_future_traj_in_ego_frame[1:] - object_future_traj_in_ego_frame[:-1])
            valid_points_num = delta_object_future_traj_in_ego_frame.shape[0]
            gt_fut_trajs[i][:valid_points_num] = delta_object_future_traj_in_ego_frame[:, :2]
            gt_fut_yaw[i][:valid_points_num] = delta_object_future_traj_in_ego_frame[:, 2]
            
            assert gt_fut_masks[i].sum() == valid_points_num
            
            # get object goal
            gt_fut_coords = np.cumsum(gt_fut_trajs[i], axis=-2)
            coord_diff = gt_fut_coords[-1] - gt_fut_coords[0]
            if coord_diff.max() < 1.0: # static
                gt_fut_goal[i] = 9
            else:
                box_mot_yaw = np.arctan2(coord_diff[1], coord_diff[0]) + np.pi
                gt_fut_goal[i] = box_mot_yaw // (np.pi / 4)  # 0-8: goal direction class
                     
        # get ego history traj (offset format)
        delta_ego_gt_history_traj = np.zeros((his_ts, 2))
        # TODO: get_history_trajectory is WRONG when his_ts > scene.scene_metadata.num_history_frames, currently scene.scene_metadata.num_history_frames == 4 (including current frame)
        ego_gt_history_traj = scene.get_history_trajectory(scene.scene_metadata.num_history_frames).poses[:, :2] 
        delta = np.array(ego_gt_history_traj[1:] - ego_gt_history_traj[:-1])
        delta_ego_gt_history_traj[-delta.shape[0]:] = delta
        if delta.shape[0] < his_ts: # NOTE: assume the padded frames have the same motion trend as the first valid frame
            pad_num = his_ts - delta.shape[0]
            delta_ego_gt_history_traj[:pad_num] = delta[0]
        
        # get ego futute traj (offset format)
        ego_fut_masks = np.ones((fut_ts + 1)) # TODO: all valid for now
        ego_gt_future_traj = scene.get_future_trajectory(fut_ts).poses[:, :2]
        ego_gt_future_traj = np.append(np.zeros((1, 2)), ego_gt_future_traj, axis=0)
        delta_ego_gt_future_traj = np.array(ego_gt_future_traj[1:] - ego_gt_future_traj[:-1])
        
        # get ego future command
        driving_command = start_ego_status.driving_command
        
        # get ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw speed
        ego_yaw = start_ego_status.ego_pose[2]
        ego_yaw_prev = scene.frames[start_frame_idx-1].ego_status.ego_pose[2] # TODO: currently all start frame has valid prev frame
        
        # TODO: check if the steering angle is correct, and, is there a way not to use metric_cache?
        # curvature (positive: turn left). 
        initial_ego_state = metric_cache.ego_state
        steering = initial_ego_state.tire_steering_angle
        # flip x axis if in left-hand traffic (singapore)
        flip_flag = True if scene.scene_metadata.map_name.startswith('sg') else False
        if flip_flag:
            steering *= -1
        Kappa = 2 * steering / 2.588
        
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[:2] = start_ego_status.ego_velocity
        ego_lcf_feat[2:4] = start_ego_status.ego_acceleration
        ego_lcf_feat[4] = (ego_yaw - ego_yaw_prev) / 0.5 # 0.5 is the dt
        ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
        ego_lcf_feat[7] = start_ego_status.ego_velocity[0] # TODO: longitudinal velocity, is it vx? m/s
        ego_lcf_feat[8] = Kappa
        
        # cameras input
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
                        'timestamp': start_frame_data.timestamp, # TODO: nuplan does not has unique token for its sensors data
                        'cam_intrinsic': camera_info['cam_intrinsic'],
                        }
            info['cams'].update({camera_type: want_info})
        
        
        # TODO: lidar input, seems no use in original VAD
        # obtain sweeps for a single key-frame
        lidar_path = raw_data['lidar_path'] 
        lidar_data = start_frame_data.lidar # (6, n)
        sweeps = []
        while len(sweeps) < max_sweeps:
            break
        # TODO: why always use prev?
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
                # break
        
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
        
        info['gt_boxes'] = start_gt_boxes
        info['gt_names'] = gt_names
        info['gt_velocity'] = gt_velocity.reshape(-1, 2)
        info['valid_flag'] = valid_flag
        info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
        info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
        info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
        info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
        info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
        info['gt_ego_his_trajs'] = delta_ego_gt_history_traj[:, :2].astype(np.float32)
        info['gt_ego_fut_trajs'] = delta_ego_gt_future_traj[:, :2].astype(np.float32)
        info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
        info['gt_ego_fut_cmd'] = driving_command.astype(np.float32)
        info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)
        
        # additionally add
        info['log_name'] = scene.scene_metadata.log_name
    
        infos.append(info)
    
    return infos


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

parser.add_argument('--extra-tag', type=str, default='navsim_dataset')

parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')

args = parser.parse_args()

if __name__ == '__main__':

    logs_path = f"{args.logs_path}/{args.dataset_type}"
    sensor_path = f"{args.sensor_path}/{args.dataset_type}"
    info_prefix =  f"{args.extra_tag}-{args.dataset_type}"
    
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