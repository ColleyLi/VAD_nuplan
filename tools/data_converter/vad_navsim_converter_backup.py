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


import pickle


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

ego_width, ego_length = 1.85, 4.084

def quart_to_rpy(qua):
    x, y, z, w = qua
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i


def create_navsim_infos(root_path,
                        sencor_path,
                         out_path,
                         info_prefix,
                         dataset_type='mini',
                         max_sweeps=10):
    
    """Create info file of opendrive dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        dataset_type (str): .
            Default: 'mini'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    
    
    ## Step1: 读取数据 根据给定的文档进行数据的划分
    
    # 事实上这里就需要使用 Scene 这些抽象的类 我在想怎么用
    

    available_vers = ['trainval', 'mini']
    assert dataset_type in available_vers
    
    # 读取处理好的 split 文件
    import yaml
    with open(f'/home/gyf/E2EDrivingScale/VAD/VAD-main/tools/data_converter/{dataset_type}_split.yaml', 'r') as f:
        splits = yaml.safe_load(f)

    if dataset_type == 'trainval':    
        train_logs = splits["train_logs"]
        val_logs = splits["val_logs"]
    elif dataset_type == 'mini':
        train_logs = splits["train_logs"]
        val_logs =  splits["val_logs"]
    else:
        raise ValueError('unknown')


    ## Step2: 过滤现有的场景
    
    
    # 根据我观察相应的 navsim 中 scenes 的划分方式 发现这里的 scene_filter 可能不适合在 vad 框架下使用
    # 上述结论的原因是
    # from navsim_dataloader import filter_scenes
    # from navsim_dataclasses import AgentInput, Scene, SceneFilter, SensorConfig
    
    # scene_frames_dicts = filter_scenes(root_path, scene_filter)
    # available_scenes = filter_scenes(root_path, scene_filter)
    
   
    # 在 NuScenes 里面有 train_scenes 和 val_scenes
    # train_scenes {
    # 'de7d80a1f5fb4c3e82ce8a4f213b450a', 
    # 'e233467e827140efa4b42d2b4c435855', 
    # 'd25718445d89453381c659b9c8734939', 
    # 'cc8c0bf57f984915a77078b10eb33198', 
    # '6f83169d067343658251f72e1dd17dbc', 
    # 'bebf5f5b2a674631ab5c88fd1aa9e87a', 
    # 'c5224b9b454b4ded9b5d2d2634bbda8a', 
    # '2fc3753772e241f2ab2cd16a784cc680'}
   
    print('train logs: {}, val logs: {}'.format(
        len(train_logs), len(val_logs)))
    
    
    ##  Step3: 得到处理后的数据 并将其用pkl文件格式进行保存

    # 得到 train 和 val 的数据
    # 在 nuscenes 中直接利用 scene_name 就可以得到相应的数据
    # 但是在 navsim 中需要利用 .pkl文件才能够得到很多数据
    train_openscene_infos = _fill_navsim_trainval_infos(
        root_path, sensor_path, train_logs, max_sweeps=max_sweeps)
    
    val_openscene_infos = _fill_navsim_trainval_infos(
        root_path, sensor_path, val_logs, max_sweeps=max_sweeps)

    
    print('train sample: {}, val sample: {}'.format(
        len(train_openscene_infos), len(val_openscene_infos)))
    
    # 保存 train 数据
    metadata = dict(dataset_type=dataset_type)
    data = dict(infos=train_openscene_infos, metadata=metadata)
    info_path = osp.join(out_path,
                            '{}_infos_temporal_train.pkl'.format(info_prefix))
    mmcv.dump(data, info_path)
    
    # 保存 val 数据
    data['infos'] = val_openscene_infos
    info_val_path = osp.join(out_path,
                                '{}_infos_temporal_val.pkl'.format(info_prefix))
    mmcv.dump(data, info_val_path)
    
    

# 最核心的改动数据
def _fill_navsim_trainval_infos(root_path,
                                sensor_path,
                                train_logs, 
                                # val_logs,  
                                test=False,
                                max_sweeps=10,
                                fut_ts = 8,
                                his_ts = 4):
    
    """Generate the train/val infos from the raw data.

    Args:
        
        train_logs (list[.pkl]): Raw Data from openscene_metadata .pkl type.
        train_logs (list[.pkl]): Raw Data from openscene_metadata .pkl type.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
   
    # 
    # train_navsim_infos = []
    # val_navsim_infos = []
    infos = []
    
    # TODO 这里的 cat2idx 是什么意思暂时不清楚
    cat2idx = {}
    # for idx, dic in enumerate(nusc.category):
    #     cat2idx[dic['name']] = idx

    
    train_logs_num = len(train_logs)
    
    # 仅仅挑选 300 个数据
    for train_log in train_logs[:3]:
        
        # 这里是要先得到相应的 .pkl 文件
        # train_log 就是一个相应的 .pkl文件
       
        train_log = train_log + ".pkl"
        train_log = f"{root_path}/{train_log}"
        with open(train_log, 'rb') as f:
            openscene_data = pickle.load(f)
            # scene_dict_list = pickle.load(f)
        
        tokens_num = len(openscene_data)
        # 事实上 token 可以和 navsim 中frame 对应上但是在这里我主要使用的还是 token
        
        for index in range(tokens_num):
            cur_token = openscene_data[index] # 这里的 cur_token 是一个字典包含众多信息
            map_location = cur_token['map_location']
            
            # 得到前一个时刻 token 的信息
            if cur_token['sample_prev'] != None:
                sample_prev = openscene_data[index-1]
                #  TODO 注意这里实际上不需要得到 pose_recode_prev 它仅仅起到一个占位作用
                pose_record_prev = sample_prev
            else:
                pose_record_prev = None
            
            # 得到后一个时刻 token 的信息
            if cur_token['sample_next'] != None:
                sample_next = openscene_data[index+1]
                #  TODO 注意这里实际上不需要得到 pose_recode_next 
                pose_record_next = sample_next
            else:
                pose_record_next = None
        
            # TODO 对于这里的 fut_ts 要进行确认 再就是这里的逻辑看起来是没问题 但是也需要进行确认
            # 根据我对于navsim数据的检查 发现使用 token['sample_next']会存在很大的问题
            future_valid_flag = True
            test_token = copy.deepcopy(cur_token)
            index_copy = index
            for i in range(fut_ts):
                # 这说明 test_token 不是整个 pkl 文件中最后一个token
                if index_copy+ 1 < tokens_num:
                    index_copy += 1
                    next_token = openscene_data[index_copy]
                    # 这说明当前 token 和 next_token 是同一个场景中的
                    if test_token['scene_token'] == next_token['scene_token']:
                        test_token = next_token
                    else:
                        future_valid_flag = False
                # 当前的 test_token 是整个 pkl 文件的最后一个token
                else:
                    if i != fut_ts - 1:
                        future_valid_flag = False
                    
            
            # 我们需要检查这里的 lidar_path 是否存在
            lidar_path = cur_token['lidar_path']
            lidar_path = f"{sensor_path}/{lidar_path}"
            # print("lidar_path", lidar_path)
            assert os.path.exists(lidar_path), "lidar_path does not exist"
            
            info = {
                "lidar_path": lidar_path,
                "token": cur_token['token'],
                
                "prev": cur_token['sample_prev'],
                "next": cur_token['sample_next'],
                'can_bus': cur_token['can_bus'],
                'frame_idx': cur_token['frame_idx'],
                'sweeps': [],
                'cams': dict(),  
                "scene_token": cur_token['scene_token'],
                "lidar2ego_translation": cur_token['lidar2ego_translation'],
                "lidar2ego_rotation": cur_token['lidar2ego_rotation'],
                "ego2global_translation": cur_token['ego2global_translation'],
                "ego2global_rotation": cur_token['ego2global_rotation'],
                "timestamp": cur_token['timestamp'],
                "fut_valid_flag": future_valid_flag,
                'map_location': map_location
            }
            # Quaternion  旋转 姿态操控等命令
            l2e_r = info['lidar2ego_rotation']
            l2e_t = info['lidar2ego_translation']
            e2g_r = info['ego2global_rotation']
            e2g_t = info['ego2global_translation']
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix
            
            # obtain 8 image's information per frame
            camera_types = [
                'CAM_B0',
                'CAM_F0',
                'CAM_L0',
                'CAM_L1',
                'CAM_L2',
                'CAM_R0',
                'CAM_R1',
                'CAM_R2',
            ]
            
            # TODO obtain_openscene_sensor2top函数是问题最多的 会影响 cam 和 lidart 的使用
            for cam in camera_types:
                cam_token = cur_token["cams"][cam]
                
                cam_path = cam_token['data_path']
                
                cam_info = obtain_sensor2top_cam(
                    cur_token,
                    cam_token,
                    l2e_t,
                    l2e_r_mat,
                    e2g_t,
                    e2g_r_mat,
                    sensor_path,
                    cam,
                )
                
                info["cams"].update({cam:cam_info})
            
            # obtain sweeps for a single key-frame
            sweeps = []
            sweep_index = index
            while len(sweeps) < max_sweeps:
                # not openscene_data[sweep_index]['sample_prev'] == None
                prev_index = sweep_index - 1
                if (prev_index) > 0 and openscene_data[prev_index]["scene_token"] == openscene_data[sweep_index]["scene_token"] :
                    
                    prev_token = openscene_data[prev_index]
                    sweep = obtain_sensor2top_lidar(
                        prev_token,
                        l2e_t,
                        l2e_r_mat,
                        e2g_t,
                        e2g_r_mat,
                        sensor_path,
                        'lidar'
                    )
                    sweeps.append(sweep)
                    sweep_index = prev_index
                    # print("len(sweeps)", len(sweeps))
                else:   
                    break
            info['sweeps'] = sweeps

            # obtain annotation
            if not test:
                annotations = cur_token['anns']
                
                
                # TODO 在这里 Navsim (Openscene) 是直接给出了相应的 gt_boxes
                gt_boxes = cur_token["anns"]['gt_boxes']
                velocity = cur_token["anns"]['gt_velocity_3d']
                # 正常来讲这里需要检查一下相应的 len(gt_boxes) 是否等于 len(velocity)我这里偷懒了
                
                # convert velo from global to lidar
                # 在这里我不知道是否应该执行下面的代码 因为在 navsim 那里这些都是不需要的
                # for i in range(len(gt_boxes)):
                #     velo = np.array([*velocity[i], 0.0])
                #     velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                #         l2e_r_mat).T
                #     velocity[i] = velo[:2]

                # get future coords for each box
                # [num_box, fut_ts*2]
                num_box = gt_boxes.shape[0]
                
                names = []
                
                NameMapping = {'movable_object.barrier': 'barrier', 
                               'vehicle.bicycle': 'bicycle', 
                               'vehicle.bus.bendy': 'bus', 
                               'vehicle.bus.rigid': 'bus', 
                               'vehicle.car': 'car', 
                               'vehicle.construction': 'construction_vehicle', 
                               'vehicle.motorcycle': 'motorcycle', 
                               'human.pedestrian.adult': 'pedestrian', 
                               'human.pedestrian.child': 'pedestrian', 
                               'human.pedestrian.construction_worker': 'pedestrian', 
                               'human.pedestrian.police_officer': 'pedestrian', 
                               'movable_object.trafficcone': 'traffic_cone',
                               'vehicle.trailer': 'trailer', 
                               'vehicle.truck': 'truck'}
                
                # 我在 navsim 中 print 得到的 names 如下
                NavimMapping = [" vehicle", "pedestrian", "generic_object", "traffic_cone",]
                
                for i in range(num_box):
                    names.append(annotations["gt_names"][i])
                    # 这里应该就是师兄提到的 需要去注意不同车辆名称的细节
                    if names[i] in NameMapping:
                        names[i] = NameMapping[names[i]]
                names = np.array(names)
                
                
                gt_fut_trajs = np.zeros((num_box, fut_ts, 2))
                gt_fut_yaw = np.zeros((num_box, fut_ts))
                gt_fut_masks = np.zeros((num_box, fut_ts))
                gt_boxes_yaw = -(gt_boxes[:,6] + np.pi / 2)
                # agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
                agent_lcf_feat = np.zeros((num_box, 9))
                gt_fut_goal = np.zeros((num_box))
                
               
                
                for i in range(num_box):
                    cur_box = gt_boxes[i]
                    # cur_anno = anno
                    agent_lcf_feat[i, 0:2] = cur_box[:2]	
                    agent_lcf_feat[i, 2] = gt_boxes_yaw[i]
                    agent_lcf_feat[i, 3:5] = velocity[i][:2]
                    
                    # TODO 暂时没有在 metadata 里面找到对应的 size 和 category_name
                    # 暂时找到了一些但是不完全 肯定需要再检验
                    agent_lcf_feat[i, 5:8] = cur_box[3:6] # width,length,height
                    agent_lcf_feat[i, 8] = annotations["gt_names"][i] if annotations["gt_names"][i] in cat2idx.keys() else -1
                    
                # get ego history traj (offset format)
                ego_his_trajs = np.zeros((his_ts+1, 3))
                ego_his_trajs_diff = np.zeros((his_ts+1, 3))
                # sample_cur = openscene_data[index]
                current_index = index 
                for i in range(his_ts, -1, -1):
                    if cur_token is not None:
                        pose_mat = get_global_sensor_pose_openscene(cur_token, inverse=False)
                        ego_his_trajs[i] = pose_mat[:3, 3]
                        # TODO 这里有 bug 本来以为直接调用就可以了但是这里的 None 仅仅在pkl最后出现
                        has_prev = cur_token['sample_prev'] != None
                        has_next = cur_token['sample_next'] != None
                        if has_next:
                            next_index = current_index + 1
                            next_token = openscene_data[next_index]
                            pose_mat_next = get_global_sensor_pose_openscene(next_token, inverse=False)
                            ego_his_trajs_diff[i] = pose_mat_next[:3, 3] - ego_his_trajs[i]
                        if has_prev:
                            current_index -= 1
                            cur_token = openscene_data[current_index]
                        else:
                            None
                    else:
                        ego_his_trajs[i] = ego_his_trajs[i+1] - ego_his_trajs_diff[i+1]
                        ego_his_trajs_diff[i] = ego_his_trajs_diff[i+1]
          
                # 下面这部分的代码可以帮助我们理解相应的 坐标转换
                # 将历史轨迹从全局坐标系转换到传感器的局部坐标系中，并计算出每个时间步的位移向量。
                # global to ego at lcf
                ego_his_trajs = ego_his_trajs - np.array(cur_token['ego2global_translation'])
                rot_mat = Quaternion(cur_token['ego2global_rotation']).inverse.rotation_matrix
                ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
                # ego to lidar at lcf
                ego_his_trajs = ego_his_trajs - np.array(cur_token['lidar2ego_translation'])
                rot_mat = Quaternion(cur_token['lidar2ego_rotation']).inverse.rotation_matrix
                ego_his_trajs = np.dot(rot_mat, ego_his_trajs.T).T
                ego_his_trajs = ego_his_trajs[1:] - ego_his_trajs[:-1]
                
                
                 # get ego futute traj (offset format)
                ego_fut_trajs = np.zeros((fut_ts+1, 3))
                ego_fut_masks = np.zeros((fut_ts+1))
                # sample_cur = openscene_data[index]
                current_index= index
                for i in range(fut_ts+1):
                    pose_mat = get_global_sensor_pose_openscene(cur_token, inverse=False)
                    ego_fut_trajs[i] = pose_mat[:3, 3]
                    ego_fut_masks[i] = 1
                    # if cur_token['sample_next'] == None:
                    if current_index+1 < tokens_num and openscene_data[current_index+ 1]["frame_idx"] == 0:
                        ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                        break
                    elif current_index + 1 == tokens_num:
                        ego_fut_trajs[i+1:] = ego_fut_trajs[i]
                        break
                    else:
                        current_index += 1
                        cur_token = openscene_data[current_index]
                
                
                 # global to ego at lcf
                ego_fut_trajs = ego_fut_trajs - np.array(cur_token['ego2global_translation'])
                rot_mat = Quaternion(cur_token['ego2global_rotation']).inverse.rotation_matrix
                ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
                # ego to lidar at lcf
                ego_fut_trajs = ego_fut_trajs - np.array(cur_token['lidar2ego_translation'])
                rot_mat = Quaternion(cur_token['lidar2ego_rotation']).inverse.rotation_matrix
                ego_fut_trajs = np.dot(rot_mat, ego_fut_trajs.T).T
                
                
                # TODO 这里是得到未来的命令 因此这样是ok 的
                # drive command according to final fut step offset from lcf
                if ego_fut_trajs[-1][0] >= 2:
                    command = np.array([1, 0, 0])  # Turn Right
                elif ego_fut_trajs[-1][0] <= -2:
                    command = np.array([0, 1, 0])  # Turn Left
                else:
                    command = np.array([0, 0, 1])  # Go Straight
                # offset from lcf -> per-step offset
                ego_fut_trajs = ego_fut_trajs[1:] - ego_fut_trajs[:-1]
                
                
                ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
                ego_lcf_feat = np.zeros(9)
                # 根据odom推算自车速度及加速度
                _, _, ego_yaw = quart_to_rpy(cur_token['ego2global_rotation'])
                ego_pos = np.array(cur_token['ego2global_translation'])
                
                # 前一帧的 ego 信息
                if pose_record_prev is not None:
                    _, _, ego_yaw_prev = quart_to_rpy(pose_record_prev['ego2global_rotation'])
                    ego_pos_prev = np.array(pose_record_prev['ego2global_translation'])
                
                # 下一帧的 ego 信息
                if pose_record_next is not None:
                    _, _, ego_yaw_next = quart_to_rpy(pose_record_next['ego2global_rotation'])
                    ego_pos_next = np.array(pose_record_next['ego2global_translation'])
                assert (pose_record_prev is not None) or (pose_record_next is not None), 'prev token and next token all empty'
                
                # 利用当前帧和前一帧的信息计算速度 这里时间间隔是 0.5
                if pose_record_prev is not None:
                    ego_w = (ego_yaw - ego_yaw_prev) / 0.5
                    ego_v = np.linalg.norm(ego_pos[:2] - ego_pos_prev[:2]) / 0.5
                    ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)
                else:
                    ego_w = (ego_yaw_next - ego_yaw) / 0.5
                    ego_v = np.linalg.norm(ego_pos_next[:2] - ego_pos[:2]) / 0.5
                    ego_vx, ego_vy = ego_v * math.cos(ego_yaw + np.pi/2), ego_v * math.sin(ego_yaw + np.pi/2)

                # can_bus  scene_token
                can_bus = cur_token['can_bus']
                ref_scene = cur_token['scene_token']
                # TODO 这里没怎么看懂 需要后续再操作
                # try:
                #     pose_msgs = nusc_can_bus.get_messages(ref_scene['name'],'pose')
                #     steer_msgs = nusc_can_bus.get_messages(ref_scene['name'], 'steeranglefeedback')
                #     pose_uts = [msg['utime'] for msg in pose_msgs]
                #     steer_uts = [msg['utime'] for msg in steer_msgs]
                #     ref_utime = sample['timestamp']
                #     pose_index = locate_message(pose_uts, ref_utime)
                #     pose_data = pose_msgs[pose_index]
                #     steer_index = locate_message(steer_uts, ref_utime)
                #     steer_data = steer_msgs[steer_index]
                #     # initial speed
                #     v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s
                #     # curvature (positive: turn left)
                #     steering = steer_data["value"]
                #     # flip x axis if in left-hand traffic (singapore)
                #     flip_flag = True if map_location.startswith('singapore') else False
                #     if flip_flag:
                #         steering *= -1
                #     Kappa = 2 * steering / 2.588
                # except:
                #     delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
                #     delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
                #     v0 = np.sqrt(delta_x**2 + delta_y**2)
                #     Kappa = 0
                
                delta_x = ego_his_trajs[-1, 0] + ego_fut_trajs[0, 0]
                delta_y = ego_his_trajs[-1, 1] + ego_fut_trajs[0, 1]
                v0 = np.sqrt(delta_x**2 + delta_y**2)
                Kappa = 0
                
                ### ego lcf feat (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度
                # 在 navsim 中直接提供了相应的 vx vy ax ay
                ego_dynamic_state = cur_token["ego_dynamic_state"]
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32)
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32)
                
                ego_lcf_feat[:2] = np.array([ego_vx, ego_vy]) #can_bus[13:15]
                ego_lcf_feat[2:4] = can_bus[7:9]
                ego_lcf_feat[4] = ego_w #can_bus[12]
                ego_lcf_feat[5:7] = np.array([ego_length, ego_width])
                ego_lcf_feat[7] = v0
                ego_lcf_feat[8] = Kappa  
                
                # TODO 
                fake_num_lidar_pts = []
                fake_num_radar_pts = []
                for i in  range(num_box):
                    fake_num_lidar_pts.append(i)
                    fake_num_radar_pts.append(i)
                valid_flag = np.array(
                [(fake_num_lidar_pts[i] + fake_num_lidar_pts[i]) > 0
                 for i in range(num_box)],
                dtype=bool).reshape(-1)
                
                info['gt_boxes'] = gt_boxes
                info["gt_names"] = names
                info["gt_velocity"] = velocity[:, :2]
                
                # 下面这俩我在 nuplan 中没有找到对应的数据啊 TODO
                info['num_lidar_pts'] = np.array(
                    [fake_num_lidar_pts[i] for i in range(num_box)])
                info['num_radar_pts'] = np.array(
                    [ fake_num_radar_pts[i] for i in range(num_box)])
                
                info['valid_flag'] = valid_flag
                info['gt_agent_fut_trajs'] = gt_fut_trajs.reshape(-1, fut_ts*2).astype(np.float32)
                info['gt_agent_fut_masks'] = gt_fut_masks.reshape(-1, fut_ts).astype(np.float32)
                info['gt_agent_lcf_feat'] = agent_lcf_feat.astype(np.float32)
                info['gt_agent_fut_yaw'] = gt_fut_yaw.astype(np.float32)
                info['gt_agent_fut_goal'] = gt_fut_goal.astype(np.float32)
                info['gt_ego_his_trajs'] = ego_his_trajs[:, :2].astype(np.float32)
                info['gt_ego_fut_trajs'] = ego_fut_trajs[:, :2].astype(np.float32)
                info['gt_ego_fut_masks'] = ego_fut_masks[1:].astype(np.float32)
                info['gt_ego_fut_cmd'] = command.astype(np.float32)
                info['gt_ego_lcf_feat'] = ego_lcf_feat.astype(np.float32)
         
    
            infos.append(info)      
        
            # if cur_token['log_name'] in train_logs:
            #         train_navsim_infos.append(info)
            # else:
            #     val_navsim_infos.append(info)

    return infos
    

def get_global_sensor_pose_openscene(cur_token, inverse=False):


    if inverse is False:
        global_from_ego = transform_matrix(cur_token['ego2global_translation'], Quaternion(cur_token['ego2global_rotation']), inverse=False)
        ego_from_sensor = transform_matrix(cur_token['lidar2ego_translation'], Quaternion(cur_token['lidar2ego_rotation']), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(cur_token['ego2global_translation'], Quaternion(cur_token['ego2global_rotation']), inverse=True)
        ego_from_global = transform_matrix(cur_token['lidar2ego_translation'], Quaternion(cur_token['lidar2ego_rotation']), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose


  
def obtain_sensor2top_cam(cur_token, 
                        cam_token,
                        l2e_t, 
                        l2e_r_mat, 
                        e2g_t, 
                        e2g_r_mat, 
                        sensor_blobs_path,
                        sensor_type):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    
    # 这边刚好卡住了  我们能够从 cam_token 中直接拿到相应的 sensor2ego_translation 和 sensor2ego_rotation
    # 但是在这里nuscene 里面是默认能够拿到 sensor2ego_translation 和 sensor2ego_rotation 存在矛盾
    sweep = {
        "data_path": f"{sensor_blobs_path}/{cam_token['data_path']}",
        "type": sensor_type,
        "sample_data_token": cur_token["token"],
        
        # TODO 已经完成
        # 'sensor2ego_translation': [0, 0, 0],
        # 'sensor2ego_rotation': [ 0, 0, 0, 0],
        
        'sensor2lidar_translation': cam_token['sensor2lidar_translation'],
        'sensor2lidar_rotation': cam_token['sensor2lidar_rotation'],
        
        'ego2global_translation': cur_token['ego2global_translation'],
        'ego2global_rotation': cur_token['ego2global_rotation'],
        'timestamp': cur_token['timestamp']
        
    }
    
    sweep['cam_intrinsic'] = cam_token['cam_intrinsic']
    
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    
    ##  我需要在这里进行视角转移
    # 得到 sensor2lidar_translation
    fx_l2e_t_s_1 = sweep['sensor2lidar_translation'] + e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    
    fx_l2e_t_s = (fx_l2e_t_s_1 @ l2e_r_mat.T @ e2g_r_mat.T - e2g_t_s) @ np.linalg.inv(e2g_r_s_mat).T
    fx_l2e_t_s = fx_l2e_t_s.tolist()
    sweep["sensor2ego_translation"] = fx_l2e_t_s
    
    # 得到 sensor2ego_rotation
    # print("cam_sweep['sensor2lidar_rotation']", sweep['sensor2lidar_rotation'])
    # print("sweep['sensor2lidar_rotation']", np.array(sweep['sensor2lidar_rotation']).shape)
    fx_l2e_r_s_mat = np.linalg.inv(e2g_r_s_mat) @ e2g_r_mat @  l2e_r_mat @ sweep['sensor2lidar_rotation']
    
    # 从 matrix 转化至 四元数[a + bi + cj + dk] 得到其中 [a, b, c, d]
    # 同时我发现这里是否进行 四元数的 正负转化都ok 
    fx_l2e_r_s = Quaternion(matrix=fx_l2e_r_s_mat)
    fx_l2e_r_s = fx_l2e_r_s.elements.tolist()
    # print("fx_l2e_r_s", fx_l2e_r_s)
    sweep["sensor2ego_rotation"] = fx_l2e_r_s
    
    return sweep
    

  
def obtain_sensor2top_lidar(
                        prev_token,
                        l2e_t, 
                        l2e_r_mat, 
                        e2g_t, 
                        e2g_r_mat, 
                        sensor_blobs_path,
                        sensor_type):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    
    # 这边刚好卡住了  我们能够从 cam_token 中直接拿到相应的 sensor2ego_translation 和 sensor2ego_rotation
    # 但是在这里nuscene 里面是默认能够拿到 sensor2ego_translation 和 sensor2ego_rotation 存在矛盾
    sweep = {
        "data_path": f"{sensor_blobs_path}/{prev_token['lidar_path']}",
        "type": sensor_type,
        "sample_data_token": prev_token["token"],
        
        # TODO 这里好像因为是 lidar 传感器导致这里还有写 bug 
        'sensor2ego_translation': [0, 0, 0],
        'sensor2ego_rotation': [ 0, 0, 0, 0],
        
        'sensor2lidar_translation': prev_token['lidar2ego_translation'],
        'sensor2lidar_rotation': prev_token['lidar2ego_rotation'],
        
        'ego2global_translation': prev_token['ego2global_translation'],
        'ego2global_rotation': prev_token['ego2global_rotation'],
        'timestamp': prev_token['timestamp']
        
    }
    
    # sweep['cam_intrinsic'] = cam_token['cam_intrinsic']
    
  
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

    
    ##  我需要在这里进行视角转移
    # 得到 sensor2lidar_translation
    fx_l2e_t_s_1 = sweep['sensor2lidar_translation'] + e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    
    fx_l2e_t_s = (fx_l2e_t_s_1 @ l2e_r_mat.T @ e2g_r_mat.T - e2g_t_s) @ np.linalg.inv(e2g_r_s_mat).T
    fx_l2e_t_s = fx_l2e_t_s.tolist()
    sweep["sensor2ego_translation"] = fx_l2e_t_s
    
    # 得到 sensor2ego_rotation
    
    # 这里甚至都不需要把这个 四元列表变成 矩阵 因为这是一个单位阵
    fx_l2e_r_s_mat = np.linalg.inv(e2g_r_s_mat) @ e2g_r_mat @  l2e_r_mat 
    # print("fx_l2e_r_s_mat_shape", fx_l2e_r_s_mat.shape) # 3,3
    
    sweep['sensor2lidar_rotation'] = Quaternion(sweep['sensor2lidar_rotation']).rotation_matrix
    # print("lidar_sweep['sensor2lidar_rotation']", sweep['sensor2lidar_rotation'])
    # print("sweep['sensor2lidar_rotation']", np.array(sweep['sensor2lidar_rotation']).shape)
    # fx_l2e_r_s_mat = np.linalg.inv(e2g_r_s_mat) @ e2g_r_mat @  l2e_r_mat @ sweep['sensor2lidar_rotation']

    
    # 从 matrix 转化至 四元数[a + bi + cj + dk] 得到其中 [a, b, c, d]
    # 同时我发现这里是否进行 四元数的 正负转化都ok 
    fx_l2e_r_s = Quaternion(matrix=fx_l2e_r_s_mat)
    fx_l2e_r_s = fx_l2e_r_s.elements.tolist()
    sweep["sensor2ego_rotation"] = fx_l2e_r_s
    print("sweep[sensor2ego_rotation]", sweep["sensor2ego_rotation"])
    print("sweep[sensor2ego_translation]", sweep["sensor2ego_translation"])
    
    # print("sweep", sweep)
    
    return sweep


def get_global_sensor_pose(rec, nusc, inverse=False):
    lidar_sample_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])

    sd_ep = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    sd_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    if inverse is False:
        global_from_ego = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False)
        ego_from_sensor = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False)
        pose = global_from_ego.dot(ego_from_sensor)
        # translation equivalent writing
        # pose_translation = np.array(sd_cs["translation"])
        # rot_mat = Quaternion(sd_ep['rotation']).rotation_matrix
        # pose_translation = np.dot(rot_mat, pose_translation)
        # # pose_translation = pose[:3, 3]
        # pose_translation = pose_translation + np.array(sd_ep["translation"])
    else:
        sensor_from_ego = transform_matrix(sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True)
        ego_from_global = transform_matrix(sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True)
        pose = sensor_from_ego.dot(ego_from_global)
    return pose

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep



def navsim_data_prep(root_path,
                     sensor_path,
                     info_prefix,
                     dataset_type,
                     out_dir,
                     max_sweeps=10):
    """Prepare data related to NAVSIM Opendrive dataset. """
    
    create_navsim_infos(
        root_path, sensor_path, out_dir, info_prefix, dataset_type=dataset_type, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
# parser.add_argument('dataset', help='name of the dataset')
parser.add_argument(
    '--root_path',
    type=str,
    default='./data/openscene-v1.1/navsim_logs',
    help='specify the root path of dataset')

parser.add_argument(
    '--sensor_path',
    type=str,
    default='./data/openscene-v1.1/sensor_blobs',
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
    default='./data/openscene-v1.1/navsim_vad_converter_pkl',
    # required='False',
    help='name of info pkl')

parser.add_argument('--extra-tag', type=str, default='navsim_dataset')

parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if  args.dataset_type == 'trainval':
        
        navsim_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            dataset_type=args.dataset_type,
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
      
    elif args.dataset_type == 'mini':
       
        root_path = f"{args.root_path}/{args.dataset_type}"
        sensor_path = f"{args.sensor_path}/{args.dataset_type}"
        info_prefix =  f"{args.extra_tag}-{args.dataset_type}"
        navsim_data_prep(
            root_path=root_path,
            sensor_path = sensor_path,
            info_prefix=info_prefix,
            dataset_type=args.dataset_type,
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)


# python tools/data_converter/vad_navsim_converter.py --dataset_type mini