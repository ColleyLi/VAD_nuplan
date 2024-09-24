
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
import math

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version

import tqdm
from tqdm import *

import cv2
cv2.setNumThreads(1)

import sys
sys.path.append('')

from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from projects.mmdet3d_plugin.core.bbox.structures.openscene_box import CustomNuscenesBox, CustomDetectionBox, color_map
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
import nuplan
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.geometry.transform import translate_longitudinally
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.car_footprint import CarFootprint

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def bezier_normal(Ps, n, t):
    """普通方式实现贝塞尔曲线

    Args:
        Ps (_type_): 控制点，格式为numpy数组：array([[x1,y1],[x2,y2],...,[xn,yn]])
        n (_type_): n个控制点，即Ps的第一维度
        t (_type_): 时刻t

    Returns:
        _type_: 当前t时刻的贝塞尔点
    """
    if n==1:
        return Ps[0]
    p_t = np.array([0,0])
    n = len(Ps)-1
    for i in range(n + 1):
        C_n_i = math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
        p_t =p_t+C_n_i*(1-t)**(n-i)*t**i*Ps[i]
    return p_t

def calc_angle(pt1, pt2): 
    angle=0
    # dy = pt2[1]-pt1[1]
    # dx = pt2[0]-pt1[0]
    dx = pt2[1]-pt1[1]
    dy = pt2[0]-pt1[0]
    if dx==0 and dy>0:
        angle = 0
    if dx==0 and dy<0:
        angle = 180
    if dy==0 and dx>0:
        angle = 90
    if dy==0 and dx<0:
        angle = 270
    if dx>0 and dy>0:
       angle = math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy>0:
       angle = 360 + math.atan(dx/dy)*180/math.pi
    elif dx<0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    elif dx>0 and dy<0:
       angle = 180 + math.atan(dx/dy)*180/math.pi
    return angle

def render_traj(ax, center, traj, color):

    traj[abs(traj) < 0.01] = 0.0
    traj = traj.cumsum(axis=0)
    traj = traj + np.array([center])

    traj = np.concatenate((np.array([center]), traj), axis=0)

    ax.plot(
        traj[:, 1],
        traj[:, 0],
        color=color,
        alpha=0.5,
        linewidth=1,
        linestyle="-",
        marker='o',
        markersize=2,
        markeredgecolor='black',
        # zorder=config["zorder"],
    )

def main():
    args = parse_args()

    import time
    s0 = time.time()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    s1 = time.time()
    print('time for loading config:', s1-s0)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            # from projects.mmdet3d_plugin.VAD.apis.train import custom_train_model
        s2 = time.time()
        print('time for loading plugin:', s2-s1)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    datasets = [build_dataset(cfg.data.train)]
    print("datasets_dir", cfg.data.train)

    # custom checking dataset
    ## 1. check data preparing
    # index = 3
    # classes = datasets[0].CLASSES
    # print("dataset[0]", datasets[0])
    # print("prepared_data", prepared_data.keys())
    # print("prepared_data_gt_attr_lables", prepared_data["gt_attr_labels"])
    
    ## 2. check wheather the data is right, by visualization
    # dict_keys(['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 
    #           'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 
    #           'ego_lcf_feat', 'gt_attr_labels', 'map_gt_labels_3d', 'map_gt_bboxes_3d'])

    def render_gt_data(data, frame_index): 

        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        plt.xlim(xmin=-30, xmax=30)
        plt.ylim(ymin=-30, ymax=30)
        
        # 1. visualize gt map
        colors_plt = ['cornflowerblue', 'royalblue', 'slategrey']
        map_vectors = data['map_gt_bboxes_3d'].data.fixed_num_sampled_points # torch.Tensor([N, fixed_num(20), 2])
        map_vectors_types = data['map_gt_labels_3d'].data
        for index, line in enumerate(map_vectors):
            gt_label_3d = map_vectors_types[index]
            pts_x = np.array([pt[0] for pt in line])
            pts_y = np.array([pt[1] for pt in line])

            # NOTE: y first, x second, which makes x and y reversed
            axes.plot(pts_y, pts_x, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
            axes.scatter(pts_y, pts_x, color=colors_plt[gt_label_3d], s=1, alpha=0.8, zorder=-1)
            # red denotes direction
            axes.plot(pts_y[:5], pts_x[:5], color='red', linewidth=0.5, alpha=0.5, zorder=-1)
            axes.scatter(pts_y[:5], pts_x[:5], color='red', s=0.5, alpha=0.5, zorder=-1)
        
        # 2. visualize other vehicles
        gt_bboxes_3d = data['gt_bboxes_3d'].data
        gt_bboxes_3d_types = data['gt_labels_3d'].data
        gt_fut_trajs = data['gt_attr_labels'].data[:, :16].reshape(-1, 8, 2)

        for i in range(gt_bboxes_3d.gravity_center.shape[0]):
            # box params
            location = gt_bboxes_3d.gravity_center[i]
            heading = gt_bboxes_3d.yaw[i]
            box_length = gt_bboxes_3d.dims[i][0]
            box_width = gt_bboxes_3d.dims[i][1]
            box_height = gt_bboxes_3d.dims[i][2]
            # plot box
            box = OrientedBox(StateSE2(location[0], location[1], heading), box_length, box_width, box_height)
            box_corners = box.all_corners()
            corners = [[corner.x, corner.y] for corner in box_corners]
            corners = np.asarray(corners + [corners[0]])
            axes.plot(
                    corners[:, 1], # NOTE: y first, x second, which makes x and y reversed
                    corners[:, 0],
                    color='tomato',
                    alpha=0.8,
                    linewidth=1,
                    )
            # plot direction
            direction = translate_longitudinally(box.center, distance=box.length / 2 + 0.)
            line = np.array([[box.center.x, box.center.y], [direction.x, direction.y]])
            axes.plot(
                line[:, 1],
                line[:, 0],
                color='tomato',
                alpha=1,
                linewidth=1,
            )
            # plot gt future
            render_traj(axes, (box.center.x, box.center.y), gt_fut_trajs[i], color='tomato')
 
        # 3. visualize ego gt planning
        car_footprint = CarFootprint.build_from_rear_axle(
            rear_axle_pose=StateSE2(0, 0, 0),
            vehicle_parameters=get_pacifica_parameters(),
        )
        # car_footprint = CarFootprint.build_from_center(
        #     center=StateSE2(0, 0, 0),
        #     vehicle_parameters=get_pacifica_parameters(),
        # )
        # plot ego box
        box = car_footprint.oriented_box 
        box_corners = box.all_corners()
        corners = [[corner.x, corner.y] for corner in box_corners]
        corners = np.asarray(corners + [corners[0]])
        axes.plot(
                corners[:, 1], # NOTE: y first, x second, which makes x and y reversed
                corners[:, 0],
                color='mediumseagreen',
                alpha=0.8,
                linewidth=1,
                )
        # plot direction
        direction = translate_longitudinally(box.center, distance=box.length / 2 + 0.)
        line = np.array([[box.center.x, box.center.y], [direction.x, direction.y]])
        axes.plot(
            line[:, 1],
            line[:, 0],
            color='mediumseagreen',
            alpha=1,
            linewidth=1,
        )
        # plot future
        ego_gt_fut_trajs = data['ego_fut_trajs'].data[0]
                
        control_points = np.concatenate((np.array([(0., 0.)]), ego_gt_fut_trajs), axis=0)
        control_points[abs(control_points) < 0.01] = 0.0
        control_points = control_points.cumsum(axis=0)
        bezier_ego_gt_fut_trajs = []
        for t in np.arange(0, 1, 0.05):
            pt = bezier_normal(control_points, control_points.shape[0], t)
            bezier_ego_gt_fut_trajs.append(pt)
        
        bezier_ego_gt_fut_trajs = np.array(bezier_ego_gt_fut_trajs)
        bezier_ego_gt_fut_trajs[abs(bezier_ego_gt_fut_trajs) < 0.01] = 0.0
        
        bezier_ego_gt_fut_trajs_with_heading = np.zeros([bezier_ego_gt_fut_trajs.shape[0], 3])
        bezier_ego_gt_fut_trajs_with_heading[0] = np.append(bezier_ego_gt_fut_trajs[0], 0.)
        
        for i in range(bezier_ego_gt_fut_trajs.shape[0] - 1):
            angle = calc_angle(bezier_ego_gt_fut_trajs[i], bezier_ego_gt_fut_trajs[i+1])
            bezier_ego_gt_fut_trajs_with_heading[i] = np.append(bezier_ego_gt_fut_trajs[i+1], angle)
        bezier_ego_gt_fut_trajs_with_heading[-1][2] = bezier_ego_gt_fut_trajs_with_heading[-2][2]
        render_traj(axes, (0,0), ego_gt_fut_trajs, color='mediumseagreen') # cmap='winter'

        # 4. 
        # plt.gcf().canvas.draw()
        # plt.gcf().canvas.renderer.rotate(90)
        # NOTE: left is y positive, right is y negative
        axes.invert_xaxis()
        axes.set_xticks([])
        axes.set_yticks([])

        # 5. save gt figure
        savepath = '/data/zyp/Projects/VAD_open/VAD-main/tools'
        plt.savefig(savepath + '/test_images/' + str(frame_index) + '_gt_map.png', bbox_inches='tight', dpi=200, color='mediumseagreen')
        plt.close()

    # for i in tqdm(range(8000)):
    #     prepared_data = datasets[0].prepare_train_data(index=i)
    #     render_gt_data(prepared_data, i)

    i = 0
    prepared_data = datasets[0].prepare_train_data(index=i)
    render_gt_data(prepared_data, i)

    ## 3. check evaluation after val/test
    import pandas as pd
    # bbox_results = pd.read_pickle('/home/gyf/E2EDrivingScale/VAD/VAD-main/test/results.pkl')
    # datasets[0].evaluate(bbox_results)
    

    # add an attribute for visualization convenience
    # model.CLASSES = datasets[0].CLASSES # TODO: seems no change compared with nuScenes, is it ok?

    print('testing done')

    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)

    # custom_train_model(
    #     model,
    #     datasets,
    #     cfg,
    #     distributed=distributed,
    #     validate=(not args.no_validate),
    #     timestamp=timestamp,
    #     meta=meta)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()


# python tools/gyf_data_test.py "projects/configs/VAD/VAD_base_e2e_nuscene.py" --launcher=none --deterministic --work-dir=openscen_train
# python tools/gyf_data_test.py "projects/configs/VAD/VAD_base_e2e_custom.py" --launcher=none --deterministic --work-dir=openscen_train