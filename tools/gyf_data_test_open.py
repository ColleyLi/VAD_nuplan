
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

def render_gradual_line(ax, traj, cmap):
    '''
    traj: (fut_ts, 2)
    colors: 
    '''
    
    cmap = cmap
    y = np.sin(np.linspace(1/2*np.pi, 3/2*np.pi, 301))
    colors = color_map(y[:-1], cmap)

    traj[abs(traj) < 0.01] = 0.0
    traj = traj.cumsum(axis=0)
    traj = np.stack((traj[:-1], traj[1:]), axis=1)

    plan_vecs = None
    for i in range(traj.shape[0]):
        plan_vec_i = traj[i]
        x_linspace = np.linspace(plan_vec_i[0, 0], plan_vec_i[1, 0], 51)
        y_linspace = np.linspace(plan_vec_i[0, 1], plan_vec_i[1, 1], 51)
        xy = np.stack((x_linspace, y_linspace), axis=1)
        xy = np.stack((xy[:-1], xy[1:]), axis=1)
        if plan_vecs is None:
            plan_vecs = xy
        else:
            plan_vecs = np.concatenate((plan_vecs, xy), axis=0)
    
    line_segments = LineCollection(plan_vecs, colors=colors, linewidths=1, linestyles='solid', cmap=cmap)
    ax.add_collection(line_segments)

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
    
# CONFIG_PATH = "/data/zyp/Projects/VAD_open/navsim_gyf/navsim/planning/script/config/pdm_scoring"
# CONFIG_NAME = "default_run_pdm_score"
from pathlib import Path
# import hydra
# from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig, SceneFilter

# @hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main():
    args = parse_args()

    import time

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

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
    # # if args.resume_from is not None:
    # if args.resume_from is not None and osp.isfile(args.resume_from):
    #     cfg.resume_from = args.resume_from
    # if args.gpu_ids is not None:
    #     cfg.gpu_ids = args.gpu_ids
    # else:
    #     cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
    #     cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    # if args.autoscale_lr:
    #     # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
    #     cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    distributed = False
    # if args.launcher == 'none':
        # distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)
    #     # re-set gpu_ids with distributed training mode
    #     _, world_size = get_dist_info()
    #     cfg.gpu_ids = range(world_size)

    # create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    # if cfg.model.type in ['EncoderDecoder3D']:
    #     logger_name = 'mmseg'
    # else:
    #     logger_name = 'mmdet'

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
    # meta['env_info'] = env_info
    # meta['config'] = cfg.pretty_text

    # log some basic info
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    datasets = [build_dataset(cfg.data.train)]
    print("datasets_dir", cfg.data.train)

    # if len(cfg.workflow) == 2:
    #     val_dataset = copy.deepcopy(cfg.data.val)
    #     # in case we use a dataset wrapper
    #     if 'dataset' in cfg.data.train:
    #         val_dataset.pipeline = cfg.data.train.dataset.pipeline
    #     else:
    #         val_dataset.pipeline = cfg.data.train.pipeline
    #     # set test_mode=False here in deep copied config
    #     # which do not affect AP/AR calculation later
    #     # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
    #     val_dataset.test_mode = False
    #     datasets.append(build_dataset(val_dataset))

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

    def render_gt_data(data, frame_token): 

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
        plt.savefig(savepath + '/dataset_test_images/' + str(frame_token) + '_gt_labels_vis.png', bbox_inches='tight', dpi=200, color='mediumseagreen')
        plt.close()
        
    
    # try to find one index for one scene (like navsim)
    SPLIT = 'mini'
    scene_filter = SceneFilter()
    sensor_blobs_path = Path(os.getenv("OPENSCENE_DATA_ROOT")) / f"sensor_blobs/{SPLIT}"
    navsim_log_path = Path(os.getenv("OPENSCENE_DATA_ROOT")) / f"navsim_logs/{SPLIT}"
    metric_cache_path = Path(os.getenv("NAVSIM_EXP_ROOT")) / "metric_cache"
    scene_loader = SceneLoader(sensor_blobs_path=sensor_blobs_path,
                                data_path=navsim_log_path,
                                scene_filter=scene_filter,
                                sensor_config=SensorConfig.build_all_sensors(),
                            )
    # metric_cache_loader = MetricCacheLoader(metric_cache_path)
    
    tokens_to_evaluate = scene_loader.tokens
    print('{} frame tokens that needed to be evaluate'.format(len(tokens_to_evaluate)))
    
    tokens_index_list = []
    for index, info in enumerate(datasets[0].data_infos):
        if info['token'] in tokens_to_evaluate:
            tokens_index_list.append(index)
    
    # varify vad labels by making visualization 
    for i in tqdm(tokens_index_list):
        prepared_data = datasets[0].prepare_train_data(index=i)
        frame_token = datasets[0].data_infos[i]['token']
        render_gt_data(prepared_data, frame_token)

    # i = 0
    # prepared_data = datasets[0].prepare_train_data(index=i)
    # render_gt_data(prepared_data, i)

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
    # torch.multiprocessing.set_start_method('fork')
    main()


# python tools/gyf_data_test.py "projects/configs/VAD/VAD_base_e2e_nuscene.py" --launcher=none --deterministic --work-dir=openscen_train
# python tools/gyf_data_test.py "projects/configs/VAD/VAD_base_e2e_custom.py" --launcher=none --deterministic --work-dir=openscen_train