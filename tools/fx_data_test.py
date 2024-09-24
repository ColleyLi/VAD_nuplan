
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

import cv2
cv2.setNumThreads(1)

import sys
sys.path.append('')

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
    # meta['seed'] = args.seed
    # meta['exp_name'] = osp.basename(args.config)

    # model = build_model(
    #     cfg.model,
    #     train_cfg=cfg.get('train_cfg'),
    #     test_cfg=cfg.get('test_cfg'))
    # model.init_weights()
    # logger.info(f'Model:\n{model}')

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
    prepared_data = datasets[0].prepare_train_data(index=4)
    print("dataset[0]", datasets[0])
    print("prepared_data", prepared_data.keys())
    # print("prepared_data_gt_attr_lables", prepared_data["gt_attr_labels"])
    
    ## 2. check wheather the data is right, by visualization
    # dict_keys(['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 
    #           'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 
    #           'ego_lcf_feat', 'gt_attr_labels', 'map_gt_labels_3d', 'map_gt_bboxes_3d'])
    def render_gt_data(data, name): 
        import numpy as np
        import matplotlib.pyplot as plt
        from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map

        # visualize gt map
        colors_plt = ['cornflowerblue', 'royalblue', 'slategrey']

        fig, axes = plt.subplots(1, 1, figsize=(4, 4))
        plt.xlim(xmin=-30, xmax=30)
        plt.ylim(ymin=-30, ymax=30)
        
        map_vectors = data['map_gt_bboxes_3d'].data.fixed_num_sampled_points # torch.Tensor([N, fixed_num(20), 2])
        map_vectors_types = data['map_gt_labels_3d'].data
        for index, line in enumerate(map_vectors):
            gt_label_3d = map_vectors_types[index]
            pts_x = np.array([pt[0] for pt in line])
            pts_y = np.array([pt[1] for pt in line])

            axes.plot(pts_x, pts_y, color=colors_plt[gt_label_3d], linewidth=1, alpha=0.8, zorder=-1)
            axes.scatter(pts_x, pts_y, color=colors_plt[gt_label_3d], s=1, alpha=0.8, zorder=-1)

        # visualize gt bboxes
        from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox, CustomDetectionBox, color_map
        # TODO: gt_bboxes_3d.gravity_center == content['translation'] ?
        # TODO: gt_bboxes_3d.dims == content['size'] ?
        # TODO: nuplan.database.utils.geometry.yaw_to_quaternion(gt_bboxes_3d.yaw[i]) == content['rotation']?
        # TODO: gt_bboxes_3d[:, -2:] == content['velocity'] ?
        gt_bboxes_3d = data['gt_bboxes_3d'].data
        gt_bboxes_3d_types = data['gt_labels_3d'].data
        ignore_list = ['barrier', 'bicycle', 'traffic_cone']

        bbox_gt_list = []
        for i in range(gt_bboxes_3d_types.shape[0]):
            bbox_gt_list.append(CustomDetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                fut_trajs=tuple(gt_fut_trajs),
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=category_to_detection_name(content['category_name']),
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))

        savepath = '/home/gyf/E2EDrivingScale/VAD/VAD-main/tools'
        plt.savefig(savepath+'/gt_map.png', bbox_inches='tight', dpi=200)



    render_gt_data(prepared_data, 'test_data_prepare')

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


# python tools/fx_data_test.py "projects/configs/VAD/VAD_base_e2e_openscen.py" --launcher=none --deterministic --work-dir=openscen_train
# python tools/gyf_data_test.py "projects/configs/VAD/VAD_base_e2e_custom.py" --launcher=none --deterministic --work-dir=openscen_train