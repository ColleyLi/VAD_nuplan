import pandas as pd
from tqdm import tqdm
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
import numpy as np
import logging
import lzma
import pickle
import os
import uuid
import math
import matplotlib.pyplot as plt

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataclasses import PDMResults, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log', filemode='w')
logger = logging.getLogger(__name__)

CONFIG_PATH = "/data/zyp/Projects/VAD_open/navsim_gyf/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    build_logger(cfg)
    # TODO: for debug
    cfg.worker.debug_mode = True
    
    worker = build_worker(cfg)
    
    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_all_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    
    # TODO: for debug
    # score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)
    score_rows = run_pdm_score(data_points)
    
    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}. 
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
    """)
    
def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))

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

def add_heading_to_trajectory(future_trajectory, delta_format: bool):
    '''
    future_trajectory: np.ndarray, shape=(num_points(8, contains origin), 2)
    
    return:
    future_trajectory_custom_heading: np.ndarray, shape=(num_points(8, no origin), 3)
    '''
    
    if delta_format:
        # if delta_format is True, it stores (dx, dy); 
        # we should make it as a series of (x, y) points
        future_trajectory = future_trajectory.cumsum(axis=0) 
    
    # add origin point
    future_trajectory = np.concatenate([np.zeros((1, 2)), future_trajectory], axis=0)
    # add heading dim to future_trajectory
    future_trajectory_custom_heading = np.concatenate([future_trajectory, np.zeros((future_trajectory.shape[0], 1))], axis=1)
    
    # calculate headings
    last_rads = 0
    for i in range(future_trajectory_custom_heading.shape[0] - 1):
        degrees = calc_angle(future_trajectory_custom_heading[i], future_trajectory_custom_heading[i+1])
        speed = np.linalg.norm(future_trajectory_custom_heading[i+1][:2] - future_trajectory_custom_heading[i][:2])
        rads = math.radians(degrees)
        rads = normalize_angle(rads)
        # the positions of the ego appear to be confusion, in the case of ego is nearly stationary
        if speed < 0.2: # TODO: 0.4?
            rads = last_rads
        else:
            last_rads = rads
        future_trajectory_custom_heading[i][2] = rads
    # the last point's heading is set the same as the second last point
    future_trajectory_custom_heading[-1][2] = future_trajectory_custom_heading[-2][2]
    
    # remove origin point, since it is not used in pdm scoring
    return future_trajectory_custom_heading[1:]

def plot_scene(scene, future_trajectory):
    '''
    scene: Scene
    future_trajectory: np.ndarray, shape=(num_points(8, no origin), 3(x, y, heading))
    '''
    from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
    from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
    from navsim.visualization.plots import configure_bev_ax, configure_ax
    
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
    
    human_trajectory = scene.get_future_trajectory()
    add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
    add_trajectory_to_bev_ax(ax, future_trajectory, TRAJECTORY_CONFIG["agent"])
    configure_bev_ax(ax)
    configure_ax(ax)
    
    return fig, ax

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    # one log file at a time
    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens # if give tokens, then scene loader will only load these tokens; else load all tokens from log
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # Load the trajectory data and corresponding tokens
    
    ## load labels
    # with open('/data/zyp/Projects/VAD_open/VAD-main/data/openscene-v1.1/gyf_navsim_mini_train.pkl', 'rb') as file:
    #     data = pickle.load(file)
    # data_trajectories = [data['infos'][i]['gt_ego_fut_trajs'] for i in range(len(data['infos']))]
    # data_tokens = [data['infos'][i]['token'] for i in range(len(data['infos']))] # frame_token
    
    ## load test results
    with open('/data/zyp/Projects/VAD_open/VAD-main/tools/token_traj_dict.pkl', 'rb') as file:
        data = pickle.load(file)
    data_tokens = [] # frame_token
    data_trajectories = []
    for token, value in data.items():
        data_tokens.append(token)
        multi_traj = value['ego_fut_preds'] # shape=(4, 8, 2)
        command = np.argmax(value['ego_fut_cmd'][0,0,0]) # int
        data_trajectories.append(multi_traj[command].tolist())
    
    # convert to np array
    trajectories_array = np.array(data_trajectories, dtype=np.float32)
    # NOTE: preprocess the trajectories_array, to change format(if needed), and add headings
    trajectories_array_custom_heading = np.zeros((trajectories_array.shape[0], trajectories_array.shape[1], 3))
    for i in range(trajectories_array.shape[0]):
        traj = trajectories_array[i]
        traj_custom_heading = add_heading_to_trajectory(traj, delta_format=True)
        trajectories_array_custom_heading[i] = traj_custom_heading
    trajectories_dict = dict(zip(data_tokens, trajectories_array_custom_heading))
    
    # define which tokens that need to be evaluated
    tokens_in_both_data_and_sceneloader = list(set(data_tokens) & set(scene_loader.tokens))
    tokens_to_evaluate = list(set(tokens_in_both_data_and_sceneloader) & set(metric_cache_loader.tokens))
    
    pdm_results: List[Dict[str, Any]] = []
    # for idx, (token) in enumerate(tokens_to_evaluate):
    for idx in tqdm(range(len(tokens_to_evaluate))):
        token = tokens_to_evaluate[idx]
        # logger.info(
        #     f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        # )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        
        # TODO: assign token id
        # token = '512a7051fa4c5554'
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)

            # trajectory loaded from pkl file in this token
            loaded_trajectory = Trajectory(poses=np.array(trajectories_dict[token], dtype=np.float32),
                                    trajectory_sampling=TrajectorySampling(interval_length=0.5,time_horizon=4.0))
            
            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                scene = scene_loader.get_scene_from_token(token)
                gt_trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                gt_trajectory = agent.compute_trajectory(agent_input)
                
            # NOTE: check heading calculation
            gt_traj_array_custom_heading = add_heading_to_trajectory(gt_trajectory.poses[:, :2], delta_format=False)
            heading_errors = np.degrees((gt_traj_array_custom_heading - gt_trajectory.poses)[:, -1])
            if sum(heading_errors) > 20: # each point allow 2.5 degree error
                print(token, ' sum over 20') # '290ef639ccbb50da' '614792f42a2153a0' '1bf100f880f558d6' '512a7051fa4c5554'
            elif any(abs(heading_errors)) > 10:
                print(token, ' any over 10')
            
            # compare outcomes for gt_trajectory and gt_trajectory_custom_heading
            gt_trajectory_custom_heading = Trajectory(poses=np.array(gt_traj_array_custom_heading, dtype=np.float32),
                                                      trajectory_sampling=TrajectorySampling(interval_length=0.5,time_horizon=4.0))
            
            # simulation and scoring
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=loaded_trajectory, # NOTE: change here to evaluate different trajectories
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
            
            # render the scene
            if not scene:
                scene = scene_loader.get_scene_from_token(token)
            fig, ax = plot_scene(scene, loaded_trajectory) # NOTE: change here to visualize different trajectories
            savepath = '/data/zyp/Projects/VAD_open/VAD-main/tools/dataset_test_images/open_tiny_mini_new_60' # NOTE: change here to visualize different trajectories
            plt.savefig(savepath + '/test_' + str(token) + '.png')
            # plt.savefig(savepath + '/test.png')
            plt.close()
            
            error = (loaded_trajectory.poses - gt_trajectory_custom_heading.poses).sum()
            # print(error)
            
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
        
    return pdm_results

if __name__ == "__main__":
    main()
