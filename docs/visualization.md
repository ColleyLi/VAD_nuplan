# Visualization

We provide the script to visualize the VAD prediction to a video [here](../tools/analysis_tools/visualization.py).

## Visualize prediction

```shell
cd /path/to/VAD/
conda activate vad
python tools/analysis_tools/visualization.py --result-path '/home/gyf/E2EDrivingScale/VAD/VAD-main/test/VAD_base_e2e/Wed_Apr_17_13_07_20_2024/pts_bbox/results_nusc.pkl' --save-path '/home/gyf/E2EDrivingScale/VAD/VAD-main/test/VAD_base_e2e/Wed_Apr_17_13_07_20_2024/vis/'
```

The inference results is a prefix_results_nusc.pkl automaticly saved to the work_dir after running evaluation. It's a list of prediction results for each validation sample.
