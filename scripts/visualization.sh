export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python /data/zyp/Projects/VAD_open/VAD-main/tools/analysis_tools/visualization_navsim.py \
    --result-path /data5/zyp/VAD_open_unc/VAD-main/test/VAD_base_e2e_custom/Tue_Apr_30_14_49_07_2024/pts_bbox/results_nusc.pkl \
    --save-path ypx_vis_nuplan