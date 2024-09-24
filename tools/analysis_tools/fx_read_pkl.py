import pickle

# 指定.pkl文件的路径
file_path = "/home/gyf/E2EDrivingScale/VAD/VAD-main/test/VAD_base_e2e/Wed_Apr_17_13_07_20_2024/pts_bbox/results_nusc.pkl"

# 使用pickle模块打开文件并读取数据
with open(file_path, "rb") as file:
    bevformer_results = pickle.load(file)


print(bevformer_results.keys())
sample_token_list = list(bevformer_results['results'].keys())

print("token_num", len(sample_token_list))


# python tools/analysis_tools/visualization_fx_debug.py --result-path '/home/gyf/E2EDrivingScale/VAD/VAD-main/test/VAD_base_e2e/Wed_Apr_17_13_07_20_2024/pts_bbox/results_nusc.pkl' --save-path '/home/gyf/E2EDrivingScale/VAD/VAD-main/test/fx_test'

# 08e76760a8c64a92a86686baf68f6aff