import pickle
import numpy as np

with open('/data/zyp/Projects/VAD_open/VAD-main/data/openscene-v1.1/navsim_logs/mini/2021.05.12.22.00.38_veh-35_01008_01518.pkl', 'rb') as f:  
    loaded_dict = pickle.load(f)
with open('/data/zyp/Projects/VAD_open/VAD-main/tools/token_full_dict.pkl', 'rb') as f:  
    loaded_dict = pickle.load(f)
print("aa")