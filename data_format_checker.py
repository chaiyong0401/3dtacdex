import torch

data = torch.load("data/expert_dataset_holodex/tactile_play_data_train/tactiles/xela_data.pth")
print(data.keys())               # ['dict_raw_data', 'timestamp']
print(type(data['dict_raw_data']))  # list of dict
print(len(data['dict_raw_data']))   # e.g., 1000
print(data['dict_raw_data'][0])     # {'xela': np.array((16, 3), dtype=float32)}