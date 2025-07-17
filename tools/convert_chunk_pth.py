import torch
import numpy as np
import os

# 설정
pth_file = 'data/expert_dataset_holodex/tactile_play_data_train/xela_data.pth'
output_dir = 'data/expert_dataset_holodex/tactile_play_data_train/tactiles'
chunk_size = 1000  # 원하는 trajectory 길이
sensor_key = 'xela'

# 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 데이터 로딩
data = torch.load(pth_file)
raw_data = data['dict_raw_data']
timestamps = data['timestamp']
total_frames = len(raw_data)

# 시퀀스로 분할 및 저장
num_chunks = total_frames // chunk_size
print(f"총 {total_frames} 프레임 → {num_chunks}개 chunk 저장")

for i in range(num_chunks):
    chunk_data = raw_data[i * chunk_size : (i + 1) * chunk_size]
    chunk_ts   = timestamps[i * chunk_size : (i + 1) * chunk_size]

    out_dict = {
        "dict_raw_data": chunk_data,
        "timestamp": np.asarray(chunk_ts),
    }
    out_path = os.path.join(output_dir, f"{i:04d}.pth")
    torch.save(out_dict, out_path)

print(f"✔️ {num_chunks}개 sequence 저장 완료 at {output_dir}")
