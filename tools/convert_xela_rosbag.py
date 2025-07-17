# import argparse
# import os
# from pathlib import Path
# import numpy as np
# import torch

# from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
# from rosidl_runtime_py.utilities import get_message
# from rclpy.serialization import deserialize_message

# def load_xela_rosbag(bag_path: Path, topic: str = "/xServTopic", sensor_name: str = "xela"):
#     bag_path = Path(bag_path)
#     assert bag_path.exists(), f"[ERROR] Bag path does not exist: {bag_path}"

#     storage_opts = StorageOptions(uri=str(bag_path), storage_id='sqlite3')
#     conv_opts = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

#     reader = SequentialReader()
#     reader.open(storage_opts, conv_opts)

#     topics = reader.get_all_topics_and_types()
#     type_map = {t.name: t.type for t in topics}
#     if topic not in type_map:
#         raise RuntimeError(f"[ERROR] Topic {topic} not found in bag.")

#     msg_type = get_message(type_map[topic])

#     dict_raw_data = []
#     timestamps = []

#     while reader.has_next():
#         tpc, raw_data, _ = reader.read_next()
#         if tpc != topic:
#             continue

#         msg = deserialize_message(raw_data, msg_type)

#         # assume 1 sensor per message
#         if not hasattr(msg, 'sensors') or not msg.sensors:
#             continue

#         sensor = msg.sensors[0]
#         taxels = sensor.taxels
#         # z_vals = [float(t.z) for t in sensor.taxels]
#         if len(taxels) != 16:
#             print(f"[WARN] Taxel count != 16, got {len(taxels)}. Skipping.")
#             continue

#         # array_vals = np.array(z_vals, dtype=np.float32).reshape(16, 1)
#         # dict_raw_data.append({sensor_name: array_vals})
#         forces = np.array([[tx.x, tx.y, tx.z] for tx in taxels], dtype=np.float32)   # (16,3)
#         dict_raw_data.append({sensor_name: forces})
#         timestamps.append(sensor.time)

#     timestamps = np.asarray(timestamps, dtype=np.float64)
#     return dict_raw_data, timestamps


# def main():
#     parser = argparse.ArgumentParser(description="Convert XELA rosbag2 to .pth")
#     parser.add_argument("bag", type=Path, help="Path to rosbag2 folder")
#     parser.add_argument("output", type=Path, help="Path to output .pth file")
#     parser.add_argument("--topic", default="/xServTopic", help="Topic name")
#     parser.add_argument("--sensor-name", default="xela", help="Sensor name key")
#     args = parser.parse_args()

#     dict_raw_data, timestamps = load_xela_rosbag(args.bag, args.topic, args.sensor_name)
#     output_dict = {"dict_raw_data": dict_raw_data, "timestamp": timestamps}

#     args.output.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(output_dict, args.output)
#     print(f"✅ Saved tactile data to: {args.output}")


# if __name__ == "__main__":
#     main()


# python tools/convert_xela_rosbag.py   /home/dyros-recruit/mcy_ws/xela_ros2_ws/src/xela_server_ros2/data/xela_2   output/xela_data.pth

import os
import argparse
from pathlib import Path
import numpy as np
import torch

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

def read_xela_from_bag(bag_path: Path, topic: str = "/xServTopic", sensor_name: str = "xela"):
    storage_opts = StorageOptions(uri=str(bag_path), storage_id='sqlite3')
    conv_opts = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics}
    if topic not in type_map:
        print(f"[WARN] Topic {topic} not found in {bag_path.name}")
        return [], []

    msg_type = get_message(type_map[topic])
    dict_raw_data = []
    timestamps = []

    while reader.has_next():
        tpc, raw_data, _ = reader.read_next()
        if tpc != topic:
            continue

        msg = deserialize_message(raw_data, msg_type)
        if not hasattr(msg, 'sensors') or not msg.sensors:
            continue

        sensor = msg.sensors[0]
        taxels = sensor.taxels
        if len(taxels) != 16:
            continue

        force_vecs = np.array([[t.x, t.y, t.z] for t in taxels], dtype=np.float32)
        dict_raw_data.append({sensor_name: force_vecs})
        timestamps.append(sensor.time)

    return dict_raw_data, np.asarray(timestamps, dtype=np.float64)


def merge_all_rosbags(base_dir: Path, output_path: Path, topic: str = "/xServTopic", sensor_name: str = "xela"):
    all_data = []
    all_timestamps = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        if not folder.name.startswith("rosbag2_"):
            continue
        db3_files = list(folder.glob("*.db3"))
        if not db3_files:
            continue

        print(f" Processing: {folder.name}")
        try:
            data, ts = read_xela_from_bag(folder, topic=topic, sensor_name=sensor_name)
            all_data.extend(data)
            all_timestamps.extend(ts)
        except Exception as e:
            print(f"[ERROR] Failed to read {folder.name}: {e}")

    print(f" Total messages: {len(all_data)}")
    output = {
        "dict_raw_data": all_data,
        "timestamp": np.asarray(all_timestamps, dtype=np.float64)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f" Saved merged data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple XELA rosbag2 into one .pth")
    parser.add_argument("input_dir", type=Path, help="Directory containing rosbag2_xServTopic_* folders")
    parser.add_argument("output", type=Path, help="Output .pth file path")
    parser.add_argument("--topic", default="/xServTopic")
    parser.add_argument("--sensor-name", default="xela")
    args = parser.parse_args()

    merge_all_rosbags(args.input_dir, args.output, args.topic, args.sensor_name)
