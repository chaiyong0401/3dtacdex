import os
import pathlib
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import roma
from diffusion_policy.real_world.fk.jaka_leap_paxini_kdl import JakaLeapPaxiniKDL
from diffusion_policy.real_world.data_preprocess.tactile_processer import tactile_process
from diffusion_policy.real_world.constants import TACTILE_DATA_TYPE

class TactilePlayDataset(Dataset):
    def __init__(self, dataset_dir, device, resultant_type=None, aug_type=None, valid=False):
        self.resultant_type = resultant_type
        self.aug_type = aug_type
        self.device = device
        self.valid = valid

        total_demo_names = []
        total_data_directory = []

        data_directory = pathlib.Path(dataset_dir)
        print(data_directory, valid)
        demo_paths = os.listdir(os.path.join(data_directory,'tactiles'))
        demo_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        demo_names = []
        data_directories = []
        for demo_path in demo_paths:
            demo_names.append(demo_path.split('.')[0])
            data_directories.append(data_directory)

        total_demo_names.extend(demo_names)
        total_data_directory.extend(data_directories)

        data_type = "tactiles"
        # Initialize the JakaLeapPaxiniKDL, which used for tactile 3D position fk computation
        jaka_leap_paxini_kdl = JakaLeapPaxiniKDL()

        all_tactile_data = []
        all_resultant_data = []

        for (data_directory, demo) in zip(total_data_directory, total_demo_names):
            if self.resultant_type is not None:
                # tactile process is used for tactile data processing, which will return tactile data in canonical representation and resultant data, relative to hand base
                # the tactile data is represent with canonical rep: 6D pose of sensor origin + 3D position of taxel + 3D force, which correspond to mask index as 9
                tactile_data, resultant_data = tactile_process(data_directory, data_type, demo, jaka_leap_paxini_kdl, resultant_type=resultant_type, base='hand')
                all_resultant_data.append(resultant_data)
            else:
                raise ValueError('No resultant_type not specified')
            all_tactile_data.append(tactile_data)

        self.all_tactile_data = np.concatenate(all_tactile_data)
        if self.resultant_type is not None:
            self.all_resultant_data = np.concatenate(all_resultant_data)

    def get_info(self):
        info = {}
        num_nodes = self.all_tactile_data.shape[1]
        num_features = self.all_tactile_data.shape[2]
        info['num_nodes'] = num_nodes
        info['num_features'] = num_features
        return info

    def random_euler_angles(self):
        batch_size = 1
        euler_angles = np.random.rand(batch_size, 3) * 2 * np.pi
        return euler_angles

    def convert_6d_pose_to_4x4matrix(self, pos=None, euler=None):
        if pos is None:
            batch_size = euler.shape[0]
            pos = np.zeros((batch_size, 3))
        elif euler is None:
            batch_size = pos.shape[0]
            euler = np.zeros((batch_size, 3))
        elif pos is not None and euler is not None:
            batch_size = pos.shape[0]
        else:
            raise ValueError('Invalid input')

        rotation_matrix = R.from_euler('xyz', euler).as_matrix()

        pose_matrix = np.zeros((batch_size, 4, 4))
        pose_matrix[:, :3, :3] = rotation_matrix
        pose_matrix[:, :3, 3] = pos
        pose_matrix[:, 3, 3] = 1  # Homogeneous coordinate

        return pose_matrix

    def convert_4x4matrix_to_6d_pose(self, pose_matrix):
        pos = pose_matrix[:, :3, 3]
        rotation_matrix = pose_matrix[:, :3, :3]
        euler = R.from_matrix(rotation_matrix).as_euler('xyz')

        return pos, euler
    
    def __len__(self):
        return len(self.all_tactile_data)

    def __getitem__(self, idx):
        tactile_data = self.all_tactile_data[idx]
        if self.resultant_type is not None:
            resultant_data = self.all_resultant_data[idx]
            # augment the rotation, prevent overfitting
            if self.aug_type == 'rotation' and not self.valid:
                random_euler_angles = self.random_euler_angles()
                random_transform_matrix = self.convert_6d_pose_to_4x4matrix(euler=random_euler_angles)
                # transform the resultant data
                if self.resultant_type == 'force':
                    resultant_data_transform_matrix = self.convert_6d_pose_to_4x4matrix(pos=resultant_data.reshape(-1,3))
                    resultant_data_transform_matrix = random_transform_matrix @ resultant_data_transform_matrix
                    resultant_data, _ = self.convert_4x4matrix_to_6d_pose(resultant_data_transform_matrix)
                    resultant_data = resultant_data.reshape(-1)
                if TACTILE_DATA_TYPE == ['3d_canonical_data']:
                    tactile_data[:, 3:6] *= np.pi
                    tactile_data_transform_matrix = self.convert_6d_pose_to_4x4matrix(pos=tactile_data[:, :3], euler=tactile_data[:, 3:6])
                    tactile_data_transform_matrix = random_transform_matrix @ tactile_data_transform_matrix
                    tactile_data[:, :3], tactile_data[:, 3:6] = self.convert_4x4matrix_to_6d_pose(tactile_data_transform_matrix)
                    tactile_data[:, 3:6] /= np.pi
            sample = {'tactile_data': tactile_data, 'resultant_data': resultant_data}
        return sample