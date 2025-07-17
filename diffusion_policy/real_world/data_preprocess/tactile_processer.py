import os
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from diffusion_policy.real_world.constants import TACTILE_RAW_DATA_SCALE, TACTILE_DATA_TYPE

def convert_force_from_paxini_to_link(forces):
    # force paixni coord x y z, link coord z y x
    # change x and z axis
    forces = forces.copy()
    forces[:,[0,2]] = forces[:,[2,0]]
    return forces

def compute_resultant(coord, forces, type):
    if type == 'force':
        # convert force in sensor frame to link frame
        force_2_link = convert_force_from_paxini_to_link(forces)
        rot_rpy = coord[:,3:6]*np.pi
        rot_matrix = R.from_euler('xyz', rot_rpy).as_matrix()
        force_2_base = np.einsum('ijk,ik->ij', rot_matrix, force_2_link)
        # compute net force
        resultant_force = np.sum(force_2_base, axis=0)
        return resultant_force
    else:
        raise NotImplementedError
    
def compute_resultant(coord, forces, type, force_convert_func=None):
    if type == 'force':
        # convert force in sensor frame to link frame if needed
        if force_convert_func is None:
            force_2_link = forces
        else:
            force_2_link = force_convert_func(forces)
        rot_rpy = coord[:,3:6]*np.pi
        rot_matrix = R.from_euler('xyz', rot_rpy).as_matrix()
        force_2_base = np.einsum('ijk,ik->ij', rot_matrix, force_2_link)
        # compute net force
        resultant_force = np.sum(force_2_base, axis=0)
        return resultant_force
    else:
        raise NotImplementedError
    
# def tactile_process(data_directory, data_type, demo, jaka_leap_paxini_kdl, end_idx=-0, resultant_type=None, base='arm'):
def tactile_process(
    data_directory,
    data_type,
    demo,
    kdl_model,
    end_idx=0,
    resultant_type=None,
    base='arm',
    force_convert_func=None,
):
    tactiles = torch.load(os.path.join(data_directory, data_type, f"{demo}.pth"))
    dp_tactiles = []
    for tactile in TACTILE_DATA_TYPE:
        if tactile == "3d_canonical_data":
            ori_tactile_data = tactiles['dict_raw_data']
            tactile_link_list = list(ori_tactile_data[0].keys())
            # state = torch.load(os.path.join(data_directory, "states", f"{demo}.pth"))
            # arm_joint_states = state['arm_abs_joint'].cpu().numpy()
            # hand_joint_states = state['hand_abs_joint'].cpu().numpy()


            for tactile_force in ori_tactile_data:
                tactile_fxfyfz = []
                for sensor_name in tactile_force:
                    force = tactile_force[sensor_name]
                    tactile_fxfyfz.append(force)
                tactile_fxfyfz = np.stack(tactile_fxfyfz, axis=0)  # shape: (num_taxels, 3)

                # 정규화
                tactile_fxfyfz = tactile_fxfyfz / TACTILE_RAW_DATA_SCALE

                # 위치 정보 없이 force만 사용 (상대좌표 기반)
                dp_tactiles.append(tactile_fxfyfz)

            dp_tactiles = np.stack(dp_tactiles, axis=0)  # shape: (T, num_taxels, 3)
            dp_tactiles = dp_tactiles[:len(dp_tactiles) + end_idx]

            return dp_tactiles

            # dp_tactiles = []
            # if resultant_type is not None:
            #     dp_resultant = []
            # for (arm_joint_state, hand_joint_state, tactile_force) in zip(arm_joint_states, hand_joint_states, ori_tactile_data):
            #     # compute canonical tactile point
            #     if tactile == "3d_canonical_data":
            #         # tactile_point = jaka_leap_paxini_kdl.forward_kinematics(arm_joint_state, hand_joint_state, tactile_link_list, coords_type='original', coords_space='canonical', base=base)
            #         tactile_point = kdl_model.forward_kinematics(
            #             arm_joint_state,
            #             hand_joint_state,
            #             tactile_link_list,
            #             coords_type='original',
            #             coords_space='canonical',
            #             base=base,
            #         )
            #     tactile_xyz = np.concatenate(tactile_point)
            #     # compute taxel force
            #     tactile_fxfyfz = []
            #     for sensor_name in tactile_force:
            #         tactile_fxfyfz.append(tactile_force[sensor_name])
            #     tactile_fxfyfz = np.concatenate(tactile_fxfyfz) / TACTILE_RAW_DATA_SCALE
            #     # compute net force
            #     if resultant_type is not None and (tactile == "3d_canonical_data"):
            #         # resultant_force = compute_resultant(tactile_xyz, tactile_fxfyfz, resultant_type)
            #         resultant_force = compute_resultant(
            #             tactile_xyz,
            #             tactile_fxfyfz,
            #             resultant_type,
            #             force_convert_func
            #         )
            #         dp_resultant.append(resultant_force)
            #     # concatenate tactile xyz and force to get representation
            #     tactile_xyzfxfyfz = np.concatenate([tactile_xyz, tactile_fxfyfz], axis=1)
            #     dp_tactiles.append(tactile_xyzfxfyfz)
            # dp_tactiles = np.stack(dp_tactiles, axis=0)
            # dp_tactiles = dp_tactiles[:len(dp_tactiles)+end_idx]
            # if resultant_type is not None:
            #     dp_resultant = np.stack(dp_resultant, axis=0)
            #     dp_resultant = dp_resultant[:len(dp_resultant)+end_idx]
            # if tactile == "3d_canonical_data":
            #     print(np.max(dp_tactiles[:, :,9:].reshape(-1,3)*TACTILE_RAW_DATA_SCALE, 0), np.min(dp_tactiles[:, :,9:].reshape(-1,3)*TACTILE_RAW_DATA_SCALE, 0), np.min(np.where(dp_tactiles[:,:,9:]==np.max(np.max(dp_tactiles[:, :,9:],1)))[0]), np.min(np.where(dp_tactiles[:,:,9:]==np.min(np.min(dp_tactiles[:, :,9:],1)))[0]), len(dp_tactiles))

    # if resultant_type is not None:
    #     return dp_tactiles, dp_resultant
    # else:
    #     return dp_tactiles