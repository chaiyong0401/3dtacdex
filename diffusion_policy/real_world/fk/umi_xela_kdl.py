from scipy.spatial.transform import Rotation as R
import numpy as np

from diffusion_policy.real_world.fk.constants import (
    XELA_USPA44_COORD,
    XELA_TACTILE_ORI_COORD,
)

class UmiXelaKDL:
    """Minimal FK wrapper for UMI gripper with a single XELA uSPa44 sensor."""

    def __init__(self):
        # the sensor has 16 taxels
        self.point_per_sensor = len(XELA_USPA44_COORD)

    def forward_kinematics(
        self,
        arm_joint_state,
        hand_joint_state,
        tactile_list=None,
        coords_type="full",
        coords_space="base",
        base="arm",
    ):
        # For the portable UMI gripper the tactile sensor pose is assumed fixed
        link_pose = np.eye(4)   # 센서가 고정된 pose로 가정
        tactile_points = self.get_tactile_points(
            XELA_TACTILE_ORI_COORD,
            XELA_USPA44_COORD,
            link_pose,
            coords_type,
            coords_space,
        )
        return [tactile_points] # Return a list with a single element (16 taxel point 3d positions + angles) -> [16,6]

    def get_tactile_points(self, tactile_ori, tactile_points, link_pose, coords_type, coords_space):
        if coords_type == "full":
            local_points = tactile_ori + tactile_points # 각 taxel 상대 좌표 + 센서 원점
            rotation = link_pose[:3, :3]    
            translation = link_pose[:3, 3]
            real_points = (rotation @ local_points.T).T + translation   # 센서 기준 taxel 좌표를 로봇 base frame으로 변환 여기서는 link pose 고정이라 생각해 무시
            real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi
            real_points = np.concatenate(
                [real_points, np.repeat(real_angle, self.point_per_sensor, axis=0)], axis=1
            )   # [16,6] (x,y,z,roll,pitch,yaw) -> 상대좌표를 사용하기 있으므로 taxel 위치 정보만 사용된다고 생각하면 될 듯 
        elif coords_type == "original": # not used in UMI
            local_points = np.array([tactile_ori])
            rotation = link_pose[:3, :3]
            translation = link_pose[:3, 3]
            real_points = (rotation @ local_points.T).T + translation
            real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi

            if coords_space == "canonical":
                min_point = np.min(tactile_points, axis=0)
                max_point = np.max(tactile_points, axis=0)
                diagonal_length = np.linalg.norm(max_point - min_point)
                center_point = (min_point + max_point) / 2
                tactile_points = 2 * (tactile_points - center_point) / diagonal_length
                real_points = np.concatenate(
                    [
                        np.repeat(real_points, self.point_per_sensor, axis=0),
                        np.repeat(real_angle, self.point_per_sensor, axis=0),
                        tactile_points,
                    ],
                    axis=1,
                )
        return real_points
