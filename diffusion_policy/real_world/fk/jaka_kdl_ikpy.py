import numpy as np
from ikpy import chain

class JakaKDL(object):
    def __init__(self):
        # Getting the URDF path
        urdf_path = "jakamini_leaphand_ori.urdf"
        self.chain = chain.Chain.from_urdf_file(
            urdf_path, 
            base_elements = ['Link_0'], 
            name = 'robot'
        )
        print('finished parsing chains')
    
    def forward_kinematics(self, joint_values, full_kinematics=False):
        joint_values = np.insert(joint_values, 0, 0.0)
        output_frame = self.chain.forward_kinematics(joint_values, full_kinematics)
        return output_frame

if __name__ == "__main__":
    jaka_kdl = JakaKDL()
    # joint_values = [0.0, -1.5398,  0.2241, -1.4109, -0.0064, -1.8088, -2.1648]
    arm_joint_state = np.array([-1.5707487,   0.24192421, -1.4037328,   0.02739489, -1.8208425,  -2.1729174])
    hand_joint_state = np.array([-0.01073527,  0.02301216,  0.00613856,  0.00000262,  0.00000262,  0.02147818,
    0.00613856,  0.23470163, -0.00613332,  0.02301216,  0.00307059,  0.00000262,
    0.02301216, -0.00306535, -0.00306535, -0.00153136])
    joint_values = np.concatenate([arm_joint_state, hand_joint_state])
    end_effector_position, end_effector_orientation = jaka_kdl.forward_kinematics(joint_values)
    print(f"End effector position: {end_effector_position}")
    print(f"End effector orientation: {end_effector_orientation}")