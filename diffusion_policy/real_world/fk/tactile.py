import numpy as np

PAXINI_PULP_ORI_COORDS = np.array([-4.70000,  3.30000, 2.94543,
                            -4.70000,  7.80000, 2.94543,
                            -4.70000, 12.30000, 2.94543,
                            -4.70000, 16.80000, 2.94543,
                            -4.70000, 21.30000, 2.94543,
                            0.00000,  3.30000, 3.09994,
                            0.00000,  7.80000, 3.09994,
                            0.00000, 12.30000, 3.09994,
                            0.00000, 16.80000, 3.09994,
                            0.00000, 21.30000, 3.09994,
                            4.70000,  3.30000, 2.94543,
                            4.70000,  7.80000, 2.94543,
                            4.70000, 12.30000, 2.94543,
                            4.70000, 16.80000, 2.94543,
                            4.70000, 21.30000, 2.94543]).reshape(-1,3)

PAXINI_TIP_ORI_COORDS = np.array([-4.50000,  4.30109, 2.97814,
                            -4.50000,  8.15109, 2.89349,
                            -4.50000, 12.18660, 2.64440,
                            -4.50000, 15.99390, 2.06277,
                            -2.25000, 21.45300, 0.09510,
                            -0.00000,  4.30109, 3.10726,
                            -0.00000,  8.15109, 3.03111,
                            -0.00000, 12.20620, 2.80133,
                            -0.00000, 16.01800, 2.25633,
                            -0.00000, 24.50520,-2.49584,
                            4.50000,  4.30109, 2.97814,
                            4.50000,  8.15109, 2.89349,
                            4.50000, 12.18660, 2.64440,
                            4.50000, 15.99390, 2.06277,
                            2.25000, 21.45300, 0.09510]).reshape(-1,3)

def convert_coord_from_sw2real(tact_2_target):
    real_axis = tact_2_target['real_axis']
    sw_axis = tact_2_target['sw_axis']
    tactile_origin_coord_in_sw = tact_2_target['tactile_origin_coord_in_sw']
    target_link_coord_in_sw = tact_2_target['target_link_coord_in_sw']

    tact2target_in_sw = tactile_origin_coord_in_sw - target_link_coord_in_sw

    tact2target_in_sw_dict = {}
    for idx, value in enumerate(tact2target_in_sw):
        if idx == 0:
            axis = 'x'
        elif idx == 1:
            axis = 'y'
        elif idx == 2:
            axis = 'z'
        tact2target_in_sw_dict[axis] = value

    tact2target_in_real_dict = {}
    for axis_ori_real in real_axis:
        idx = real_axis.index(axis_ori_real)
        axis_ori_in_sw = sw_axis[idx]
        assert len(axis_ori_in_sw)==3, "The length of axis_ori_in_sw should be 3."
        sign = axis_ori_in_sw[0]
        axis = axis_ori_in_sw[2]
        if sign == '+':
            tact2target_in_real_dict[axis_ori_real] = tact2target_in_sw_dict[axis]
        elif sign == '-':
            tact2target_in_real_dict[axis_ori_real] = -tact2target_in_sw_dict[axis]

    # convert tact2fingertip_in_real_dict to a list
    tact2target_in_real = [tact2target_in_real_dict[axis] for axis in real_axis]
    return tact2target_in_real

def convert_coord_from_paixni2urdf(tact_2_target):
    paxini_axis = tact_2_target['paxini_axis']
    urdf_axis = tact_2_target['urdf_axis']

    tact_coord_in_paxini = tact_2_target['tact_coord_in_paxini']

    tact_in_paxini_dict = {}
    for idx in range(3):
        if idx == 0:
            axis = 'x'
        elif idx == 1:
            axis = 'y'
        elif idx == 2:
            axis = 'z'
        tact_in_paxini_dict[axis] = tact_coord_in_paxini[:, idx]
    
    tact_in_urdf_dict = {}
    for axis_ori_urdf in urdf_axis:
        idx = urdf_axis.index(axis_ori_urdf)
        axis_ori_in_paxini = paxini_axis[idx]
        assert len(axis_ori_in_paxini)==3, "The length of axis_ori_in_paxini should be 3."
        sign = axis_ori_in_paxini[0]
        axis = axis_ori_in_paxini[2]
        if sign == '+':
            tact_in_urdf_dict[axis_ori_urdf] = tact_in_paxini_dict[axis]
        elif sign == '-':
            tact_in_urdf_dict[axis_ori_urdf] = -tact_in_paxini_dict[axis]

    # convert tact2fingertip_in_real_dict to a list
    tact_in_urdf = np.array([tact_in_urdf_dict[axis] for axis in urdf_axis]).T
    return tact_in_urdf

if __name__ == "__main__":

    tip_tact_2_fingertip = {}
    tip_tact_2_fingertip['real_axis'] = ['x', 'y', 'z']
    tip_tact_2_fingertip['sw_axis'] = ['-_y', '+_x', '+_z']
    tip_tact_2_fingertip['tactile_origin_coord_in_sw'] = np.array([99.57, 48.88, 0.02]) / 1000
    tip_tact_2_fingertip['target_link_coord_in_sw'] = np.array([113.55, 33.35, -15.9]) / 1000
    tip_tact_2_fingertip_in_real = convert_coord_from_sw2real(tip_tact_2_fingertip)
    print(tip_tact_2_fingertip_in_real)

    pulp_tact_2_dip = {}
    pulp_tact_2_dip['real_axis'] = ['x', 'y', 'z']
    pulp_tact_2_dip['sw_axis'] = ['+_y', '-_x', '+_z']
    pulp_tact_2_dip['tactile_origin_coord_in_sw'] = np.array([-9.19, 0.57, 17.78]) / 1000
    pulp_tact_2_dip['target_link_coord_in_sw'] = np.array([-21.56, 16.37, 1.88]) / 1000
    pulp_tact_2_dip_in_real = convert_coord_from_sw2real(pulp_tact_2_dip)
    print(pulp_tact_2_dip_in_real)

    tip_tact_2_thumbfingertip = {}
    tip_tact_2_thumbfingertip['real_axis'] = ['x', 'y', 'z']
    tip_tact_2_thumbfingertip['sw_axis'] = ['-_x', '+_y', '-_z']
    tip_tact_2_thumbfingertip['tactile_origin_coord_in_sw'] = np.array([-16.42, -138.43, 7.49]) / 1000
    tip_tact_2_thumbfingertip['target_link_coord_in_sw'] = np.array([-32.74, -138.53, -8.41]) / 1000
    tip_tact_2_thumbfingertip_in_real = convert_coord_from_sw2real(tip_tact_2_thumbfingertip)
    print(tip_tact_2_thumbfingertip_in_real)

    pulp_tact_2_thumbfingertip = {}
    pulp_tact_2_thumbfingertip['real_axis'] = ['x', 'y', 'z']
    pulp_tact_2_thumbfingertip['sw_axis'] = ['-_x', '+_y', '-_z']
    pulp_tact_2_thumbfingertip['tactile_origin_coord_in_sw'] = np.array([-17.22, -168.11, 7.49]) / 1000
    pulp_tact_2_thumbfingertip['target_link_coord_in_sw'] = np.array([-32.74, -138.53, -8.41]) / 1000
    pulp_tact_2_thumbfingertip_in_real = convert_coord_from_sw2real(pulp_tact_2_thumbfingertip)
    print(pulp_tact_2_thumbfingertip_in_real)

    pulp_paxini_2_tact = {}
    pulp_paxini_2_tact['urdf_axis'] = ['x', 'y', 'z']
    pulp_paxini_2_tact['paxini_axis'] = ['-_z', '-_y', '-_x']
    pulp_paxini_2_tact['tact_coord_in_paxini'] = PAXINI_PULP_ORI_COORDS
    pulp_paxini_2_tact_in_urdf = convert_coord_from_paixni2urdf(pulp_paxini_2_tact)
    print(pulp_paxini_2_tact_in_urdf)

    tip_paxini_2_tact = {}
    tip_paxini_2_tact['urdf_axis'] = ['x', 'y', 'z']
    tip_paxini_2_tact['paxini_axis'] = ['-_z', '-_y', '-_x']
    tip_paxini_2_tact['tact_coord_in_paxini'] = PAXINI_TIP_ORI_COORDS
    tip_paxini_2_tact_in_urdf = convert_coord_from_paixni2urdf(tip_paxini_2_tact)
    print(tip_paxini_2_tact_in_urdf)

    thumb_pulp_paxini_2_tact = {}
    thumb_pulp_paxini_2_tact['urdf_axis'] = ['x', 'y', 'z']
    thumb_pulp_paxini_2_tact['paxini_axis'] = ['-_z', '-_y', '-_x']
    thumb_pulp_paxini_2_tact['tact_coord_in_paxini'] = PAXINI_PULP_ORI_COORDS
    thumb_pulp_paxini_2_tact_in_urdf = convert_coord_from_paixni2urdf(thumb_pulp_paxini_2_tact)
    print(thumb_pulp_paxini_2_tact_in_urdf)

    thumb_tip_paxini_2_tact = {}
    thumb_tip_paxini_2_tact['urdf_axis'] = ['x', 'y', 'z']
    thumb_tip_paxini_2_tact['paxini_axis'] = ['-_z', '-_y', '-_x']
    thumb_tip_paxini_2_tact['tact_coord_in_paxini'] = PAXINI_TIP_ORI_COORDS
    thumb_tip_paxini_2_tact_in_urdf = convert_coord_from_paixni2urdf(thumb_tip_paxini_2_tact)
    print(thumb_tip_paxini_2_tact_in_urdf)

