import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data, Batch

from diffusion_policy.real_world.fk.constants import *

def data_to_gnn_batch(data, edge_type='four+sensor'):
    assert len(data.shape) == 3
    num_batch = data.shape[0]
    num_nodes = data.shape[1]
    num_feature_dim = data.shape[2]
    edge_index = create_edge_index(num_nodes, edge_type).to(data.device)
    data = list(map(lambda x: Data(x=x,  edge_index=edge_index), data))
    data = Batch.from_data_list(data)
    return data, num_batch, num_nodes, num_feature_dim

def create_edge_index(num_nodes, edge_type='four+sensor'):
    # tip    pulp     tip    pulp    tip     pulp    tip      pulp
    # 6 10  21 25   36 40   51 55   66 70   81 85   96 100   111 115
    # 6 10   6 10    6 10    6 10    6 10    6 10    6 10     6 10
    # 1-15 16-30 31-45 46-60 61-75 76-90 91-105 106-120
    # tip:
    #     15--14--13--12--11
    #   /
    # 10-- 9--     8-- 7-- 6
    #   \ 
    #      5-- 4-- 3-- 2-- 1

    # pulp:
        # 15--14--13--12--11
        # 10-- 9-- 8-- 7-- 6
        #  5-- 4-- 3-- 2-- 1
    
    tip_points_number = PAXINI_TIP_COORD.shape[0]
    pulp_points_number = PAXINI_PULP_COORD.shape[0]
    assert tip_points_number == pulp_points_number

    if 'four' in edge_type:
        tip_edge_index = np.array([[ 1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  
                                    5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  
                                    8,  8,  9,  9,  9,  9, 10, 10, 11, 11, 11, 12, 12, 12, 
                                    12,13, 13, 13, 13, 14, 14, 14, 15, 15, 15],
                                    [1,  2,  6,  2,  1,  3,  7,  3,  2,  4,  8,  4,  3,  5,  
                                    5,  4,  9,  6,  1,  7, 11,  7,  2,  6,  8, 12,  8,  3,  
                                    7, 13,  9,  5, 10, 15, 10,  9, 11,  6, 12, 12,  7, 11,
                                    13,13,  8, 12, 14, 14, 13, 15, 15,  9, 14]]) - 1   
        
        pulp_edge_index = np.array([[ 1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4,
                                    5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  
                                    8,  8,  9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 12,
                                    12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15], 
                                    [ 1,  2,  6,  2,  1,  3,  7,  3,  2,  4,  8,  4,  3,  5,  9,
                                    5,  4, 10,  6,  1,  7, 11,  7,  2,  6,  8, 12,  8,  3,  7,
                                    9, 13,  9,  4,  8, 10, 14, 10,  5,  9, 15, 11,  6, 12, 12,
                                    7, 11, 13, 13,  8, 12, 14, 14,  9, 13, 15, 15, 10, 14]]) - 1
    elif 'eight' in edge_type:
        tip_edge_index = np.array([[ 1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  4,
                                     4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,
                                     7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,
                                     9,  9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12,
                                     12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15,
                                     15],
                                    [ 1,  2,  6,  7,  2,  1,  3,  6,  7,  8,  3,  2,  4,  7,  8,  4,
                                      3,  5,  8,  9,  5,  4,  9, 10,  6,  1,  2,  7, 11, 12,  7,  1,
                                      2,  3,  6,  8, 11, 12, 13,  8,  2,  3,  4,  7, 12, 13, 14,  9,
                                      4,  5, 10, 14, 15, 10,  5,  9, 15, 11,  6,  7, 12, 12,  6,  7,
                                      8, 11, 13, 13,  7,  8, 12, 14, 14,  8,  9, 13, 15, 15,  9, 10,
                                      14]]) - 1
        
        pulp_edge_index = np.array([[ 1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,
                                        4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,
                                        7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,
                                        8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10, 10, 10, 10, 10,
                                        10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13,
                                        13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15],
                                    [ 1,  2,  6,  7,  2,  1,  3,  6,  7,  8,  3,  2,  4,  7,  8,  9,
                                        4,  3,  5,  8,  9, 10,  5,  4,  9, 10,  6,  1,  2,  7, 11, 12,
                                        7,  1,  2,  3,  6,  8, 11, 12, 13,  8,  2,  3,  4,  7,  9, 12,
                                        13, 14,  9,  3,  4,  5,  8, 10, 13, 14, 15, 10,  4,  5,  9, 14,
                                        15, 11,  6,  7, 12, 12,  6,  7,  8, 11, 13, 13,  7,  8,  9, 12,
                                        14, 14,  8,  9, 10, 13, 15, 15,  9, 10, 14]]) -1
    elif 'all' in edge_type:
        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes))
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        return edge_index
    else:
        raise NotImplementedError(f"{edge_type} is not implemented.")

    edge_index = []
    # tip pulp tip pulp tip pulp tip pulp
    assert list(TACTILE_INFO.keys())==['thumb_tip', 'thumb_pulp', 'index_tip', 'index_pulp', 'middle_tip', 'middle_pulp', 'ring_tip', 'ring_pulp']

    start_idx = 0
    for sensor in TACTILE_INFO:
        if 'tip' in sensor:
            edge_index.append(start_idx+tip_edge_index)
            start_idx+=tip_points_number
        elif 'pulp' in sensor:
            edge_index.append(start_idx+pulp_edge_index)
            start_idx+=pulp_points_number
    
    if 'sensor' in edge_type:
        sensor_edge_index = np.array([[ 6, 25, 21, 21,  21,   36, 55, 51, 51, 51,   66, 85, 81, 81, 81,    96, 115, 111, 111, 111],
                                  [25,  6, 51, 81, 111,   55, 36, 21, 81,111,   85, 66, 21, 51,111,   115,  96,  21,  51,  81]]) - 1
        edge_index.append(sensor_edge_index)

    edge_index = np.concatenate(edge_index, axis=1)
    edge_index = torch.from_numpy(edge_index)
    return edge_index

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")