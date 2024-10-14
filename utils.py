import numpy as np
import torch
import copy


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def calculate_distance(a, b):
    return abs(a - b)


def calculate_distance_matrix(vehicles, rsus):
    distance_matrix = np.zeros((len(vehicles), len(rsus)))
    for i, vehicle in enumerate(vehicles):
        for j, rsu in enumerate(rsus):
            distance_matrix[i][j] = calculate_distance(vehicle.position, rsu.position)
    return distance_matrix

def restrict_connections(actions, max_connections):
    connected_vehicle_indices = np.where(actions != 0)[0]
    if len(connected_vehicle_indices) > max_connections:
        drop_indices = np.random.choice(
            connected_vehicle_indices,
            len(connected_vehicle_indices) - max_connections,
            replace=False,
        )
        actions[drop_indices] = 0
        
    return actions