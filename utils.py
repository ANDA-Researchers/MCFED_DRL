import copy
import json
import os

import numpy as np
import torch
import yaml


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def load_args():
    with open("./configs/simulation.yml") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    args = type("args", (object,), configs)()
    return args, configs


def save_results(save_dir, cache, delivery, results):

    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f)
