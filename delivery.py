import numpy as np
import torch


def random_delivery(env):
    return torch.randint(0, 2, (env.args.num_vehicle,))


def greedy_delivery(env):
    action = []
    requested_vehicles = env.mobility.request.nonzero()[0]
    requested_data = env.mobility.request[requested_vehicles].nonzero()[1]

    for v, r in zip(requested_vehicles, requested_data):
        a = env.args.num_rsu + 1  # default to the cloud
        for rsu_idx, rsu in enumerate(env.rsu):
            if rsu.had(r):
                a = rsu_idx + 1
                if rsu_idx == env.get_local_rsu_of_vehicle(v):
                    break
        action.append(a)

    return torch.tensor(action)


def nocache_delivery(env):
    return torch.zeros(
        env.args.num_vehicle,
    )
