import numpy as np
import torch


def random_delivery(env):
    random_action = torch.randint(0, env.args.num_rsu + 2, (env.args.num_vehicle,))
    max_connection = env.args.max_connections
    rsu_count = [0] * (env.args.num_rsu)
    for i, a in enumerate(random_action):
        if a != 0:
            local_rsu = env.get_local_rsu_of_vehicle(i)
            if rsu_count[local_rsu] < max_connection:
                rsu_count[local_rsu] += 1
            else:
                random_action[i] = 0

    return random_action


def greedy_delivery(env):
    action = []
    requested_vehicles = env.mobility.request.nonzero()[0]
    requested_data = env.mobility.request[requested_vehicles].nonzero()[1]

    rsu_count = [0] * (env.args.num_rsu)

    for v, r in zip(requested_vehicles, requested_data):
        a = 0  # default to the cloud
        if not env.rsu[env.get_local_rsu_of_vehicle(v)].is_interrupt():
            a = env.args.num_rsu + 1
        for rsu_idx, rsu in enumerate(env.rsu):
            if rsu.had(r):
                a = rsu_idx + 1
                if rsu_idx == env.get_local_rsu_of_vehicle(v):
                    break
        if a != 0:  # download via local RSU
            local_rsu = env.get_local_rsu_of_vehicle(v)
            if rsu_count[local_rsu] < env.args.max_connections:
                rsu_count[local_rsu] += 1
            else:
                a = 0

        action.append(a)

    return torch.tensor(action)


def nocache_delivery(env):
    return torch.zeros(
        env.args.num_vehicle,
    )
