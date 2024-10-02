import numpy as np


def random_delivery(args):
    actions = np.random.random(args.num_vehicles * (args.num_rsu + 1)).reshape(
        args.num_vehicles, (args.num_rsu + 1)
    )
    actions = np.argmax(actions, axis=1)
    return actions


def greedy_delivery(args, env, timestep, reverse_coverage, requests):
    actions = np.zeros(args.num_vehicles)
    for vehicle_idx, ts in enumerate(requests):
        if ts == timestep % args.time_step_per_round:
            requested_content = env.vehicles[vehicle_idx].request
            all_rsus = env.rsu
            local_rsu_idx = reverse_coverage[vehicle_idx]
            local_rsu = all_rsus[local_rsu_idx]
            neighbor_rsus = [
                (rsu, idx) for idx, rsu in enumerate(all_rsus) if idx != local_rsu
            ]

            if requested_content in local_rsu.cache:
                actions[vehicle_idx] = local_rsu_idx + 1
            else:
                for rsu, idx in neighbor_rsus:
                    if requested_content in rsu.cache:
                        actions[vehicle_idx] = idx + 1
                        break
    return actions
