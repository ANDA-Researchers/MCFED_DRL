import numpy as np


def random_delivery(args):
    actions = np.random.random(args.num_vehicles * (args.num_rsu + 2)).reshape(
        args.num_vehicles, (args.num_rsu + 2)
    )
    actions = np.argmax(actions, axis=1)

    # restrict the number of connections
    connected_vehicle_indices = np.where(actions != 0)[0]
    if len(connected_vehicle_indices) > args.max_connections:
        drop_indices = np.random.choice(
            connected_vehicle_indices,
            len(connected_vehicle_indices) - args.max_connections,
            replace=False,
        )
        actions[drop_indices] = 0

    return actions


def greedy_delivery(args, env, timestep, requests):
    actions = np.zeros(args.num_vehicles)
    for vehicle_idx, ts in enumerate(requests):
        if ts == timestep % args.time_step_per_round:
            requested_content = env.vehicles[vehicle_idx].request
            all_rsus = env.rsus
            local_rsu_idx = env.reverse_coverage[vehicle_idx]
            local_rsu = all_rsus[local_rsu_idx]
            neighbor_rsus = [
                (rsu, idx) for idx, rsu in enumerate(all_rsus) if idx != local_rsu
            ]

            if requested_content in local_rsu.cache:
                actions[vehicle_idx] = local_rsu_idx + 1
            else:
                actions[vehicle_idx] = args.num_rsu + 1
                for rsu, idx in neighbor_rsus:
                    if requested_content in rsu.cache:
                        actions[vehicle_idx] = idx + 1
                        break

    # restrict the number of connections
    connected_vehicle_indices = np.where(actions != 0)[0]
    if len(connected_vehicle_indices) > args.max_connections:
        drop_indices = np.random.choice(
            connected_vehicle_indices,
            len(connected_vehicle_indices) - args.max_connections,
            replace=False,
        )
        actions[drop_indices] = 0

    return actions
