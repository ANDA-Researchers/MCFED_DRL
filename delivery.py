import numpy as np


class RandomDelivery:
    def __init__(self, args):
        self.args = args

    def select_action(self, env):
        actions = np.zeros(self.args.num_vehicles)

        for idx in env.request:
            actions[idx] = np.random.randint(0, self.args.num_rsu + 2)

        # restrict the number of connections
        connected_vehicle_indices = np.where(actions != 0)[0]
        if len(connected_vehicle_indices) > self.args.max_connections:
            drop_indices = np.random.choice(
                connected_vehicle_indices,
                len(connected_vehicle_indices) - self.args.max_connections,
                replace=False,
            )
            actions[drop_indices] = 0

        return actions


class GreedyDelivery:
    def __init__(self, args):
        self.args = args

    def select_action(self, env):
        actions = np.zeros(self.args.num_vehicles)
        for idx in env.request:
            requested_content = env.vehicles[idx].request
            all_rsus = env.rsus
            local_rsu_idx = env.reverse_coverage[idx]
            local_rsu = all_rsus[local_rsu_idx]
            neighbor_rsus = [
                (rsu, idx) for idx, rsu in enumerate(all_rsus) if idx != local_rsu
            ]

            if requested_content in local_rsu.cache:
                actions[idx] = local_rsu_idx + 1
            else:
                actions[idx] = self.args.num_rsu + 1
                for rsu, idx in neighbor_rsus:
                    if requested_content in rsu.cache:
                        actions[idx] = idx + 1
                        break

        # restrict the number of connections
        connected_vehicle_indices = np.where(actions != 0)[0]
        if len(connected_vehicle_indices) > self.args.max_connections:
            drop_indices = np.random.choice(
                connected_vehicle_indices,
                len(connected_vehicle_indices) - self.args.max_connections,
                replace=False,
            )
            actions[drop_indices] = 0

        return actions
