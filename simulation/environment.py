import numpy as np
import torch
from .channel import V2X
from .mobility import Mobility
from itertools import product


class Environment:
    def __init__(self, args) -> None:
        self.mobility = Mobility(args)
        self.channel = V2X(args)
        self.args = args

    def reset(self):
        self.mobility.reset()
        self.channel.reset(distance=self.distance)
        self.update_state()
        return self.state

    def step(self, action):
        action = action.cpu().numpy()
        avg_delay, total_request, total_hits, total_fails = self.compute_delay(action)
        hit_ratio = total_hits / total_request
        success_ratio = (total_request - total_fails) / total_request

        reward = (
            torch.tensor(
                -avg_delay * self.args.alpha
                + hit_ratio * self.args.beta
                + success_ratio * self.args.mu,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(self.args.device)
        )

        self.mobility.step()
        self.channel.step(distance=self.distance)
        self.update_state()
        return (
            self.state,
            reward,
            (avg_delay, total_request, total_hits, total_request - total_fails),
        )

    def get_local_rsu_of_vehicle(self, vehicle):
        return self.mobility.reverse_coverage[vehicle]

    def get_request_of_vehicle(self, vehicle):
        request = self.mobility.request[vehicle]
        return request.nonzero()[0]

    def compute_delay(self, action):
        delays = []
        total_hits = 0  # number of download from the cache
        total_fails = 0  # number of successful connections (no fallback)

        # count total number of requests
        total_request = np.count_nonzero(self.request)

        # get the index of the vehicles that have requests
        request_vehicles = np.where(self.request != 0)[0]

        # count the number of connections to each rsu
        connection_count = torch.zeros(self.args.num_rsu, dtype=torch.int32)
        for vehicle_idx, a in enumerate(action):
            if 0 < a:
                local_rsu = self.get_local_rsu_of_vehicle(vehicle_idx)
                connection_count[local_rsu] += 1

        # compute the power consumption of each rsu and step the interruption
        for i in range(self.args.num_rsu):
            power = self.channel.P_rsu * connection_count[i]
            self.rsu[i].step(power)

        for vehicle_idx in request_vehicles:
            # Get vehicle wireless rate
            bs_rate = self.channel.data_rate[vehicle_idx][0]
            rsu_rate = self.channel.data_rate[vehicle_idx][1]

            # compute the delay for the vehicle
            rsu_delay = self.args.content_size / rsu_rate
            bs_delay = self.args.content_size / bs_rate
            fiber_delay = self.args.content_size / self.args.fiber_rate
            backhaul_delay = self.args.content_size / self.args.cloud_rate

            # get index of local rsu
            local_rsu = self.get_local_rsu_of_vehicle(vehicle_idx)

            # get the requested data
            requested = self.get_request_of_vehicle(vehicle_idx)

            # get the index of destination rsu
            current_rsu = action[vehicle_idx] - 1

            # get the action of the vehicle
            vehicle_action = action[vehicle_idx]

            # get the interruption status of the local rsu
            local_interrupted = self.rsu[local_rsu].is_interrupt()

            if vehicle_action == 0:  # Download from the BS
                delays.append(bs_delay)
            elif (
                vehicle_action == self.args.num_rsu + 1
            ):  # Download from the BS via local RSU
                if not local_interrupted:
                    delays.append(rsu_delay + backhaul_delay)
                else:
                    delays.append(bs_delay)  # Fallback to BS
                    total_fails += 1
            else:
                if local_interrupted:
                    delays.append(bs_delay)  # Fallback to BS
                    total_fails += 1
                else:
                    # get the interruption status of the current rsu
                    current_interrupted = self.rsu[current_rsu].is_interrupt()
                    if current_interrupted:
                        delays.append(
                            rsu_delay + backhaul_delay
                        )  # Fallback to BS via local RSU
                    else:
                        if self.rsu[current_rsu].had(requested):
                            delay = rsu_delay
                            total_hits += 1
                            if local_rsu != current_rsu:
                                hop = abs(local_rsu - current_rsu)
                                delay += hop * fiber_delay
                            delays.append(delay)
                        else:
                            delays.append(
                                rsu_delay + backhaul_delay
                            )  # Fallback to BS via local RSU

        # compute the average delay, hit ratio, and success ratio
        avg_delay = np.mean(delays)

        return avg_delay, total_request, total_hits, total_fails

    @property
    def vehicle(self):
        return self.mobility.vehicle

    @property
    def rsu(self):
        return self.mobility.rsu

    @property
    def bs(self):
        return self.mobility.bs

    @property
    def distance(self):
        return self.mobility.distance

    @property
    def state_dim(self):
        return (
            +1 * self.args.num_vehicle
            + 1 * self.args.num_rsu
            + 2 * self.args.num_vehicle
            + self.args.num_vehicle * (self.args.num_rsu + 2)
        )

    @property
    def action_dim(self):
        return (self.args.num_rsu + 2) ** self.args.num_vehicle

    @property
    def library(self):
        return self.mobility.library

    def update_state(self):
        channel_state = self.channel.channel_gain.flatten()
        distance = self.distance.flatten()

        max_position = self.args.num_rsu * self.args.rsu_coverage

        vehicle_position = torch.tensor(
            [v.position for v in self.vehicle], dtype=torch.float32
        )
        rsu_position = torch.tensor([r.position for r in self.rsu], dtype=torch.float32)

        normalized_vehicle_position = vehicle_position / max_position
        normalized_rsu_position = rsu_position / max_position
        request = self.mobility.request  # (num_vehicle, num_items)
        storage = self.mobility.storage  # (num_rsu + 2, num_items)

        interrupt = np.ones(self.args.num_rsu + 2)  # (num_rsu + 2)
        for i, rsu in enumerate(self.rsu):
            if rsu.is_interrupt():
                interrupt[i + 1] = 0

        interrupt = interrupt.reshape(1, -1)
        interrupt = interrupt.repeat(self.args.num_vehicle, 0)

        possible_moves = np.matmul(request, storage.T)  # (num_vehicle, num_rsu + 2)

        possible_moves = possible_moves * interrupt

        state = np.concatenate(
            [
                x.flatten()
                for x in [
                    normalized_vehicle_position,  # N * 1
                    normalized_rsu_position,  # M * 1
                    channel_state,  # N * 2
                    possible_moves,  # N * (M + 2)
                ]
            ]
        )

        self.state = torch.tensor(state, dtype=torch.float32).to(self.args.device)

    @property
    def request(self):
        return self.mobility.request


if __name__ == "__main__":
    args = {
        "length": 2000,
        "num_vehicle": 5,
        "num_rsu": 2,
        "rsu_capacity": 10,
        "rsu_coverage": 1000,
        "min_velocity": 10,
        "max_velocity": 30,
        "std_velocity": 5,
    }

    args = type("args", (object,), args)()

    env = Environment(args)
    env.reset()

    for i in range(20000):
        env.step(0)
