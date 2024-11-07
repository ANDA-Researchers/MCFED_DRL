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

    # def decision(self, action):
    #     action = action.squeeze().cpu().numpy()
    #     return self.action_spaces[action]

    def reset(self):
        self.mobility.reset()
        self.channel.reset(distance=self.distance)
        self.update_state()
        return self.state, self.mask

    def step(self, action):
        action = action.cpu().numpy()
        avg_delay, total_request, total_hits, total_success = self.compute_delay(action)
        hit_ratio = total_hits / total_request
        success_ratio = total_success / total_request

        reward = (
            torch.tensor(
                -avg_delay * self.args.alpha
                + hit_ratio * self.args.mu
                + success_ratio * self.args.beta,
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
            self.mask,
            reward,
            (avg_delay, total_request, total_hits, total_success),
        )

    def get_local_rsu_of_vehicle(self, vehicle):
        return self.mobility.reverse_coverage[vehicle]

    def get_request_of_vehicle(self, vehicle):
        request = self.mobility.request[vehicle]
        return request.nonzero()[0]

    def compute_delay(self, action):
        delays = []
        total_hits = 0
        total_success = 0
        total_request = np.count_nonzero(self.request)
        request_vehicles = np.where(self.request != 0)[0]
        # action = self.decision(action) # when using DQN

        connection_count = torch.zeros(self.args.num_rsu, dtype=torch.int32)

        for a in action:
            if 0 < a < self.args.num_rsu + 1:
                connection_count[a - 1] += 1

        for i in range(self.args.num_rsu):
            power = self.channel.P_rsu * connection_count[i]
            self.rsu[i].step(power)

        for vehicle_idx in request_vehicles:

            # get the data rate of the vehicle
            bs_rate = max(self.channel.data_rate[0][vehicle_idx], 4e6)
            rsu_rate = self.channel.data_rate[1][vehicle_idx]

            # compute the delay for the vehicle
            rsu_delay = self.args.content_size / rsu_rate
            bs_delay = self.args.content_size / bs_rate
            fiber_delay = self.args.content_size / self.args.fiber_rate
            backhaul_delay = self.args.content_size / self.args.cloud_rate

            local_rsu = self.get_local_rsu_of_vehicle(vehicle_idx)
            requested = self.get_request_of_vehicle(vehicle_idx)
            current_rsu = action[vehicle_idx] - 1

            local_interrupted = self.rsu[local_rsu].is_interrupt()

            # download from BS
            if action[vehicle_idx] == 0:
                delay = bs_delay
                total_success += 1

            # download from BS via local RSU
            elif action[vehicle_idx] > self.args.num_rsu:
                # local rsu is not interruped
                if not local_interrupted:
                    delay = rsu_delay + backhaul_delay
                    total_success += 1

                # local rsu is interruped, fallback to BS
                else:
                    delay = bs_delay

            # download from RSU
            else:
                current_interrupted = self.rsu[current_rsu].is_interrupt()
                # local rsu is not interruped and the requested data is in the rsu
                if (
                    self.rsu[current_rsu].had(requested)
                    and not current_interrupted
                    and not local_interrupted
                ):
                    delay = rsu_delay
                    total_hits += 1
                    total_success += 1

                    # if the current rsu is not the local rsu, add the fiber delay
                    if local_rsu != current_rsu:
                        distance = abs(
                            self.rsu[current_rsu].position
                            - self.vehicle[vehicle_idx].position
                        )
                        hop = distance // 1000
                        delay += fiber_delay * hop

                # fallback to BS
                else:
                    delay = bs_delay

            delays.append(delay)

        # compute the average delay, hit ratio, and success ratio
        avg_delay = np.mean(delays)
        hit_ratio = total_hits / total_request
        success_ratio = total_success / total_request

        return avg_delay, total_request, total_hits, total_success

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
            2 * self.args.num_vehicle
            + 1 * self.args.num_vehicle
            + 1 * self.args.num_rsu
            + 2 * self.args.num_vehicle
            + self.args.num_vehicle * (self.mobility.library.max_movie_id + 1)
            + self.args.num_rsu * (self.mobility.library.max_movie_id + 1)
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
        normalize_distance = distance / max_position
        request = self.mobility.request
        storage = self.mobility.storage

        interrupt = np.ones(len(self.rsu))
        for i, rsu in enumerate(self.rsu):
            if rsu.is_interrupt():
                interrupt[i] = -1

        normalized_rsu_position = normalized_rsu_position * interrupt

        mask = torch.ones(self.args.num_vehicle, self.args.num_rsu + 2) * -10

        for vehicle_idx in range(self.args.num_vehicle):
            mask[vehicle_idx][0] = 0
            mask[vehicle_idx][self.args.num_rsu + 1] = 0
            for rsu_idx in range(self.args.num_rsu):
                if self.rsu[rsu_idx].had(self.request[vehicle_idx].nonzero()[0]):
                    mask[vehicle_idx][rsu_idx] = 0

        state = np.concatenate(
            [
                x.flatten()
                for x in [
                    normalize_distance,  # N * 2
                    normalized_vehicle_position,  # N * 1
                    normalized_rsu_position,  # M * 1
                    channel_state,  # N * 2
                    request,  # N * (L + 1)
                    storage,  # M * (L + 1)
                ]
            ]
        )

        # create the action mask: vehicle can only connect to rsu that has the requested data

        self.state = torch.tensor(state, dtype=torch.float32).to(self.args.device)
        self.mask = mask.to(self.args.device)

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
