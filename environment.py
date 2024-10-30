import numpy as np
import torch
from channel import V2X
from mobility import Mobility
from itertools import product


class Environment:
    def __init__(self, args) -> None:
        self.mobility = Mobility(args)
        self.channel = V2X(args)
        self.interupt = lambda x: 1.012**x - 1
        self.args = args

        # self.action_spaces = np.array(
        #     list(
        #         product(
        #             *[
        #                 range(self.args.num_rsu + 2)
        #                 for _ in range(self.args.num_vehicle)
        #             ]
        #         )
        #     )
        # )

    # def decision(self, action):
    #     action = action.squeeze().cpu().numpy()
    #     return self.action_spaces[action]

    def reset(self):
        self.mobility.reset()
        self.channel.reset(distance=self.distance)
        self.update_state()

        return self.state

    def step(self, action):
        action = action.cpu().numpy()
        avg_delay, hit_ratio, success_ratio = self.compute_delay(action)

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

        return self.state, reward

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

        for vehicle_idx in request_vehicles:

            # get the data rate of the vehicle
            bs_rate = 3000000  # self.channel.data_rate[vehicle_idx][0]
            rsu_rate = self.channel.data_rate[vehicle_idx][1]

            # compute the delay for the vehicle
            rsu_delay = self.args.content_size / rsu_rate
            bs_delay = self.args.content_size / bs_rate
            fiber_delay = self.args.content_size / self.args.fiber_rate
            backhaul_delay = self.args.content_size / 5000000

            # download from BS
            if action[vehicle_idx] == 0:
                delay = bs_delay
                total_success += 1

            # download from BS via local RSU
            elif action[vehicle_idx] > self.args.num_rsu:
                # local rsu is not interruped
                if True:
                    delay = rsu_delay + backhaul_delay
                    total_success += 1

                # local rsu is interruped, fallback to BS
                else:
                    delay = bs_delay

            # download from RSU
            else:
                local_rsu = self.get_local_rsu_of_vehicle(vehicle_idx)
                requested = self.get_request_of_vehicle(vehicle_idx)
                current_rsu = action[vehicle_idx] - 1

                # local rsu is not interruped and the requested data is in the rsu
                if self.rsu[current_rsu].had(requested) and True:
                    delay = rsu_delay
                    total_hits += 1
                    total_success += 1

                    # if the current rsu is not the local rsu, add the fiber delay
                    if local_rsu != current_rsu:
                        hop = 1
                        delay += fiber_delay * hop

                # fallback to BS
                else:
                    delay = bs_delay

            delays.append(delay)

        # compute the average delay, hit ratio, and success ratio
        avg_delay = np.mean(delays)
        hit_ratio = total_hits / total_request
        success_ratio = total_success / total_request

        return avg_delay, hit_ratio, success_ratio

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
            self.args.num_vehicle * 2
            + self.args.num_vehicle * 2
            + self.args.num_vehicle * self.args.num_rsu
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
        channel_state = self.channel.channel_gain.flatten()  # N * 2
        distance = self.distance.flatten()  # N * 2
        coverage_matrix = np.zeros((self.args.num_vehicle, len(self.rsu)))
        for r, v in self.mobility.coverage.items():
            coverage_matrix[v, r] = 1
        coverage_matrix = coverage_matrix.flatten()  # N * M
        request = self.mobility.request.flatten()  # N * (L + 1)
        storage = self.mobility.storage.flatten()  # M * (L + 1)

        state = np.concatenate(
            [channel_state, distance, coverage_matrix, request, storage]
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
