import numpy as np
from channel import V2X
from mobility import Mobility


class Environment:
    def __init__(self, args) -> None:
        self.mobility = Mobility(args)
        self.channel = V2X(args)
        self.interupt = lambda x: 1.012**x - 1
        self.args = args

    def reset(self):
        self.mobility.reset()
        self.channel.reset(distance=self.distance)

    def step(self, action):
        self.mobility.step()
        self.channel.step(distance=self.distance)

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
        return self.args.num_vehicle * (self.num_rsu + 2)

    @property
    def library(self):
        return self.mobility.library

    @property
    def state(self):
        channel_state = self.channel.channel_gain.flatten()  # N * 2
        distance = self.distance.flatten()  # N * 2
        coverage_matrix = np.zeros((self.args.num_vehicle, len(self.rsu)))
        for r, v in self.mobility.coverage.items():
            coverage_matrix[v, r] = 1
        coverage_matrix = coverage_matrix.flatten()  # N * M
        request = self.mobility.request.flatten()  # N * (L + 1)
        storage = self.mobility.storage.flatten()  # M * (L + 1)

        # print(channel_state.shape, self.args.num_vehicle * 2)
        # print(distance.shape, self.args.num_vehicle * 2)
        # print(coverage_matrix.shape, self.args.num_vehicle * self.args.num_rsu)
        # print(request.shape, self.args.num_vehicle * (self.library.max_movie_id + 1))
        # print(storage.shape, self.args.num_rsu * (self.library.max_movie_id + 1))

        state = np.concatenate(
            [channel_state, distance, coverage_matrix, request, storage]
        )
        return state

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
