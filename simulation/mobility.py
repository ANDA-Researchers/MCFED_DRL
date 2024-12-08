import copy

import numpy as np
import torch
from scipy.stats import truncnorm
from torch import optim
from torch.utils.data import DataLoader

from .server import BS, RSU

from .interrupt import Interrupt
from .library import Library
from .mcfed import AECF, BaseModel
from .vehicle import Vehicle


class Mobility:
    def __init__(self, args) -> None:
        self.length = args.num_rsu * args.rsu_coverage
        self.num_vehicle = args.num_vehicle
        self.num_rsu = args.num_rsu
        self.rsu_capacity = args.rsu_capacity
        self.rsu_coverage = args.rsu_coverage
        self.min_velocity = args.min_velocity
        self.max_velocity = args.max_velocity
        self.std_velocity = args.std_velocity
        self.bs = BS(self.length / 2)

        self.library = Library()
        self.base_model = BaseModel(args.fl_hidden_dim, self.library.num_items)
        self.local_epochs = args.num_local_epochs
        self.device = args.device
        self.run_mode = args.run_mode
        self.interruption = args.interruption
        self._lambda = args._lambda

        assert (
            self.length % self.rsu_coverage == 0
        ), "RSU coverage should be a divisor of road length"

        self.rsu = []
        self.vehicle = []
        self.coverage = None
        self.reverse_coverage = None
        self.distance = None

        self.request = None

    def reset(self):
        positions = self.uniform()
        self.rsu = []
        self.vehicle = []
        self.library.reset()

        """ Distribute vehicles uniformly on the road """
        for i in range(self.num_vehicle):
            position = positions[i]
            self.add_vehicle(position)

        """ Distribute RSUs """
        for i in range(self.num_rsu):
            position = (i + 1) * self.rsu_coverage - self.rsu_coverage / 2
            capacity = self.rsu_capacity
            model = self.base_model

            new_rsu = RSU(
                position,
                capacity,
                model,
                self.library.num_items,
                self.interruption,
                self._lambda,
            )
            self.rsu.append(new_rsu)

        self.update_coverage()
        self.update_request()

    def truncated_gaussian(self, mean=None) -> float:
        if mean is None:
            mean = (self.min_velocity + self.max_velocity) / 2

        a, b = (self.min_velocity - mean) / self.std_velocity, (
            self.max_velocity - mean
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=mean, scale=self.std_velocity)

    def uniform(self) -> tuple:
        positions = np.random.uniform(0, self.length, self.num_vehicle)
        return positions

    def add_vehicle(self, position: float) -> None:
        velocity = self.truncated_gaussian()
        data = self.library.create_client()
        model = self.base_model

        new_vehicle = Vehicle(
            position,
            velocity,
            data,
            model,
            self.device,
            self.local_epochs,
            self.run_mode,
        )

        self.vehicle.append(new_vehicle)

    def update_coverage(self):
        self.distance = np.zeros((self.num_vehicle, self.num_rsu))
        self.coverage = {k: [] for k in range(self.num_rsu)}

        for i, vehicle in enumerate(self.vehicle):
            for j, rsu in enumerate(self.rsu):
                distance = abs(vehicle.position - rsu.position)
                if distance <= self.rsu_coverage / 2:
                    self.distance[i][j] = distance
                    self.coverage[j].append(i)

        self.reverse_coverage = {v: k for k, l in self.coverage.items() for v in l}

        # compure distance from the BS
        distance_2 = np.zeros((self.num_vehicle, 1))  # num_vehicle * 1
        for i, vehicle in enumerate(self.vehicle):
            distance_2[i] = np.sqrt((vehicle.position - self.bs.position) ** 2 + 1e6)

        self.distance = np.max(self.distance, axis=1).reshape(-1, 1)
        self.distance = np.concatenate([distance_2, self.distance], axis=1)

    def update_request(self):
        self.request = np.zeros((self.num_vehicle, self.library.num_items))

        if self.run_mode == "simulation":
            for idx, v in enumerate(self.vehicle):
                reg = int(v.request)
                self.request[idx][reg] = 1
        else:
            total_content = list(range(self.library.num_items))
            prob = (0.7, 0.3, 0)
            cached = []

            for r in self.rsu:
                cached.extend(r.cache)

            cached = list(set(cached))
            uncached = list(set(total_content) - set(cached))
            for idx, v in enumerate(self.vehicle):
                local_cache = self.rsu[self.reverse_coverage[idx]].cache

                cached_not_local = list(set(cached) - set(local_cache))

                option = np.random.choice([0, 1, 2], p=prob)

                if option == 0:
                    reg = np.random.choice(local_cache)
                elif option == 1:
                    reg = np.random.choice(cached_not_local)
                else:
                    reg = np.random.choice(uncached)

                self.request[idx][reg] = 1

    @property
    def storage(self):
        storage = np.zeros((self.num_rsu + 2, self.library.num_items))
        storage[0, :] = 1  # BS has all the content
        storage[-1, :] = 1  # BS has all the content
        for idx, rsu in enumerate(self.rsu):
            for movie in rsu.cache:
                storage[idx + 1][movie] = 1

        return storage

    def step(self):
        # Update vehicle position and velocity
        mean = np.mean([vehicle.velocity for vehicle in self.vehicle])
        for vehicle in self.vehicle:
            vehicle.update_position()
            vehicle.update_velocity(self.truncated_gaussian(mean))

        # Add new vehicle if it goes out of the road
        for vehicle in self.vehicle.copy():
            if vehicle.position < 0 or vehicle.position > self.length:
                uid = vehicle.uid
                self.vehicle.remove(vehicle)
                self.library.return_uid(uid)
                self.add_vehicle(0)

        # Update RSU coverage
        self.update_coverage()
        self.update_request()
