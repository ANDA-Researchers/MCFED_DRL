import numpy as np
from scipy.stats import truncnorm
from communication import Communication
from dataset import Library
from vehicle import Vehicle
from model import DNN


class RSU:
    def __init__(self, position: tuple, capacity: int, model) -> None:
        self.position = position
        self.capacity = capacity
        self.cache = np.random.randint(1, 3952, capacity)
        self.model = model
        self.cluster = None

    def __repr__(self) -> str:
        return f"id: {self.position}, capacity: {self.capacity}"


class Environment:
    def __init__(
        self,
        min_velocity: float,
        max_velocity: float,
        std_velocity: float,
        rsu_coverage: float,
        rsu_capacity: int,
        num_rsu: int,
        num_vehicles: int,
        args,
        time_step: int = 1,
        gpu: int = 0,
        embedding_size: int = 50,
    ) -> None:

        # Simulation parameters
        self.road_length = num_rsu * rsu_coverage
        self.time_step = time_step
        self.library = Library(num_vehicles)
        self.gpu = gpu
        self.embedding_size = embedding_size
        self.args = args

        # RSU
        self.rsu_coverage = rsu_coverage
        self.rsu_capacity = rsu_capacity
        self.num_rsu = num_rsu

        # Vehicle parameters
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.std_velocity = std_velocity
        self.num_vehicles = num_vehicles

        # state dim
        self.state_dim = (
            self.num_vehicles * self.num_rsu
            + self.num_vehicles * (self.library.max_movie_id + 1)
            + self.num_vehicles * self.num_rsu
            + (self.num_rsu + 2) * (self.library.max_movie_id + 1)
        )

    def reset(self):
        self.library.reset()
        self._initialize_rsus()
        self._initialize_vehicles()
        self._initialize_channel()
        self._update_coverage()
        self._init_request()
        self._update_state()

    def step(self):
        self._update_vehicles()
        self.communication.reset(self.vehicles, self.rsus)
        self._update_coverage()
        self._update_requests()
        self._update_state()

    def _init_model(self):
        return DNN(self.embedding_size, 1)

    def _initialize_vehicles(self) -> list:
        self.vehicles = []
        positions = self.poisson_process_on_road(self.num_vehicles, self.road_length)

        for position in positions:
            self._add_vehicle(position, self.library.genrate_clients())

    def _initialize_rsus(self) -> list:
        self.rsus = [
            RSU(
                (i + 1) * self.rsu_coverage - self.rsu_coverage / 2,
                self.rsu_capacity,
                self._init_model(),
            )
            for i in range(self.num_rsu)
        ]

    def _initialize_channel(self):
        self.communication = Communication(self.vehicles, self.rsus)

    def _init_request(self):
        requests = [
            int(reg)
            for reg in np.random.uniform(
                0, self.args.time_step_per_round, self.args.num_vehicles
            )
        ]

        self.requests = [
            [reg for reg in requests if reg == i]
            for i in range(self.args.time_step_per_round)
        ]
        self.request = self.requests.pop(0)

    def truncated_gaussian(self, mean=None) -> float:

        if mean is None:
            mean = (self.min_velocity + self.max_velocity) / 2

        a, b = (self.min_velocity - mean) / self.std_velocity, (
            self.max_velocity - mean
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=mean, scale=self.std_velocity)

    def poisson_process_on_road(self, n: int, length: float) -> tuple:
        positions = np.random.uniform(0, length, n)
        return positions

    def _add_vehicle(self, position: float, data) -> Vehicle:
        vehicle = Vehicle(
            position,
            self.truncated_gaussian(),
            data,
            self._init_model(),
            self.gpu,
        )
        self.vehicles.append(vehicle)
        return vehicle

    def _update_vehicles(self):
        self._update_vehicle_positions()
        self._update_vehicle_requests()
        self._update_vehicle_velocities()
        self._remove_and__add_vehicles()

    def _update_vehicle_positions(self) -> None:
        for vehicle in self.vehicles:
            new_x_position = vehicle.position + vehicle.velocity * self.time_step
            vehicle.update_position(new_x_position)

    def _remove_and__add_vehicles(self) -> None:
        for vehicle in self.vehicles.copy():
            if vehicle.position > self.road_length:
                self.vehicles.remove(vehicle)
                self.library.available_users.remove(vehicle.uid)
                self._add_vehicle(0, self.library.genrate_clients())

    def _update_vehicle_velocities(self) -> None:
        mean = sum([vehicle.velocity for vehicle in self.vehicles]) / len(self.vehicles)
        for vehicle in self.vehicles:
            vehicle.update_velocity(self.truncated_gaussian(mean))

    def _update_vehicle_requests(self) -> None:
        for vehicle in self.vehicles:
            vehicle.update_request()

    def _update_requests(self):
        if len(self.requests) == 0:
            self._init_request()
        else:
            self.request = self.requests.pop(0)

    def _update_coverage(self):
        self.coverage = {k: [] for k in range(self.num_rsu)}
        for i in range(self.num_vehicles):
            for j in range(self.num_rsu):
                if self.communication.distance_matrix[i][j] < self.rsu_coverage:
                    self.coverage[j].append(i)

        self.reverse_coverage = {v: k for k, l in self.coverage.items() for v in l}

    def _update_state(self):
        # self.num_vehicles * self.num_rsu
        positions_state = self.communication.distance_matrix.flatten()

        # self.num_vehicles * (self.library.max_movie_id + 1)
        request_matrix = np.zeros((self.num_vehicles, self.library.max_movie_id + 1))
        for idx in self.request:
            request_matrix[idx][self.vehicles[idx].request] = 1

        # self.num_vehicles * self.num_rsu
        channel_status = (
            self.communication.current_shadowing.flatten()
            + self.communication.current_fast_fading.flatten()
        )

        # (self.num_rsu + 2) * (self.library.max_movie_id + 1)
        cache_matrix = np.zeros((self.num_rsu + 2, self.library.max_movie_id + 1))

        for i in range(self.num_rsu + 1):
            for j in range(self.library.max_movie_id + 1):
                if i == self.num_rsu + 2:
                    cache_matrix[i][j] = 1
                else:
                    if j in self.rsus[i - 1].cache:
                        cache_matrix[i][j] = 1

        self.state = np.concatenate(
            [
                positions_state,
                request_matrix.flatten(),
                channel_status,
                cache_matrix.flatten(),
            ]
        )

        # ensure the state shape is correct
        assert self.state.shape == self.state_dim
