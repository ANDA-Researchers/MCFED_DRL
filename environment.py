from os import path
from turtle import pos
import numpy as np
from scipy.stats import truncnorm
from communication import Communication
from dataset import Library
from vehicle import Vehicle
from model import DNN
from utils import cal_distance


class RSU:
    def __init__(self, position: tuple, capacity: int, model) -> None:
        self.position = position
        self.capacity = capacity
        self.cache = np.random.randint(1, 3952, capacity)
        self.model = model

    def __repr__(self) -> str:
        return f"id: {self.position}, capacity: {self.capacity}"


class Environment:
    def __init__(
        self,
        min_velocity: float,
        max_velocity: float,
        std_velocity: float,
        road_length: float,
        road_width: float,
        rsu_coverage: float,
        rsu_capacity: int,
        num_rsu: int,
        num_vehicles: int,
        time_step: int = 1,
        rsu_highway_distance: float = 1,
        gpu: int = 0,
        embedding_size: int = 50,
        writer=None,
    ) -> None:
        self._validate_parameters(
            min_velocity, max_velocity, num_rsu, rsu_coverage, road_length
        )

        # Simulation parameters
        self.road_length = road_length
        self.road_width = road_width
        self.time_step = time_step
        self.current_time = 0
        self.library = Library(num_vehicles)
        self.communication = None
        self.gpu = gpu
        self.embedding_size = embedding_size
        self.requests = None
        self.writer = writer

        # RSU
        self.rsu_coverage = rsu_coverage
        self.rsu_capacity = rsu_capacity
        self.num_rsu = num_rsu
        self.rsu_highway_distance = rsu_highway_distance

        # Vehicle parameters
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.std_velocity = std_velocity
        self.num_vehicles = num_vehicles

        self.rsus = []
        self.vehicles = []

        self.state_dim = (
            self.num_vehicles * self.num_rsu
            + self.num_vehicles * (self.library.max_movie_id + 1)
            + self.num_vehicles * self.num_rsu
            + (self.num_rsu + 1) * (self.library.max_movie_id + 1)
        )
        self.action_dim = self.num_vehicles * (self.num_rsu + 1)

    def generate_request(self, args):
        self.requests = [
            int(reg)
            for reg in np.random.uniform(0, args.time_step_per_round, args.num_vehicles)
        ]

    def _validate_parameters(
        self,
        min_velocity: float,
        max_velocity: float,
        num_rsu: int,
        rsu_coverage: float,
        road_length: float,
    ) -> None:
        assert min_velocity <= max_velocity and min_velocity >= 0, "Invalid velocity"
        assert num_rsu * rsu_coverage <= road_length, "Invalid RSU configuration"

    def init_model(self):
        return DNN(self.embedding_size, 1)

    def truncated_gaussian(self, mean=None) -> float:

        if mean is None:
            mean = (self.min_velocity + self.max_velocity) / 2

        a, b = (self.min_velocity - mean) / self.std_velocity, (
            self.max_velocity - mean
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=mean, scale=self.std_velocity)

    def add_vehicle(self, position: float, data) -> Vehicle:
        vehicle = Vehicle(
            position,
            self.truncated_gaussian(),
            data,
            self.init_model(),
            self.gpu,
            writer=self.writer,
        )
        self.vehicles.append(vehicle)
        return vehicle

    def poisson_process_on_road(self, n: int, length: float) -> tuple:
        positions = np.random.uniform(0, length, n)
        return positions

    def _initialize_rsus(self) -> list:
        self.rsus = [
            RSU(
                (i + 1) * self.rsu_coverage - self.rsu_coverage / 2,
                self.rsu_capacity,
                self.init_model(),
            )
            for i in range(self.num_rsu)
        ]

    def _initialize_vehicles(self) -> list:
        positions = self.poisson_process_on_road(self.num_vehicles, self.road_length)

        for position in positions:
            self.add_vehicle(position, self.library.genrate_clients())

    def _update_vehicle_positions(self) -> None:
        for vehicle in self.vehicles:
            new_x_position = vehicle.position + vehicle.velocity * self.time_step
            vehicle.update_position(new_x_position)

    def _remove_and_add_vehicles(self) -> None:
        for vehicle in self.vehicles.copy():
            if vehicle.position > self.road_length:
                self.vehicles.remove(vehicle)
                self.library.available_users.remove(vehicle.uid)
                self.add_vehicle(0, self.library.genrate_clients())

    def _update_vehicle_velocities(self) -> None:
        mean = sum([vehicle.velocity for vehicle in self.vehicles]) / len(self.vehicles)
        for vehicle in self.vehicles:
            vehicle.update_velocity(self.truncated_gaussian(mean))

    def _update_vehicle_requests(self) -> None:
        for vehicle in self.vehicles:
            vehicle.update_request()

    def reset(self) -> None:
        if self.rsus == [] or self.vehicles == []:
            self._initialize_rsus()
            self._initialize_vehicles()
        else:
            self._update_vehicle_positions()
            self._remove_and_add_vehicles()
            self._update_vehicle_velocities()
            self.current_time += self.time_step

        # Get channel status
        self.communication = Communication(self.vehicles, self.rsus)

        # Get coverage status
        self.coverage = {k: [] for k in range(self.num_rsu)}
        for i in range(self.num_vehicles):
            for j in range(self.num_rsu):
                if self.communication.distance_matrix[i][j] < self.rsu_coverage:
                    self.coverage[j].append(i)

        self.reverse_coverage = {v: k for k, l in self.coverage.items() for v in l}

    def update_state(self, timestep, args):
        positions_state = (
            self.communication.distance_matrix.flatten()
        )  # num_vehicles * num_rsu
        request_matrix = np.zeros(
            (self.num_vehicles, self.library.max_movie_id + 1)
        )  # num_vehicles * num_movies
        for idx, ts in enumerate(self.requests):
            if ts == timestep % args.time_step_per_round:
                request_matrix[idx][self.vehicles[idx].request] = 1
        shadowing_state = (
            self.communication.current_shadowing.flatten()
        )  # num_vehicles * num_rsu
        fast_fading_state = (
            self.communication.current_fast_fading.flatten()
        )  # num_vehicles * num_rsu

        channel_status = shadowing_state + fast_fading_state
        cache_matrix = np.zeros((self.num_rsu + 1, self.library.max_movie_id + 1))
        for i in range(self.num_rsu + 1):
            for j in range(self.library.max_movie_id + 1):
                if i == 0:
                    cache_matrix[i][j] = 1
                else:

                    if j in self.rsus[i - 1].cache:
                        cache_matrix[i][j] = 1
        self.state = np.concatenate(
            [
                positions_state,  # num_vehicles * num_rsu
                request_matrix.flatten(),  # num_vehicles * num_movies
                channel_status,  # num_vehicles * num_rsu
                cache_matrix.flatten(),
            ]
        )
