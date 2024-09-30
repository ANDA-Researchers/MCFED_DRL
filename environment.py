import numpy as np
from scipy.stats import truncnorm
from library import ContentLibrary
from vehicle import Vehicle
from model import AutoEncoder
from utils import cal_distance

# Constants
BS_CAPACITY = 100000
BS_POSITION = (10, -2000)
MODEL_DIMENSION = 512
DATA_PATH = "./data/ml-100k/"


class RSU:
    def __init__(self, position: tuple, capacity: int, distance_from_bs: float) -> None:
        self.position = position
        self.distance_from_bs = distance_from_bs
        self.capacity = capacity
        self.cache = []

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
        bs_position: tuple = BS_POSITION,
        rsu_highway_distance: float = 1,
        gpu: int = 0,
    ) -> None:
        self._validate_parameters(
            min_velocity, max_velocity, num_rsu, rsu_coverage, road_length
        )

        # Simulation parameters
        self.road_length = road_length
        self.road_width = road_width
        self.time_step = time_step
        self.current_time = 0
        self.content_library = ContentLibrary(DATA_PATH)
        self.global_model = self.init_model()
        self.communication = None
        self.gpu = gpu

        # RSU
        self.rsu_coverage = rsu_coverage
        self.rsu_capacity = rsu_capacity
        self.num_rsu = num_rsu
        self.rsu_highway_distance = rsu_highway_distance

        # BS
        self.bs = RSU(bs_position, BS_CAPACITY, 0)

        # Vehicle parameters
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.std_velocity = std_velocity
        self.mean_velocity = (min_velocity + max_velocity) / 2
        self.num_vehicles = num_vehicles

        # RSU/BS placement
        self.rsu = self._initialize_rsus()

        # Vehicle initialization
        self.vehicles = []
        self.vehicles = self._initialize_vehicles()

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

    def init_model(self) -> AutoEncoder:
        return AutoEncoder(self.content_library.max_item_id + 1, MODEL_DIMENSION)

    def truncated_gaussian(self, mean=None) -> float:

        if mean is None:
            mean = self.mean_velocity

        a, b = (self.min_velocity - mean) / self.std_velocity, (
            self.max_velocity - mean
        ) / self.std_velocity
        return truncnorm.rvs(a, b, loc=mean, scale=self.std_velocity)

    def add_vehicle(self, x_position: float, y_position: float) -> Vehicle:
        user_id = self.content_library.get_user()
        user_info = self.content_library.load_user_info(user_id)
        user_data = self.content_library.load_ratings(user_id)
        vehicle = Vehicle(
            (x_position, y_position),
            self.truncated_gaussian(),
            user_id,
            user_info,
            user_data,
            self.init_model(),
            self.gpu,
        )
        self.vehicles.append(vehicle)
        return vehicle

    def poisson_process_on_road(self, n: int, length: float, width: float) -> tuple:
        x_positions = np.random.uniform(0, length, n)
        y_positions = np.random.uniform(0, width, n)
        return x_positions, y_positions

    def _initialize_rsus(self) -> list:
        return [
            RSU(
                (
                    (i + 1) * self.rsu_coverage - self.rsu_coverage / 2,
                    self.road_width // 2,
                ),
                self.rsu_capacity,
                cal_distance(
                    self.bs.position,
                    (
                        (i + 1) * self.rsu_coverage - self.rsu_coverage / 2,
                        self.road_width + self.rsu_highway_distance,
                    ),
                ),
            )
            for i in range(self.num_rsu)
        ]

    def _initialize_vehicles(self) -> list:
        x_positions, y_positions = self.poisson_process_on_road(
            self.num_vehicles, self.road_length, self.road_width
        )
        return [
            self.add_vehicle(x, y)
            for x, y in zip(x_positions, y_positions)
            if self.add_vehicle(x, y) is not None
        ]

    def _update_vehicle_positions(self) -> None:
        for vehicle in self.vehicles:
            new_x_position = vehicle.position[0] + vehicle.velocity * self.time_step
            vehicle.update_position((new_x_position, vehicle.position[1]))

    def _remove_and_add_vehicles(self) -> None:
        for vehicle in self.vehicles.copy():
            if vehicle.position[0] > self.road_length:
                self.content_library.return_user(vehicle.user_id)
                self.vehicles.remove(vehicle)
                _, y_positions = self.poisson_process_on_road(
                    1, self.road_length, self.road_width
                )
                self.add_vehicle(0, y_positions[0])

    def _update_vehicle_velocities(self) -> None:
        mean = sum([vehicle.velocity for vehicle in self.vehicles]) / len(self.vehicles)
        for vehicle in self.vehicles:
            vehicle.update_velocity(self.truncated_gaussian(mean))

    def _update_vehicle_requests(self) -> None:
        for vehicle in self.vehicles:
            vehicle.update_request()

    def step(self) -> None:
        self._update_vehicle_positions()
        self._remove_and_add_vehicles()
        self._update_vehicle_velocities()
        self.current_time += self.time_step
