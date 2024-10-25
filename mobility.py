import copy
import numpy as np
from scipy.stats import truncnorm
import torch
from torch import optim
from torch.utils.data import DataLoader

from library import Library
from model import DNN


class Vehicle:
    def __init__(self, position: tuple, velocity: float, data: dict, model) -> None:
        self.position = position
        self.velocity = velocity
        self.data = data
        self.model = model

        gpu = 0

        if gpu != -1:
            self.device = torch.device("cuda:" + str(gpu))
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        self.user_info = data["user_info"]

        self.train_cosine, self.train_semantic, self.train_labels, self.train_ids = (
            self.data["train"]
        )
        self.test_cosine, self.test_semantic, self.test_labels, self.test_ids = (
            self.data["test"]
        )
        self.uid = self.data["uid"]

        self.movies = self.data["movies"]

        a = self.get_flatten_weights()

    def update_velocity(self, velocity: float) -> None:
        self.velocity = velocity

    def update_position(self) -> None:
        self.position = self.position + self.velocity

    @property
    def request(self):
        return np.random.choice(self.test_ids)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for idx, movie_id in enumerate(self.test_ids):
                self.movies[movie_id] = self.model(
                    torch.tensor(np.array([self.test_semantic[idx]]))
                    .float()
                    .to(self.device),
                )
        return self.movies

    def local_update(self):
        self.model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.005)

        # merge x_train and x_test
        X = torch.tensor(np.array(self.train_semantic)).float().to(self.device)
        Y = torch.tensor(np.array(self.train_labels)).float().to(self.device)

        train_dataset = torch.utils.data.TensorDataset(X, Y)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        patience = 10
        best_loss = float("inf")
        epochs_no_improve = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        for _ in range(200):
            total_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                x = data[0]
                y = data[1]
                outputs = self.model(x)
                loss = criterion(outputs, y.view_as(outputs))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / len(train_loader)

            if total_loss < best_loss:
                best_loss = total_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                break
        self.model.load_state_dict(best_weights)

    def get_flatten_weights(self):
        weights = self.model.state_dict()
        return torch.cat(
            [v.view(-1) for k, v in weights.items()] + [self.user_info.to(self.device)]
        )

    def get_weights(self):
        weights = self.model.state_dict()
        weights = {k: v.cpu() for k, v in weights.items()}
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights)


class RSU:
    def __init__(self, position: tuple, capacity: int, model) -> None:
        self.position = position
        self.capacity = capacity
        self.cache = np.random.randint(1, 3952, capacity)
        self.model = model
        self.cluster = None

    def had(self, data: int) -> bool:
        return data in self.cache


class BS:
    def __init__(self, position: tuple) -> None:
        self.position = position

    def had(self, data: int) -> bool:
        return True


class Mobility:
    def __init__(self, args) -> None:
        self.length = args.length
        self.num_vehicle = args.num_vehicle
        self.num_rsu = args.num_rsu
        self.rsu_capacity = args.rsu_capacity
        self.rsu_coverage = args.rsu_coverage
        self.min_velocity = args.min_velocity
        self.max_velocity = args.max_velocity
        self.std_velocity = args.std_velocity
        self.bs = BS(self.length / 2)

        self.library = Library(args)
        self.base_model = DNN(50, 1)

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

        """ Distribute vehicles uniformly on the road """
        for i in range(self.num_vehicle):
            position = positions[i]
            self.add_vehicle(position)

        """ Distribute RSUs """
        for i in range(self.num_rsu):
            position = (i + 1) * self.rsu_coverage - self.rsu_coverage / 2
            capacity = self.rsu_capacity
            model = self.base_model

            new_rsu = RSU(position, capacity, model)

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
        data = self.library.generate_client()
        model = self.base_model

        new_vehicle = Vehicle(position, velocity, data, model)

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
            distance_2[i] = np.sqrt((vehicle.position - self.bs.position) ** 2 + 1e-6)

        self.distance = np.max(self.distance, axis=1).reshape(-1, 1)
        self.distance = np.concatenate([self.distance, distance_2], axis=1)

    def update_request(self):
        self.request = np.zeros((self.num_vehicle, self.library.max_movie_id + 1))
        for idx, v in enumerate(self.vehicle):
            reg = int(v.request)
            self.request[idx][reg] = 1

    @property
    def storage(self):
        storage = np.zeros((self.num_rsu, self.library.max_movie_id + 1))
        for idx, rsu in enumerate(self.rsu):
            for movie in rsu.cache:
                storage[idx][movie] = 1
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

    mobility = Mobility(args)
    mobility.reset()

    for i in range(10):
        """Print the state of the environment"""
        print("time step:", i)
        print(mobility.distance)
        print(mobility.coverage)
        print(mobility.request)
        print(mobility.reverse_coverage)

        mobility.step()
