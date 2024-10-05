import copy
from urllib import request
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Vehicle:
    def __init__(self, position, velocity, data, model, gpu=0, writer=None) -> None:
        self.data = data
        self.user_info = data["user_info"]
        # mobility parameters
        self.position = position
        self.velocity = velocity
        self.writer = writer
        # load the model architecture
        if gpu != -1:
            self.device = torch.device("cuda:" + str(gpu))
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)

        self.train_cosine, self.train_semantic, self.train_labels, self.train_ids = (
            self.data["train"]
        )
        self.test_cosine, self.test_semantic, self.test_labels, self.test_ids = (
            self.data["test"]
        )
        self.uid = self.data["uid"]
        (
            self.request_cosine,
            self.request_semantic,
            self.request_labels,
            self.request_ids,
        ) = self.data["request"]

        self.request = np.random.choice(self.request_ids)
        self.movies = self.data["movies"]
        a = self.get_flatten_weights()

    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_position(self, position):
        self.position = position

    def update_request(self):
        self.request = np.random.choice(self.request_ids)

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for idx, movie_id in enumerate(self.request_ids):
                self.movies[movie_id] = self.model(
                    torch.tensor(np.array([self.request_semantic[idx]]))
                    .float()
                    .to(self.device),
                )
        output = torch.sigmoid(self.movies)
        return self.movies

    def local_update(self, round):
        self.model.train()
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # merge x_train and x_test
        X = (
            torch.tensor(np.concatenate([self.train_semantic, self.test_semantic]))
            .float()
            .to(self.device)
        )
        Y = (
            torch.tensor(np.concatenate([self.train_labels, self.test_labels]))
            .float()
            .to(self.device)
        )

        train_dataset = torch.utils.data.TensorDataset(X, Y)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        patience = 10
        best_loss = float("inf")
        epochs_no_improve = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        for _ in range(100):
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

            # self.writer.add_scalar(
            #     f"[{round}] local_loss_uid_{self.uid}", total_loss, _
            # )

            if total_loss < best_loss:
                best_loss = total_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # if epochs_no_improve == patience:
            #     break
        self.model.load_state_dict(best_weights)

    def get_weights(self):
        weights = self.model.state_dict()
        weights = {k: v.cpu() for k, v in weights.items()}
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_flatten_weights(self):
        weights = self.model.state_dict()
        return torch.cat(
            [v.view(-1) for k, v in weights.items()] + [self.user_info.to(self.device)]
        )
