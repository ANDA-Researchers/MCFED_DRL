import copy
import numpy as np
import torch
import torch.optim as optim


class Vehicle:
    def __init__(self, position, velocity, user_id, info, data, model, gpu=0) -> None:
        self.user_id = user_id
        self.position = position
        self.velocity = velocity
        self.data = data
        self.info = info
        self.divider = int(len(data["contents"]) * 0.8)
        self.gpu = gpu

        self.input_shape = self.data["max"] + 1

        # load the model architecture
        if gpu != -1:
            self.model = model.cuda("cuda:" + str(gpu))
        else:
            self.model = model

    def __repr__(self) -> str:
        return (
            f"id: {self.user_id}, position: {self.position}, velocity: {self.velocity}"
        )

    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_position(self, position):
        self.position = position

    def update_request(self):
        self.divider += 1

    def create_ratings_matrix(self):
        matrix = []
        for i in range(self.data["max"] + 1):
            if i in self.data["contents"][: self.divider]:
                matrix.append(1)
            else:
                matrix.append(0)
        return np.array(matrix)

    @property
    def request(self):
        return self.data["contents"][self.divider]

    def predict(self):
        self.model.eval()
        _input = torch.tensor(self.create_ratings_matrix()).float()

        if self.gpu != -1:
            _input = _input.cuda("cuda:" + str(self.gpu))

        output = self.model(_input)
        output = output.cpu().detach().numpy()

        return output

    def local_update(self):
        self.model.train()
        criterion = torch.nn.L1Loss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        _input = torch.tensor(self.create_ratings_matrix()).float()
        if self.gpu != -1:
            _input = _input.cuda("cuda:" + str(self.gpu))

        patience = 10
        best_loss = float("inf")
        epochs_no_improve = 0
        best_weights = copy.deepcopy(self.model.state_dict())

        for _ in range(10):
            optimizer.zero_grad()
            output = self.model(_input)
            loss = criterion(output, _input)
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                epochs_no_improve = 0
                best_weights = copy.deepcopy(self.model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        self.model.load_state_dict(best_weights)

    def get_weights(self):
        weights = self.model.state_dict()
        weights = {k: v.cpu() for k, v in weights.items()}
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
