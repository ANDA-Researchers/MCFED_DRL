import numpy as np
import torch
import torch.optim as optim


class Vehicle:
    def __init__(
        self,
        position: tuple,
        velocity: float,
        data: dict,
        model,
        device,
        local_epochs,
        run_mode,
    ) -> None:
        self.position = position
        self.velocity = velocity
        self.data = data
        self.run_mode = run_mode
        self.device = device
        uid, r_i, Y, urh, upi = self.data
        self.uid = uid
        if self.run_mode == "simulation":
            pivot = int(len(urh) * 0.8)
            self.train = torch.tensor(urh[:pivot]).to(self.device)
            self.test = torch.tensor(urh[pivot:]).to(self.device)
            self.r_i = r_i.to(self.device)
            self.Y = Y
            self.urh = urh
            self.upi = upi.to(self.device)
            self.model = model
            self.batch_size = 128
            self.model.to(self.device)
            self.preference = self.r_i[self.test].cpu()

    def update_velocity(self, velocity: float) -> None:
        self.velocity = velocity

    def update_position(self) -> None:
        self.position = self.position + self.velocity

    @property
    def request(self):
        return self.test[
            torch.distributions.categorical.Categorical(self.preference).sample()
        ]

    def predict(self):
        self.model.eval()
        output = self.model(self.test).detach().cpu()
        prediction = torch.zeros(self.r_i.shape[0])
        prediction[self.test] = output
        return prediction

    def local_update(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        best_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(100):
            for i in range(0, len(self.train), self.batch_size):
                optimizer.zero_grad()
                output = self.model(self.train[i : i + self.batch_size])
                target = self.r_i[self.train[i : i + self.batch_size]]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and abs(loss.item() - best_loss) < 1e-5:
                break

    def get_flatten_weights(self):
        weights = self.model.state_dict()
        return torch.cat([v.view(-1) for k, v in weights.items()] + [self.upi])

    def get_weights(self):
        weights = self.model.state_dict()
        weights = {k: v.cpu() for k, v in weights.items()}
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
