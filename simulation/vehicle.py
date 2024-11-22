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
        train = True
        self.position = position
        self.velocity = velocity
        self.data = data
        self.run_mode = run_mode
        self.device = device
        uid, r_i, Y, urh, upi = self.data
        self.uid = uid
        if self.run_mode == "train":
            self.model = model.to(self.device)
            self.r_i = r_i.to(self.device)
            self.Y = Y.to(self.device)
            self.local_epochs = local_epochs
            self.urh = urh
            self.upi = upi.to(self.device)

            self.mask = self.r_i != 0
            self.preference = torch.ones_like(self.r_i) * self.r_i[self.mask].mean()
            self.preference[self.mask] = 0

    def update_velocity(self, velocity: float) -> None:
        self.velocity = velocity

    def update_position(self) -> None:
        self.position = self.position + self.velocity

    @property
    def request(self):
        prob = self.preference
        sample = torch.distributions.Categorical(prob).sample()
        return sample

    def predict(self):
        self.model.eval()
        output = self.model(self.r_i)
        output[self.mask] = 0
        self.preference = output
        return output.cpu().detach().numpy()

    def local_update(self):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()

        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            output = self.model(self.r_i)
            loss = criterion(output[self.mask], self.r_i[self.mask])
            loss.backward()
            optimizer.step()

    def get_flatten_weights(self):
        weights = self.model.state_dict()
        return torch.cat([v.view(-1) for k, v in weights.items()] + [self.upi])

    def get_weights(self):
        weights = self.model.state_dict()
        weights = {k: v.cpu() for k, v in weights.items()}
        return weights

    def set_weights(self, weights):
        self.model.load_state_dict(weights)
