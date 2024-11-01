import datetime
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from a2c import A2C
from torch.utils.tensorboard import SummaryWriter
import os

created_at = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S")


save_dir = os.path.join(
    "logs",
    f"[{created_at}] toy",
)

logger = SummaryWriter(log_dir=save_dir, flush_secs=1)


class ToyEnv(object):
    def __init__(self):
        self.M = 10
        self.N = 40
        self.K = 200

        self.server_capacity = 50
        self.content_size = 8e6
        self.mean_data_rate = 10e6
        self.std_data_rate = 1e6

        self.device = "cuda"

    def reset(self):
        self.data_rate = torch.normal(
            mean=self.mean_data_rate, std=self.std_data_rate, size=(self.N, self.M)
        )

        self.storage_status = torch.zeros(self.M, self.K)

        for server in range(self.M):
            caches = np.random.choice(self.K, size=self.server_capacity, replace=False)
            for cache in caches:
                self.storage_status[server, cache] = 1

        self.request = torch.zeros(self.N, self.K)

        for vehicle in range(self.N):
            content = np.random.choice(self.K, replace=False)
            self.request[vehicle, content] = 1

        self.state = torch.cat(
            [
                self.data_rate.flatten(),
                self.storage_status.flatten(),
                self.request.flatten(),
            ]
        ).to(self.device)

    def step(self, action):
        assert action.shape[0] == self.N
        total_reward = 0
        for vehicle_idx, action in enumerate(action):
            request = self.request[vehicle_idx].nonzero().flatten()
            server_idx = action

            if server_idx == self.M:
                total_reward += 2 * self.content_size / 1e6

            elif self.storage_status[server_idx, request] == 1:
                total_reward += (
                    self.content_size / self.data_rate[vehicle_idx, server_idx]
                )

        self.reset()

        return torch.tensor(-total_reward.mean()), self.state


env = ToyEnv()

env.reset()
state_dim = env.state.shape[0]
action_dim = env.M
num_actions = env.N

agent = A2C(
    state_dim=state_dim,
    action_dim=action_dim + 1,
    num_actions=num_actions,
    hidden_dim=512,
    gamma=0.99,
    lr=1e-3,
    device="cuda",
    logger=logger,
    capacity=1000,
    batch_size=32,
)

for episode in range(100):
    env.reset()
    total_reward = 0
    for step in range(200):
        action, action_prob = agent.act(env.state)
        reward, next_state = env.step(action)

        total_reward += reward

        agent.memory.push(
            env.state.unsqueeze(0).to("cpu"),
            action.unsqueeze(0).to("cpu"),
            next_state.unsqueeze(0).to("cpu"),
            reward.unsqueeze(0).to("cpu"),
            action_prob.unsqueeze(0).to("cpu"),
        )

        agent.learn()

    logger.add_scalar("reward", total_reward, episode)
