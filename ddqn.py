import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDQN(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=64,
        gamma=0.99,
        lr=1e-3,
        capacity=10000,
        batch_size=64,
        target_update=10,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device="cpu",
        logger=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update

        self.policy_net = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = MLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.steps = 0

        self.logger = logger

    def act(self, state, sample=False):
        # random sample
        if sample:
            return torch.randint(0, self.action_dim, (1,)).to(self.device)

        # epsilon gready
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).to(self.device)

        state = state.to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax(1)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.cat(batch.state).to(self.device)
        action = torch.cat(batch.action).to(self.device)
        next_state = torch.cat(batch.next_state).to(self.device)
        reward = torch.cat(batch.reward).to(self.device)

        state_action_values = self.policy_net(state).gather(1, action)
        next_state_values = self.target_net(next_state).max(1)[0].detach().unsqueeze(1)
        expected_state_action_values = reward + self.gamma * next_state_values

        loss = torch.nn.functional.mse_loss(
            state_action_values, expected_state_action_values
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        if self.steps % self.target_update == 0:
            self.update_target()

        if self.logger:
            self.logger.add_scalar("loss", loss.item(), self.steps)
            self.logger.add_scalar("epsilon", self.epsilon, self.steps)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
