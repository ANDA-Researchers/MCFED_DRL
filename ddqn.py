import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DDNQ(nn.Module):
    def __init__(self, input_size, num_vehicle, num_rsu):
        super(DDNQ, self).__init__()

        self.num_vehicle = num_vehicle
        self.num_rsu = num_rsu

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_vehicle * (num_rsu + 2))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # reshape to (batch_size, num_vehicle, num_rsu + 2)
        x = x.view(-1, self.num_vehicle, self.num_rsu + 2)

        # apply softmax to get probabilities best action for each vehicle
        x = F.softmax(x, dim=2)
        return x


class DDNQAgent:
    def __init__(
        self,
        state_dim,
        num_vehicle,
        num_rsu,
        device,
        args,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        writer=None,
    ):

        self.state_dim = state_dim
        self.num_vehicle = num_vehicle
        self.num_rsu = num_rsu
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.writer = writer
        self.steps = 0

        self.policy_net = DDNQ(state_dim, num_vehicle, num_rsu).to(device)
        self.target_net = DDNQ(state_dim, num_vehicle, num_rsu).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)

        self.args = args

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(
                0, self.num_vehicle * (self.num_rsu + 2), self.num_vehicle
            )
        state_tensor = (
            state.detach()
            if torch.is_tensor(state)
            else torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)
            actions = q_values.argmax(dim=1).cpu().numpy()

        # update epsilon
        self.update_epsilon()

        return self.filter_actions(actions, q_values.cpu().numpy())

    def filter_actions(self, actions, q_values=None):
        max_connections = self.args.max_connections
        connected_vehicle_indices = np.where(actions != 0)[0]

        # if q_values is not None:
        if len(connected_vehicle_indices) > max_connections:
            if q_values is None:  # randomly drop connections
                drop_indices = np.random.choice(
                    connected_vehicle_indices,
                    len(connected_vehicle_indices) - max_connections,
                    replace=False,
                )
            else:
                q_values = q_values[connected_vehicle_indices]
                drop_indices = np.argpartition(q_values, max_connections)[
                    max_connections:
                ]

            actions[connected_vehicle_indices[drop_indices]] = 0

        return actions

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_net(states)

        q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).argmax(dim=2)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(q_values, target_q_values)

        self.writer.add_scalar("Loss", loss.item(), self.steps)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        return loss.item()
