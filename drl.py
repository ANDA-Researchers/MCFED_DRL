import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, num_vehicles, num_rsu):
        super(DuelingDQN, self).__init__()

        self.num_vehicles = num_vehicles
        self.num_rsu = num_rsu

        # Common feature extraction layers
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)

        # Value stream
        self.value_stream = nn.Linear(512, 1)  # Outputs a scalar value

        # Advantage stream
        self.advantage_stream = nn.Linear(
            512, num_vehicles * (num_rsu + 1)
        )  # Output dimension (num_vehicles, num_rsu + 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Separate into value and advantage streams
        value = self.value_stream(x).unsqueeze(1)  # Scalar value for each state

        # Advantage has shape (num_vehicles * (num_rsu + 1))
        advantage = self.advantage_stream(x)
        advantage = advantage.view(
            -1, self.num_vehicles, self.num_rsu + 1
        )  # Reshape to (num_vehicles, num_rsu + 1)

        # Combine value and advantage to get the Q-values
        q_values = value + (
            advantage - advantage.mean(dim=2, keepdim=True)
        )  # Broadcast across RSU decisions
        return q_values


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)


class DuelingDQNAgent:
    def __init__(
        self,
        state_dim,
        num_vehicles,
        num_rsu,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=500,
        device=-1,
        writer=None,
    ):
        self.device = torch.device(f"cuda:{device}" if device != -1 else "cpu")
        self.num_vehicles = num_vehicles
        self.num_rsu = num_rsu
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.policy_net = DuelingDQN(state_dim, num_vehicles, num_rsu).to(self.device)
        self.target_net = DuelingDQN(state_dim, num_vehicles, num_rsu).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(10000)

        self.writer = writer

    def select_action(self, state, args, timestep, env):
        self.steps_done += 1
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        if random.random() < self.epsilon:
            actions = np.zeros(args.num_vehicles)
            for vehicle_idx, ts in enumerate(env.requests):
                if ts == timestep % args.time_step_per_round:
                    requested_content = env.vehicles[vehicle_idx].request
                    all_rsus = env.rsus
                    local_rsu_idx = env.reverse_coverage[vehicle_idx]
                    local_rsu = all_rsus[local_rsu_idx]
                    neighbor_rsus = [
                        (rsu, idx)
                        for idx, rsu in enumerate(all_rsus)
                        if idx != local_rsu
                    ]

                    if requested_content in local_rsu.cache:
                        actions[vehicle_idx] = local_rsu_idx + 1
                    else:
                        for rsu, idx in neighbor_rsus:
                            if requested_content in rsu.cache:
                                actions[vehicle_idx] = idx + 1
                                break

            return actions  # Shape: (num_vehicles,)
        else:
            # Exploitation: Select actions based on Q-values
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state).squeeze(0)
                action = (
                    q_values.argmax(dim=1).cpu().numpy()
                )  # For each vehicle, select the best RSU
            return action  # Shape: (num_vehicles,)

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch from the experience replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.LongTensor(np.array(action)).to(
            self.device
        )  # Action matrix for vehicles (batch_size, num_vehicles)
        reward = (
            torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        )  # Reshaped to (batch_size, 1)
        done = (
            torch.FloatTensor(done).unsqueeze(1).to(self.device)
        )  # Reshaped to (batch_size, 1)

        # Compute current Q-values from the policy network
        q_values = self.policy_net(
            state
        )  # Shape: (batch_size, num_vehicles, num_rsu + 1)

        # Gather the Q-values for the selected actions
        # action.unsqueeze(2) ensures we have the correct action index for each vehicle (batch_size, num_vehicles, 1)
        q_values = q_values.gather(2, action.unsqueeze(2)).squeeze(
            2
        )  # (batch_size, num_vehicles)

        # Compute the target Q-values from the target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(2)[
                0
            ]  # Get max Q-values over RSUs for each vehicle (batch_size, num_vehicles)
            target_q_values = (
                reward + (1 - done) * self.gamma * next_q_values
            )  # (batch_size, num_vehicles)

        # Compute the loss (MSE between current Q-values and target Q-values)
        loss = F.mse_loss(q_values, target_q_values)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("Loss", loss.item(), self.steps_done)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
