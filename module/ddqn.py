"""
Big thanks to the authors of the following papers:
- "Action Branching Architectures for Deep Reinforcement Learning" 
by Pierre-Luc Bacon, Jean Harb, Martin G. Bellemare, Doina Precup
"""

import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayMemory(object):
    def __init__(self, capacity):
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        return self.Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class BDQN(nn.Module):
    """
    Action branching DQN
    """

    def __init__(
        self,
        state_dim,
        num_actions,
        action_dim,
        hidden_dim=128,
        dueling=False,
        reduce="mean",
    ):
        super(BDQN, self).__init__()

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dueling = dueling
        self.reduce = reduce

        self.common = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_branches = nn.ModuleList(
            [nn.Linear(hidden_dim, action_dim) for n in range(num_actions)]
        )

        if dueling:
            self.value_branch = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        pass

    def forward(self, state):
        out = self.common(state)

        action_scores = [actor_branch(out) for actor_branch in self.actor_branches]

        if self.dueling:
            value = self.value_branch(out)

            if self.reduce == "mean":
                action_scores = [
                    value + (score - score.mean(dim=-1, keepdim=True))
                    for score in action_scores
                ]
            elif self.reduce == "max":
                action_scores = [
                    value + (score - score.max(dim=-1, keepdim=True)[0])
                    for score in action_scores
                ]

        return action_scores


class BDQNAgent:
    def __init__(
        self,
        state_dim,
        num_actions,
        action_dim,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
        capacity=10000,
        batch_size=64,
        target_update=10,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device="cpu",
        logger=None,
        dueling=False,
        reduce="mean",
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.target_update = target_update
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device
        self.logger = logger
        self.dueling = dueling
        self.reduce = reduce

        self.policy_net = BDQN(
            state_dim, num_actions, action_dim, hidden_dim, dueling, reduce
        ).to(device)

        self.target_net = BDQN(
            state_dim, num_actions, action_dim, hidden_dim, dueling, reduce
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = ReplayMemory(capacity)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.steps = 0

    def act(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)
                action_scores = self.policy_net(state)
                action = torch.stack([score.argmax() for score in action_scores])
        else:
            action = torch.tensor(
                [random.randrange(self.action_dim) for _ in range(self.num_actions)],
                device=self.device,
            )

        # Epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.policy_net.train()
        self.target_net.eval()

        # Sample a batch from the replay memory
        batch = self.memory.sample(self.batch_size)

        states = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.action).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)

        # Q(s, a)
        action_values = self.policy_net(states)
        q_values = torch.cat(action_values, dim=1)

        # Q(s', a')
        next_acion_values = self.policy_net(next_states)
        next_q_values = torch.cat(next_acion_values, dim=1)

        best_next_actions = torch.cat(
            [q.max(dim=-1)[1] for q in next_acion_values], dim=1
        )  # (batch_size, num_actions)

        # Q_target(s', argmax_a' Q(s', a'))
        with torch.no_grad():
            target_next_action_values = self.target_net(next_states)
            target_next_q_values = torch.cat(
                target_next_action_values, dim=1
            )  # batch_size, num_actions , action_dim

            target_next_q_values = target_next_q_values.gather(
                2, best_next_actions.unsqueeze(-1)
            ).squeeze(-1)

            # Compute target Q-values
            target_q_values = rewards.unsqueeze(-1) + self.gamma * torch.mean(
                target_next_q_values, dim=-1, keepdim=True
            )  # Shape: (batch_size, 1)

            target_q_values = target_q_values.repeat(1, self.num_actions)

        # Gather the Q-values for the actions taken from q_values

        actions = actions.squeeze()  # (batch_size, num_actions)

        # get the q_values for the actions taken, q_values is a tensors of shape (batch_size, num_actions, action_dim)
        q_values_taken = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Calculate the loss
        loss = F.mse_loss(q_values_taken, target_q_values)

        if self.logger is not None:
            self.logger.add_scalar("Loss", loss.item(), self.steps)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        # Update the target network
        if self.steps % self.target_update == 0:
            self.update_target()

    def update_target(self):
        """Update the target network with the weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
