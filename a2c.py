import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "prob")
)


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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_actions, hidden_dim):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.num_actions = num_actions

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, num_actions * action_dim)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        logits = self.action_head(x)

        return logits.view(-1, self.num_actions, self.action_dim)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_head(x)

        return value


class A2C(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        num_actions,
        hidden_dim,
        gamma,
        lr,
        device,
        logger=None,
        capacity=20000,
        batch_size=32,
    ):
        self.steps = 0
        self.gamma = gamma
        self.device = device
        self.logger = logger
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.bach_size = batch_size

        self.actor = Actor(state_dim, action_dim, num_actions, hidden_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = ReplayMemory(capacity)

    def act(self, state):
        state = state.to(self.device).unsqueeze(0)

        # (batch_size, num_actions, action_dim)
        logits = self.actor(state)

        # (batch_size, num_actions, action_dim)
        probs = nn.Softmax(dim=-1)(logits)

        # (batch_size, num_actions)
        action = torch.distributions.Categorical(probs).sample()

        # (batch_size, num_actions)
        action_probs = probs.gather(2, action.unsqueeze(-1)).squeeze(-1)

        return action, action_probs

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device)
        rewards = torch.cat(batch.reward).to(self.device)
        action_probs = torch.cat(batch.prob).to(self.device)
