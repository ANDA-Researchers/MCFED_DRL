import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class ReplayMemory(object):
    def __init__(self, capacity):
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward", "prob")
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


class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, action_dim, hidden_dim=128):
        """
        state_dim: The dimension of the state space (input size to the network).
        action_counts: List with the number of discrete actions for each action dimension.
        """
        super(Actor, self).__init__()
        self.num_actions = num_actions

        # Shared layers for actor
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor heads for each action dimension
        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, action_dim) for n in range(num_actions)]
        )

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Get logits for each action dimension
        action_logits = [actor_head(x) for actor_head in self.actor_heads]

        return action_logits

    def get_action_and_log_prob(self, state):
        batch_size = state.size(0)
        action_logits = self.forward(state)

        # Sample actions for each dimension from categorical distributions
        action_probs = [F.softmax(logits, dim=-1) for logits in action_logits]
        action_distributions = [
            torch.distributions.Categorical(probs) for probs in action_probs
        ]

        actions = [dist.sample() for dist in action_distributions]
        action_log_probs = [dist.logits for dist in action_distributions]

        return torch.stack(actions).view(batch_size, -1), torch.stack(
            action_log_probs
        ).view(batch_size, self.num_actions, -1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        state_dim: The dimension of the state space (input size to the network).
        action_dim: The total dimension of the action space, representing each action dimension concatenated.
        """
        super(Critic, self).__init__()

        # Shared layers for critic
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output Q-value

    def forward(self, state, action):
        # Concatenate state and action as input to the Q network
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)

        return q_value


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
        update_target=1000,
        logger=None,
        capacity=20000,
        batch_size=32,
    ):
        self.n_actions = num_actions
        self.steps = 0
        self.gamma = gamma
        self.device = device
        self.logger = logger
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.update_target = update_target
        self.memory = ReplayMemory(capacity)

        self.actor = Actor(state_dim, num_actions, action_dim, hidden_dim).to(
            self.device
        )
        self.critic = Critic(state_dim, num_actions * action_dim, hidden_dim).to(
            self.device
        )

        self.critic_target = Critic(state_dim, num_actions * action_dim, hidden_dim).to(
            self.device
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.003)

        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, state):
        self.actor.eval()
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action_and_log_prob(state)
            action = action.squeeze().cpu()
            log_prob = log_prob.squeeze().cpu()

        return action, log_prob

    def one_hot(self, action):
        batch_size = action.size(0)
        action_dim = self.action_dim
        num_actions = self.num_actions

        # Create a tensor of zeros with the required shape
        one_hot_action = torch.zeros(
            batch_size, num_actions, action_dim, device=action.device
        )

        # Use advanced indexing to set the appropriate elements to 1
        indices = torch.arange(batch_size, device=action.device).unsqueeze(1)
        one_hot_action[
            indices, torch.arange(num_actions, device=action.device), action
        ] = 1

        # Reshape the tensor to the shape of the action
        one_hot_action = one_hot_action.view(batch_size, -1)
        return one_hot_action

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.steps += 1

        # Sample experience from memory
        batch = self.memory.sample(self.batch_size)

        state = torch.cat(batch.state).to(self.device)
        action = torch.cat(batch.action).to(self.device)
        next_state = torch.cat(batch.next_state).to(self.device)
        reward = torch.cat(batch.reward).to(self.device)
        behavior_action_probs = torch.cat(batch.prob).to(self.device)

        # Get action and log prob for current state
        action, action_log_prob = self.actor.get_action_and_log_prob(state)
        one_hot_action = self.one_hot(action)
        values = self.critic(state, one_hot_action)

        # Get action and log prob for next state (for value function update)
        next_action, next_action_log_prob = self.actor.get_action_and_log_prob(
            next_state
        )
        one_hot_next_action = self.one_hot(next_action)
        next_state_values = self.critic_target(next_state, one_hot_next_action)

        # Detach next state values for advantage calculation
        next_state_values_ = next_state_values.detach()
        q_values = reward + self.gamma * next_state_values_

        # Value loss calculation
        v_loss = F.mse_loss(values, q_values)

        # Calculate advantage
        advantage = q_values - values.detach()

        # Initialize the total policy loss
        p_loss = 0

        # Calculate policy loss for each action dimension
        for i in range(self.num_actions):
            # Get the log probability of the action taken
            log_prob = action_log_prob[:, i]

            # Get the log probability of the action taken under the behavior policy
            behavior_log_prob = behavior_action_probs[:, i]

            # Calculate the policy loss
            p_loss += -torch.mean(log_prob * advantage * (1 / behavior_log_prob))

        # Combine policy loss and value loss
        loss = v_loss + p_loss

        # Update actor and critic networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # Update target network if needed
        if self.steps % self.update_target == 0:
            self.update_target_network()

        if self.logger is not None:
            self.logger.add_scalar("loss", loss.item(), self.steps)
            self.logger.add_scalar("value_loss", v_loss.item(), self.steps)
            self.logger.add_scalar("policy_loss", p_loss.item(), self.steps)

    def update_target_network(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
