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
            "Transition",
            ("state", "action", "next_state", "reward"),
        )
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        if len(self.memory) == 0:
            raise ValueError("Memory is empty")

        transitions = random.sample(self.memory, batch_size)
        return self.Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6):
        self.Transition = namedtuple(
            "Transition",
            ("state", "action", "next_state", "reward"),
        )
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = (
            alpha  # Controls the level of prioritization (0 means no prioritization)
        )

    def push(self, *args):
        """Save a transition with the maximum priority."""
        max_priority = max(
            list(self.priorities), default=1.0
        )  # Ensure non-zero initial priority
        self.memory.append(self.Transition(*args))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of transitions based on priorities."""
        if len(self.memory) == 0:
            raise ValueError("Memory is empty")

        # Calculate sampling probabilities proportional to priorities
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sampling_probabilities = scaled_priorities / scaled_priorities.sum()

        # Sample indices based on the calculated probabilities
        indices = np.random.choice(
            len(self.memory), batch_size, p=sampling_probabilities
        )
        transitions = [self.memory[idx] for idx in indices]

        # Calculate importance-sampling weights
        total = len(self.memory)
        weights = (total * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability

        return self.Transition(*zip(*transitions)), indices, weights

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

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

        for layer in self.common:
            if isinstance(layer, nn.Linear):
                layer.register_full_backward_hook(self._backward_hook)

        self.advantage_branches = nn.ModuleList(
            [nn.Linear(hidden_dim, action_dim) for n in range(num_actions)]
        )

        if dueling:
            self.value_branch = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def _backward_hook(self, module, grad_input, grad_output):
        scale_factor = 1 / self.num_actions

        scaled_grad_input = tuple(
            grad * scale_factor if grad is not None else None for grad in grad_input
        )
        return scaled_grad_input

    def forward(self, state):
        out = self.common(state)

        action_scores = [
            advantage_branch(out) for advantage_branch in self.advantage_branches
        ]

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

        return action_scores  # batch, num_actions, action_dim


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
        mini_batch=1,
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
        self.prioritized = True

        self.policy_net = BDQN(
            state_dim, num_actions, action_dim, hidden_dim, dueling, reduce
        ).to(device)

        self.target_net = BDQN(
            state_dim, num_actions, action_dim, hidden_dim, dueling, reduce
        ).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory = (
            ReplayMemory(capacity)
            if not self.prioritized
            else PrioritizedReplayMemory(capacity)
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.steps = -1

    def act(self, state, inference=False):
        if random.random() > self.epsilon or inference:
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

        self.steps += 1

        self.policy_net.train()
        self.target_net.eval()

        # Sample a batch from the replay memory
        if self.prioritized:
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)

        states = torch.cat(batch.state).to(self.device)
        actions = torch.cat(batch.action).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.cat(batch.next_state).to(self.device)

        # Q(s, a)
        action_values = self.policy_net(states)
        q_values = torch.stack(action_values, dim=1)

        # Q(s', a')
        next_acion_values = self.policy_net(next_states)
        next_q_values = torch.stack(next_acion_values, dim=1)

        # argmax_a' Q(s', a')
        best_next_actions = next_q_values.argmax(dim=-1)

        # Q_target(s', argmax_a' Q(s', a'))
        with torch.no_grad():
            target_next_action_values = self.target_net(next_states)
            target_next_q_values = torch.stack(
                target_next_action_values, dim=1
            )  # batch_size, num_actions , action_dim

            target_next_q_values = target_next_q_values.gather(
                2, best_next_actions.unsqueeze(-1)
            ).squeeze(-1)

            # Compute target Q-values
            target_q_values = rewards.unsqueeze(-1) + self.gamma * torch.mean(
                target_next_q_values, dim=1, keepdim=True
            )  # Shape: (batch_size, 1)

            target_q_values = target_q_values.repeat(1, self.num_actions)

        # get the q_values for the actions taken, q_values is a tensors of shape (batch_size, num_actions, action_dim)
        q_values_taken = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # Calculate the loss
        loss = F.mse_loss(q_values_taken, target_q_values)

        # Update the priorities
        if self.prioritized:
            new_priorities = (
                abs(q_values_taken - target_q_values).detach().cpu().numpy()
            )
            new_priorities = np.sum(new_priorities, axis=1)
            self.memory.update_priorities(indices, new_priorities)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update the target network
        if self.steps % self.target_update == 0:
            self.update_target()

        return loss.item()

    def update_target(self):
        """Update the target network with the weights from the policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
