import time
import numpy as npd
import torch
from tqdm import tqdm
from a2c import A2C
from cache import random_cache, mcfed
from ddqn import DDQN
from environment import Environment
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from options import parse_args

args = parse_args()


def main():

    created_at = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S")

    save_dir = os.path.join(
        "logs",
        f"[{created_at}] {args.num_rsu}_{args.rsu_capacity}_{args.num_vehicle}",
    )

    logger = SummaryWriter(log_dir=save_dir, flush_secs=1)

    env = Environment(
        args=args,
    )

    # agent = DDQN(
    #     state_dim=env.state_dim,
    #     action_dim=env.action_dim,
    #     hidden_dim=512,
    #     gamma=args.gamma,
    #     lr=args.lr,
    #     capacity=args.capacity,
    #     batch_size=args.batch_size,
    #     target_update=args.target_update,
    #     epsilon_decay=args.epsilon_decay,
    #     epsilon_min=args.epsilon_min,
    #     device=args.device,
    #     logger=logger,
    # )

    agent = A2C(
        state_dim=env.state_dim,
        action_dim=args.num_rsu + 2,
        num_actions=args.num_vehicle,
        hidden_dim=512,
        gamma=args.gamma,
        lr=args.lr,
        device=args.device,
        logger=logger,
        capacity=args.capacity,
        batch_size=args.batch_size,
    )

    reward_tracking = []

    # Train the agent
    for episode in range(args.episodes):

        total_reward = 0
        state = env.reset()

        for _ in tqdm(range(50000)):
            random_cache(env)

            action, action_prob = agent.act(state)

            next_state, reward = env.step(action)

            agent.memory.push(
                state.unsqueeze(0).to("cpu"),
                action.to("cpu"),
                next_state.unsqueeze(0).to("cpu"),
                reward.unsqueeze(0).to("cpu"),
                action_prob.to("cpu"),
            )

            agent.learn()

            total_reward += reward

            reward_tracking.append(reward)

            state = next_state

            if agent.steps % 100 == 0:
                agent.logger.add_scalar(
                    "Avg reward", sum(reward_tracking[-100:]) / 100, agent.steps
                )

        agent.logger.add_scalar("Cummulative reward", total_reward, episode)


if __name__ == "__main__":
    main()
