import time
import numpy as npd
import torch
from tqdm import tqdm
from a2c import A2C
from cache import random_cache, mcfed
from module.ddqn import DDQN
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
    state = env.reset()
    # Train the agent
    for episode in tqdm(range(1, 30 + 1)):

        total_reward = 0
        mcfed(env)

        for step in tqdm(range(10), leave=False):

            action, action_prob = agent.act(state)

            next_state, reward = env.step(action)

            total_reward += reward

            reward_tracking.append(reward)

            state = next_state

        agent.logger.add_scalar("Reward", total_reward / args.training_steps, episode)


if __name__ == "__main__":
    main()
