import time
import numpy as npd
import torch
from tqdm import tqdm
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
        f"{args.num_rsu}_{args.rsu_capacity}_{args.num_vehicle}_{created_at}",
    )

    logger = SummaryWriter(log_dir=save_dir, flush_secs=1)

    env = Environment(
        args=args,
    )

    agent = DDQN(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=128,
        gamma=args.gamma,
        lr=args.lr,
        capacity=args.capacity,
        batch_size=args.batch_size,
        target_update=args.target_update,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        device=args.device,
        logger=logger,
    )

    # Train the agent
    for episode in range(args.episodes):
        total_reward = 0
        state = env.reset()

        for round in tqdm(
            range(args.num_rounds), desc=f"Episode {episode}", unit="round"
        ):
            # Put cache replacement policy here
            mcfed(env)

            # Delivery phase
            for timestep in tqdm(
                range(args.time_step_per_round),
                desc=f"Round {round}",
                unit="time step",
            ):
                action = agent.act(state, sample=True)

                next_state, reward = env.step(action)

                # agent.memory.push(
                #     state.unsqueeze(0).to("cpu"),
                #     action.unsqueeze(0).to("cpu"),
                #     next_state.unsqueeze(0).to("cpu"),
                #     reward.unsqueeze(0).to("cpu"),
                # )

                # agent.learn()

                # total_reward += reward

                # state = next_state

        agent.logger.add_scalar(
            "Average reward", total_reward / args.num_rounds, episode
        )


if __name__ == "__main__":
    main()
