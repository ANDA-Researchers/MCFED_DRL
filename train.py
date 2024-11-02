import time
import numpy as npd
import torch
from tqdm import tqdm
from cache import random_cache, mcfed
from module.ddqn import BDQNAgent
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

    agent = BDQNAgent(
        state_dim=env.state_dim,
        num_actions=args.num_vehicle,
        action_dim=args.num_rsu + 2,
        hidden_dim=512,
        gamma=args.gamma,
        lr=args.lr,
        capacity=args.capacity,
        batch_size=args.batch_size,
        target_update=args.target_update,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        device=args.device,
        logger=logger,
        dueling=True,
    )
    reward_tracking = []

    # Train the agent
    for episode in tqdm(range(1, args.episodes + 1)):

        total_reward = 0
        state = env.reset()

        for step in tqdm(range(args.training_steps), leave=False):
            random_cache(env)

            action = agent.act(state)

            next_state, reward = env.step(action)

            agent.memory.push(
                state.unsqueeze(0).to("cpu"),
                action.unsqueeze(0).to("cpu"),
                next_state.unsqueeze(0).to("cpu"),
                reward.unsqueeze(0).to("cpu"),
            )

            if step % 4 == 0:
                agent.learn()

            total_reward += reward

            reward_tracking.append(reward)

            state = next_state

        agent.logger.add_scalar("Reward", total_reward / args.training_steps, episode)

    # Save the model
    torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "model.pth"))
    logger.close()


if __name__ == "__main__":
    main()
