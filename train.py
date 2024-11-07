import datetime
import os

import torch
from tqdm import tqdm

from cache import random_cache
from logger import *
from module.bdqn import BDQNAgent
from simulation import Environment
from utils import load_args

args, configs = load_args()


def main():

    created_at = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S")

    save_dir = os.path.join(
        "logs",
        f"[{created_at}] {args.num_rsu}_{args.rsu_capacity}_{args.num_vehicle}",
    )

    if args.logger == "wandb":
        logger = WandbLogger(configs)
    elif args.logger == "tensorboard":
        logger = TensorboardLogger(save_dir)
    else:
        logger = None

    env = Environment(
        args=args,
    )

    agent = BDQNAgent(
        state_dim=env.state_dim,
        num_actions=args.num_vehicle,
        action_dim=args.num_rsu + 2,
        hidden_dim=args.hidden_dim,
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
    for episode in range(0, args.episodes):
        print(f"Episode: {episode}")

        total_reward = 0
        state, mask = env.reset()

        for step in tqdm(range(args.training_steps), leave=False, desc="Steps"):
            action = agent.act(state, mask)

            _, _, reward = env.step(action)

            random_cache(env)

            next_state, next_mask = env.state, env.mask

            agent.memory.push(
                state.unsqueeze(0).to("cpu"),
                action.unsqueeze(0).to("cpu"),
                next_state.unsqueeze(0).to("cpu"),
                reward.unsqueeze(0).to("cpu"),
                mask.unsqueeze(0).to("cpu"),
                next_mask.unsqueeze(0).to("cpu"),
            )

            if step % args.train_every == 0:
                loss = agent.learn()

                if agent.steps != -1:
                    agent.logger.log("Loss", loss, agent.steps)

            total_reward += reward

            reward_tracking.append(reward)

            state = next_state
            mask = next_mask

        agent.logger.log("Reward", total_reward / args.training_steps, episode)

    # Save the model
    torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "model.pth"))


if __name__ == "__main__":
    main()
