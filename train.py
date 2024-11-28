import datetime
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from cache import random_cache_for_train_drl
from logger import *
from module.bdqn import BDQNAgent
from simulation import Environment
from utils import load_args

args, configs = load_args()

torch.manual_seed(0)
np.random.seed(0)


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
        tau=args.tau,
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
        state = env.reset()
        total_delay = 0
        total_hit_ratio = 0
        total_success_ratio = 0

        for step in tqdm(range(args.training_steps), leave=False, desc="Steps"):
            action = agent.act(state)

            agent.logger.log("Epsilon", agent.epsilon, agent.steps)

            _, reward, logs = env.step(action)

            avg_delay, total_request, total_hits, total_success = logs

            random_cache_for_train_drl(env)

            next_state = env.state

            agent.memory.push(
                state.unsqueeze(0).to("cpu"),
                action.unsqueeze(0).to("cpu"),
                next_state.unsqueeze(0).to("cpu"),
                reward.unsqueeze(0).to("cpu"),
            )

            if step % args.train_every == 0:
                loss = agent.learn()
                if agent.steps != -1:
                    agent.logger.log("Loss", loss, agent.steps)

            hit_ratio = total_hits / total_request
            success_ratio = total_success / total_request

            total_hit_ratio += hit_ratio
            total_success_ratio += success_ratio
            total_delay += avg_delay
            total_reward += reward

            reward_tracking.append(reward)

            state = next_state

        agent.logger.log("Reward", total_reward / args.training_steps, episode)
        agent.logger.log("Hit Ratio", total_hit_ratio / args.training_steps, episode)
        agent.logger.log(
            "Success Ratio", total_success_ratio / args.training_steps, episode
        )
        agent.logger.log("Avg Delay", total_delay / args.training_steps, episode)

        # save model
        if episode % 5 == 0:
            torch.save(
                agent.policy_net.state_dict(),
                os.path.join(save_dir, "model_{}.pth".format(episode)),
            )

        # Save the model
        torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "model.pth"))

        # Save args as json
        with open(os.path.join(save_dir, "args.json"), "w") as file:
            json.dump(vars(args), file, indent=4)


if __name__ == "__main__":
    main()
