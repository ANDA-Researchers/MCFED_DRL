import torch
from tqdm import tqdm
from cache import random_cache
from module.bdqn import BDQNAgent
from simulation import Environment
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from options import load_args

args = load_args()


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
    for episode in range(0, args.episodes):
        print(f"Episode: {episode}")

        total_reward = 0
        state = env.reset()

        for step in tqdm(range(args.training_steps), leave=False, desc="Steps"):
            action = agent.act(state)

            _, reward = env.step(action)

            random_cache(env)

            next_state = env.state

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
