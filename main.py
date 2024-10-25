import numpy as np
from tqdm import tqdm
from cache import random_cache, mcfed
from ddqn import DDNQAgent
from environment import Environment
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

from options import parse_args


class OutputHandler:
    def __init__(self, name, args):
        self.args = args
        self.created = datetime.datetime.now().strftime(f"%Y%m%d-%H%M%S")
        self.delays = []
        self.requests = []
        self.hits = []
        self.name = name
        self.save_dir = os.path.join(
            "logs",
            f"{self.name}_{args.rsu_capacity}_{args.num_vehicle}_{self.created}",
        )

        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.save_dir, flush_secs=1)

    def log(self, delay, total_hits, total_request, timestep):
        self.delays.extend(delay)
        self.requests.append(total_request)
        self.hits.append(total_hits)

        step_avg_delay = np.mean(delay) if len(delay) != 0 else 0
        step_hit_ratio = total_hits / total_request if total_request != 0 else 1

        self.writer.add_scalar(
            f"Average Tranmission Delay",
            step_avg_delay,
            timestep,
        )
        self.writer.add_scalar(
            f"Average Cache Hit ratio",
            step_hit_ratio,
            timestep,
        )

        print(
            f"[Timestep {timestep}] Average Tranmission Delay for {self.name}: ",
            step_avg_delay,
        )
        print(
            f"[Timestep {timestep}] Average Cache Hit ratio for {self.name}: ",
            step_hit_ratio,
        )


args = parse_args()


def main():

    drl_delivery_logger = OutputHandler("ddqn", args)

    env = Environment(
        args=args,
    )

    drl_agent = DDNQAgent(
        state_dim=env.state_dim,
        num_vehicle=args.num_vehicle,
        num_rsu=args.num_rsu,
        device=args.gpu,
        lr=0.001,
        writer=drl_delivery_logger.writer,
        args=args,
    )

    # Train the agent
    for episode in range(args.episode):

        total_reward = 0
        env.reset()

        for timestep in tqdm(range(args.num_rounds * args.time_step_per_round)):
            current_round = timestep // args.num_rounds
            step = timestep % args.time_step_per_round

            begin_round = step == 0

            if begin_round:
                # random_cache(env)  # random cache placement for train the delivery agent
                mcfed(env)

            # get state
            state = env.state

            # get actions
            actions = drl_agent.select_action(
                state,
            )

            # compute reward
            delays, total_request, total_hits = compute_delay(args, env, actions)

            env.step(actions)

            next_state = env.state

            reward = np.mean(delays) if len(delays) != 0 else 0
            reward = 1 / np.exp(reward)

            hit_ratio = total_hits / total_request

            drl_delivery_logger.writer.add_scalar(
                "Hit Ratio per step",
                hit_ratio,
                drl_agent.steps,
            )

            drl_delivery_logger.writer.add_scalar(
                "Avg delay", np.mean(delays), drl_agent.steps
            )

            done = timestep == args.num_rounds * args.time_step_per_round - 1

            drl_agent.memory.append((state, actions, reward, next_state, done))

            drl_agent.train(env.args.batch_size)

            total_reward += reward

            drl_agent.writer.add_scalar(
                "Reward per step",
                reward,
                drl_agent.steps,
            )

            if begin_round:
                drl_agent.update_target()


def compute_delay(args, env, actions):
    delays = []
    total_hits = 0

    # count request (!=0)
    total_request = np.count_nonzero(env.request)

    request_vehicle_ids = np.where(env.request != 0)[0]

    for vehicle_idx in request_vehicle_ids:
        action = int(actions[vehicle_idx])
        local_rsu = env.mobility.reverse_coverage[vehicle_idx]
        requested_content = np.where(env.request[vehicle_idx] != 0)[0][0]

        rsu_rate = env.channel.data_rate[vehicle_idx][1]
        bs_rate = env.channel.data_rate[vehicle_idx][0]

        rsu_delay = args.content_size / rsu_rate
        bs_delay = args.content_size / bs_rate
        fiber_delay = args.content_size / args.fiber_rate
        backhaul_delay = args.content_size / args.cloud_rate

        if action == 0:
            delay = bs_delay

        elif action == args.num_rsu + 1:
            delay = backhaul_delay + rsu_delay

        elif action - 1 == local_rsu and env.rsu[local_rsu].had(requested_content):
            delay = rsu_delay
            total_hits += 1
        elif env.rsu[action - 1].had(requested_content):
            delay = rsu_delay + fiber_delay
            total_hits += 1
        else:
            delay = bs_delay

        delays.append(delay)

    return delays, total_request, total_hits


if __name__ == "__main__":
    main()
