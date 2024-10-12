import numpy as np
from tqdm import tqdm
from cache import fl_cache, random_cache
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
            f"{args.content_handler}_{self.name}_{args.rsu_capacity}_{args.num_vehicles}_{self.created}",
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
        min_velocity=args.min_velocity,
        max_velocity=args.max_velocity,
        std_velocity=args.std_velocity,
        rsu_coverage=args.rsu_coverage,
        rsu_capacity=args.rsu_capacity,
        num_rsu=args.num_rsu,
        num_vehicles=args.num_vehicles,
        time_step=args.time_step,
        args=args,
        gpu=args.gpu,
    )

    drl_agent = DDNQAgent(
        state_dim=env.state_dim,
        num_vehicle=args.num_vehicles,
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
            step = timestep % args.num_rounds

            begin_round = step == 0

            if begin_round:
                if args.content_handler == "fl":
                    fl_cache(args, env)
                else:
                    random_cache(env)

            # get state
            state = env.state

            # get actions
            actions = drl_agent.select_action(
                state,
            )

            # compute reward
            delays, total_request, total_hits = compute_delay(args, env, actions)

            env.step()

            next_state = env.state

            reward = np.mean(delays) if len(delays) != 0 else 0
            reward = 1 / np.exp(reward)

            done = timestep == args.num_rounds * args.time_step_per_round - 1

            drl_agent.memory.append((state, actions, reward, next_state, done))

            if len(drl_agent.memory) > args.drl_step:
                drl_agent.train(32)

            total_reward += reward

            drl_agent.writer.add_scalar(
                "Reward per step",
                reward,
                drl_agent.steps,
            )

            if begin_round:
                drl_agent.update_target()

        drl_agent.writer.add_scalar(
            "Avg Total Reward",
            total_reward / (args.num_rounds * args.time_step_per_round),
            episode,
        )


def compute_delay(args, env, actions):
    delays = []
    total_hits = 0

    total_request = len(env.request)
    for vehicle_idx in env.request:
        action = int(actions[vehicle_idx])
        local_rsu = env.reverse_coverage[vehicle_idx]
        requested_content = env.vehicles[vehicle_idx].request
        wireless_rate = env.communication.get_data_rate(vehicle_idx, local_rsu)

        wireless_delay = args.content_size / wireless_rate
        fiber_delay = args.content_size / args.fiber_rate
        cloud_delay = args.content_size / args.cloud_rate

        if action == 0:
            delays.append(float("inf"))
        elif action == args.num_rsu + 1:
            delays.append(cloud_delay + wireless_delay)
        else:
            if requested_content in env.rsus[action - 1].cache:
                if action - 1 == local_rsu:
                    total_delay = wireless_delay
                else:
                    total_delay = fiber_delay + wireless_delay

                if total_delay < args.deadline:
                    total_hits += 1

                delays.append(total_delay)
            else:
                delays.append(cloud_delay + wireless_delay)

    return delays, total_request, total_hits


if __name__ == "__main__":
    main()
