import argparse
import numpy as np
from caching import fl_cache, random_cache
from communication import Communication
from drl import DuelingDQN, DuelingDQNAgent
from environment import Environment
from vehicle import Vehicle
from delivery import random_delivery, greedy_delivery
import pickle as pkl
import json
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

save_dir = os.path.join("results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(save_dir, exist_ok=True)


writer = SummaryWriter(log_dir=save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="MCFED DRL Simulation")
    parser.add_argument("--min_velocity", type=int, default=5)
    parser.add_argument("--max_velocity", type=int, default=10)
    parser.add_argument("--std_velocity", type=float, default=2.5)
    parser.add_argument("--road_length", type=int, default=2000)
    parser.add_argument("--road_width", type=int, default=10)
    parser.add_argument("--rsu_coverage", type=int, default=500)
    parser.add_argument("--rsu_capacity", type=int, default=100)
    parser.add_argument("--num_rsu", type=int, default=4)
    parser.add_argument("--num_vehicles", type=int, default=15)
    parser.add_argument("--time_step", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=2)
    parser.add_argument("--time_step_per_round", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel_update", type=bool, default=False)
    parser.add_argument("--content_size", type=int, default=800)
    parser.add_argument("--cloud_rate", type=float, default=1e6)
    parser.add_argument("--fiber_rate", type=float, default=2e6)
    parser.add_argument(
        "--content_handler", type=str, default="fl", choices=["fl", "random"]
    )
    parser.add_argument(
        "--va_decision", type=str, default="drl", choices=["random", "drl", "greedy"]
    )

    return parser.parse_args()


def main():
    args = parse_args()

    env = Environment(
        min_velocity=args.min_velocity,
        max_velocity=args.max_velocity,
        std_velocity=args.std_velocity,
        road_length=args.road_length,
        road_width=args.road_width,
        rsu_coverage=args.rsu_coverage,
        rsu_capacity=args.rsu_capacity,
        num_rsu=args.num_rsu,
        num_vehicles=args.num_vehicles,
        time_step=args.time_step,
        writer=writer,
    )

    global_delays = []
    global_request = []
    global_hits = []

    agent = DuelingDQNAgent(
        state_dim=env.state_dim,
        num_vehicles=args.num_vehicles,
        num_rsu=args.num_rsu,
        device=args.gpu,
        lr=0.0001,
        writer=writer,
    )

    for timestep in range(args.num_rounds * args.time_step_per_round):
        env.reset()
        # Round trigger
        if timestep % args.time_step_per_round == 0:

            # sampling round's requests
            env.generate_request(args)

            # Perform FL
            if args.content_handler == "fl":
                fl_cache(args, env, timestep, env.coverage)
            elif args.content_handler == "random":
                random_cache(args, env, timestep, env.coverage)

        # get action matrix
        env.update_state(timestep, args)

        if args.va_decision == "random":
            actions = random_delivery(args)
            delays, total_request, total_hits = compute_delay(
                args, env, timestep, env.requests, actions
            )
        elif args.va_decision == "greedy":
            actions = greedy_delivery(args, env, timestep, env.requests)
            delays, total_request, total_hits = compute_delay(
                args, env, timestep, env.requests, actions
            )
        elif args.va_decision == "drl":
            state = env.state

            total_request = sum(
                1 for ts in env.requests if ts == timestep % args.time_step_per_round
            )

            # train DRL model
            for step in range(1, 50):
                actions = agent.select_action(state, args, timestep, env)

                delays, total_request, total_hits = compute_delay(
                    args, env, timestep, env.requests, actions
                )

                hit_rate = total_hits / total_request if total_request != 0 else 0

                next_state = env.state
                done = True if step == 50 else False
                agent.replay_buffer.push(
                    state,
                    actions,
                    (
                        1 / ((sum(delays) * 1e6 / total_request) + 0.0001) + hit_rate
                        if total_request != 0
                        else 0
                    ),
                    next_state,
                    done,
                )
                state = next_state
                agent.train(batch_size=256)
                if done:
                    break

            if timestep % 10 == 0:
                agent.update_target_network()
            # inference DRL model

        if len(delays) > 0:
            print(
                f"Timestep {timestep%args.time_step_per_round + 1}, avg delay: {np.mean(delays)}, hit rate: {total_hits/total_request}"
            )
            writer.add_scalar("avg_delay", np.mean(delays), timestep)
            writer.add_scalar("hit_rate", total_hits / total_request, timestep)
            global_delays.extend(delays)
            global_request.append(total_request)
            global_hits.append(total_hits)

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    # save results
    with open(os.path.join(save_dir, "results.pkl"), "wb") as f:
        pkl.dump(
            {
                "global_delays": global_delays,
                "global_request": global_request,
                "global_hits": global_hits,
                "avg_delay": np.mean(global_delays),
                "avg_hit_rate": np.mean(global_hits) / np.mean(global_request),
            },
            f,
        )
    print(f"Average delay: {np.mean(global_delays)}")
    print(f"Average hit rate: {np.mean(global_hits)/np.mean(global_request)}")


def compute_delay(args, env, timestep, requests, actions):
    delays = []

    total_request = 0
    total_hits = 0

    # Compute delay
    for vehicle_idx, ts in enumerate(requests):
        if ts == timestep % args.time_step_per_round:
            total_request += 1
            action = int(actions[vehicle_idx])
            local_rsu = env.reverse_coverage[vehicle_idx]
            requested_content = env.vehicles[vehicle_idx].request
            wireless_rate = env.communication.get_data_rate(vehicle_idx, local_rsu)

            wireless_delay = args.content_size / wireless_rate
            fiber_delay = args.content_size / args.fiber_rate
            cloud_delay = args.content_size / args.cloud_rate

            if action != 0:
                if requested_content in env.rsus[action - 1].cache:
                    total_hits += 1
                    if action - 1 == local_rsu:
                        delays.append(wireless_delay)
                    else:
                        delays.append(fiber_delay + wireless_delay)
                else:
                    delays.append(cloud_delay + wireless_delay)

            else:
                delays.append(cloud_delay + wireless_delay)
    return delays, total_request, total_hits


if __name__ == "__main__":
    main()
