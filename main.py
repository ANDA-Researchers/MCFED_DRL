import argparse
import numpy as np
from sklearn import neighbors
from caching import fl_cache, random_cache
from communication import Communication
from environment import Environment
from vehicle import Vehicle
from delivery import random_delivery


def parse_args():
    parser = argparse.ArgumentParser(description="MCFED DRL Simulation")
    parser.add_argument("--min_velocity", type=int, default=5)
    parser.add_argument("--max_velocity", type=int, default=10)
    parser.add_argument("--std_velocity", type=float, default=2.5)
    parser.add_argument("--road_length", type=int, default=2500)
    parser.add_argument("--road_width", type=int, default=10)
    parser.add_argument("--rsu_coverage", type=int, default=500)
    parser.add_argument("--rsu_capacity", type=int, default=20)
    parser.add_argument("--num_rsu", type=int, default=5)
    parser.add_argument("--num_vehicles", type=int, default=40)
    parser.add_argument("--time_step", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=5)
    parser.add_argument("--time_step_per_round", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--parallel_update", type=bool, default=True)
    parser.add_argument("--content_size", type=int, default=800)
    parser.add_argument("--cloud_rate", type=float, default=10e6)
    parser.add_argument("--fiber_rate", type=float, default=15e6)
    parser.add_argument(
        "--content_handler", type=str, default="fl", choices=["fl", "random"]
    )
    parser.add_argument(
        "--va_decision", type=str, default="random", choices=["random", "drl"]
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
    )

    global_delays = []

    for timestep in range(args.num_rounds * args.time_step_per_round):

        # Get channel status
        env.communication = Communication(env.vehicles, env.rsu)

        # Compute hypotenuse distance
        hypotenuse_distance = np.sqrt(
            (args.rsu_coverage // 2) ** 2 + (args.road_width // 2) ** 2
        )

        # Get coverage
        coverage = {k: [] for k in range(args.num_rsu)}
        for i in range(args.num_vehicles):
            for j in range(args.num_rsu):
                if env.communication.distance_matrix[i][j] < hypotenuse_distance:
                    coverage[j].append(i)

        reverse_coverage = {v: k for k, l in coverage.items() for v in l}

        # Round trigger
        if timestep % args.time_step_per_round == 0:

            # sampling round's requests
            requests = [
                int(reg)
                for reg in np.random.uniform(
                    0, args.time_step_per_round, args.num_vehicles
                )
            ]

            # Perform FL
            if args.content_handler == "fl":
                fl_cache(args, env, timestep, coverage)
            elif args.content_handler == "random":
                random_cache(args, env, timestep, coverage)

        # State: request matrix
        request_matrix = np.zeros(
            (args.num_vehicles, len(env.content_library.total_items))
        )
        for idx, ts in enumerate(requests):
            if ts == timestep % args.time_step_per_round:
                request_matrix[idx][env.vehicles[idx].request] = 1

        # get action matrix
        actions = random_delivery(args)

        # actions = greedy_delivery(args, env, timestep, reverse_coverage, requests)

        delays = []

        # Compute delay
        for vehicle_idx, ts in enumerate(requests):
            if ts == timestep % args.time_step_per_round:
                action = int(actions[vehicle_idx])
                local_rsu = reverse_coverage[vehicle_idx]
                requested_content = env.vehicles[vehicle_idx].request
                wireless_rate = env.communication.get_data_rate(vehicle_idx, local_rsu)

                wireless_delay = args.content_size / wireless_rate
                fiber_delay = args.content_size / args.fiber_rate
                cloud_delay = args.content_size / args.cloud_rate

                if action != 0:
                    if requested_content in env.rsu[action - 1].cache:
                        if action - 1 == local_rsu:
                            delays.append(wireless_delay)
                        else:
                            delays.append(fiber_delay + wireless_delay)
                    else:
                        delays.append(cloud_delay + wireless_delay)

                else:
                    delays.append(cloud_delay + wireless_delay)

        if len(delays) > 0:
            print(
                f"Timestep {timestep%args.time_step_per_round + 1}, avg delay: {np.mean(delays)}"
            )
            global_delays.extend(delays)

        # Update the environment
        env.step()

    print(f"Average delay: {np.mean(global_delays)}")


def greedy_delivery(args, env, timestep, reverse_coverage, requests):
    actions = np.zeros(args.num_vehicles)
    for vehicle_idx, ts in enumerate(requests):
        if ts == timestep % args.time_step_per_round:
            requested_content = env.vehicles[vehicle_idx].request
            all_rsus = env.rsu
            local_rsu_idx = reverse_coverage[vehicle_idx]
            local_rsu = all_rsus[local_rsu_idx]
            neighbor_rsus = [
                (rsu, idx) for idx, rsu in enumerate(all_rsus) if idx != local_rsu
            ]

            if requested_content in local_rsu.cache:
                actions[vehicle_idx] = local_rsu_idx + 1
            else:
                for rsu, idx in neighbor_rsus:
                    if requested_content in rsu.cache:
                        actions[vehicle_idx] = idx + 1
                        break
    return actions


if __name__ == "__main__":
    main()
