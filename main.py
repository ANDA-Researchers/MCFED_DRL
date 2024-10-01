import argparse
from audioop import avg
import concurrent.futures
import re
from turtle import delay
import numpy as np
from communication import Communication
from environment import Environment
from utils import cal_distance_matrix, average_weights
from cluster import clustering


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
    parser.add_argument("--content_size", type=int, default=800)
    parser.add_argument("--cloud_rate", type=float, default=1e6)
    parser.add_argument("--fiber_rate", type=float, default=15e6)
    parser.add_argument(
        "--content_handler", type=str, default="fl", choices=["fl", "random"]
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
        # Compute distance matrix
        distance_matrix = cal_distance_matrix(env.vehicles, env.rsu)

        # Get communication status
        env.communication = Communication(distance_matrix)

        # Get coverage
        coverage = {k: [] for k in range(args.num_rsu)}
        for i in range(args.num_vehicles):
            for j in range(args.num_rsu):
                if env.communication.distance_matrix[i][j] < np.sqrt(
                    (args.rsu_coverage // 2) ** 2 + (args.road_width // 2) ** 2
                ):
                    coverage[j].append(i)

        reverse_coverage = {v: k for k, l in coverage.items() for v in l}

        # Round trigger
        if timestep % args.time_step_per_round == 0:

            # Generate requests
            requests = [
                int(reg)
                for reg in np.random.uniform(
                    0, args.time_step_per_round, args.num_vehicles
                )
            ]

            # Perform FL
            FL(args, env, timestep, coverage)

            # Random cache replacement
            # random_cache(args, env, timestep, coverage)

        # request matrix
        request_matrix = np.zeros(
            (args.num_vehicles, len(env.content_library.total_items))
        )

        for idx, ts in enumerate(requests):
            if ts == timestep % args.time_step_per_round:
                request_matrix[idx][env.vehicles[idx].request()] = 1

        # get action matrix
        actions = np.random.random(args.num_vehicles * (args.num_rsu + 1)).reshape(
            args.num_vehicles, (args.num_rsu + 1)
        )

        actions = np.argmax(actions, axis=1)

        delays = []

        # for each vehicle that requests in this time step, print their desired request
        for idx, ts in enumerate(requests):
            if ts == timestep % args.time_step_per_round:
                action = actions[idx]
                vehicle = env.vehicles[idx]
                local_rsu = reverse_coverage[idx]
                requested_content = vehicle.request()
                wireless_rate = env.communication.calculate_V2R_data_rate(
                    vehicle, env.rsu[local_rsu], idx, local_rsu
                )

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
                f"Timestep {timestep//args.time_step_per_round + 1}, avg delay: {np.mean(delays)}"
            )
            global_delays.extend(delays)
        # Update the environment
        env.step()

    print(f"Average delay: {np.mean(global_delays)}")


def random_cache(args, env, timestep, coverage):

    for rsu in env.rsu:
        rsu.cache = np.random.choice(
            env.content_library.total_items, rsu.capacity, replace=False
        )


def FL(args, env, timestep, coverage):
    print(f"Round {timestep // args.time_step_per_round + 1}, performing FL")

    # Select vehicles to join the Federated Learning
    selected_vehicles = env.vehicles

    # Local update: perform in parallel
    def local_update(vehicle):
        vehicle.local_update()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(local_update, selected_vehicles)

    # for vehicle in selected_vehicles:
    #     vehicle.local_update()

    # Get the weights then flatten
    weights = [vehicle.get_weights() for vehicle in selected_vehicles]
    flattened_weights = [
        np.concatenate([np.array(v).flatten() for v in w.values()]) for w in weights
    ]

    # Clustering
    clusters = []
    for i in range(args.num_rsu):
        if len(coverage[i]) >= args.num_clusters:
            cluster, _ = clustering(
                args.num_clusters, [flattened_weights[j] for j in coverage[i]]
            )
        else:
            cluster, _ = clustering(1, [flattened_weights[j] for j in coverage[i]])

        cluster = [[coverage[i][j] for j in c] for c in cluster]
        clusters.extend(cluster)

    # Perform global aggregation then download to vehicles
    for cluster in clusters:
        cluster_weights = average_weights([weights[i] for i in cluster])
        for idx in cluster:
            env.vehicles[idx].set_weights(cluster_weights)

    # Cache replacement
    for r in range(args.num_rsu):
        predictions = [env.vehicles[i].predict() for i in coverage[r]]
        if len(predictions) == 0:
            continue
        popularity = np.mean(predictions, axis=0)
        cache = np.argsort(popularity)[::-1][: env.rsu[r].capacity]
        env.rsu[r].cache = cache


if __name__ == "__main__":
    main()
