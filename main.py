import argparse
import concurrent.futures
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
    parser.add_argument("--road_length", type=int, default=2000)
    parser.add_argument("--road_width", type=int, default=10)
    parser.add_argument("--rsu_coverage", type=int, default=400)
    parser.add_argument("--rsu_capacity", type=int, default=20)
    parser.add_argument("--num_rsu", type=int, default=5)
    parser.add_argument("--num_vehicles", type=int, default=40)
    parser.add_argument("--time_step", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=5)
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

    for _ in range(1):
        distance_matrix = cal_distance_matrix(env.vehicles, env.rsu)
        env.communication = Communication(distance_matrix)

        coverage = {k: [] for k in range(args.num_rsu)}
        for i in range(args.num_vehicles):
            for j in range(args.num_rsu):
                if env.communication.distance_matrix[i][j] < args.rsu_coverage // 2:
                    coverage[j].append(i)

        selected_vehicles = env.vehicles

        def local_update(vehicle):
            vehicle.local_update()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(local_update, selected_vehicles)

        weights = [vehicle.get_weights() for vehicle in selected_vehicles]
        flattened_weights = [
            np.concatenate([np.array(v).flatten() for v in w.values()]) for w in weights
        ]

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

        for cluster in clusters:
            cluster_weights = average_weights([weights[i] for i in cluster])
            for idx in cluster:
                env.vehicles[idx].set_weights(cluster_weights)

        for r in range(args.num_rsu):
            predictions = [
                env.vehicles[i].predict().detach().numpy() for i in coverage[r]
            ]
            popularity = np.mean(predictions, axis=0)
            cache = np.argsort(popularity)[::-1][: env.rsu[r].capacity]
            env.rsu[r].cache = cache

        for rsu in env.rsu:
            print(rsu.cache)

        env.step()


if __name__ == "__main__":
    main()
