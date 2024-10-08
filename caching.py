import numpy as np
from cluster import clustering
from utils import average_weights
import concurrent.futures
from tqdm import tqdm


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def random_cache(args, env, timestep, coverage):
    for rsu in env.rsus:
        rsu.cache = np.random.choice(
            env.library.max_movie_id + 1, rsu.capacity, replace=False
        )


def fl_cache(args, env, timestep, coverage):
    print(f"Round {timestep // args.time_step_per_round + 1}, performing FL")

    # Select vehicles to join the Federated Learning
    selected_vehicles = []
    for idx, vehicle in enumerate(env.vehicles):
        next_location = vehicle.velocity + vehicle.position
        local_rsu = env.rsus[env.reverse_coverage[idx]]

        if abs(next_location - local_rsu.position) < args.rsu_coverage:
            selected_vehicles.append(idx)

    for rsu_idx, rsu in enumerate(env.rsus):
        if rsu.cluster is None:
            for vehicle_idx in coverage[rsu_idx]:
                env.vehicles[vehicle_idx].set_weights(rsu.model.state_dict())
        else:
            print()
            for vehicle_idx in coverage[rsu_idx]:
                similar = 0
                vehicle_flatten_weights = (
                    env.vehicles[vehicle_idx].get_flatten_weights().cpu().numpy()
                )
                new_weights = None
                for cluster in rsu.cluster:
                    centroid = cluster["centroid"]
                    cosine = cosine_similarity(vehicle_flatten_weights, centroid)
                    if cosine > similar:
                        similar = cosine
                        new_weights = cluster["weights"]

                if new_weights is not None:
                    env.vehicles[vehicle_idx].set_weights(new_weights)

    # Local update:
    if args.parallel_update:
        pbar = tqdm(total=len(selected_vehicles), desc="Local update")

        def local_update(vehicle):
            vehicle.local_update(round=timestep // args.time_step_per_round)
            pbar.update()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(local_update, [env.vehicles[i] for i in selected_vehicles])
    else:
        for vehicle in tqdm(selected_vehicles, desc="Local update"):
            env.vehicles[vehicle].local_update(
                round=timestep // args.time_step_per_round
            )

    # Clustering
    for i in range(args.num_rsu):
        # get the selected vehicles in the coverage area
        vehicle_ids = [j for j in coverage[i] if j in selected_vehicles]

        # get weights of the selected vehicles
        flattened_weights = [
            env.vehicles[idx].get_flatten_weights() for idx in vehicle_ids
        ]

        # do clustering
        if len(coverage[i]) >= args.num_clusters:
            clusters, _, centroids = clustering(args.num_clusters, flattened_weights)
        else:
            clusters, _, centroids = clustering(1, flattened_weights)

        clusters_save_in_rsu = []

        for idx, cluster in enumerate(clusters):
            vehicles_id_in_cluster = []
            for v_id in cluster:
                vehicles_id_in_cluster.append(vehicle_ids[v_id])

            # update the weights of the vehicles in the cluster
            avg_weights = average_weights(
                [env.vehicles[v_id].get_weights() for v_id in vehicles_id_in_cluster]
            )

            # update the weights of the vehicles in the cluster
            for v_id in vehicles_id_in_cluster:
                env.vehicles[v_id].set_weights(avg_weights)

            clusters_save_in_rsu.append(
                {
                    "centroid": centroids[idx],
                    "weights": avg_weights,
                }
            )

        env.rsus[i].cluster = clusters_save_in_rsu

    # Cache replacement
    for r in range(args.num_rsu):
        predictions = [env.vehicles[i].predict() for i in coverage[r]]
        if len(predictions) == 0:
            continue
        popularity = np.mean(predictions, axis=0)
        cache = np.argsort(popularity)[::-1][: env.rsus[r].capacity]
        env.rsus[r].cache = cache
