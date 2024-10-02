import numpy as np
from cluster import clustering
from utils import average_weights
import concurrent.futures


def random_cache(args, env, timestep, coverage):

    for rsu in env.rsu:
        rsu.cache = np.random.choice(
            env.content_library.total_items, rsu.capacity, replace=False
        )


def fl_cache(args, env, timestep, coverage):
    print(f"Round {timestep // args.time_step_per_round + 1}, performing FL")

    # TODO: Select vehicles to join the Federated Learning
    selected_vehicles = env.vehicles

    # Local update:
    if args.parallel_update:

        def local_update(vehicle):
            vehicle.local_update()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(local_update, selected_vehicles)
    else:
        for vehicle in selected_vehicles:
            vehicle.local_update()

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
