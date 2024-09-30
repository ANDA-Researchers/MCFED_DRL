from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def clustering(clients_num, flattened_weights):
    if len(flattened_weights) == 0:
        return [], 0
    else:
        if clients_num is None:
            silhouette_score_lst = []
            for n_clusters in range(2, len(flattened_weights)):
                kmeans = KMeans(n_clusters=n_clusters)
                cluster_labels = kmeans.fit_predict(flattened_weights)
                silhouette_avg = silhouette_score(flattened_weights, cluster_labels)
                silhouette_score_lst.append(silhouette_avg)

            best_n_clusters = silhouette_score_lst.index(max(silhouette_score_lst)) + 2
        else:
            best_n_clusters = clients_num

        kmeans = KMeans(n_clusters=best_n_clusters)
        cluster_labels = kmeans.fit_predict(flattened_weights)

        clusters_veh = [
            [
                index
                for index, value in enumerate(cluster_labels)
                if value == sublist_index
            ]
            for sublist_index in range(best_n_clusters)
        ]

        return clusters_veh, best_n_clusters
