import json
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm


def load_glove(
    words=None, glove_path="./glove.6B/glove.6B.50d.txt", save_path="./data/glove.json"
):
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)

    glove = {}
    with open(glove_path, "r", encoding="utf8") as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            if words is None or word in words:
                vector = np.asarray(values[1:], dtype="float32")
                glove[word] = vector.tolist()

    with open(save_path, "w") as f:
        json.dump(glove, f)

    return glove


def load_dataset(path):
    dataset_path = "./data/dataset.pkl"
    if os.path.exists(dataset_path):
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    genres_list = load_genres(os.path.join(path, "u.genre"))
    ratings = load_ratings(os.path.join(path, "u.data"))
    movies = load_movies(os.path.join(path, "u.item"), genres_list)
    users = load_users(os.path.join(path, "u.user"))

    dataset, ratings = preprocess_data(movies, ratings, users, genres_list)
    glove = load_glove(words=genres_list)

    sparse_vecs, semantic_vecs = create_vectors(dataset, genres_list, glove)
    cosine_sparse_matrix, cosine_semantic_matrix = compute_cosine_similarities(
        sparse_vecs, semantic_vecs
    )

    dataset = {
        "sparse_vec": sparse_vecs,
        "semantic_vec": semantic_vecs,
        "cosine_sparse": cosine_sparse_matrix,
        "cosine_semantic": cosine_semantic_matrix,
        "ratings": ratings,
        "users": users,
    }

    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset


def load_genres(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    return [
        d.strip().split("|")[0].lower().replace("'", "") if "oir" not in d else "noir"
        for d in data
    ][
        1:
    ]  # skip the "unknown" genre


def load_ratings(file_path):
    ratings = pd.read_csv(
        file_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    ratings = (
        ratings.dropna()
        .drop_duplicates(subset=["user_id", "item_id"])
        .sort_values(by=["timestamp"])
    )
    return ratings


def load_movies(file_path, genres_list):
    movies = pd.read_csv(
        file_path,
        sep="|",
        encoding="latin-1",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            *genres_list,
        ],
    )
    return movies


def load_users(file_path):
    users = pd.read_csv(
        file_path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return users.drop(["zip_code"], axis=1)


def preprocess_data(movies, ratings, users, genres_list):
    dataset = movies.drop(
        ["release_date", "title", "video_release_date", "IMDb_URL"], axis=1
    )
    dataset = (
        dataset[dataset["unknown"] != 1]
        .drop(["unknown"], axis=1)
        .drop_duplicates(subset="item_id")
    )

    ratings = ratings[
        ratings["item_id"].isin(dataset["item_id"])
        & ratings["user_id"].isin(users["user_id"])
    ]

    min_id = dataset["item_id"].min()
    dataset["item_id"] -= min_id
    ratings["item_id"] -= min_id

    substraction = 0
    for i in range(dataset["item_id"].max() + 1):
        if i not in dataset["item_id"].values:
            substraction += 1
        dataset.loc[dataset["item_id"] == i, "item_id"] -= substraction
        ratings.loc[ratings["item_id"] == i, "item_id"] -= substraction

    return dataset, ratings


def create_vectors(dataset, genres_list, glove):
    sparse_vecs = {}
    semantic_vecs = {}

    for index, row in dataset.iterrows():
        sparse_vec = row[1:].to_numpy()
        semantic_vec = [glove[genre] for genre in genres_list if row[genre] == 1]
        semantic_vec = np.mean(semantic_vec, axis=0)

        sparse_vecs[row["item_id"]] = sparse_vec
        semantic_vecs[row["item_id"]] = semantic_vec

    dataset.drop(genres_list, axis=1, inplace=True)
    return sparse_vecs, semantic_vecs


def compute_cosine_similarities(sparse_vecs, semantic_vecs):
    item_ids = sorted(sparse_vecs.keys())
    sparse_matrix = np.array([sparse_vecs[item_id] for item_id in item_ids])
    semantic_matrix = np.array([semantic_vecs[item_id] for item_id in item_ids])

    sparse_norms = np.linalg.norm(sparse_matrix, axis=1)
    semantic_norms = np.linalg.norm(semantic_matrix, axis=1)

    cosine_sparse_matrix = np.dot(sparse_matrix, sparse_matrix.T) / np.outer(
        sparse_norms, sparse_norms
    )
    cosine_semantic_matrix = np.dot(semantic_matrix, semantic_matrix.T) / np.outer(
        semantic_norms, semantic_norms
    )

    return cosine_sparse_matrix, cosine_semantic_matrix
