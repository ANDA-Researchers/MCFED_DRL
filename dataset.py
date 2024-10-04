import json
from turtle import goto
from urllib import request
import numpy as np
import pandas as pd
import os
from requests import get
import torch
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class Library:
    def __init__(self, num_clients) -> None:
        if "ml-1m" not in os.listdir("./data"):
            print("Dataset not found. Downloading from grouplens...")
            self.download_dataset()
        else:
            print("Dataset found. Loading...")

        self.ratings, self.user_info = load_ratings()
        self.sparse_vecs, self.semantic_vecs = load_movies()
        self.cosine_sparse_matrix, self.cosine_semantic_matrix = (
            compute_cosine_similarities(self.sparse_vecs, self.semantic_vecs)
        )

        self.max_user_id = self.ratings["user_id"].max()
        self.max_movie_id = self.ratings["movie_id"].max()
        self.available_users = []

    def genrate_clients(self):
        top_user = self.ratings["user_id"].value_counts().index[:200]
        choosen = np.random.choice(top_user, replace=False)
        while choosen in self.available_users:
            choosen = np.random.choice(top_user, replace=False)
        self.available_users.append(choosen)
        history, ratings = self.get_inputs(choosen)

        end_train = int(len(history) * 0.8)
        end_test = int(len(history) * 0.9)

        cosine_inputs = [self.cosine_semantic_matrix[his - 1] for his in history]
        semantic_inputs = [self.semantic_vecs[his - 1] for his in history]
        labels = [ratings[i] for i in range(len(ratings))]

        train_cosine = cosine_inputs[:end_train]
        train_semantic = semantic_inputs[:end_train]
        train_labels = labels[:end_train]
        train_ids = history[:end_train]

        test_cosine = cosine_inputs[end_train:end_test]
        test_semantic = semantic_inputs[end_train:end_test]
        test_labels = labels[end_train:end_test]
        test_ids = history[end_train:end_test]

        request_cosine = cosine_inputs[end_test:]
        request_semantic = semantic_inputs[end_test:]
        request_labels = labels[end_test:]
        request_ids = history[end_test:]

        return {
            "train": (train_cosine, train_semantic, train_labels, train_ids),
            "test": (test_cosine, test_semantic, test_labels, test_ids),
            "request": (request_cosine, request_semantic, request_labels, request_ids),
            "movies": torch.zeros((self.max_movie_id + 1)),
            "uid": choosen,
        }

    def get_inputs(self, user_id, request_ratio=0):
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        movie_ids = user_ratings["movie_id"].values
        ratings = user_ratings["rating"].values

        return torch.tensor(movie_ids), torch.tensor(ratings)


def load_movies():
    movies = pd.read_csv(
        "./data/ml-1m/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        encoding="latin1",
        engine="python",
    )

    movies.drop(["title"], axis=1, inplace=True)

    genres = set()
    for movie in movies["genres"]:
        genres.update(movie.split("|"))

    genres_new = []

    for genre in genres:
        if genre == "Children's":
            genres_new.append("Children".lower())
        elif genre == "Film-Noir":
            genres_new.append("Noir".lower())
        else:
            genres_new.append(genre.lower())

    genres = genres_new

    for genre in genres:
        movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x.lower() else 0)

    movies = movies.drop(["genres"], axis=1)

    glove = load_glove(words=genres)

    sparse_vecs = {}
    semantic_vecs = {}

    with tqdm(range(len(movies)), desc="Creating vectors...") as pbar:
        for index, row in movies.iterrows():
            sparse_vec = row[1:].to_numpy().astype(int)
            semantic_vec = [glove[genre] for genre in genres if row[genre] == 1]
            semantic_vec = np.mean(semantic_vec, axis=0)
            sparse_vecs[row["movie_id"]] = sparse_vec
            semantic_vecs[row["movie_id"]] = semantic_vec
            pbar.update(1)

    movies.drop(genres, axis=1, inplace=True)

    for movie_id in range(1, movies["movie_id"].max() + 1):
        if movie_id not in sparse_vecs:
            sparse_vecs[movie_id] = np.zeros(len(genres))
            semantic_vecs[movie_id] = np.zeros((50,))

    item_ids = sorted(sparse_vecs.keys())
    sparse_matrix = np.array([sparse_vecs[item_id] for item_id in item_ids])
    semantic_matrix = np.array([semantic_vecs[item_id] for item_id in item_ids])

    return sparse_matrix, semantic_matrix


def load_ratings():
    ratings = pd.read_csv(
        "./data/ml-1m/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
    )

    ratings = ratings.sort_values(by="timestamp")
    ratings["rating"] = (ratings["rating"]) / 5.0

    users_info = pd.read_csv(
        "./data/ml-1m/users.dat",
        sep="::",
        names=["user_id", "gender", "age", "occupation", "zip"],
        engine="python",
    )

    users_info = process_user_info(users_info)

    # ratings = ratings.merge(users_info, on="user_id")

    return ratings, users_info


def download_dataset():
    os.system("wget https://files.grouplens.org/datasets/movielens/ml-1m.zip -P ./data")
    os.system("unzip ./data/ml-1m.zip -d ./data")


def load_glove(
    words=None,
    glove_path="./data/glove.6B/glove.6B.50d.txt",
    save_path="./data/emb.json",
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


def compute_cosine_similarities(sparse_matrix, semantic_matrix):
    sparse_norms = np.linalg.norm(sparse_matrix, axis=1)
    semantic_norms = np.linalg.norm(semantic_matrix, axis=1)

    with np.errstate(invalid="ignore"):
        cosine_sparse_matrix = np.dot(sparse_matrix, sparse_matrix.T) / np.outer(
            sparse_norms, sparse_norms
        )
        cosine_sparse_matrix = np.nan_to_num(cosine_sparse_matrix)

        cosine_semantic_matrix = np.dot(semantic_matrix, semantic_matrix.T) / np.outer(
            semantic_norms, semantic_norms
        )
        cosine_semantic_matrix = np.nan_to_num(cosine_semantic_matrix)

    return cosine_sparse_matrix, cosine_semantic_matrix


def process_user_info(user_info):
    user_info["occupation"] = user_info["occupation"] / 20

    user_info["gender"] = user_info["gender"].map({"M": 0.3, "F": 0.15})

    def age_map(age):
        if 0 <= age <= 10:
            return 1 / 7
        elif 11 <= age <= 20:
            return 2 / 7
        elif 21 <= age <= 29:
            return 3 / 7
        elif 30 <= age <= 38:
            return 4 / 7
        elif 39 <= age <= 47:
            return 5 / 7
        elif 48 <= age <= 55:
            return 6 / 7
        else:
            return 1

    user_info["age"] = user_info["age"].apply(lambda age: age_map(age))

    user_info = user_info.drop(["zip"], axis=1)

    return user_info


if __name__ == "__main__":
    dataset = Library(5)
