import torch
from dataset import (
    load_movies,
    load_ratings,
    compute_cosine_similarities,
    download_dataset,
)
import os
import numpy as np


class Library:
    def __init__(self, args) -> None:
        print("Initializing Content Library...")
        if "ml-1m" not in os.listdir("./data"):
            print("Dataset not found. Downloading from grouplens...")
            download_dataset()
        else:
            print("Dataset found. Loading...")
        self.ratings, self.user_info = load_ratings()
        self.sparse_vecs, self.semantic_vecs = load_movies()
        self.cosine_sparse_matrix, self.cosine_semantic_matrix = (
            compute_cosine_similarities(self.sparse_vecs, self.semantic_vecs)
        )

        self.max_user_id = self.ratings["user_id"].max()
        self.max_movie_id = self.ratings["movie_id"].max()
        self.unavailable_users = []

    def generate_client(self):
        # get top 10% of users with most ratings
        top_user = self.ratings["user_id"].value_counts().index[:100]

        choosen = np.random.choice(top_user)
        while choosen in self.unavailable_users:
            choosen = np.random.choice(top_user)
        self.unavailable_users.append(choosen)
        history, ratings = self.get_inputs(choosen)

        end_train = int(len(history) * 0.98)

        cosine_inputs = [self.cosine_semantic_matrix[his - 1] for his in history]
        semantic_inputs = [self.semantic_vecs[his - 1] for his in history]
        labels = [ratings[i] for i in range(len(ratings))]

        train_cosine = cosine_inputs[:end_train]
        train_semantic = semantic_inputs[:end_train]
        train_labels = labels[:end_train]
        train_ids = history[:end_train]

        test_cosine = cosine_inputs[end_train:]
        test_semantic = semantic_inputs[end_train:]
        test_labels = labels[end_train:]
        test_ids = history[end_train:]

        return {
            "train": (train_cosine, train_semantic, train_labels, train_ids),
            "test": (test_cosine, test_semantic, test_labels, test_ids),
            "movies": torch.zeros((self.max_movie_id + 1)),
            "uid": choosen,
            "user_info": torch.tensor(
                self.user_info[self.user_info["user_id"] == choosen].values[0]
            ),
        }

    def get_inputs(self, user_id, request_ratio=0):
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        movie_ids = user_ratings["movie_id"].values
        ratings = user_ratings["rating"].values
        return torch.tensor(movie_ids), torch.tensor(ratings)

    def return_uid(self, uid):
        self.unavailable_users.remove(uid)

    def reset(self):
        self.unavailable_users = []

    def step(self):
        pass


if __name__ == "__main__":
    library = Library(10)
    library.genrate_clients()
    library.reset()
