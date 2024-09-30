import numpy as np
from dataset import load_dataset


class ContentLibrary:
    def __init__(self, path):
        self.dataset = load_dataset(path)
        self.ratings = self.dataset["ratings"]
        self.semantic_vecs = self.dataset["semantic_vec"]
        self.sparse_vecs = self.dataset["sparse_vec"]
        self.total_items = list(set(self.ratings["item_id"]))
        self.total_users = list(set(self.ratings["user_id"]))
        self.used_user_ids = set()
        self.max_item_id = max(self.total_items)

    def load_ratings(self, user_id):
        ratings = self.ratings[self.ratings["user_id"] == user_id].copy()
        ratings["semantic_vec"] = ratings["item_id"].apply(
            lambda x: self.semantic_vecs[int(x)]
        )
        ratings["sparse_vec"] = ratings["item_id"].apply(
            lambda x: self.sparse_vecs[int(x)]
        )
        ratings = ratings.sort_values(by="timestamp", ascending=False)
        return {
            "contents": ratings["item_id"].values,
            "ratings": ratings["rating"].values,
            "semantic_vecs": np.stack(ratings["semantic_vec"].values),
            "sparse_vecs": np.stack(ratings["sparse_vec"].values),
            "max": self.max_item_id,
        }

    def load_user_info(self, user_id):
        return self.ratings[self.ratings["user_id"] == user_id]

    def get_user(self):
        available_users = list(set(self.total_users) - self.used_user_ids)
        if not available_users:
            raise ValueError("No available users left to sample.")
        user_id = np.random.choice(available_users)
        self.used_user_ids.add(user_id)
        return user_id

    def return_user(self, user_id):
        self.used_user_ids.discard(user_id)
