import numpy as np
import torch
from .dataset import get_data
import copy


class Library:
    def __init__(self) -> None:
        user_data, item_data, rating_data, user_request_sequences = get_data()

        self.num_users = len(user_data)
        self.num_items = len(item_data)

        # Create item latent matrix Y
        self.Y = torch.tensor(
            np.array(item_data[["semantic"]].values.tolist()), dtype=torch.float32
        ).squeeze()

        # Create user-item interaction matrix R
        self.R = torch.zeros(self.num_users, self.num_items)
        self.R[rating_data["user_id"], rating_data["item_id"]] = torch.tensor(
            rating_data["rating"].values / 5, dtype=torch.float32
        )

        self.user_request_history = np.array(user_request_sequences["temporal"])

        self.user_personal_info = torch.tensor(
            np.array(user_data["data"].values.tolist()), dtype=torch.float32
        )

        self.unavailable_users = []

    def create_client(self):
        uid = np.random.choice(
            [i for i in range(self.num_users) if i not in self.unavailable_users]
        )
        self.unavailable_users.append(uid)

        r_i = self.R[uid]
        urh = self.user_request_history[uid]
        upi = self.user_personal_info[uid]

        return uid, r_i, copy.deepcopy(self.Y), urh, upi

    def return_uid(self, uid):
        self.unavailable_users.remove(uid)

    def reset(self):
        self.unavailable_users = []


if __name__ == "__main__":
    lib = Library()

    uid, r_i, Y, urh, upi = lib.create_client()
