import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--num_clients", type=int, default=10)
args_parser.add_argument("--train_ratio", type=float, default=0.8)
args = args_parser.parse_args()


from simulation.library import Library


class FedModel(nn.Module):
    def __init__(self, hidden_dim, num_items):
        super(FedModel, self).__init__()
        self.user_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, item_id):
        user_embedding = self.user_embedding.repeat(item_id.shape[0], 1)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x


class CentralizedModel(nn.Module):
    def __init__(self, hidden_dim, num_items, num_users):
        super(CentralizedModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x


def custom_train_test_split(data, ratio):
    pivot = int(len(data) * ratio)
    return data[:pivot], data[pivot:]


def custom_train_loop(
    model,
    optimizer,
    criterion,
    train_data,
    num_epochs,
    batch_size,
    patience=5,
    early_stopping=True,
):
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_data)

        if early_stopping:
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    return model


if __name__ == "__main__":

    library = Library()
    num_items = library.num_items
    num_users = library.num_users

    uid, r_i, Y, urh, upi = library.create_client()

    train, test = custom_train_test_split(urh, 0.95)

    print(len(train), len(test))
