import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

from cluster import clustering
from utils import average_weights

args_parser = argparse.ArgumentParser()
args_parser.add_argument("--num_clients", type=int, default=10)
args_parser.add_argument("--train_ratio", type=float, default=0.8)
args_parser.add_argument("--device", type=str, default="cuda:1")
args_parser.add_argument("--temporal", action="store_true")
args_parser.add_argument("--use_semantic", action="store_true")
args_parser.add_argument("--similarity", action="store_true")
args_parser.add_argument("--num_clusters", type=int, default=3)
args_parser.add_argument("--runs", type=int, default=1)
args_parser.add_argument("--learning_rate", type=float, default=0.001)
args_parser.add_argument("--batch_size", type=int, default=512)
args = args_parser.parse_args()


from simulation.library import Library


class FedModel(nn.Module):
    def __init__(self, hidden_dim, num_items, feature_dim, temporal=False, similarity=False):
        super(FedModel, self).__init__()
        self.user_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.temporal = temporal
        self.similarity = similarity

        self.extra_dim = 0

        if temporal:
            self.bi_lstm = nn.LSTM(
                feature_dim, hidden_dim, batch_first=True, bidirectional=True
            )
            self.extra_dim += hidden_dim
        if similarity:
            self.similarity_embedding = nn.Linear(num_items, hidden_dim)
            self.extra_dim += hidden_dim

        self.fc1 = nn.Linear(hidden_dim * 2 + self.extra_dim, hidden_dim)

    def forward(self, user_id, item_id, h, sim):
        user_embedding = self.user_embedding.repeat(item_id.shape[0], 1)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        if self.temporal:
            lstm_out, _ = self.bi_lstm(h)
            x = torch.cat([x, lstm_out[:, -1, :]], dim=1)
        if self.similarity:
            sim_embedding = self.similarity_embedding(sim)
            x = torch.cat([x, sim_embedding], dim=1)   
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x

    def get_flatten_params(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])


class CentralizedModel(nn.Module):
    def __init__(self, hidden_dim, num_items, num_users, feature_dim, temporal=False, similarity=False):
        super(CentralizedModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.item_embedding = nn.Embedding(num_items, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.temporal = temporal
        self.similarity = similarity
        self.extra_dim = 0

        if temporal:
            self.bi_lstm = nn.LSTM(
                feature_dim, hidden_dim, batch_first=True, bidirectional=True
            )
            self.extra_dim += hidden_dim

        if similarity:
            self.similarity_embedding = nn.Linear(num_items, hidden_dim)
            self.extra_dim += hidden_dim

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)


    def forward(self, user_id, item_id, h, sim):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)
        x = torch.cat([user_embedding, item_embedding], dim=1)
        if self.temporal:
            lstm_out, _ = self.bi_lstm(h)
            x = torch.cat([x, lstm_out[:, -1, :]], dim=1)
        if self.similarity:
            sim_embedding = self.similarity_embedding(sim)
            x = torch.cat([x, sim_embedding], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x).squeeze(-1)
        return x


def batch_padding(sequences):
    """
    Pad sequences to the same length
    sequences: sequence of lists of size([50]) torch tensors with different lengths
    """
    max_length = max([len(sequence) for sequence in sequences])
    feature_dim = 50 if args.use_semantic else 19
    pad = torch.zeros(feature_dim)
    out = []
    for sequence in sequences:
        new_sequence = []
        while len(new_sequence) + len(sequence) < max_length:
            new_sequence.append(pad)
        new_sequence.extend(sequence)
        new_sequence = torch.stack(new_sequence)
        out.append(new_sequence)

    return torch.stack(out)


def weights_averaging(models, weights, do_clustering=False):
    if do_clustering:
        new_models = []
        clusters, _, _ = clustering(args.num_clusters, weights)
        for cluster in clusters:
            cluster_weights = [weights[i] for i in cluster]
            cluster_models = [models[i] for i in cluster]
            new_models.extend(weights_averaging(cluster_models, cluster_weights))

        return new_models

    weights = [{k: v.cpu() for k, v in model.state_dict().items()} for model in models]
    averaged_weights = average_weights(weights)

    for model in models:
        model.load_state_dict(averaged_weights)

    return models


def custom_train_test_split(inputs, outputs, ratio):
    pivot = int(len(outputs) * ratio)
    train_inputs = {key: value[:pivot] for key, value in inputs.items()}
    test_inputs = {key: value[pivot:] for key, value in inputs.items()}
    train_outputs = outputs[:pivot]
    test_outputs = outputs[pivot:]

    return train_inputs, test_inputs, train_outputs, test_outputs


def custom_train_loop(
    model,
    optimizer,
    criterion,
    X_train,
    y_train,
    num_epochs,
    batch_size,
    patience=10,
    early_stopping=True,
    device="cpu",
):
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for i in tqdm(
            range(0, len(y_train), batch_size), desc=f"Epoch {epoch}", leave=False
        ):
            item_ids = X_train["item_id"][i : i + batch_size].to(device)
            user_ids = X_train["user_id"][i : i + batch_size].to(device)
            history = batch_padding(X_train["history"][i : i + batch_size]).to(device)
            similarity = X_train["similarity"][i : i + batch_size].to(device)
            labels = y_train[i : i + batch_size].to(device)
            optimizer.zero_grad()
            output = model(user_ids, item_ids, history, similarity)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(X_train)

        if early_stopping:
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

    return model


def evaluate(model, X_test, y_test, batch_size, device="cpu"):
    y_true = []
    y_pred = []

    for i in range(0, len(X_test), batch_size):
        item_ids = X_test["item_id"][i : i + batch_size].to(device)
        user_ids = X_test["user_id"][i : i + batch_size].to(device)
        history = batch_padding(X_test["history"][i : i + batch_size]).to(device)
        similarity = X_test["similarity"][i : i + batch_size].to(device)
        labels = y_test[i : i + batch_size].to(device)
        output = model(user_ids, item_ids, history, similarity)
        y_true.extend(labels.tolist())
        y_pred.extend(output.tolist())
    return y_true, y_pred


def preprocess(uid, r_i, Y, urh, similarites):
    user_ids = []
    items_ids = []
    ratings = []
    history = []
    sims = []
    feature_dim = 50 if args.use_semantic else 19

    start = torch.zeros(feature_dim).unsqueeze(0)
    for pivot in range(len(urh)):
        item_id = urh[pivot]
        rating = r_i[item_id]
        if pivot == 0:
            h = start
        else:
            h = Y[urh[:pivot]]
        similarity = similarites[item_id]
        user_ids.append(uid)
        items_ids.append(item_id)
        ratings.append(rating)
        history.append(h)
        sims.append(similarity)

    return {
        "user_id": torch.tensor(user_ids),
        "item_id": torch.tensor(items_ids),
        "similarity": torch.stack(sims),
        "history": history,
    }, torch.tensor(ratings)


def eval_mcfed(library, client_data, avg=False):
    num_items = library.num_items
    num_users = library.num_users
    num_clients = args.num_clients
    feature_dim = 50 if args.use_semantic else 19
    errors = []
    for _ in tqdm(range(args.runs), desc="Evaluating..."):
        library.reset()
        models = []
        weights = []
        y_true = []
        y_pred = []
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for client in tqdm(range(num_clients), desc="Handling clients...", leave=False):
            model = FedModel(
                hidden_dim=128,
                num_items=num_items,
                feature_dim=feature_dim,
                temporal=args.temporal,
                similarity=args.similarity,
            ).to(args.device)

            uid, r_i, Y, urh, upim, sims = client_data[client]
            inputs, outputs = preprocess(uid, r_i, Y, urh, sims)
            sub_X_train, sub_X_test, sub_y_train, sub_y_test = custom_train_test_split(
                inputs, outputs, args.train_ratio
            )
            X_train.append(sub_X_train)
            y_train.append(sub_y_train)
            X_test.append(sub_X_test)
            y_test.append(sub_y_test)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
            criterion = nn.MSELoss()

            model = custom_train_loop(
                model,
                optimizer,
                criterion,
                sub_X_train,
                sub_y_train,
                num_epochs=100,
                batch_size=args.batch_size,
                device=args.device,
            )
            upi = upi.clone().detach().to(args.device)
            models.append(model)
            weight = model.get_flatten_params()
            weight = torch.cat([weight, upi])
            weights.append(weight)

        models = weights_averaging(models, weights, do_clustering=not avg)

        for i in range(num_clients):
            model = models[i]
            sub_y_true, sub_y_pred = evaluate(
                model,
                X_test[i],
                y_test[i],
                batch_size=args.batch_size,
                device=args.device,
            )

            y_true.extend(sub_y_true)
            y_pred.extend(sub_y_pred)

        rmse = root_mean_squared_error(y_true, y_pred)
        errors.append(rmse)

    mean_rmse = np.mean(errors)
    std_rmse = np.std(errors)

    print(f"Mean RMSE: {mean_rmse} - Std RMSE: {std_rmse}")


def eval_avgfed(library, client_data):
    eval_mcfed(library, client_data, avg=True)


def eval_centralized(library, client_data):
    num_items = library.num_items
    num_users = library.num_users
    num_clients = args.num_clients
    feature_dim = 50 if args.use_semantic else 19
    errors = []
    for _ in tqdm(range(args.runs), desc="Evaluating..."):
        library.reset()
        model = CentralizedModel(
            hidden_dim=128,
            num_items=num_items,
            num_users=num_users,
            feature_dim=feature_dim,
            temporal=args.temporal,
            similarity=args.similarity,
        ).to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        criterion = nn.MSELoss()

        X_train = {}
        y_train = []
        X_test = {}
        y_test = []

        for client in tqdm(range(num_clients), desc="Handling clients...", leave=False):
            uid, r_i, Y, urh, upi, sims = client_data[client]
            inputs, outputs = preprocess(uid, r_i, Y, urh, sims)
            sub_X_train, sub_X_test, sub_y_train, sub_y_test = custom_train_test_split(
                inputs, outputs, args.train_ratio
            )
            y_train.extend(sub_y_train)
            y_test.extend(sub_y_test)

            for key in sub_X_train.keys():
                if key not in X_train:
                    X_train[key] = []
                    X_test[key] = []
                X_train[key].extend(sub_X_train[key])
                X_test[key].extend(sub_X_test[key])

        # shuffle data
        indices = np.random.permutation(len(y_train))
        for key in X_train.keys():
            X_train[key] = [X_train[key][i] for i in indices]
            if key != "history":
                if key == "similarity":
                    X_train[key] = torch.stack(X_train[key])
                    X_test[key] = torch.stack(X_test[key])

                X_train[key] = torch.tensor(np.array(X_train[key]))
                X_test[key] = torch.tensor(np.array(X_test[key]))

        y_train = torch.tensor(y_train)[indices]
        y_test = torch.tensor(y_test)

        model = custom_train_loop(
            model,
            optimizer,
            criterion,
            X_train,
            y_train,
            num_epochs=100,
            batch_size=args.batch_size,
            device=args.device,
        )

        y_true, y_pred = evaluate(
            model, X_test, y_test, batch_size=args.batch_size, device=args.device
        )
        rmse = root_mean_squared_error(y_true, y_pred)
        errors.append(rmse)

    mean_rmse = np.mean(errors)
    std_rmse = np.std(errors)

    print(f"Mean RMSE: {mean_rmse} - Std RMSE: {std_rmse}")


if __name__ == "__main__":

    library = Library(semantic=args.use_semantic)
    client_data = []
    for client in range(args.num_clients):
        uid, r_i, Y, urh, upi = library.create_client()
        similarites = library.Y_similarities
        client_data.append((uid, r_i, Y, urh, upi, similarites))

    print("====================================")
    print(args)
    print("Centralized Training")
    eval_centralized(library, client_data)
    print("Average Fed Training")
    eval_avgfed(library, client_data)
    print("MC Fed Training")
    eval_mcfed(library, client_data)
