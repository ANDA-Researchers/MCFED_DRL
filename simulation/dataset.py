import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import json

# Define the path to the data directory
data_dir = "./data/ml-100k/"

# File paths (adjust to your file locations)
file_paths = {
    "item": os.path.join(data_dir, "u.item"),  # Item data
    "data": os.path.join(data_dir, "u.data"),  # Rating data
    "user": os.path.join(data_dir, "u.user"),  # User information
    "genre": os.path.join(data_dir, "u.genre"),  # Genre information
}


# Function to preprocess the user data
def preprocess_user_data(file_path):
    """
    Preprocess user data: Encode categorical variables, one-hot encode occupation, and normalize numerical variables.
    """
    user_columns = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_data = pd.read_csv(
        file_path, sep="|", header=None, names=user_columns, encoding="latin-1"
    )

    # Handle Missing Data (if any)
    user_data = user_data.dropna()  # Drop rows with missing values

    # Encode the categorical variable "gender"
    user_data["gender"] = user_data["gender"].map({"M": 1, "F": 0})  # 'M' → 1, 'F' → 0

    # One-hot encode the categorical variable "occupation"
    user_data = pd.get_dummies(
        user_data,
        columns=["occupation"],
        prefix="occupation",
        drop_first=True,
        dtype=int,
    )

    # Normalize the "age" column
    scaler = MinMaxScaler()
    user_data["age"] = scaler.fit_transform(
        user_data[["age"]]
    )  # Normalize age to a range [0, 1]

    # Drop 'zip_code' column as it is not needed for collaborative filtering
    user_data = user_data.drop(columns=["zip_code"])

    user_data = user_data.copy()
    temp = user_data.iloc[:, 1:].values.tolist()

    user_data = user_data.drop(columns=user_data.columns[1:])
    user_data["data"] = temp

    return user_data


# Function to preprocess the item data (with one-hot genre embeddings)
def preprocess_item_data(file_path):
    """
    Preprocess item data: One-hot encode the genres and keep item_id.
    """
    # Load the item data (item information such as title, genre)
    item_columns = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "IMDb_URL",
    ] + [f"genre_{i}" for i in range(19)]
    item_data = pd.read_csv(
        file_path, sep="|", header=None, names=item_columns, encoding="latin-1"
    )

    # Keep only the item_id and genre columns from the item data
    item_data_genres = item_data[
        ["item_id"] + [f"genre_{i}" for i in range(19)]
    ]  # Selecting item_id and genre columns

    item_data_genres = item_data_genres.copy()
    item_data_genres["genres"] = item_data_genres.iloc[:, 1:].values.tolist()

    # Drop the original genre columns
    item_data_genres = item_data_genres.drop(columns=[f"genre_{i}" for i in range(19)])

    return item_data_genres


# Function to preprocess rating data
def preprocess_rating_data(file_path):
    """
    Preprocess rating data: Handle missing ratings and ensure clean dataset.
    """
    rating_columns = ["user_id", "item_id", "rating", "timestamp"]
    rating_data = pd.read_csv(file_path, sep="\t", header=None, names=rating_columns)

    # Handle Missing Data (if any)
    rating_data = rating_data.dropna()  # Drop rows with missing values

    return rating_data


# Function to create user request sequences
def create_user_request_sequences(rating_data):
    """
    Create user request sequences based on the interaction timestamps.
    Each user’s request sequence is an ordered list of item_ids they interacted with over time.
    """
    # Sort by user_id and timestamp to ensure temporal order
    rating_data["timestamp"] = pd.to_datetime(rating_data["timestamp"], unit="s")
    rating_data = rating_data.sort_values(by=["user_id", "timestamp"])

    # Group by user_id and create the sequence of item_ids they interacted with
    user_request_sequences = (
        rating_data.groupby("user_id")["item_id"].apply(list).reset_index()
    )

    user_request_sequences.columns = ["user_id", "temporal"]

    return user_request_sequences


def load_genre_data():
    genre_columns = ["genre_id", "genre"]
    genre_data = pd.read_csv(
        file_paths["genre"], sep="|", header=None, names=genre_columns
    )

    genre_data_dict = dict(zip(genre_data["genre"], genre_data["genre_id"]))

    with open("./data/emb.json", "r") as f:
        genre_embeddings = json.load(f)

    genre_vector = np.zeros((len(genre_data), 50))

    for k, v in genre_data_dict.items():
        t = v.lower()

        if t.find("unknown") != -1:
            continue

        if t.find("child") != -1:
            t = "children"
        if t.find("sci") != -1:
            t = "sci-fi"
        if t.find("noir") != -1:
            t = "noir"

        genre_vector[k] = genre_embeddings[t]

    return genre_vector


def get_data():
    # Load and preprocess the item data
    item_data = preprocess_item_data(file_paths["item"])

    # Load and preprocess the user data
    user_data = preprocess_user_data(file_paths["user"])

    # Load and preprocess the rating data
    rating_data = preprocess_rating_data(file_paths["data"])

    # Create user request sequences
    user_request_sequences = create_user_request_sequences(rating_data)

    # Load genre vectors
    genres_vector = load_genre_data()
    vecs = []
    # iterate over each row and multiply the genre_vector
    for i in range(len(item_data)):
        item_id = item_data.loc[i]["item_id"]
        genre = np.array(item_data.loc[i]["genres"])
        vecs.append(np.sum(genres_vector[genre], axis=0) / genre.nonzero()[0].shape[0])

    item_data.loc[:, "semantic"] = vecs

    item_data.columns = ["item_id", "sparse", "semantic"]

    user_data["user_id"] = user_data["user_id"].apply(lambda x: x - 1)
    item_data["item_id"] = item_data["item_id"].apply(lambda x: x - 1)
    rating_data["user_id"] = rating_data["user_id"].apply(lambda x: x - 1)
    rating_data["item_id"] = rating_data["item_id"].apply(lambda x: x - 1)

    # substract 1 from user_request_sequences
    user_request_sequences["user_id"] = user_request_sequences["user_id"].apply(
        lambda x: x - 1
    )

    user_request_sequences["temporal"] = user_request_sequences["temporal"].apply(
        lambda x: [i - 1 for i in x]
    )

    return user_data, item_data, rating_data, user_request_sequences
