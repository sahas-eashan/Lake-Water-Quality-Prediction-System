import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class ChunkedDataset(Dataset):
    def __init__(self, chunk_files):
        self.chunk_files = chunk_files
        self.chunk_sizes = []
        self.cumulative_sizes = [0]

        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
                size = len(chunk_data["X"])
                self.chunk_sizes.append(size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)

        self.total_size = self.cumulative_sizes[-1]

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chunk_idx = 0
        while idx >= self.cumulative_sizes[chunk_idx + 1]:
            chunk_idx += 1

        local_idx = idx - self.cumulative_sizes[chunk_idx]

        with open(self.chunk_files[chunk_idx], "rb") as f:
            chunk_data = pickle.load(f)

        return torch.tensor(chunk_data["X"][local_idx]), torch.tensor(
            chunk_data["y"][local_idx]
        )


def create_chunks(df, chunk_size=10000):
    input_seq_len = 12
    output_seq_len = 12
    features = ["pH", "Cond", "Tur"]

    os.makedirs("data_chunks", exist_ok=True)

    chunk_files = []
    chunk_idx = 0
    current_sequences = []
    current_targets = []
    current_loc_ids = []

    total_processed = 0

    for loc, group in df.groupby("location_id"):
        group_sorted = group.sort_values(by=["year", "month"])
        X = group_sorted[features].values

        if len(X) < input_seq_len + output_seq_len:
            continue

        for i in range(len(X) - input_seq_len - output_seq_len + 1):
            seq_x = X[i : i + input_seq_len]
            seq_y = X[i + input_seq_len : i + input_seq_len + output_seq_len]

            current_sequences.append(seq_x)
            current_targets.append(seq_y)
            current_loc_ids.append(loc)
            total_processed += 1

            if total_processed % 100000 == 0:
                print(f"Processed {total_processed} sequences...")

            if len(current_sequences) >= chunk_size:
                chunk_file = f"data_chunks/chunk_{chunk_idx}.pkl"

                chunk_data = {
                    "X": np.stack(current_sequences).astype(np.float32),
                    "y": np.stack(current_targets).astype(np.float32),
                    "loc_ids": np.array(current_loc_ids),
                }

                with open(chunk_file, "wb") as f:
                    pickle.dump(chunk_data, f)

                chunk_files.append(chunk_file)
                print(f"Saved chunk {chunk_idx}")

                current_sequences = []
                current_targets = []
                current_loc_ids = []
                chunk_idx += 1

    if current_sequences:
        chunk_file = f"data_chunks/chunk_{chunk_idx}.pkl"
        chunk_data = {
            "X": np.stack(current_sequences).astype(np.float32),
            "y": np.stack(current_targets).astype(np.float32),
            "loc_ids": np.array(current_loc_ids),
        }

        with open(chunk_file, "wb") as f:
            pickle.dump(chunk_data, f)

        chunk_files.append(chunk_file)
        print(f"Saved final chunk {chunk_idx}")

    return chunk_files


def split_chunks_by_location(chunk_files, test_size=0.3, val_size=0.5):
    all_loc_ids = set()

    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            chunk_data = pickle.load(f)
            all_loc_ids.update(chunk_data["loc_ids"])

    unique_locations = list(all_loc_ids)
    train_locs, temp_locs = train_test_split(
        unique_locations, test_size=test_size, random_state=42
    )
    val_locs, test_locs = train_test_split(
        temp_locs, test_size=val_size, random_state=42
    )

    train_chunks = []
    val_chunks = []
    test_chunks = []

    for chunk_file in chunk_files:
        with open(chunk_file, "rb") as f:
            chunk_data = pickle.load(f)

        train_mask = np.isin(chunk_data["loc_ids"], train_locs)
        val_mask = np.isin(chunk_data["loc_ids"], val_locs)
        test_mask = np.isin(chunk_data["loc_ids"], test_locs)

        if train_mask.any():
            train_chunk_file = chunk_file.replace(".pkl", "_train.pkl")
            train_chunk_data = {
                "X": chunk_data["X"][train_mask],
                "y": chunk_data["y"][train_mask],
                "loc_ids": chunk_data["loc_ids"][train_mask],
            }
            with open(train_chunk_file, "wb") as f:
                pickle.dump(train_chunk_data, f)
            train_chunks.append(train_chunk_file)

        if val_mask.any():
            val_chunk_file = chunk_file.replace(".pkl", "_val.pkl")
            val_chunk_data = {
                "X": chunk_data["X"][val_mask],
                "y": chunk_data["y"][val_mask],
                "loc_ids": chunk_data["loc_ids"][val_mask],
            }
            with open(val_chunk_file, "wb") as f:
                pickle.dump(val_chunk_data, f)
            val_chunks.append(val_chunk_file)

        if test_mask.any():
            test_chunk_file = chunk_file.replace(".pkl", "_test.pkl")
            test_chunk_data = {
                "X": chunk_data["X"][test_mask],
                "y": chunk_data["y"][test_mask],
                "loc_ids": chunk_data["loc_ids"][test_mask],
            }
            with open(test_chunk_file, "wb") as f:
                pickle.dump(test_chunk_data, f)
            test_chunks.append(test_chunk_file)

    return train_chunks, val_chunks, test_chunks


df = pd.read_csv("all_locations_timeseries_cleaned.csv")
print(f"Loaded data with {len(df)} rows")

if not os.path.exists("data_chunks"):
    print("Creating chunks...")
    chunk_files = create_chunks(df, chunk_size=5000)
    print(f"Created {len(chunk_files)} chunks")
else:
    chunk_files = [
        f"data_chunks/{f}"
        for f in os.listdir("data_chunks")
        if f.endswith(".pkl") and not ("_train" in f or "_val" in f or "_test" in f)
    ]
    print(f"Found {len(chunk_files)} existing chunks")

print("Splitting chunks by location...")
train_chunks, val_chunks, test_chunks = split_chunks_by_location(chunk_files)

train_dataset = ChunkedDataset(train_chunks)
val_dataset = ChunkedDataset(val_chunks)
test_dataset = ChunkedDataset(test_chunks)

print(f"Train sequences: {len(train_dataset)}")
print(f"Validation sequences: {len(val_dataset)}")
print(f"Test sequences: {len(test_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Data loaders created!")
