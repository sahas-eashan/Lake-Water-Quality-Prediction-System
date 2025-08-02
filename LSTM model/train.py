import torch
import torch.nn as nn
import torch.optim as optim
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

        return torch.tensor(
            chunk_data["X"][local_idx], dtype=torch.float32
        ), torch.tensor(chunk_data["y"][local_idx], dtype=torch.float32)


# Function to load chunk files (train/val/test)
def load_existing_chunks():
    if not os.path.exists("data_chunks"):
        raise FileNotFoundError(
            "No data_chunks folder found. Run the chunking script first."
        )

    train_chunks = [
        f"data_chunks/{f}"
        for f in os.listdir("data_chunks")
        if f.endswith("_train.pkl")
    ]
    val_chunks = [
        f"data_chunks/{f}" for f in os.listdir("data_chunks") if f.endswith("_val.pkl")
    ]
    test_chunks = [
        f"data_chunks/{f}" for f in os.listdir("data_chunks") if f.endswith("_test.pkl")
    ]

    if not train_chunks:
        raise FileNotFoundError(
            "No train chunks found. Make sure you've run the chunking script completely."
        )

    train_chunks.sort()
    val_chunks.sort()
    test_chunks.sort()

    print(f"Found {len(train_chunks)} train chunks")
    print(f"Found {len(val_chunks)} validation chunks")
    print(f"Found {len(test_chunks)} test chunks")

    return train_chunks, val_chunks, test_chunks


# Create DataLoaders from chunk files
def create_data_loaders(batch_size=128):
    train_chunks, val_chunks, test_chunks = load_existing_chunks()

    train_dataset = ChunkedDataset(train_chunks)
    val_dataset = ChunkedDataset(val_chunks)
    test_dataset = ChunkedDataset(test_chunks)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Define the Multi-step LSTM model
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * seq_len)
        self.output_size = output_size
        self.seq_len = seq_len

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last timestep output
        out = self.fc(out)
        out = out.view(-1, self.seq_len, self.output_size)
        return out


def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate_epoch(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_data_loaders(batch_size=128)

    model = MultiStepLSTM(
        input_size=3, hidden_size=64, num_layers=2, output_size=3, seq_len=12
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 30
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss = validate_epoch(model, device, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "multistep_lstm_best.pt")
            print("Saved new best model")

    print("Training complete!")


if __name__ == "__main__":
    main()
