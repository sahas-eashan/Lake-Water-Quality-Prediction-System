import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import random


class ChunkedDataset(Dataset):
    def __init__(self, chunk_files, cache_size=5):
        self.chunk_files = chunk_files
        self.chunk_sizes = []
        self.cumulative_sizes = [0]
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []

        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(f)
                size = len(chunk_data["X"])
                self.chunk_sizes.append(size)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)

        self.total_size = self.cumulative_sizes[-1]

    def _load_chunk(self, chunk_idx):
        if chunk_idx in self.cache:
            return self.cache[chunk_idx]

        with open(self.chunk_files[chunk_idx], "rb") as f:
            chunk_data = pickle.load(f)

        if len(self.cache) >= self.cache_size:
            oldest_chunk = self.cache_order.pop(0)
            del self.cache[oldest_chunk]

        self.cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)

        return chunk_data

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chunk_idx = 0
        while idx >= self.cumulative_sizes[chunk_idx + 1]:
            chunk_idx += 1

        local_idx = idx - self.cumulative_sizes[chunk_idx]
        chunk_data = self._load_chunk(chunk_idx)

        return torch.tensor(
            chunk_data["X"][local_idx], dtype=torch.float32
        ), torch.tensor(chunk_data["y"][local_idx], dtype=torch.float32)


def create_subset_dataset(dataset, subset_ratio=0.05):
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    indices = random.sample(range(total_size), subset_size)
    return torch.utils.data.Subset(dataset, indices)


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


def create_data_loaders(subset_ratio=0.05, batch_size=1024):
    train_chunks, val_chunks, test_chunks = load_existing_chunks()

    train_dataset = ChunkedDataset(train_chunks, cache_size=10)
    val_dataset = ChunkedDataset(val_chunks, cache_size=5)
    test_dataset = ChunkedDataset(test_chunks, cache_size=5)

    train_dataset = create_subset_dataset(train_dataset, subset_ratio)
    val_dataset = create_subset_dataset(val_dataset, subset_ratio)

    print(f"Using {subset_ratio*100}% subset for training")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


class MultiStepLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, output_size, seq_len, dropout=0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size * seq_len)
        self.output_size = output_size
        self.seq_len = seq_len

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        out = out.view(-1, self.seq_len, self.output_size)
        return out


def train_epoch(model, device, train_loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)

    for xb, yb in pbar:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(xb)
            loss = criterion(pred, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)


def validate_epoch(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = criterion(pred, yb)

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    return total_loss / len(val_loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = create_data_loaders(
        subset_ratio=0.05, batch_size=1024
    )

    model = MultiStepLSTM(
        input_size=3,
        hidden_size=128,
        num_layers=2,
        output_size=3,
        seq_len=12,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    epochs = 30
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training with {len(train_loader)} batches per epoch")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, device, train_loader, optimizer, criterion, scaler
        )
        val_loss = validate_epoch(model, device, val_loader, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} -- Train Loss: {train_loss:.5f} -- "
            f"Val Loss: {val_loss:.5f} -- LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                "multistep_lstm_best.pt",
            )
            print("Saved new best model")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                f"checkpoint_epoch_{epoch+1}.pt",
            )

    print("Training complete!")


if __name__ == "__main__":
    main()
