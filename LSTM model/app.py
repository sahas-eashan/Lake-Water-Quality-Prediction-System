from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from mangum import Mangum

app = FastAPI()
handler = Mangum(app)


class InputData(BaseModel):
    input_sequence: list  # list of 12 timesteps, each timestep a list of 3 features


# Define your MultiStepLSTM model class here or import it
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
        self.fc = nn.Linear(hidden_size, output_size * seq_len)
        self.output_size = output_size
        self.seq_len = seq_len

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(-1, self.seq_len, self.output_size)
        return out


# Load model and set to eval mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiStepLSTM(
    input_size=3, hidden_size=128, num_layers=2, output_size=3, seq_len=12
)
checkpoint = torch.load("multistep_lstm_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()


@app.post("/predict/")
def predict(data: InputData):
    input_seq = np.array(data.input_sequence, dtype=np.float32)
    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy()
    return {"prediction": pred.tolist()}
