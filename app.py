import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

# ----- TCN Layer -----
class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size - 1) * dilation,
                              dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x[:, :, :-self.conv.padding[0]]  # remove padding

# ----- TimeNet Model -----
class TimeNetModel(nn.Module):
    def __init__(self, in_features=5, seq_len=6, hidden_dim=64):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNLayer(in_features, 64, dilation=1),
            TCNLayer(64, 64, dilation=2),
        )
        self.gru = nn.GRU(64, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))

# ----- Load Model -----
model = TimeNetModel()
model.load_state_dict(torch.load("tcn_timesnet_model_final.pt", map_location="cpu"))
model.eval()

# ----- Flask App -----
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sequence = np.array(data['sequence']).astype(np.float32)  # shape: (6, 5)
    input_tensor = torch.tensor([sequence])  # shape: (1, 6, 5)

    with torch.no_grad():
        output = model(input_tensor).item()

    return jsonify({'bolus_prediction': round(output, 2)})

# ----- Run the App (for Render) -----
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
