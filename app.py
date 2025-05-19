from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Your model class (must match what you trained in Colab)
class TimeNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=3, padding=2, dilation=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=2)
        self.relu2 = nn.ReLU()
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        _, h_n = self.gru(x)
        return self.fc(h_n.squeeze(0))

# Load the trained model
model = TimeNetModel()
model.load_state_dict(torch.load("tcn_timesnet_model_final.pt", map_location="cpu"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Expecting shape: [6, 5]
    sequence = np.array(data['sequence']).astype(np.float32)
    
    # Add batch dimension: [1, 6, 5]
    input_tensor = torch.tensor([sequence])
    
    with torch.no_grad():
        output = model(input_tensor).item()

    return jsonify({'bolus_prediction': round(output, 2)})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render
    app.run(host='0.0.0.0', port=port)

