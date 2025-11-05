import torch.nn as nn
import torch.nn.functional as F

class ModelSimpleLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.load()

    def forward(self, x):
        x = self.linear1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
