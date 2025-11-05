import torch.nn as nn
import torch.nn.functional as F

class ModelSimple(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = F.relu(x)
        # x = F.softmax(x, dim=1)
        return x
