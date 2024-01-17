from torch import nn
from .utils import get_last_visit


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # self.proj = nn.Linear(input_dim, hidden_dim)
        # self.act = act_layer()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x, mask, **kwargs):
        x, _ = self.lstm(x)
        out = get_last_visit(x, mask)
        return out