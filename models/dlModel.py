import torch
import torch.nn as nn

from .utils import generate_mask


class DLModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        task_name: str,
        demo_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super(DLModel, self).__init__()
        self.backbone_name = backbone_name
        self.task_name = task_name
        self.demo_dim = demo_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if backbone_name == 'transformer':
            from .transformer import Transformer
            self.backbone = Transformer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        if backbone_name == 'gru':
            from .gru import GRU
            self.backbone = GRU(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        if backbone_name == 'rnn':
            from .rnn import RNN
            self.backbone = RNN(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        if backbone_name == 'lstm':
            from .lstm import LSTM
            self.backbone = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        elif backbone_name == 'adacare':
            from .adacare import AdaCare
            self.backbone = AdaCare(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        elif backbone_name == 'mhagru':
            from .mhagru import MHAGRU
            self.backbone = MHAGRU(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        elif backbone_name == 'retain':
            from .retain import RETAIN
            self.backbone = RETAIN(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        elif backbone_name == 'concare':
            from .concare import ConCare
            self.backbone = ConCare(lab_dim=self.input_dim-self.demo_dim, demo_dim=self.demo_dim, hidden_dim=self.hidden_dim)
        elif backbone_name == 'safari':
            from .safari import SAFARI
            self.backbone = SAFARI(input_dim=self.input_dim-self.demo_dim, hidden_dim=self.hidden_dim)
        elif backbone_name == 'm3care':
            from .m3care import M3Care
            self.backbone = M3Care(input_dim=self.input_dim, hidden_dim=self.hidden_dim)

        if task_name == "outcome":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())
        elif task_name == "los":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0))
        elif task_name== "multitask":
            from .heads import MultitaskHead
            self.head = MultitaskHead(self.hidden_dim, self.output_dim, drop=0.0)
    
    def forward(
        self,
        x: torch.tensor,
        lens
    ):
        if self.backbone_name == "concare":
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens).to(x.device)
            embedding, decov_loss = self.backbone(x_lab, x_demo, mask)
            embedding, decov_loss = embedding.to(x.device), decov_loss.to(x.device)
            y_hat = self.head(embedding).squeeze()
            return y_hat, decov_loss
        elif self.backbone_name in ['transformer', 'retain', 'gru', 'rnn', 'adacare', 'lstm']:
            mask = generate_mask(lens).to(x.device)
            embedding = self.backbone(x, mask).to(x.device)
            y_hat = self.head(embedding).squeeze()
            return y_hat
        elif self.backbone_name in ['mhagru']:
            embedding, _ = self.backbone(x)
            y_hat = self.head(embedding).squeeze()
            return y_hat
        elif self.backbone_name in ['m3care']:
            embedding = self.backbone(x)
            y_hat = self.head(embedding).squeeze()
            return y_hat
        elif self.backbone_name in ['safari']:
            x_demo, x_lab = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:]
            embedding = self.backbone(x_lab, x_demo)
            y_hat = self.head(embedding).squeeze()
            return y_hat