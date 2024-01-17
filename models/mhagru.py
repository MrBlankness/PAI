import torch
import torch.nn as nn


class MHAGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim: int=32, feat_dim: int=8, output_dim: int=2, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        self.num_heads = 4
        self.act = act_layer()
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.grus = nn.ModuleList(
            [
                nn.GRU(1, feat_dim, num_layers=1, batch_first=True)
                for _ in range(input_dim)
            ]
        )
        self.mha = nn.MultiheadAttention(feat_dim, self.num_heads, dropout=drop, batch_first=True)
        self.time_step_score = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(input_dim * feat_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop)
    
    def forward(self, x, **kwargs):
        # x: [bs, time_steps, input_dim]
        bs, time_steps, input_dim = x.shape

        x_unrolled = x.contiguous().view(-1, input_dim)    # [bs * t, h]
        time_step_importance = self.time_step_score(x_unrolled).view(bs, time_steps) # [bs, t]

        x = self.input_proj(x)   # [bs, time_steps, input_dim] -> [bs, time_steps, hidden_dim]
        out = torch.zeros(bs, time_steps, self.input_dim, self.feat_dim).to(x.device)
        attention = torch.zeros(bs, time_steps, self.input_dim, self.feat_dim).to(x.device)
        for i, gru in enumerate(self.grus):
            cur_feat = x[:, :, i].unsqueeze(-1)     # [bs, time_steps, 1]
            cur_feat = gru(cur_feat)[0]             # [bs, time_steps, feat_dim]
            out[:, :, i] = cur_feat                 # [bs, time_steps, input_dim, feat_dim]
            
            attn_feat = self.mha(cur_feat, cur_feat, cur_feat)[0]
            attention[:, :, i] = attn_feat

        out = out.flatten(2)        # [bs, time, input, feat] -> [bs, time, input * feat]
        out = self.out_proj(out)    # [bs, time, input * feat] -> [bs, time, hidden_dim]

        feature_importance = self.sigmoid(attention.transpose(1, 2).reshape(bs, input_dim, -1).sum(-1).squeeze(-1)) # [bs, input_dim]

        time_step_feature_importance = self.sigmoid(attention.sum(-1).squeeze(-1))  # [bs, time_steps, input_dim]
        
        scores = {
            'feature_importance': feature_importance,
            'time_step_importance': time_step_importance,
            'time_step_feature_importance': time_step_feature_importance
        }
        return out[:, -1, :], scores