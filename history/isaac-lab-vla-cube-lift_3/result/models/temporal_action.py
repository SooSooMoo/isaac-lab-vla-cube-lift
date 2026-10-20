import torch
from torch import nn

class TemporalAction(nn.Module):
    """Causal recurrent action decoder; no teacher phase or step input."""
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU())
        self.memory = nn.GRU(128, 128, batch_first=True)
        self.output = nn.Linear(128, 7)
        self.register_buffer('feature_mean', torch.zeros(256))
        self.register_buffer('feature_scale', torch.ones(256))
        self.register_buffer('action_mean', torch.zeros(6))
        self.register_buffer('action_scale', torch.ones(6))

    def forward(self, features, hidden=None):
        x = self.project((features-self.feature_mean)/self.feature_scale)
        x, hidden = self.memory(x, hidden)
        y = self.output(x)
        arm = (y[..., :6]*self.action_scale+self.action_mean).clamp(-1, 1)
        return arm, y[..., 6], hidden
