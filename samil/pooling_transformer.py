import torch, torch.nn as nn

class TransformerPool(nn.Module):
    def __init__(self, d, nhead=4, depth=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                           dim_feedforward=2*d, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.cls = nn.Parameter(torch.randn(1,1,d))

    def forward(self, H):       # H: [K,d]
        x = torch.cat([self.cls, H.unsqueeze(0)], dim=1)  # [1,1+K,d]
        y = self.encoder(x)
        z = y[:,0]                                        # CLS
        return z.squeeze(0)                               # [d]
