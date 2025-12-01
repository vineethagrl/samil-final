import torch, torch.nn as nn
from torchvision.models import resnet18

class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
    def forward(self, H):                # H: [K,d]
        w = self.a(H).squeeze(-1)
        alpha = torch.softmax(w, dim=0)
        Z = (alpha.unsqueeze(-1) * H).sum(0)
        return Z, alpha

class ABMIL(nn.Module):
    def __init__(self, num_classes=3, att_dim=256):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        d = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(d, att_dim)
        self.att = Attention(att_dim)
        self.cls = nn.Linear(att_dim, num_classes)

    def encode_bag(self, bag):           # [K,1,H,W]
        feats = self.backbone(bag)       # [K,d]
        H = self.proj(feats)             # [K,att_dim]
        return H

    def forward(self, bag):
        H = self.encode_bag(bag)
        Z, alpha = self.att(H)
        logits = self.cls(Z)
        return {"logits": logits, "att": alpha.detach()}
