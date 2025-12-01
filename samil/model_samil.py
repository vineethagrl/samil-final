import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from .losses import supervised_attention_kld, focal_kl

class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1))
    def forward(self, H):
        w = self.a(H).squeeze(-1)
        alpha = torch.softmax(w, dim=0)
        Z = (alpha.unsqueeze(-1) * H).sum(0)
        return Z, alpha

class SAMIL(nn.Module):
    def __init__(self, num_classes=3, att_dim=256, lambda_sa=0.5,
                 use_bag_contrastive=True, use_focal_kl=False, gamma=2.0):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
        d = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.proj = nn.Linear(d, att_dim)
        self.att = Attention(att_dim)
        self.sup_att = Attention(att_dim)
        self.cls = nn.Linear(att_dim, num_classes)
        self.lambda_sa = lambda_sa
        self.use_bag_contrastive = use_bag_contrastive
        self.use_focal_kl = use_focal_kl
        self.gamma = gamma
        if use_bag_contrastive:
            self.bn = nn.BatchNorm1d(att_dim)

    def encode_bag(self, bag):
        feats = self.backbone(bag)
        H = self.proj(feats)
        return H

    def forward(self, bag, target=None, att_target=None):
        H = self.encode_bag(bag)
        Z, alpha = self.att(H)
        logits = self.cls(Z)
        out = {"logits": logits, "att": alpha.detach()}
        loss = None
        if target is not None:
            ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            if att_target is not None and self.lambda_sa>0:
                att_target = att_target / (att_target.sum()+1e-8)
                sa = (focal_kl(alpha, att_target, self.gamma)
                      if self.use_focal_kl else supervised_attention_kld(alpha, att_target))
                ce = ce + self.lambda_sa * sa
            loss = ce
        out["loss"] = loss
        if self.use_bag_contrastive and target is not None:
            out["rep"] = self.bn(Z)
        return out
