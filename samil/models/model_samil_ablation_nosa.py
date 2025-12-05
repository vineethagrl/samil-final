



import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMIL_NoSA(nn.Module):
    def __init__(self, embed_dim=256, num_classes=3):
        super().__init__()


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_embed = nn.Linear(64 * 24 * 24, embed_dim)


        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )


        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images):

        feats = []
        for img in images:
            x = self.feature_extractor(img)
            x = x.view(x.size(0), -1)
            x = self.fc_embed(x)
            feats.append(x)

        feats = torch.stack(feats, dim=1)               
        attn = self.attention(feats).squeeze(-1)        
        attn = torch.softmax(attn, dim=1)             

        bag_rep = torch.sum(attn.unsqueeze(-1) * feats, dim=1)
        output = self.classifier(bag_rep)
        return output, attn
