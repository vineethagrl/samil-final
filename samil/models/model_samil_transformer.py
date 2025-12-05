


import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMIL_TransformerPool(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.fc_embed = nn.Linear(64 * 24 * 24, embed_dim)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images):
        feats = []
        for img in images:
            x = self.feature_extractor(img)
            x = x.view(x.size(0), -1)
            x = self.fc_embed(x)
            feats.append(x)

        feats = torch.stack(feats, dim=1)
        trans_out = self.transformer(feats)
        bag_rep = trans_out.mean(dim=1)

        output = self.classifier(bag_rep)
        return output
