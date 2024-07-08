import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNet121(nn.Module):
    def __init__(self, hidden_units, out_size, drop_rate=0):
        super(DenseNet121, self).__init__()
        self.densenet121 = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = self.densenet121.classifier.in_features

        self.densenet121.fc_layer = nn.Linear(num_ftrs, hidden_units)
        self.densenet121.classifier = nn.Linear(hidden_units, out_size)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        fmaps_b4 = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(
            fmaps_b4, (1, 1)).view(fmaps_b4.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)

        feature4 = self.densenet121.fc_layer(out)
        logit_b4 = self.densenet121.classifier(feature4)
        return feature4, logit_b4
