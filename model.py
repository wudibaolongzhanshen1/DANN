from torch import nn
import torch
from GRL import GRL


class DANN(nn.module):
    def __init__(self,lambda_):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32, 48, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.label_predictor = nn.Sequential(
            nn.Linear(28*28*4, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        self.domain_predictor = nn.Sequential(
            GRL(lambda_), # GRL层
            nn.Linear(28*28*4, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        class_pred = self.label_predictor(features)
        domain_pred = self.domain_predictor(features)
        return class_pred, domain_pred