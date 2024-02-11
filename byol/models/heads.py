import torch.nn as nn

class BYOLProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, output_dim: int = 256) -> None:
        super(BYOLProjectionHead, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BYOLPredictionHead(nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 1024, output_dim: int = 256) -> None:
        super(BYOLPredictionHead, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=True)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)