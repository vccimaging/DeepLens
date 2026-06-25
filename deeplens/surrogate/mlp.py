import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Fully-connected network for low-frequency PSF prediction.

    Predicts PSFs as flattened vectors using stacked linear layers with ReLU
    activations and a Sigmoid output. The output is L1-normalized so it sums to 1
    (valid as a PSF energy distribution).

    Args:
        in_features (int): Number of input features (e.g., field angle + wavelength).
        out_features (int): Number of output features (flattened PSF size).
        hidden_features (int): Width of hidden layers. Defaults to 64.
        hidden_layers (int): Number of hidden layers. Defaults to 3.
    """

    def __init__(self, in_features, out_features, hidden_features=64, hidden_layers=3):
        super(MLP, self).__init__()

        layers = [
            nn.Linear(in_features, hidden_features // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features // 4, hidden_features, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_features, hidden_features, bias=True),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.extend(
            [nn.Linear(hidden_features, out_features, bias=True), nn.Sigmoid()]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.

        Returns:
            x (torch.Tensor): L1-normalized output tensor of shape
                `(batch_size, out_features)`, summing to 1 along the last dim.
        """
        x = self.net(x)
        x = F.normalize(x, p=1, dim=-1)
        return x


if __name__ == "__main__":
    # Test the network
    mlp = MLP(4, 64, hidden_features=64, hidden_layers=3)
    print(mlp)
    x = torch.rand(100, 4)
    y = mlp(x)
    print(y.size())
