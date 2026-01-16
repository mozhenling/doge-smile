import torch
import torch.nn as nn
from algorithms.ERM import ERM
from utils.network.basenet import Featurizer, Classifier
import os

class AGLU(ERM):
    """
    Adaptive Generalised Linear Unit (AGLU)
    Ref.: [1] K. P. Alexandridis, J. Deng, A. Nguyen, and S. Luo,
         “Adaptive Parametric Activation,” in Computer Vision – ECCV 2024, vol. 15112,
          A. Leonardis, E. Ricci, S. Roth, O. Russakovsky, T. Sattler, and G. Varol, Eds.,
          in Lecture Notes in Computer Science, vol. 15112. , Cham: Springer Nature Switzerland,
          2025, pp. 455–476. doi: 10.1007/978-3-031-72949-2_26.

    """
    def __init__(self, config, train_examples, adapt_examples=None):
        super(AGLU, self).__init__(config, train_examples, adapt_examples)
        # -- model
        self.featurizer = Featurizer(in_channel=self.input_shape[0])
        self.aglu = Unified()
        self.classifier = Classifier(in_features=self.featurizer.n_outputs, out_features=self.num_classes)
        self.model = nn.Sequential(self.featurizer, self.aglu, self.classifier)

        if self.model_path is not None and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Loaded model from {self.model_path}")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class Unified(nn.Module):
    """Unified activation function module."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(1, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(1, **factory_kwargs))
        self.softplus = nn.Softplus(beta=-1.0)
        self.lambda_param = nn.Parameter(lambda_param)
        self.kappa_param = nn.Parameter(kappa_param)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass of the Unified activation function."""
        l = torch.clamp(self.lambda_param, min=0.0001)
        p = torch.exp((1 / l) * self.softplus((self.kappa_param * input) - torch.log(l)))
        return p  # for AGLU simply return p*input
