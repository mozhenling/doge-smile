import torch
import random
import torch.nn as nn
from algorithms.ERM import ERM

class MixStyle(ERM):
    """
    MixStyle adapted from the following, manipulating the same features as other methods:

    Ref.:
        [1] K. Zhou, Y. Yang, Y. Qiao, and T. Xiang, “MixStyle Neural Networks for Domain
        Generalization and Adaptation,” Int J Comput Vis, vol. 132, no. 3, pp. 822–836,
        Mar. 2024, doi: 10.1007/s11263-023-01913-8.

    """
    def __init__(self, config, train_examples, adapt_examples=None):
        super(MixStyle, self).__init__(config, train_examples, adapt_examples)

        self.mixstyle = MixStyleMechanism(config["mixup_alpha"])

    def _train_step(self, step):
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            if step >= self.train_steps:
                break
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            x_batch = x_batch.view(-1, *self.input_shape)

            # Obtain latent features, mix style, and get logits
            z_batch = self.featurizer(x_batch)
            z_mix_batch = self.mixstyle(z_batch)
            logits = self.classifier(z_mix_batch)

            self.optimizer.zero_grad()
            loss_train = self.loss(logits, y_batch)
            loss_train.backward()
            self.optimizer.step()

            step += 1
            if step % self.check_freq == 0:
                print(f"[Step {step}] Training Loss: {loss_train.item():.4f}")

        return step

class MixStyleMechanism(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix