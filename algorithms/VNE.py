import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from algorithms.ERM import ERM

class VNE(ERM):
    """
        VNE:  von Neumann entropy
        @InProceedings{Kim_2023_CVPR,
        author    = {Kim, Jaeill and Kang, Suhyun and Hwang, Duhun and Shin, Jungwook and Rhee, Wonjong},
        title     = {VNE: An Effective Method for Improving Deep Representation by Manipulating Eigenvalue Distribution},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {3799-3810}
        }
        https://github.com/jaeill/CVPR23-VNE
    """
    def __init__(self, config, train_examples, adapt_examples=None):
        super(VNE, self).__init__(config, train_examples, adapt_examples)

    def get_vne(self, H):
        Z = torch.nn.functional.normalize(H, dim=1)
        # Regularization
        epsilon = 1e-16
        EP = epsilon * torch.eye(Z.shape[0], Z.shape[1]).to(Z.device)
        sing_val = torch.svd((Z + EP) / (np.sqrt(Z.shape[0]) + epsilon))[1]
        eig_val = sing_val ** 2
        return - (eig_val * torch.log(eig_val)).nansum()

    def _train_step(self, step):
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            if step >= self.train_steps:
                break
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            x_batch = x_batch.view(-1, *self.input_shape)
            z = self.featurizer(x_batch)

            # In case svd is not convergent, we wet it to zero
            try:
                vne = self.get_vne(z)
            except:
                vne = 0.

            loss_erm = F.cross_entropy(self.classifier(z), y_batch)
            # Minimizing the von Neumann entropy
            objective = loss_erm - self.config["vne_coef"] * vne

            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()

            step += 1
            if step % self.check_freq == 0:
                print(f"[Step {step}] Training Loss: {objective.item():.4f}")

        return step