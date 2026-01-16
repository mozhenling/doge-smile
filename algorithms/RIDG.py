import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.ERM import ERM

class RIDG(ERM):
    """
    Rational Invariance for Domain Generalization (RIDG)

    @InProceedings{Chen_2023_ICCV,
    author    = {Chen, Liang and Zhang, Yong and Song, Yibing and van den Hengel, Anton and Liu, Lingqiao},
    title     = {Domain Generalization via Rationale Invariance},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {1751-1760}
    }
    """
    def __init__(self, config, train_examples, adapt_examples=None):
        super(RIDG, self).__init__(config, train_examples, adapt_examples)
        self.rational_bank = torch.zeros(self.num_classes, self.num_classes, self.featurizer.n_outputs, device=self.device)
        self.init = torch.ones(self.num_classes, device=self.device)

    def _train_step(self, step):
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            if step >= self.train_steps:
                break
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            x_batch = x_batch.view(-1, *self.input_shape)

            features = self.featurizer(x_batch)
            logits = self.model(x_batch)
            rational = torch.zeros(self.num_classes, x_batch.shape[0], self.featurizer.n_outputs, device=self.device)
            for i in range(self.num_classes):
                rational[i] = (self.classifier.weight[i] * features)

            classes = torch.unique(y_batch)
            loss_rational = 0
            for i in range(classes.shape[0]):
                rational_mean = rational[:, y_batch == classes[i]].mean(dim=1)
                if self.init[classes[i]]:
                    self.rational_bank[classes[i]] = rational_mean
                    self.init[classes[i]] = False
                else:
                    self.rational_bank[classes[i]] = (1 - self.config['momentum']) * self.rational_bank[classes[i]] + \
                                                     self.config['momentum'] * rational_mean
                loss_rational += ((rational[:, y_batch == classes[i]] - (
                    self.rational_bank[classes[i]].unsqueeze(1)).detach()) ** 2).sum(dim=2).mean()

            loss = F.cross_entropy(logits, y_batch)
            loss += self.config['ridg_reg'] * loss_rational

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            step += 1
            if step % self.check_freq == 0:
                print(f"[Step {step}] Training Loss: {loss.item():.4f}")

        return step