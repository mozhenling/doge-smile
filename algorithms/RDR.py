import torch
from algorithms.ERM import ERM

class RDR(ERM):
    """
    Re-weighting with Density Ratio (RDR):

    Ref.:
        [1] J. Luo, F. Hong, J. Yao, B. Han, Y. Zhang, and Y. Wang, “Revive Re-weighting
        in Imbalanced Learning by Density Ratio Estimation”, in 38th Conference on Neural Information Processing
        Systems (NeurIPS 2024).

        https://github.com/GoodMorningPeter/RDR/tree/main

    """
    def __init__(self, config, train_examples, adapt_examples=None):
        super(RDR, self).__init__(config, train_examples, adapt_examples)
        self.config = config
        self.Phi_nu = torch.zeros(self.featurizer.n_outputs, self.num_classes, device=self.device)
        self.updated_labels = set() # Unordered collections of unique elements
        self.cls_num_list = [len([l for l in self.train_set["label"] if l == i]) for i in range(self.num_classes)]

    def _train_step(self, step):
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            if step >= self.train_steps:
                break
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            x_batch = x_batch.view(-1, *self.input_shape)

            #
            feature = self.featurizer(x_batch)
            logits = self.classifier(feature)


            if step < self.config["warmup"]:
                sample_weights = torch.ones_like(y_batch, dtype=torch.float, device=self.device)
            else:
                sample_weights = self.rdr_weights(feature, y_batch, self.cls_num_list, self.Phi_nu, self.device)

            self.optimizer.zero_grad()
            loss_train_batch = self.loss(logits, y_batch, reduction='none')* sample_weights
            loss_train = loss_train_batch.mean()
            loss_train.backward()
            self.optimizer.step()

            self.update_Phi_nu(feature, y_batch, self.Phi_nu, self.config["rdr_momentum"], self.updated_labels)

            step += 1
            if step % self.check_freq == 0:
                print(f"[Step {step}] Training Loss: {loss_train.item():.4f}")

        return step

    def update_Phi_nu(self, feature, target, Phi_nu, rdr_momentum, updated_labels):
        with torch.no_grad():
            class_feature_sum = {}
            class_sample_count = {}

            for index in range(feature.size(0)):
                label = target[index].item()
                if label in class_feature_sum:
                    class_feature_sum[label] += feature[index].detach()
                    class_sample_count[label] += 1
                else:
                    class_feature_sum[label] = feature[index].detach()
                    class_sample_count[label] = 1

            for label in class_feature_sum:
                class_mean_feature = class_feature_sum[label] / class_sample_count[label]
                if label in updated_labels:
                    Phi_nu[:, label] = rdr_momentum* Phi_nu[:, label] + (1 - rdr_momentum) * class_mean_feature
                else:
                    Phi_nu[:, label] = class_mean_feature
                    updated_labels.add(label)

    def rdr_weights(self, feature, target, cls_num_list, Phi_nu, device):
        num_classes = len(cls_num_list)
        sample_weights = torch.zeros(target.size(0), dtype=torch.float, device=device)

        with torch.no_grad():
            for cls in range(num_classes):
                class_mask = (target == cls)
                class_features = feature[class_mask]

                if class_features.size(0) > 0:
                    Phi_de = class_features.T
                    Phi_deT_Phi_de = torch.matmul(Phi_de.T, Phi_de)
                    try:
                        Phi_de_pinv = torch.inverse(Phi_deT_Phi_de)
                    except RuntimeError as e:
                        Phi_de_pinv = torch.pinverse(Phi_deT_Phi_de)

                    Phi_de_pinv_Phi_deT = torch.matmul(Phi_de_pinv, Phi_de.T)
                    Phi_nu_cls = Phi_nu[:, cls].unsqueeze(1)

                    r = torch.matmul(Phi_de_pinv_Phi_deT, Phi_nu_cls) * class_features.size(0)
                    r = torch.clamp(r, min=1e-9)

                    sample_weights[class_mask] = r.squeeze()



            for cls in range(num_classes):
                class_mask = (target == cls)
                sample_weights[class_mask] *= (sum(cls_num_list) / cls_num_list[cls])

            # normalize
            sample_weights = sample_weights / sample_weights.sum() * sample_weights.size(0)

        return sample_weights