# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.network.basenet import Featurizer, Classifier
import os

class ERM():
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, config, train_examples, adapt_examples=None):
        self.is_llm = False
        self.train_required = True
        # -- all configurations
        self.config = config
        self.input_shape = config["input_shape"]
        self.num_classes = config["num_classes"]
        self.num_domains = config["num_domains"]

        self.lr = config["lr"]
        self.device = config["device"]
        self.train_steps = config["train_steps"]
        self.weight_decay = config["weight_decay"]
        self.valid_ratio = config["valid_ratio"]
        self.batch_size = config["batch_size"]
        self.model_path = config["model_path"]
        self.patience = config["patience"]
        self.check_freq = config["check_freq"]

        # available datasets (pool all available domains into one)
        self.avail_data = train_examples["data"]
        self.avail_label = train_examples["label"]
        self.avail_env = train_examples["env"]

        if adapt_examples is not None:
            self.avail_data=torch.cat([self.avail_data, adapt_examples["data"]], dim=0)
            self.avail_label=torch.cat([self.avail_label, adapt_examples["label"]], dim=0)
            self.avail_label=torch.cat([self.avail_env, adapt_examples["env"]], dim=0)

        self.update_count = 0

        # # -- train-validation split (applicable when data size is large)
        if self.valid_ratio is not None:
            train_num = len(self.avail_data) - int(self.valid_ratio * len(self.avail_data))
            self.train_set = {"data": self.avail_data[:train_num], "label": self.avail_label[:train_num]}
            self.valid_set = {"data": self.avail_data[train_num:], "label": self.avail_label[train_num:]}
        else:
            # --train-validation no-split (applicable when data size is too small)
            self.train_set = {"data": self.avail_data, "label": self.avail_label}
            self.valid_set = {"data": self.avail_data, "label": self.avail_label}

        # -- model
        self.featurizer = Featurizer(in_channel=self.input_shape[0])
        self.classifier = Classifier(in_features=self.featurizer.n_outputs, out_features=self.num_classes)
        self.model = nn.Sequential(self.featurizer, self.classifier)

        if self.model_path is not None and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Loaded model from {self.model_path}")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss = F.cross_entropy

        # -- data loaders
        self.train_dataset = TensorDataset(self.train_set["data"], self.train_set["label"])
        self.valid_dataset = TensorDataset(self.valid_set["data"], self.valid_set["label"])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def train_val(self):
        best_val_acc = 0.0
        best_model_state = None
        steps_since_improvement = 0
        step = 0

        while step < self.train_steps:
            step = self._train_step(step)

            val_acc, val_loss = self._validate()
            print(f"[Step {step}] Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict()
                steps_since_improvement = 0
                print(f"New best model at step {step}, val_acc = {val_acc:.4f}")
            else:
                steps_since_improvement += 1
                print(f"No improvement for {steps_since_improvement} steps")

            if steps_since_improvement >= self.patience:
                print(f"Early stopping at step {step}. No improvement for {self.patience} steps.")
                break

        if self.model_path is not None and best_model_state:
            torch.save(best_model_state, self.model_path)
            print(f"Best model saved to {self.model_path} with val_acc = {best_val_acc:.4f}")

    def _train_step(self, step):
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            if step >= self.train_steps:
                break
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            x_batch = x_batch.view(-1, *self.input_shape)
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss_train = self.loss(logits, y_batch)
            loss_train.backward()
            self.optimizer.step()

            step += 1
            if step % self.check_freq == 0:
                print(f"[Step {step}] Training Loss: {loss_train.item():.4f}")

        return step

    def _validate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in self.valid_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch = x_batch.view(-1, *self.input_shape)
                logits = self.model(x_batch)
                val_loss += self.loss(logits, y_batch).item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total
        avg_val_loss = val_loss / len(self.valid_loader)
        return val_acc, avg_val_loss

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.view(-1, *self.input_shape).to(self.device)
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()