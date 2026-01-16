import copy
from algorithms.ERM import ERM
import torch

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

class ERMppMovAvg:
    def __init__(self, network):
        self.model = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 600
        self.global_iter = 0
        self.sma_count = 0

    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.model.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone())/(1.+self.sma_count))
        else:
            for (name,param_q), (_,param_k) in zip(self.model.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)

class ERMpp(ERM, ERMppMovAvg):
    """
    Empirical Risk Minimization with improvements (ERM++)

    Ref.:
        [1] P. Teterwak, K. Saito, T. Tsiligkaridis, K. Saenko and B. A. Plummer,
        "ERM++: An Improved Baseline for Domain Generalization," 2025
        IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),
        Tucson, AZ, USA, 2025, pp. 8525-8535, doi: 10.1109/WACV61041.2025.00826.
        [2] https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """
    def __init__(self, config, train_examples, adapt_examples=None):
        ERM.__init__(self, config, train_examples, adapt_examples)
        if self.config["lars"]:
            self.optimizer = LARS(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config['weight_decay'],
                # foreach=False
            )

        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
                weight_decay=self.config['weight_decay'],
                foreach=False
            )

        linear_parameters = []
        for n, p in self.model[1].named_parameters():
            linear_parameters.append(p)

        if self.config["lars"]:
            self.linear_optimizer = LARS(
                linear_parameters,
                lr=self.config["linear_lr"],
                weight_decay=self.config['weight_decay'],
                # foreach=False
            )

        else:
            self.linear_optimizer = torch.optim.Adam(
                linear_parameters,
                lr=self.config["linear_lr"],
                weight_decay=self.config['weight_decay'],
                foreach=False
            )
        self.lr_schedule = []
        self.lr_schedule_changes = 0
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=1)
        ERMppMovAvg.__init__(self, self.model)

    def _train_step(self, step):
        self.model.train()
        for x_batch, y_batch in self.train_loader:
            if step >= self.train_steps:
                break
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            x_batch = x_batch.view(-1, *self.input_shape)

            if self.global_iter > self.config["linear_steps"]:
                selected_optimizer = self.optimizer
            else:
                selected_optimizer = self.linear_optimizer

            loss = self.loss(self.model(x_batch), y_batch)

            selected_optimizer.zero_grad()
            loss.backward()
            selected_optimizer.step()
            self.update_sma()
            if not self.config["freeze_bn"]:
                self.network_sma.train()
                self.network_sma(x_batch)

            step += 1
            if step % self.check_freq == 0:
                print(f"[Step {step}] Training Loss: {loss.item():.4f}")

        return step