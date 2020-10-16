import torch.optim as optim
import numpy as np


class Transformer_ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self._optimizer = optimizer
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", factor=0.5, patience=2
        )
        self.n_warmup_steps = n_warmup_steps
        self.warmup_coeff = 8e-4 / n_warmup_steps
        self.n_current_steps = 0

    def step(self):
        cur_step = self.n_current_steps + 1
        if cur_step <= self.n_warmup_steps:
            lr = cur_step * self.warmup_coeff
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = lr

        self._optimizer.step()
        self.n_current_steps += 1

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def schedule(self, score):
        self._scheduler.step(score)


class MyScheduledOptim:
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", factor=0.5, patience=4, verbose=True, eps=1e-12
        )

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def schedule(self, fscore):
        self._scheduler.step(fscore)
