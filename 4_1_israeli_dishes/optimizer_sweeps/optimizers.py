# Lion algorithm adapted from Google Research's Apache-2.0 implementation.
# Copyright 2023 Google Research. All Rights Reserved.
# https://github.com/google/automl/blob/master/lion/lion_pytorch.py
# Licensed under Apache-2.0; see LICENSE-LION.txt in this directory.
"""Explicit optimizer construction; only the trainable LoRA matrices are passed."""
from __future__ import annotations

import math
import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not math.isfinite(lr) or lr <= 0 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid Lion learning rate or betas")
        super().__init__(params, dict(lr=lr, betas=tuple(betas), weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Lion requires dense gradients")
                state = self.state[parameter]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                momentum = state["exp_avg"]
                direction = (beta1 * momentum + (1 - beta1) * gradient).sign()
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"])
                momentum.mul_(beta2).add_(gradient, alpha=1 - beta2)
        return loss


def make_optimizer(config, parameters):
    params = list(parameters)
    if not params or any(not parameter.requires_grad or parameter.ndim != 2 for parameter in params):
        raise ValueError("Expected only trainable 2D LoRA A/B parameters")
    kwargs = dict(config["optimizer_kwargs"])
    for key in ("betas", "ns_coefficients"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    classes = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD, "lion": Lion, "muon": torch.optim.Muon}
    return classes[config["optimizer_name"]](params, **kwargs)
