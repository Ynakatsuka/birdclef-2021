"""
Ref: https://github.com/vballoli/nfnets-pytorch
"""
from collections import Iterable

import torch
from torch import nn
from torch.optim.optimizer import required


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError("Wrong input dimensions")

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5


class AGC(torch.optim.Optimizer):
    """Generic implementation of the Adaptive Gradient Clipping
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
      model (torch.nn.Module, optional): The original model
      ignore_agc (str, Iterable, optional): Layers for AGC to ignore
    """

    def __init__(
        self,
        params,
        optim: torch.optim.Optimizer,
        clipping: float = 1e-2,
        eps: float = 1e-3,
        model=None,
        ignore_agc=["fc"],
    ):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}

        if not isinstance(ignore_agc, Iterable):
            ignore_agc = [ignore_agc]

        if model is not None:
            assert ignore_agc not in [
                None,
                [],
            ], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
            names = [name for name, module in model.named_modules()]

            for module_name in ignore_agc:
                if module_name not in names:
                    raise ModuleNotFoundError(
                        "Module name {} not found in the model".format(module_name)
                    )
            parameters = [
                {"params": module.parameters()}
                for name, module in model.named_modules()
                if name not in ignore_agc
            ]

        super(AGC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_norm = torch.max(
                    unitwise_norm(p.detach()), torch.tensor(group["eps"]).to(p.device)
                )
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * group["clipping"]

                trigger = grad_norm < max_norm

                clipped_grad = p.grad * (
                    max_norm
                    / torch.max(grad_norm, torch.tensor(1e-6).to(grad_norm.device))
                )
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)


class SGD_AGC(torch.optim.Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__
    AGC from NFNets: https://arxiv.org/abs/2102.06171.pdf.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        dampening (float, optional): dampening for momentum (default: 0.01)
        clipping (float, optional): clipping value (default: 1e-3)
        eps (float, optional): eps (default: 1e-3)
    Example:
        >>> optimizer = torch.optim.SGD_AGC(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    .. note::
        The implementation has been adapted from the PyTorch framework and the official
        NF-Nets paper.
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        clipping=1e-2,
        eps=1e-3,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            clipping=clipping,
            eps=eps,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_AGC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_AGC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_norm = torch.max(
                    unitwise_norm(p.detach()), torch.tensor(group["eps"]).to(p.device)
                )
                grad_norm = unitwise_norm(p.grad.detach())
                max_norm = param_norm * group["clipping"]

                trigger = grad_norm < max_norm

                clipped_grad = p.grad * (
                    max_norm
                    / torch.max(grad_norm, torch.tensor(1e-6).to(grad_norm.device))
                )
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
