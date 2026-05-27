"""
tests/adapters.py  —  Bind student implementations to the test harness
======================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor

from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


# ------------------------------------------------------------------
# VP SDE (Part 5.A)
# ------------------------------------------------------------------

def make_vpsde(beta_min: float = 0.01, beta_max: float = 5.0, T: int = 1000) -> VPSDE:
    return VPSDE(beta_min=beta_min, beta_max=beta_max, T=T)


def run_beta(sde: VPSDE, t: Tensor) -> Tensor:
    return sde.beta(t)


def run_c(sde: VPSDE, t: Tensor) -> Tensor:
    return sde.c(t)


def run_sigma(sde: VPSDE, t: Tensor) -> Tensor:
    return sde.sigma(t)


def run_marginal(sde: VPSDE, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    return sde.marginal(x0, t)


# ------------------------------------------------------------------
# Rectified Flow (Part 6.A)
# ------------------------------------------------------------------

def make_rectflow() -> RectifiedFlow:
    return RectifiedFlow()


def run_rf_forward(flow: RectifiedFlow, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    return flow.forward_process(x1, t)
