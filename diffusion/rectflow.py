"""
diffusion/rectflow.py  —  Rectified Flow
=========================================
Part 6 of EE/CS 148B HW4.

Reference: Liu et al. (2023) "Flow Straight and Fast: Learning to Generate
and Transfer Data with Rectified Flow" (ICLR 2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RectifiedFlow:
    """Rectified Flow forward process, training loss, and ODE sampler.

    The interpolation is:
        X_t = (1 - t) X_0 + t X_1,   t ∈ [0, 1]

    where X_0 ~ π_0 = N(0, I)  and  X_1 ~ π_1  (data).

    The regression target is the velocity  v = X_1 - X_0.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 6.A  Forward process and loss
    # ------------------------------------------------------------------

    def forward_process(
        self, x1: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Sample from the rectified flow interpolation at time t.

        Args:
            x1: Clean data samples, shape (B, *).
            t:  Continuous time in [0, 1], shape (B,).

        Returns:
            (x_t, x0, vel): interpolated point, noise used, and regression
                            target velocity (x1 - x0), all shape (B, *).
        """
        x0 = torch.randn_like(x1)
        # broadcast t: (B,) -> (B, 1, 1, 1) for image tensors
        t_broad = t.view(x1.size(0), *([1] * (x1.dim() - 1)))
        x_t = (1 - t_broad) * x0 + t_broad * x1
        vel = x1 - x0
        return x_t, x0, vel

    def loss(self, v_theta: nn.Module, x1: Tensor) -> Tensor:
        """Rectified Flow training loss (RF objective).

        L_RF(θ) = E_{t,X_0,X_1} [ ‖(X_1 - X_0) - v_θ(X_t, t)‖² ]

        Args:
            v_theta: Velocity network; called as v_theta(x_t, t).
                     t is a float tensor of shape (B,) in [0, 1].
            x1:      Clean data batch, shape (B, C, H, W).

        Returns:
            Scalar loss.
        """
        B = x1.size(0)
        t = torch.rand(B, device=x1.device)
        x_t, _, vel = self.forward_process(x1, t)
        v_pred = v_theta(x_t, t)
        return F.mse_loss(v_pred, vel)

    # ------------------------------------------------------------------
    # 6.B  Euler ODE sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def euler_sample(
        self,
        v_theta: nn.Module,
        shape: tuple[int, ...],
        num_steps: int = 100,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Euler ODE sampler for rectified flow (Problem 6.B).

        Integrates  dX/dt = v_θ(X_t, t)  from t=0 to t=1 using
        uniform step size Δt = 1 / num_steps.

        Args:
            v_theta:   Trained velocity network.
            shape:     Output shape (B, C, H, W).
            num_steps: Number of Euler integration steps.
                       After reflow, a single step (num_steps=1) should
                       produce reasonable samples.
            device:    Target device.

        Returns:
            Generated samples X_1, shape (B, C, H, W).
        """
        dt = 1.0 / num_steps
        x = torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = i * dt
            t_batch = torch.full((shape[0],), t_val, device=device)
            v = v_theta(x, t_batch)
            x = x + v * dt

        return x

    # ------------------------------------------------------------------
    # 6.C  Reflow  (data generation only — retraining uses loss() above)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_reflow_pairs(
        self,
        v_theta: nn.Module,
        n_pairs: int,
        image_shape: tuple[int, ...],
        num_steps: int = 100,
        batch_size: int = 128,
        device: str | torch.device = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Generate (X̂_0, X̂_1) pairs for the reflow procedure (Problem 6.C).

        For each fresh noise sample X̂_0 ~ N(0, I), run the Euler ODE to
        obtain X̂_1 = Φ_1(X̂_0).  The resulting pairs are used to retrain
        the velocity network, producing straighter trajectories.

        Args:
            v_theta:     Trained velocity network (first-round).
            n_pairs:     Total number of pairs to generate (e.g. 50 000).
            image_shape: Spatial shape of one image (C, H, W).
            num_steps:   Euler steps used for the ODE integration.
            batch_size:  Number of pairs to generate per forward pass.
            device:      Target device.

        Returns:
            (x0_all, x1_all): tensors of shape (n_pairs, C, H, W) on CPU.
        """
        v_theta.eval()
        dt = 1.0 / num_steps
        x0_list, x1_list = [], []
        n_done = 0

        while n_done < n_pairs:
            bsz = min(batch_size, n_pairs - n_done)
            x0 = torch.randn(bsz, *image_shape, device=device)
            x = x0.clone()

            for i in range(num_steps):
                t_val = i * dt
                t_batch = torch.full((bsz,), t_val, device=device)
                v = v_theta(x, t_batch)
                x = x + v * dt

            x0_list.append(x0.cpu())
            x1_list.append(x.cpu())
            n_done += bsz

            if n_done % 5000 == 0 or n_done >= n_pairs:
                print(f"  Generated {n_done}/{n_pairs} pairs")

        return torch.cat(x0_list, dim=0)[:n_pairs], torch.cat(x1_list, dim=0)[:n_pairs]
