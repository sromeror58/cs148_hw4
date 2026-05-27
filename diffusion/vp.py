"""
diffusion/vp.py  —  Variance-Preserving (VP) SDE
=================================================
Part 5 of EE/CS 148B HW4.

Reference: Song et al. (2021) "Score-Based Generative Modeling through
Stochastic Differential Equations" (Song21), Appendix B & D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class VPSDE:
    """Variance-Preserving SDE forward process and samplers.

    The VP-SDE is:
        dx = -½ β(t) x dt + √β(t) dB_t

    with β(t) = β_min + (β_max - β_min) * t  (linear schedule).

    Args:
        beta_min: Minimum noise schedule value β_min.
        beta_max: Maximum noise schedule value β_max.
        T:        Number of discrete time steps (used by the EM/PC samplers).
    """

    def __init__(self, beta_min: float = 0.01, beta_max: float = 5.0, T: int = 1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    # ------------------------------------------------------------------
    # 5.A  Defining the VP SDE
    # ------------------------------------------------------------------

    def beta(self, t: Tensor) -> Tensor:
        """β(t) — the linear noise schedule.

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            β(t), same shape as t.

        Reference: Eq. (32) of Song21.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def c(self, t: Tensor) -> Tensor:
        """c(t) = exp(-½ ∫_0^t β(s) ds) — the signal decay factor.

        For a linear β schedule:
            ∫_0^t β(s) ds = β_min * t + ½ (β_max - β_min) * t²

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            c(t), same shape as t.

        Reference: Eq. (33) of Song21.
        """
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def sigma(self, t: Tensor) -> Tensor:
        """σ(t) = √(1 - c(t)²) — the noise standard deviation.

        Args:
            t: Continuous time in [0, 1], shape (*).

        Returns:
            σ(t), same shape as t.
        """
        return torch.sqrt((1 - self.c(t) ** 2).clamp(min=0.0))

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """Drift coefficient  f(x, t) = -½ β(t) x.

        Args:
            x: State tensor, shape (B, *).
            t: Time tensor, shape (B,) broadcast-compatible with x.

        Returns:
            Drift f(x, t), same shape as x.
        """
        beta_t = self.beta(t)
        # broadcast (B,) -> (B, 1, 1, 1) for image tensors
        for _ in range(x.dim() - 1):
            beta_t = beta_t.unsqueeze(-1)
        return -0.5 * beta_t * x

    def diffusion(self, t: Tensor) -> Tensor:
        """Diffusion coefficient  g(t) = √β(t).

        Args:
            t: Time tensor, shape (*).

        Returns:
            g(t), same shape as t.
        """
        return torch.sqrt(self.beta(t))

    def marginal(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Sample from the forward marginal  q(x_t | x_0).

        The marginal satisfies:
            x_t = c(t) * x_0 + σ(t) * ε,   ε ~ N(0, I)

        Args:
            x0: Clean data, shape (B, *).
            t:  Continuous time in [0, 1], shape (B,).

        Returns:
            (x_t, eps): noised sample and the noise used, both shape (B, *).
        """
        c_t = self.c(t)
        s_t = self.sigma(t)
        # broadcast (B,) -> (B, 1, 1, 1)
        for _ in range(x0.dim() - 1):
            c_t = c_t.unsqueeze(-1)
            s_t = s_t.unsqueeze(-1)
        eps = torch.randn_like(x0)
        x_t = c_t * x0 + s_t * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # 5.B  Samplers
    # ------------------------------------------------------------------

    def _broadcast(self, v: Tensor, ndim: int) -> Tensor:
        """Unsqueeze v from (B,) to (B, 1, ..., 1) with total ndim dims."""
        for _ in range(ndim - 1):
            v = v.unsqueeze(-1)
        return v

    @torch.no_grad()
    def euler_maruyama(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Euler-Maruyama reverse-SDE sampler (Problem 5.B.i).

        Starting from x(T=1) ~ N(0, σ(1)² I), integrates the reverse VP-SDE:
            dx = [-½ β(t) x - β(t) ∇_x log p_t(x)] dt + √β(t) dB̄_t

        Args:
            score_model: Trained score network s_θ(x, t).
                         Called as `score_model(x, t)` where t is a float
                         tensor of shape (B,) with values in [0, 1].
            shape:       Output shape (B, C, H, W).
            num_steps:   Number of discretisation steps (default: self.T).
            device:      Target device.

        Returns:
            Generated samples, shape (B, C, H, W), values in [-1, 1].
        """
        num_steps = num_steps or self.T
        dt = 1.0 / num_steps
        B = shape[0]
        ndim = len(shape)

        # Initialise x ~ N(0, σ(1)² I)
        t1 = torch.ones(B, device=device)
        sigma_T = self._broadcast(self.sigma(t1), ndim)
        x = sigma_T * torch.randn(shape, device=device)

        # Reverse: t from 1 down to dt
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_val = max(t_val, 1e-5)
            t_batch = torch.full((B,), t_val, device=device)

            # Score: model predicts noise ε_θ, score = -ε_θ / σ(t)
            eps_theta = score_model(x, t_batch)
            sigma_t = self._broadcast(self.sigma(t_batch), ndim).clamp(min=1e-8)
            score = -eps_theta / sigma_t

            beta_t = self._broadcast(self.beta(t_batch), ndim)

            # Reverse SDE EM update:
            # x_{t-dt} = x + [f(x,t) - g²·score]·dt + g·√dt·z
            # f = -½β·x, g² = β
            z = torch.randn_like(x) if i < num_steps - 1 else torch.zeros_like(x)
            drift_term = -0.5 * beta_t * x - beta_t * score
            x = x + drift_term * dt + torch.sqrt(beta_t * dt) * z

        return x.clamp(-1, 1)

    @torch.no_grad()
    def predictor_corrector(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        n_corrector: int = 1,
        snr: float = 0.16,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Predictor-Corrector sampler with EM predictor (Problem 5.B.ii).

        Follows Algorithm 5 of Song21.  Each predictor step is an EM step;
        each corrector step is one step of annealed Langevin dynamics.

        Args:
            score_model:  Trained score network s_θ(x, t).
            shape:        Output shape (B, C, H, W).
            num_steps:    Number of predictor steps (default: self.T).
            n_corrector:  Number of Langevin corrector steps per predictor step.
            snr:          Signal-to-noise ratio for the corrector step size.
            device:       Target device.

        Returns:
            Generated samples, shape (B, C, H, W), values in [-1, 1].
        """
        num_steps = num_steps or self.T
        dt = 1.0 / num_steps
        B = shape[0]
        ndim = len(shape)

        # Initialise x ~ N(0, σ(1)² I)
        t1 = torch.ones(B, device=device)
        sigma_T = self._broadcast(self.sigma(t1), ndim)
        x = sigma_T * torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_val = max(t_val, 1e-5)
            t_batch = torch.full((B,), t_val, device=device)

            # ---- Corrector: annealed Langevin dynamics ----
            for _ in range(n_corrector):
                eps_theta = score_model(x, t_batch)
                sigma_t = self._broadcast(self.sigma(t_batch), ndim).clamp(min=1e-8)
                score = -eps_theta / sigma_t

                z = torch.randn_like(x)
                # Per-sample L2 norms for adaptive step size
                score_norm = score.flatten(1).norm(dim=1)  # (B,)
                z_norm = z.flatten(1).norm(dim=1)           # (B,)
                score_norm = self._broadcast(score_norm, ndim).clamp(min=1e-8)
                z_norm = self._broadcast(z_norm, ndim)

                eps_step = 2.0 * (snr * z_norm / score_norm) ** 2
                x = x + eps_step * score + torch.sqrt(2.0 * eps_step) * z

            # ---- Predictor: EM step ----
            eps_theta = score_model(x, t_batch)
            sigma_t = self._broadcast(self.sigma(t_batch), ndim).clamp(min=1e-8)
            score = -eps_theta / sigma_t

            beta_t = self._broadcast(self.beta(t_batch), ndim)
            z = torch.randn_like(x) if i < num_steps - 1 else torch.zeros_like(x)
            drift_term = -0.5 * beta_t * x - beta_t * score
            x = x + drift_term * dt + torch.sqrt(beta_t * dt) * z

        return x.clamp(-1, 1)

    @torch.no_grad()
    def ddim_sample(
        self,
        score_model: nn.Module,
        shape: tuple[int, ...],
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Deterministic DDIM / probability-flow ODE sampler.

        Uses the continuous-time DDIM update rule (η=0):
            x_s = (c(s)/c(t))·x_t + [σ(s) - σ(t)·c(s)/c(t)]·ε_θ(x_t, t)

        Args:
            score_model: Trained score network.
            shape:       Output shape (B, C, H, W).
            num_steps:   Number of ODE steps.
            device:      Target device.

        Returns:
            Generated samples, shape (B, C, H, W).
        """
        num_steps = num_steps or self.T
        B = shape[0]
        ndim = len(shape)

        # Time schedule from t=1 to t=0
        times = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        t1 = torch.ones(B, device=device)
        sigma_T = self._broadcast(self.sigma(t1), ndim)
        x = sigma_T * torch.randn(shape, device=device)

        for i in range(num_steps):
            t_cur = times[i].item()
            t_next = times[i + 1].item()
            t_batch = torch.full((B,), t_cur, device=device)

            eps_theta = score_model(x, t_batch)

            c_t = self._broadcast(self.c(t_batch), ndim).clamp(min=1e-8)
            s_t = self._broadcast(self.sigma(t_batch), ndim)

            t_next_batch = torch.full((B,), max(t_next, 1e-5), device=device)
            c_s = self._broadcast(self.c(t_next_batch), ndim)
            s_s = self._broadcast(self.sigma(t_next_batch), ndim)

            # Continuous-time DDIM (η=0)
            x = (c_s / c_t) * x + (s_s - s_t * c_s / c_t) * eps_theta

        return x.clamp(-1, 1)

    # ------------------------------------------------------------------
    # 5.D  Inverse problems (EC)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self,
        score_model: nn.Module,
        corrupted: Tensor,
        mask: Tensor,
        num_steps: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Tensor:
        """Conditional reverse diffusion for inpainting (EC Problem 5.D).

        At each reverse step, replaces the known pixels with their
        forward-diffused ground-truth values, conditioning the reverse
        process on the observed measurements.

        Reference: Song et al. (2022) "Solving Inverse Problems in Medical
        Imaging with Score-Based Generative Models".

        Args:
            score_model: Trained score network s_θ(x, t).
            corrupted:   Observed (corrupted) image, shape (B, C, H, W).
                         Unknown pixels are set to 0.
            mask:        Binary mask, shape (B, 1, H, W).
                         1 = observed pixel, 0 = missing pixel.
            num_steps:   Reverse steps (default: self.T).
            device:      Target device.

        Returns:
            Reconstructed images, shape (B, C, H, W).
        """
        num_steps = num_steps or self.T
        dt = 1.0 / num_steps
        B = corrupted.shape[0]
        ndim = corrupted.dim()
        shape = corrupted.shape

        corrupted = corrupted.to(device)
        mask = mask.to(device)

        # Initialise x ~ N(0, σ(1)² I)
        t1 = torch.ones(B, device=device)
        sigma_T = self._broadcast(self.sigma(t1), ndim)
        x = sigma_T * torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_val = max(t_val, 1e-5)
            t_batch = torch.full((B,), t_val, device=device)

            # Replace known pixels with forward-noised clean values
            x_known, _ = self.marginal(corrupted, t_batch)
            x = mask * x_known + (1 - mask) * x

            # Standard EM reverse step
            eps_theta = score_model(x, t_batch)
            sigma_t = self._broadcast(self.sigma(t_batch), ndim).clamp(min=1e-8)
            score = -eps_theta / sigma_t

            beta_t = self._broadcast(self.beta(t_batch), ndim)
            z = torch.randn_like(x) if i < num_steps - 1 else torch.zeros_like(x)
            drift_term = -0.5 * beta_t * x - beta_t * score
            x = x + drift_term * dt + torch.sqrt(beta_t * dt) * z

        # Final replacement of known pixels with clean values
        x = mask * corrupted + (1 - mask) * x
        return x.clamp(-1, 1)
