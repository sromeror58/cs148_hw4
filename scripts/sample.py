"""
scripts/sample.py  —  Generate and compare samples (Parts 5C, 6B, 6D)
=======================================================================

Usage::
    # EM samples  (5.C.iii)
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000

    # PC samples  (5.C.iv)
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 1
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 3

    # Rectified Flow Euler  (6.B)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \\
        --num_steps 100

    # One-step reflow  (6.C)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow_reflow/best.pt \\
        --num_steps 1

    # Side-by-side grid  (6.D): pass a fixed seed file
    python scripts/sample.py --method all --vp_checkpoint runs/vp/best.pt \\
        --rf_checkpoint runs/rectflow/best.pt \\
        --reflow_checkpoint runs/rectflow_reflow/best.pt \\
        --seed 42 --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, samples.size(0) // nrow + 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      type=str, default="em",
                   choices=["em", "pc", "rectflow", "all"],
                   help="Sampler to run (or 'all' for side-by-side grid).")
    # VP checkpoints
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--vp_checkpoint", type=str, default=None)
    # Rect-flow checkpoints
    p.add_argument("--rf_checkpoint",     type=str, default=None)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    # VP schedule
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T",        type=int,   default=1000)
    # Sampler params
    p.add_argument("--num_steps",   type=int, default=1000)
    p.add_argument("--n_corrector", type=int, default=1)
    p.add_argument("--snr",         type=float, default=0.16)
    p.add_argument("--n_samples",   type=int, default=64)
    # Output
    p.add_argument("--out",    type=str, default="samples.png")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vp_model(checkpoint: str, beta_min: float, beta_max: float, T: int, device):
    sde = VPSDE(beta_min=beta_min, beta_max=beta_max, T=T)
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return sde, model


def load_rf_model(checkpoint: str, device):
    flow = RectifiedFlow()
    model = UNet(in_channels=1, base_channels=64).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return flow, model


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.n_samples, 1, 28, 28)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.method == "em":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, args.beta_min, args.beta_max, args.T, device)
        print(f"Running EM sampler ({args.num_steps} steps) ...")
        samples = sde.euler_maruyama(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out, title=f"VP EM ({args.num_steps} steps)")

    elif args.method == "pc":
        ckpt = args.checkpoint or args.vp_checkpoint
        sde, model = load_vp_model(ckpt, args.beta_min, args.beta_max, args.T, device)
        print(f"Running PC sampler ({args.num_steps} steps, {args.n_corrector} corrector) ...")
        samples = sde.predictor_corrector(
            model, shape,
            num_steps=args.num_steps,
            n_corrector=args.n_corrector,
            snr=args.snr,
            device=device,
        )
        save_grid(samples, args.out,
                  title=f"VP PC ({args.num_steps} steps, K={args.n_corrector})")

    elif args.method == "rectflow":
        ckpt = args.checkpoint or args.rf_checkpoint
        flow, model = load_rf_model(ckpt, device)
        print(f"Running Rectified Flow Euler ({args.num_steps} steps) ...")
        samples = flow.euler_sample(model, shape, num_steps=args.num_steps, device=device)
        save_grid(samples, args.out,
                  title=f"Rectified Flow Euler ({args.num_steps} steps)")

    elif args.method == "all":
        # Problem 6.D: 4×8 grid with 8 fixed seeds, 4 methods
        # Methods: DDPM EM (1000), Rect Flow (100), Rect Flow (1), Reflow (1)
        n_fixed = 8
        fixed_shape = (n_fixed, 1, 28, 28)

        # Fixed noise vectors (same seed for all methods)
        torch.manual_seed(args.seed)
        z_fixed = torch.randn(n_fixed, 1, 28, 28, device=device)

        # 1) VP DDPM EM (1000 steps)
        ckpt_vp = args.vp_checkpoint or args.checkpoint
        sde, vp_model = load_vp_model(
            ckpt_vp, args.beta_min, args.beta_max, args.T, device
        )

        @torch.no_grad()
        def em_from_z(z):
            dt = 1.0 / 1000
            x = z.clone()
            for i in range(1000):
                t_val = max(1.0 - i * dt, 1e-5)
                t_b = torch.full((n_fixed,), t_val, device=device)
                eps_th = vp_model(x, t_b)
                sigma_t = sde.sigma(t_b).view(n_fixed, 1, 1, 1).clamp(min=1e-8)
                score = -eps_th / sigma_t
                beta_t = sde.beta(t_b).view(n_fixed, 1, 1, 1)
                zz = torch.randn_like(x) if i < 999 else torch.zeros_like(x)
                x = x + (-0.5 * beta_t * x - beta_t * score) * dt + torch.sqrt(beta_t * dt) * zz
            return x.clamp(-1, 1)

        # Scale z_fixed by sigma(1) for VP initialisation
        t1 = torch.ones(n_fixed, device=device)
        sigma_T = sde.sigma(t1).view(n_fixed, 1, 1, 1)
        z_vp = z_fixed * sigma_T
        print("Generating DDPM EM (1000 steps) ...")
        row_em = em_from_z(z_vp)

        # 2) Rectified Flow (100 steps) from z_fixed directly
        ckpt_rf = args.rf_checkpoint
        flow, rf_model = load_rf_model(ckpt_rf, device)

        @torch.no_grad()
        def rf_from_z(z, steps):
            dt = 1.0 / steps
            x = z.clone()
            for i in range(steps):
                t_val = i * dt
                t_b = torch.full((n_fixed,), t_val, device=device)
                v = rf_model(x, t_b)
                x = x + v * dt
            return x

        print("Generating Rect Flow (100 steps) ...")
        row_rf100 = rf_from_z(z_fixed, 100)

        print("Generating Rect Flow (1 step) ...")
        row_rf1 = rf_from_z(z_fixed, 1)

        # 3) Reflow (1 step)
        ckpt_reflow = args.reflow_checkpoint
        _, reflow_model = load_rf_model(ckpt_reflow, device)

        @torch.no_grad()
        def reflow_from_z(z, steps):
            dt = 1.0 / steps
            x = z.clone()
            for i in range(steps):
                t_val = i * dt
                t_b = torch.full((n_fixed,), t_val, device=device)
                v = reflow_model(x, t_b)
                x = x + v * dt
            return x

        print("Generating Reflow (1 step) ...")
        row_reflow = reflow_from_z(z_fixed, 1)

        # Build 4×8 grid: each row is a method, each column is a seed
        rows = [row_em, row_rf100, row_rf1, row_reflow]
        labels = [
            "DDPM EM (1000 steps)",
            "Rect Flow (100 steps)",
            "Rect Flow (1 step)",
            "Reflow (1 step)",
        ]
        all_imgs = torch.cat(rows, dim=0)  # (32, 1, 28, 28)
        grid = make_grid(all_imgs.clamp(-1, 1) * 0.5 + 0.5, nrow=n_fixed)

        fig, ax = plt.subplots(figsize=(n_fixed * 1.5, 4 * 1.5))
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        ax.axis("off")

        # Row labels on the left
        h = grid.shape[1]
        row_h = h / 4
        for r, lbl in enumerate(labels):
            ax.text(-5, row_h * r + row_h / 2, lbl,
                    ha="right", va="center", fontsize=9, transform=ax.transData)

        plt.tight_layout()
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
