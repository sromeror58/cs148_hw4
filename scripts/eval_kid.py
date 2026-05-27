"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def save_real_data(real_dir: str, n_samples: int):
    """Save real FashionMNIST images to a directory for KID reference."""
    os.makedirs(real_dir, exist_ok=True)
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    count = 0
    for img, _ in ds:
        if count >= n_samples:
            break
        # Save as 3-channel PNG (torch-fidelity expects RGB or handles 1ch)
        save_image(img, os.path.join(real_dir, f"{count:05d}.png"))
        count += 1


def compute_kid(generated_dir: str, real_dir: str, n_samples: int) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(n_samples, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def generate_em(sde, model, n_samples, num_steps, device, batch_size=128):
    shape = (batch_size, 1, 28, 28)
    all_samples = []
    n_done = 0
    model.eval()
    while n_done < n_samples:
        bsz = min(batch_size, n_samples - n_done)
        s = sde.euler_maruyama(model, (bsz, 1, 28, 28), num_steps=num_steps, device=device)
        all_samples.append(s.cpu())
        n_done += bsz
    return torch.cat(all_samples, dim=0)[:n_samples]


def generate_ddim(sde, model, n_samples, num_steps, device, batch_size=128):
    all_samples = []
    n_done = 0
    model.eval()
    while n_done < n_samples:
        bsz = min(batch_size, n_samples - n_done)
        s = sde.ddim_sample(model, (bsz, 1, 28, 28), num_steps=num_steps, device=device)
        all_samples.append(s.cpu())
        n_done += bsz
    return torch.cat(all_samples, dim=0)[:n_samples]


def generate_rf(flow, model, n_samples, num_steps, device, batch_size=128):
    all_samples = []
    n_done = 0
    model.eval()
    while n_done < n_samples:
        bsz = min(batch_size, n_samples - n_done)
        s = flow.euler_sample(model, (bsz, 1, 28, 28), num_steps=num_steps, device=device)
        all_samples.append(s.cpu())
        n_done += bsz
    return torch.cat(all_samples, dim=0)[:n_samples]


def main():
    args = get_args()
    device = torch.device(args.device)

    # Load models
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_model.load_state_dict(torch.load(args.vp_checkpoint, map_location=device))
    vp_model.eval()

    flow = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_model.load_state_dict(torch.load(args.rf_checkpoint, map_location=device))
    rf_model.eval()

    # Save real data reference once
    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = os.path.join(tmpdir, "real")
        save_real_data(real_dir, args.n_samples)

        results = {}

        for method in METHODS:
            results[method] = {}
            for steps in STEP_COUNTS:
                # EM only makes sense for steps >= 1; skip EM at steps 1 (it's bad but run it)
                print(f"  [{method}] steps={steps} ...", end=" ", flush=True)

                gen_dir = os.path.join(tmpdir, f"{method}_{steps}")

                with torch.no_grad():
                    if method == "rectflow":
                        samples = generate_rf(flow, rf_model, args.n_samples, steps, device)
                    elif method == "ddim":
                        samples = generate_ddim(sde, vp_model, args.n_samples, steps, device)
                    elif method == "em":
                        if steps > 200:
                            # EM at 1000 steps is the baseline
                            samples = generate_em(sde, vp_model, args.n_samples, steps, device)
                        else:
                            samples = generate_em(sde, vp_model, args.n_samples, steps, device)

                save_samples_to_dir(samples, gen_dir)
                metrics = compute_kid(gen_dir, real_dir, args.n_samples)
                kid_mean = metrics["kernel_inception_distance_mean"]
                kid_std  = metrics["kernel_inception_distance_std"]
                results[method][steps] = (kid_mean, kid_std)
                print(f"KID = {kid_mean:.4f} ± {kid_std:.4f}")

        # Print markdown table
        print("\n| Steps | Flow Matching | DDIM | DDPM EM |")
        print("|-------|--------------|------|---------|")
        em_na = {1, 5, 10, 50, 100, 200}
        for steps in STEP_COUNTS:
            rf_m, rf_s   = results["rectflow"][steps]
            dd_m, dd_s   = results["ddim"][steps]
            em_m, em_s   = results["em"][steps]
            em_str = f"{em_m:.4f} ± {em_s:.4f}"
            if steps in em_na:
                em_str = "—"
            print(f"| {steps:5d} | {rf_m:.4f} ± {rf_s:.4f} | {dd_m:.4f} ± {dd_s:.4f} | {em_str} |")


if __name__ == "__main__":
    main()
