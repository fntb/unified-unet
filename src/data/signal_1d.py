# src/data/signal_1d.py
"""
Synthetic 1D signal dataset for denoising experiments.

Each sample is generated as:
    x_clean = sines + (optional) steps + (optional) spikes
    y_noisy = x_clean + Gaussian(sigma) + impulsive_noise(prob)

Notes on reproducibility:
- Randomness is driven by a CPU torch.Generator seeded with (base_seed + idx).
- This makes samples deterministic regardless of GPU availability.

Returned batch format (default):
    (y_noisy, x_clean)  # input, target
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


# Configuration

@dataclass(frozen=True)
class Signal1DConfig:
    # Signal length
    L: int = 512

    # Clean signal composition
    use_steps: bool = False
    use_spikes: bool = False

    # Sines
    sines_n_min: int = 3
    sines_n_max: int = 8
    sines_f_min: float = 1.0
    sines_f_max: float = 10.0
    sines_amp_min: float = 0.6
    sines_amp_max: float = 1.2

    # Steps
    steps_n_min: int = 0
    steps_n_max: int = 1
    steps_amp_min: float = -0.3
    steps_amp_max: float = 0.3

    # Spikes in the clean signal
    spikes_n_min: int = 0
    spikes_n_max: int = 1
    spikes_amp_min: float = 0.2
    spikes_amp_max: float = 0.5
    spikes_width_min: float = 0.01
    spikes_width_max: float = 0.03

    # Noise parameters (sampled per example if overrides are not provided)
    sigma_noise_min: float = 0.05
    sigma_noise_max: float = 0.60

    impulsive_prob_min: float = 0.00
    impulsive_prob_max: float = 0.10
    impulsive_amp_min: float = 0.8
    impulsive_amp_max: float = 3.0

    # Normalization
    # If True, x and y are standardized using mean/std of x_clean (per sample).
    do_standardize: bool = True



# Small helpers (deterministic sampling)
def _rand_uniform(rng: torch.Generator, a: float, b: float) -> float:
    return float(a + (b - a) * torch.rand((), generator=rng).item())


def _rand_int_inclusive(rng: torch.Generator, low: int, high_inclusive: int) -> int:
    return int(torch.randint(low=low, high=high_inclusive + 1, size=(1,), generator=rng).item())


def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x - mean) / (std + eps)


def make_time_grid(L: int, device: Optional[torch.device] = None) -> torch.Tensor:
    # Use [0, 1] so frequencies are interpretable as "cycles per unit interval".
    return torch.linspace(0.0, 1.0, L, device=device, dtype=torch.float32)


# Clean signal components

def sines_component(
    t: torch.Tensor,
    rng: torch.Generator,
    n_min: int,
    n_max: int,
    f_min: float,
    f_max: float,
    amp_min: float,
    amp_max: float,
) -> torch.Tensor:
    L = t.shape[0]
    n_sines = _rand_int_inclusive(rng, n_min, n_max)
    x = torch.zeros(L, dtype=torch.float32, device=t.device)

    for _ in range(n_sines):
        amp = _rand_uniform(rng, amp_min, amp_max)
        freq = _rand_uniform(rng, f_min, f_max)
        phase = _rand_uniform(rng, 0.0, 2.0 * math.pi)
        x = x + amp * torch.sin(2.0 * math.pi * freq * t + phase)

    return x


def steps_component(
    L: int,
    rng: torch.Generator,
    n_min: int,
    n_max: int,
    amp_min: float,
    amp_max: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    n_steps = _rand_int_inclusive(rng, n_min, n_max)
    x = torch.zeros(L, dtype=torch.float32, device=device)

    for _ in range(n_steps):
        p = int(torch.randint(low=0, high=L, size=(1,), generator=rng).item())
        a = _rand_uniform(rng, amp_min, amp_max)
        x[p:] = x[p:] + a

    return x


def spikes_component(
    t: torch.Tensor,
    rng: torch.Generator,
    n_min: int,
    n_max: int,
    amp_min: float,
    amp_max: float,
    width_min: float,
    width_max: float,
) -> torch.Tensor:
    n_spikes = _rand_int_inclusive(rng, n_min, n_max)
    x = torch.zeros_like(t)

    for _ in range(n_spikes):
        center = _rand_uniform(rng, 0.0, 1.0)
        width = _rand_uniform(rng, width_min, width_max)
        amp = _rand_uniform(rng, amp_min, amp_max)
        sign = -1.0 if torch.rand((), generator=rng).item() < 0.5 else 1.0

        bump = (sign * amp) * torch.exp(-((t - center) ** 2) / (2.0 * (width ** 2)))
        x = x + bump

    return x



# Noise models

def add_gaussian_noise(x: torch.Tensor, rng: torch.Generator, sigma: float) -> torch.Tensor:
    # Generate on CPU for full determinism, then move to device.
    noise_cpu = torch.randn(x.shape, generator=rng, device=torch.device("cpu"), dtype=x.dtype)
    return x + float(sigma) * noise_cpu.to(x.device)


def add_impulsive_noise(
    x: torch.Tensor,
    rng: torch.Generator,
    prob: float,
    amp_min: float,
    amp_max: float,
) -> torch.Tensor:
    """
    Impulsive noise: independently for each time index, with probability 'prob',
    add a large spike with random sign and amplitude in [amp_min, amp_max].

    This is useful to stress high-frequency reconstruction.
    """
    if prob <= 0.0:
        return x

    # CPU randomness for determinism
    mask_cpu = (torch.rand(x.shape, generator=rng, device=torch.device("cpu")) < float(prob))
    amps_cpu = torch.empty(x.shape, device=torch.device("cpu")).uniform_(
        float(amp_min), float(amp_max), generator=rng
    )
    signs_cpu = torch.where(
        torch.rand(x.shape, generator=rng, device=torch.device("cpu")) < 0.5,
        torch.tensor(-1.0, dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float32),
    )

    spikes_cpu = mask_cpu.to(amps_cpu.dtype) * amps_cpu * signs_cpu.to(amps_cpu.dtype)
    return x + spikes_cpu.to(x.device, dtype=x.dtype)



# Sample generation

def generate_pair(
    cfg: Signal1DConfig,
    seed: int,
    device: Optional[torch.device] = None,
    return_components: bool = False,
    # Optional overrides, useful for fixed test cases and qualitative plots
    sigma_override: Optional[float] = None,
    impulsive_prob_override: Optional[float] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]:
    """
    Returns:
        x_clean, y_noisy  (or plus a components dict if return_components=True)
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    t = make_time_grid(cfg.L, device=device)

    x_sines = sines_component(
        t, rng,
        n_min=cfg.sines_n_min, n_max=cfg.sines_n_max,
        f_min=cfg.sines_f_min, f_max=cfg.sines_f_max,
        amp_min=cfg.sines_amp_min, amp_max=cfg.sines_amp_max,
    )

    x_steps = torch.zeros_like(t)
    if cfg.use_steps:
        x_steps = steps_component(
            cfg.L, rng,
            n_min=cfg.steps_n_min, n_max=cfg.steps_n_max,
            amp_min=cfg.steps_amp_min, amp_max=cfg.steps_amp_max,
            device=t.device,
        )

    x_spikes = torch.zeros_like(t)
    if cfg.use_spikes:
        x_spikes = spikes_component(
            t, rng,
            n_min=cfg.spikes_n_min, n_max=cfg.spikes_n_max,
            amp_min=cfg.spikes_amp_min, amp_max=cfg.spikes_amp_max,
            width_min=cfg.spikes_width_min, width_max=cfg.spikes_width_max,
        )

    x_clean = x_sines + x_steps + x_spikes

    sigma = float(sigma_override) if sigma_override is not None else _rand_uniform(rng, cfg.sigma_noise_min, cfg.sigma_noise_max)
    prob = float(impulsive_prob_override) if impulsive_prob_override is not None else _rand_uniform(rng, cfg.impulsive_prob_min, cfg.impulsive_prob_max)

    y_noisy = add_gaussian_noise(x_clean, rng, sigma=sigma)
    y_noisy = add_impulsive_noise(
        y_noisy, rng,
        prob=prob,
        amp_min=cfg.impulsive_amp_min,
        amp_max=cfg.impulsive_amp_max,
    )

    if cfg.do_standardize:
        m = x_clean.mean()
        s = x_clean.std(unbiased=False)
        x_clean_std = _standardize(x_clean, m, s)
        y_noisy_std = _standardize(y_noisy, m, s)
    else:
        x_clean_std = x_clean
        y_noisy_std = y_noisy

    x = x_clean_std.to(dtype=torch.float32)
    y = y_noisy_std.to(dtype=torch.float32)

    if not return_components:
        return x, y

    # Components are provided for interpretability in plots.
    # By default they are returned in the original (non-standardized) scale,
    # while x_clean/y_noisy are returned after standardization
    components: Dict[str, torch.Tensor] = {
        "x_sines": x_sines.to(dtype=torch.float32),
        "x_steps": x_steps.to(dtype=torch.float32),
        "x_spikes": x_spikes.to(dtype=torch.float32),
        "x_clean_raw": x_clean.to(dtype=torch.float32),
        "y_noisy_raw": y_noisy.to(dtype=torch.float32),
        "noise_sigma": torch.tensor(sigma, dtype=torch.float32),
        "impulsive_prob": torch.tensor(prob, dtype=torch.float32),
    }
    return x, y, components



# Dataset

class Signal1DDataset(Dataset):
    """
    Returns (y_noisy, x_clean) for denoising training.
    """

    def __init__(self, n: int, cfg: Signal1DConfig, seed: int = 0):
        self.n = int(n)
        self.cfg = cfg
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Deterministic sample for a given idx
        s = self.seed + int(idx)
        x, y = generate_pair(self.cfg, seed=s, device=None, return_components=False)
        return y, x


# Lightning DataModule

class Signal1DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_train: int,
        n_val: int,
        n_test: int,
        # Signal config
        L: int = 512,
        use_steps: bool = False,
        use_spikes: bool = False,
        sines_n_min: int = 3,
        sines_n_max: int = 8,
        sines_f_min: float = 1.0,
        sines_f_max: float = 10.0,
        sines_amp_min: float = 0.6,
        sines_amp_max: float = 1.2,
        steps_n_min: int = 0,
        steps_n_max: int = 1,
        steps_amp_min: float = -0.3,
        steps_amp_max: float = 0.3,
        spikes_n_min: int = 0,
        spikes_n_max: int = 1,
        spikes_amp_min: float = 0.2,
        spikes_amp_max: float = 0.5,
        spikes_width_min: float = 0.01,
        spikes_width_max: float = 0.03,
        sigma_noise_min: float = 0.05,
        sigma_noise_max: float = 0.60,
        impulsive_prob_min: float = 0.00,
        impulsive_prob_max: float = 0.10,
        impulsive_amp_min: float = 0.8,
        impulsive_amp_max: float = 3.0,
        do_standardize: bool = True,
        # Repro and dataloader
        seed: int = 0,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)

        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)

        self.cfg = Signal1DConfig(
            L=int(L),
            use_steps=bool(use_steps),
            use_spikes=bool(use_spikes),
            sines_n_min=int(sines_n_min),
            sines_n_max=int(sines_n_max),
            sines_f_min=float(sines_f_min),
            sines_f_max=float(sines_f_max),
            sines_amp_min=float(sines_amp_min),
            sines_amp_max=float(sines_amp_max),
            steps_n_min=int(steps_n_min),
            steps_n_max=int(steps_n_max),
            steps_amp_min=float(steps_amp_min),
            steps_amp_max=float(steps_amp_max),
            spikes_n_min=int(spikes_n_min),
            spikes_n_max=int(spikes_n_max),
            spikes_amp_min=float(spikes_amp_min),
            spikes_amp_max=float(spikes_amp_max),
            spikes_width_min=float(spikes_width_min),
            spikes_width_max=float(spikes_width_max),
            sigma_noise_min=float(sigma_noise_min),
            sigma_noise_max=float(sigma_noise_max),
            impulsive_prob_min=float(impulsive_prob_min),
            impulsive_prob_max=float(impulsive_prob_max),
            impulsive_amp_min=float(impulsive_amp_min),
            impulsive_amp_max=float(impulsive_amp_max),
            do_standardize=bool(do_standardize),
        )

        self.train_dataset: Optional[Signal1DDataset] = None
        self.val_dataset: Optional[Signal1DDataset] = None
        self.test_dataset: Optional[Signal1DDataset] = None
        self.predict_dataset: Optional[Signal1DDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = Signal1DDataset(n=self.n_train, cfg=self.cfg, seed=self.seed)
            self.val_dataset = Signal1DDataset(n=self.n_val, cfg=self.cfg, seed=self.seed + 10_000)

        if stage in (None, "test"):
            self.test_dataset = Signal1DDataset(n=self.n_test, cfg=self.cfg, seed=self.seed + 20_000)

        if stage in (None, "predict"):
            self.predict_dataset = Signal1DDataset(n=self.n_test, cfg=self.cfg, seed=self.seed + 30_000)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        assert self.predict_dataset is not None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )