# src/data/signal_1d.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


# Utils

def make_time_grid(L: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Returns t in [0, 1] of shape (L,)."""
    return torch.linspace(0.0, 1.0, L, device=device, dtype=torch.float32)


def standardize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-std per sample (1D vector)."""
    m = x.mean()
    s = x.std(unbiased=False)
    return (x - m) / (s + eps)


def _rand_uniform(rng: torch.Generator, a: float, b: float) -> float:
    """Scalar uniform in [a, b], using a CPU generator."""
    u = torch.rand((), generator=rng).item()
    return a + (b - a) * u


def _rand_int(rng: torch.Generator, low: int, high_inclusive: int) -> int:
    """Integer uniform in [low, high_inclusive]."""
    return int(torch.randint(low=low, high=high_inclusive + 1, size=(1,), generator=rng).item())


# Signal components

def sines_component(
    t: torch.Tensor,
    rng: torch.Generator,
    n_min: int = 1,
    n_max: int = 5,
    f_min: float = 1.0,
    f_max: float = 20.0,
) -> torch.Tensor:
    """Sum of random sinusoids. Returns x_sines of shape (L,)."""
    L = t.shape[0]
    n_sines = _rand_int(rng, n_min, n_max)

    x = torch.zeros(L, dtype=torch.float32, device=t.device)
    for _ in range(n_sines):
        amp = _rand_uniform(rng, 0.3, 1.0)
        freq = _rand_uniform(rng, f_min, f_max)
        phase = _rand_uniform(rng, 0.0, 2.0 * math.pi)
        x = x + float(amp) * torch.sin((2.0 * math.pi * float(freq)) * t + float(phase))
    return x


def steps_component(
    L: int,
    rng: torch.Generator,
    n_min: int = 0,
    n_max: int = 3,
    amp_min: float = -1.0,
    amp_max: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Piecewise-constant jumps (discontinuities).
    Returns x_steps of shape (L,).
    """
    n_steps = _rand_int(rng, n_min, n_max)
    x = torch.zeros(L, dtype=torch.float32, device=device)

    for _ in range(n_steps):
        p = int(torch.randint(low=0, high=L, size=(1,), generator=rng).item())
        a = _rand_uniform(rng, amp_min, amp_max)
        x[p:] = x[p:] + float(a)

    return x


def spikes_component(
    t: torch.Tensor,
    rng: torch.Generator,
    n_min: int = 0,
    n_max: int = 6,
    amp_min: float = 0.5,
    amp_max: float = 2.0,
    width_min: float = 0.005,
    width_max: float = 0.03,
) -> torch.Tensor:
    """
    Local Gaussian bumps (spikes).
    width is expressed in units of t in [0,1], not samples.
    Returns x_spikes of shape (L,).
    """
    n_spikes = _rand_int(rng, n_min, n_max)
    x = torch.zeros_like(t)

    for _ in range(n_spikes):
        c = _rand_uniform(rng, 0.0, 1.0)
        w = _rand_uniform(rng, width_min, width_max)
        a = _rand_uniform(rng, amp_min, amp_max)

        sign = -1.0 if torch.rand((), generator=rng).item() < 0.5 else 1.0
        a = float(sign) * float(a)

        bump = a * torch.exp(-((t - float(c)) ** 2) / (2.0 * (float(w) ** 2)))
        x = x + bump

    return x


def add_gaussian_noise(x: torch.Tensor, rng: torch.Generator, sigma: float) -> torch.Tensor:
    """
    Adds iid Gaussian noise N(0, sigma^2).
    Robust across CPU/CUDA/MPS: sample on CPU then move to x.device.
    """
    noise = torch.randn(x.shape, generator=rng, device=torch.device("cpu"), dtype=x.dtype).to(x.device)
    return x + float(sigma) * noise


# Full generator

@dataclass
class Signal1DConfig:
    L: int = 512
    sigma_noise: float = 0.2

    use_steps: bool = True
    use_spikes: bool = True
    do_standardize: bool = True


def generate_pair(
    cfg: Signal1DConfig,
    seed: int,
    device: Optional[torch.device] = None,
    return_components: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]:
    """
    Generates a clean signal X and a noisy observation Y.

    Returns:
      if return_components=False:
        X: (L,), Y: (L,)
      else:
        X: (L,), Y: (L,), components: dict with keys
          - x_sines, x_steps, x_spikes, x_clean
    """
    # IMPORTANT: CPU generator for robustness (MPS/CUDA)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    t = make_time_grid(cfg.L, device=device)

    x_sines = sines_component(t, rng)

    x_steps = torch.zeros_like(t)
    if cfg.use_steps:
        x_steps = steps_component(cfg.L, rng, device=t.device)

    x_spikes = torch.zeros_like(t)
    if cfg.use_spikes:
        x_spikes = spikes_component(t, rng)

    x = x_sines + x_steps + x_spikes

    if cfg.do_standardize:
        x = standardize(x)
        # Pour rester cohérent, on standardise aussi les composantes "affichées"
        # en les re-projetant sur la même transformation affine :
        # x_std = (x - m)/s => x = x_sines + x_steps + x_spikes
        # Ici on reconstruit les composantes standardisées via:
        # comp_std = comp / s  (car mean est retirée globalement sur x)
        m = (x_sines + x_steps + x_spikes).mean()
        s = (x_sines + x_steps + x_spikes).std(unbiased=False) + 1e-8
        x_sines = (x_sines) / s
        x_steps = (x_steps) / s
        x_spikes = (x_spikes) / s
        # et le mean retiré globalement est porté par "x" lui-même

    y = add_gaussian_noise(x, rng, sigma=cfg.sigma_noise)

    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    if not return_components:
        return x, y

    components = {
        "x_sines": x_sines.to(dtype=torch.float32),
        "x_steps": x_steps.to(dtype=torch.float32),
        "x_spikes": x_spikes.to(dtype=torch.float32),
        "x_clean": x,
        "y_noisy": y,
    }
    return x, y, components



# Dataset

class Signal1DDataset(Dataset):
    """
    Returns (Y, X) for denoising/regression:
      input  = Y (noisy)
      target = X (clean)
    """
    def __init__(self, n: int, cfg: Signal1DConfig, seed: int = 0, device: Optional[torch.device] = None):
        self.n = int(n)
        self.cfg = cfg
        self.seed = int(seed)
        self.device = device

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.seed + int(idx)
        x, y = generate_pair(self.cfg, seed=s, device=self.device, return_components=False)
        return y, x


# Lightning DataModule

class Signal1DDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for 1D denoising/regression.

    Returns batches of (Y, X):
      input  = Y (noisy)
      target = X (clean)
    """
    def __init__(
        self,
        batch_size: int,
        n_train: int,
        n_val: int,
        n_test: int,
        L: int = 512,
        sigma_noise: float = 0.2,
        use_steps: bool = True,
        use_spikes: bool = True,
        do_standardize: bool = True,
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
            sigma_noise=float(sigma_noise),
            use_steps=bool(use_steps),
            use_spikes=bool(use_spikes),
            do_standardize=bool(do_standardize),
        )

        self.train_dataset: Optional[Signal1DDataset] = None
        self.val_dataset: Optional[Signal1DDataset] = None
        self.test_dataset: Optional[Signal1DDataset] = None
        self.predict_dataset: Optional[Signal1DDataset] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = Signal1DDataset(n=self.n_train, cfg=self.cfg, seed=self.seed)
            self.val_dataset = Signal1DDataset(n=self.n_val, cfg=self.cfg, seed=self.seed + 10_000)

        if stage == "test" or stage is None:
            self.test_dataset = Signal1DDataset(n=self.n_test, cfg=self.cfg, seed=self.seed + 20_000)

        if stage == "predict" or stage is None:
            # (Y, X) pairs kept to allow metrics if you want
            self.predict_dataset = Signal1DDataset(n=self.n_test, cfg=self.cfg, seed=self.seed + 30_000)

    def train_dataloader(self):
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=max(1, self.num_workers // 2),
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        assert self.predict_dataset is not None
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )