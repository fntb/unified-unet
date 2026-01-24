# src/data/signal_1d.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


def make_time_grid(L: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, L, device=device, dtype=torch.float32)


def standardize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = x.mean()
    s = x.std(unbiased=False)
    return (x - m) / (s + eps)


def _rand_uniform(rng: torch.Generator, a: float, b: float) -> float:
    u = torch.rand((), generator=rng).item()
    return a + (b - a) * u


def _rand_int(rng: torch.Generator, low: int, high_inclusive: int) -> int:
    return int(torch.randint(low=low, high=high_inclusive + 1, size=(1,), generator=rng).item())


def add_gaussian_noise(x: torch.Tensor, rng: torch.Generator, sigma: float) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=rng, device=torch.device("cpu"), dtype=x.dtype).to(x.device)
    return x + float(sigma) * noise


@dataclass
class Signal1DConfig:
    # length + noise
    L: int = 512
    sigma_noise: float = 0.35

    # standardization
    do_standardize: bool = True

    # composition toggles
    use_steps: bool = True
    use_spikes: bool = True

    # sines complexity (multi-frequency)
    n_sines_min: int = 4
    n_sines_max: int = 10
    f_min: float = 1.0
    f_max: float = 80.0
    amp_sine_min: float = 0.2
    amp_sine_max: float = 1.2

    # steps (discontinuities)
    n_steps_min: int = 1
    n_steps_max: int = 6
    step_amp_min: float = -1.5
    step_amp_max: float = 1.5

    # spikes (localized high-freq)
    n_spikes_min: int = 2
    n_spikes_max: int = 10
    spike_amp_min: float = 0.6
    spike_amp_max: float = 2.5
    spike_width_min: float = 0.002   # plus étroit
    spike_width_max: float = 0.015   # plus étroit


def generate_pair(
    cfg: Signal1DConfig,
    seed: int,
    device: Optional[torch.device] = None,
    return_components: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
]:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    t = make_time_grid(cfg.L, device=device)

    # --- sines
    n_sines = _rand_int(rng, cfg.n_sines_min, cfg.n_sines_max)
    x_sines = torch.zeros(cfg.L, dtype=torch.float32, device=t.device)
    for _ in range(n_sines):
        amp = _rand_uniform(rng, cfg.amp_sine_min, cfg.amp_sine_max)
        freq = _rand_uniform(rng, cfg.f_min, cfg.f_max)
        phase = _rand_uniform(rng, 0.0, 2.0 * math.pi)
        x_sines = x_sines + float(amp) * torch.sin((2.0 * math.pi * float(freq)) * t + float(phase))

    # --- steps
    x_steps = torch.zeros_like(t)
    if cfg.use_steps:
        n_steps = _rand_int(rng, cfg.n_steps_min, cfg.n_steps_max)
        for _ in range(n_steps):
            p = int(torch.randint(low=0, high=cfg.L, size=(1,), generator=rng).item())
            a = _rand_uniform(rng, cfg.step_amp_min, cfg.step_amp_max)
            x_steps[p:] = x_steps[p:] + float(a)

    # --- spikes
    x_spikes = torch.zeros_like(t)
    if cfg.use_spikes:
        n_spikes = _rand_int(rng, cfg.n_spikes_min, cfg.n_spikes_max)
        for _ in range(n_spikes):
            c = _rand_uniform(rng, 0.0, 1.0)
            w = _rand_uniform(rng, cfg.spike_width_min, cfg.spike_width_max)
            a = _rand_uniform(rng, cfg.spike_amp_min, cfg.spike_amp_max)
            sign = -1.0 if torch.rand((), generator=rng).item() < 0.5 else 1.0
            bump = float(sign) * float(a) * torch.exp(-((t - float(c)) ** 2) / (2.0 * (float(w) ** 2)))
            x_spikes = x_spikes + bump

    x = x_sines + x_steps + x_spikes

    if cfg.do_standardize:
        x = standardize(x)

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


class Signal1DDataset(Dataset):
    def __init__(self, n: int, cfg: Signal1DConfig, seed: int = 0):
        self.n = int(n)
        self.cfg = cfg
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        s = self.seed + int(idx)
        x, y = generate_pair(self.cfg, seed=s, device=None, return_components=False)
        return y, x


class Signal1DDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        n_train: int,
        n_val: int,
        n_test: int,
        seed: int = 0,
        num_workers: int = 1,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        **cfg_kwargs,
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

        # everything in cfg_kwargs maps to Signal1DConfig
        self.cfg = Signal1DConfig(**cfg_kwargs)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in ("fit", None):
            self.train_dataset = Signal1DDataset(self.n_train, self.cfg, seed=self.seed)
            self.val_dataset = Signal1DDataset(self.n_val, self.cfg, seed=self.seed + 10_000)
        if stage in ("test", None):
            self.test_dataset = Signal1DDataset(self.n_test, self.cfg, seed=self.seed + 20_000)
        if stage in ("predict", None):
            self.predict_dataset = Signal1DDataset(self.n_test, self.cfg, seed=self.seed + 30_000)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=max(1, self.num_workers // 2), pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers
        )