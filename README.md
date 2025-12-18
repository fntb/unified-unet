
# Unified UNet

_This is a **WIP**._

An exploration of the paper _A Unified Framework for U-Net Design and Analysis, by Williams, Falck, et al._

We aim to implement

- a basic UNet for reference ;
- a Haar Wavelet Residual UNet _which replaces the learnable encoder with a non-learnable wavelet based encoder_ ;
- a Haar Wavelet Residual UNet on complex geometry _(e.g. a sphere rather than a 2d square image)_ ;
- a Haar Wavelet Residual UNet with an enforced functionnal constraint _(e.g. conservation laws in weather simulation)_.

Currently implemented :

- basic UNet for classification and segmentation (mnist and kvasir) ;
- wavelet Unet for classification and segmentation (mnist and kvasir).

## Quick Start

We use the `uv` package and project manager and the `torch` - `lightning` - `hydra` framework.

Run the demo with

```bash
uv run src/main.py
```