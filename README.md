# Simulating and Estimating Fractional Brownian Motion

This project accompanies a bachelor thesis on **fractional Brownian motion (fBm)**.
It is split into two parts:

1. **Simulation** — generating sample paths of fBm for different values of the
   Hurst parameter *H*.
2. **Estimation** *(planned)* — estimating the Hurst parameter *H* from an
   observed fBm path.

## Background

Fractional Brownian motion is a centered Gaussian process `B_H(t)` with
covariance

```
Cov(B_H(s), B_H(t)) = 1/2 (|s|^{2H} + |t|^{2H} - |s - t|^{2H}),   H in (0, 1).
```

The Hurst parameter *H* controls the roughness and correlation of the paths:

- `H < 0.5` — anti-persistent, rough paths,
- `H = 0.5` — ordinary Brownian motion (independent increments),
- `H > 0.5` — persistent, smooth paths with long-range dependence.

## Simulation

`fbm_simulation.py` builds the covariance matrix on a grid of timepoints,
factorises it with a Cholesky decomposition, and multiplies by standard normal
noise to produce sample paths. The same noise vector is reused across all *H*
so that the differences in the plot are due to *H* alone.

Running the script produces `fbm_paths.pdf`, a vector figure showing one sample
path per Hurst value (`H = 0.1, 0.3, 0.5, 0.7, 0.9`).

## Usage

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fbm_simulation.py
```

## Requirements

See `requirements.txt` (NumPy and Matplotlib).