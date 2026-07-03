from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def cov(s, t, H: float) -> np.ndarray:
    """Covariance of fractional Brownian motion with Hurst index H at times s and t.
    s and t may be scalars or broadcastable arrays.
    """
    if not (0 < H < 1):
        raise ValueError(f"H must be in (0, 1), got {H}")
    if np.any(s < 0) or np.any(t < 0):
        raise ValueError("s and t must be non-negative")
    return np.asarray(1 / 2 * (s ** (2 * H) + t ** (2 * H) - np.abs(s - t) ** (2 * H)))


def cov_matrix(timepoints: np.ndarray, H: float) -> np.ndarray:
    """Covariance matrix of fBm sampled at the given timepoints."""
    column = timepoints[:, None]
    row = timepoints[None, :]
    return cov(column, row, H)


def simulate_fbm(timepoints: np.ndarray, H: float, noise: np.ndarray) -> np.ndarray:
    """Simulate one fBm path at the given timepoints via Cholesky factorisation
    of the covariance matrix. `noise` is a vector of standard normals; reusing
    the same `noise` across different H isolates the effect of H on the path.
    """
    chol = np.linalg.cholesky(cov_matrix(timepoints, H))
    return chol @ noise

def simulate_fgn(timepoints: np.ndarray, H: float, noise: np.ndarray) -> np.ndarray:
    b_H = simulate_fbm(timepoints, H, noise)
    x = b_H[1:] - b_H[:-1]
    return x


def plot_paths(timepoints: np.ndarray, hurst_values, paths) -> Figure:
    """Plot each fBm path in its own panel, styled like a textbook figure"""
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, axes = plt.subplots(len(hurst_values), 1, figsize=(7, 9), sharex=True)
    for ax, H, path in zip(axes, hurst_values, paths):
        ax.plot(timepoints, path, color="black", linewidth=0.5)

        # x-axis through y=0 and y-axis through x=0
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)

        ax.text(1.02, 0, f"$H = {H}$", transform=ax.get_yaxis_transform(),
                va="center", fontsize=13)
        # label the origin
        ax.annotate("$0$", xy=(0, 0), xytext=(-12, -10),
                    textcoords="offset points", fontsize=11)

        # round the bounds outward to a 0.1 step and label the y-range
        lo = np.floor(float(path.min()) / 0.1) * 0.1
        hi = np.ceil(float(path.max()) / 0.1) * 0.1
        ax.set_ylim(lo, hi)
        ax.annotate(f"${hi:.1f}$", xy=(0, hi), xytext=(-6, 0), ha="right",
                    va="center", textcoords="offset points", fontsize=10)
        ax.annotate(f"${lo:.1f}$", xy=(0, lo), xytext=(-6, 0), ha="right",
                    va="center", textcoords="offset points", fontsize=10)
        # remove the box borders
        for spine in ax.spines.values():
            spine.set_visible(False)
        # no numbers on the axes
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def main() -> None:
    rng = np.random.default_rng(404)
    timepoints = np.linspace(0.01, 1, 999)
    noise = rng.normal(0, 1, len(timepoints))

    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    fbm_paths = [simulate_fbm(timepoints, H, noise) for H in hurst_values]
    fgn_paths = [simulate_fgn(timepoints, H, noise) for H in hurst_values]

    fig1 = plot_paths(timepoints, hurst_values, fbm_paths)
    fig2 = plot_paths(timepoints[:-1], hurst_values, fgn_paths)
    # save as a vector PDF
    output_fbm = Path(__file__).with_name("fbm_paths.pdf")
    output_fgn = Path(__file__).with_name("fgn_paths.pdf")
    fig1.savefig(output_fbm, format="pdf", bbox_inches="tight")
    fig2.savefig(output_fgn, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()