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
    of the covariance matrix. `noise` is a vector of standard normals, and
    reusing the same one across different H isolates the effect of H.
    """
    if not (0 < H < 1):
        raise ValueError(f"H must be in (0, 1), got {H}")
    chol = np.linalg.cholesky(cov_matrix(timepoints, H))
    return chol @ noise

def simulate_fgn(b_H: np.ndarray, lag: int) -> np.ndarray:
    """Increments of a precomputed fBm path at the given lag,
    x_t = B_H(t + lag) - B_H(t). Taking the path as input instead of recomputing
    it avoids repeating the Cholesky factorisation across many lags."""
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    return b_H[lag:] - b_H[:-lag]


def _round_step(value: float) -> float:
    """Rounding step for a bound: 0.1 for magnitudes >= 1, otherwise the place of
    the biggest non-zero decimal, so 0.0048 gives a step of 0.001."""
    if value == 0:
        return 0.1
    return min(0.1, 10.0 ** np.floor(np.log10(abs(value))))


def _round_bound(value: float, direction: str) -> tuple[float, str]:
    """Round `value` outward ('up' or 'down') to its step, returning the rounded
    value and a label with a matching number of decimals."""
    step = _round_step(value)
    rounded = float((np.ceil if direction == "up" else np.floor)(value / step) * step)
    decimals = max(1, int(-np.floor(np.log10(step))))
    return rounded, f"{rounded:.{decimals}f}"


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

        # round the bounds outward and label the y-range
        lo, lo_label = _round_bound(float(path.min()), "down")
        hi, hi_label = _round_bound(float(path.max()), "up")
        ax.set_ylim(lo, hi)
        ax.annotate(f"${hi_label}$", xy=(0, hi), xytext=(-6, 0), ha="right",
                    va="center", textcoords="offset points", fontsize=10)
        ax.annotate(f"${lo_label}$", xy=(0, lo), xytext=(-6, 0), ha="right",
                    va="center", textcoords="offset points", fontsize=10)
        for spine in ax.spines.values():  # no box border
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
    fgn_paths = [simulate_fgn(path, 1) for path in fbm_paths]

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