"""
Collects the Hurst-parameter estimates for all assets used in the thesis and
checks the stylized facts the estimator relies on.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import fbm_simulation as fbms
import hurst_param_estimation as hpe

DATA_DIR = Path(__file__).parent / "data"
# (file, RV column). BTC is a single series, the VOLARE files are panels of many
# symbols; their rv5 is the 5-minute estimator, matching BTC's rv.
ASSET_FILES = {
    "crypto": (DATA_DIR / "BTCUSDT_daily_rv.csv", "rv"),
    "stocks": (DATA_DIR / "realized_variance_stocks.csv", "rv5"),
    "futures": (DATA_DIR / "realized_variance_futures.csv", "rv5"),
}
CLASS_COLOURS = {"crypto": "C0", "stocks": "C1", "futures": "C2"}

FIGURE_DIR = Path(__file__).parent / "real_data_figures"
PLOT_SYMBOLS = ["BTCUSDT", "BTCUSDT_noweekend", "AAPL", "ES"]
DISPLAY_NAMES = {
    "BTCUSDT_noweekend": "BTCUSDT (weekends dropped)",
}

LAGS = list(range(1, 31))
MOMENTS = [0.5, 1.0, 1.5, 2.0]
GAUSSIAN_LAGS = [1, 5, 10, 30]
REFERENCE_COLOUR = "#c0392b"
SCALED_COLOUR = "#1f4e9c"


class Series(NamedTuple):
    asset_class: str
    log_rv: np.ndarray
    dates: pd.DatetimeIndex


def load_log_rv() -> dict[str, Series]:
    """Daily log realized variance per symbol."""
    series: dict[str, Series] = {}

    for asset_class, (path, column) in ASSET_FILES.items():
        df = pd.read_csv(path, parse_dates=["date"])
        if "symbol" not in df.columns:
            df["symbol"] = path.stem.split("_")[0]

        for symbol, group in df.groupby("symbol", sort=True):
            group = group.sort_values("date")
            if group["date"].duplicated().any():
                raise ValueError(f"{symbol}: duplicate dates in {path.name}")
            if (group[column] <= 0).any():
                raise ValueError(f"{symbol}: {column} must be positive to take logs")
            series[str(symbol)] = Series(asset_class, np.log(group[column].to_numpy()),
                                         pd.DatetimeIndex(group["date"]))

    return series


def add_weekday_variant(series: dict[str, Series], symbol: str = "BTCUSDT") -> dict[str, Series]:
    """Drop Saturday and Sunday from a 24/7 series."""
    asset_class, log_rv, dates = series[symbol]
    keep = dates.dayofweek.to_numpy() < 5
    series[f"{symbol}_noweekend"] = Series(asset_class, log_rv[keep], dates[keep])

    return series


def display_name(symbol: str) -> str:
    return DISPLAY_NAMES.get(symbol, symbol)


def increments(log_rv: np.ndarray, lags: list[int]) -> dict[int, np.ndarray]:
    """Lag-m increments Y(t+m) - Y(t), keyed by lag, as ols_estimation expects."""
    return {lag: fbms.simulate_fgn(log_rv, lag) for lag in lags}


def estimate_hurst(log_rv: np.ndarray, lags: list[int], k: float) -> float:
    """H_hat from the log-log regression of the k-th absolute moment."""
    return hpe.ols_estimation(k, increments(log_rv, lags))[1]


def hurst_table(series: dict[str, Series], lags: list[int], moments: list[float]) -> pd.DataFrame:
    """H_hat for every symbol at every moment order."""
    k_grid = np.array(moments)
    rows = []
    for symbol, (asset_class, log_rv, dates) in series.items():
        estimates = np.array([estimate_hurst(log_rv, lags, k) for k in moments])
        exponents = estimates * k_grid
        h_mono = float(k_grid @ exponents / (k_grid @ k_grid))
        rows.append({"symbol": symbol, "class": asset_class,
                     "start": dates[0].date(), "end": dates[-1].date(),
                     "n": len(log_rv),
                     **{f"H(k={k})": h for k, h in zip(moments, estimates)},
                     "H_mono": h_mono,
                     "max_dev": float(np.abs(exponents - h_mono * k_grid).max())})
    return pd.DataFrame(rows).set_index("symbol")


def log_log_scatter(series: dict[str, Series], symbol: str, lags: list[int], k: float):
    """log S_N(k, m) against log m for one symbol, with the fitted line of slope
    k * H_hat. Real-data counterpart of hpe.plot_log_log_scatter."""
    asset_class, log_rv, _ = series[symbol]
    colour = CLASS_COLOURS[asset_class]

    incr = increments(log_rv, lags)
    log_moments = [np.log(hpe.k_mom(k, incr[lag])) for lag in lags]
    intercept, slope = hpe.ols_estimation(k, incr)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(np.log(lags), log_moments, s=14, alpha=0.85, color=colour, zorder=3)
    ax.axline((0, intercept), slope=slope * k, color=REFERENCE_COLOUR, linewidth=1.2,
              label=rf"OLS fit, $\hat H={slope:.3f}$")

    ax.set_xlabel(r"$\log(m)$")
    ax.set_ylabel(rf"$\log(S_N({k},m))$")
    ax.set_title(f"{display_name(symbol)}: log-log scaling of absolute moments with OLS fit",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return fig


def moment_stability(series: dict[str, Series], symbol: str, moments: list[float],
                     lags: list[int]):
    """
    Scaling exponent kH against k for one symbol and all. Monoscaling means the points should lie on a line through the
    origin.
    """
    asset_class, log_rv, _ = series[symbol]
    k_grid = np.array(moments)
    upper_k = float(k_grid.max()) * 1.04

    incr = increments(log_rv, lags)
    exponents = np.array([hpe.ols_estimation(k, incr)[1] * k for k in moments])
    # least squares through the origin, i.e. the single H fitting every moment
    h_mono = float(k_grid @ exponents / (k_grid @ k_grid))

    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    colour = CLASS_COLOURS[asset_class]

    ax.plot([0, upper_k], [0, h_mono * upper_k], color=REFERENCE_COLOUR, linestyle="--",
            linewidth=1.0, zorder=2, label=rf"fit through origin, $\hat H = {h_mono:.3f}$")
    ax.plot(k_grid, exponents, color=colour, linewidth=0.9, alpha=0.55, zorder=3)
    ax.scatter(k_grid, exponents, s=14, color=colour, zorder=4, label=r"$k\hat H$")

    ax.set_xlim(0, upper_k)
    ax.set_ylim(0, float(max(exponents.max(), h_mono * upper_k)) * 1.06)
    ax.set_xticks([0.0, *k_grid])
    ax.set_xlabel("$k$", fontsize=10)
    ax.set_ylabel(r"$k\hat H$", fontsize=10)
    ax.set_title(f"{display_name(symbol)}: scaling exponent of the absolute moments",
                 fontsize=11, pad=10)
    ax.tick_params(labelsize=9, length=3, width=0.8)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=False, handlelength=1.6)

    return fig


def increment_gaussianity(series: dict[str, Series], symbol: str, lags: list[int],
                          k: float = 2.0):
    """Histogram of the increments X^(m) with a fitted normal density, one panel
    per lag m, following the argument in Gatheral et al.

    Two claims are checked at once, which is why the increments are left on their
    own scale instead of being standardised:

    red    N(mean, var) fitted to that lag's increments; closeness to the
           histogram is Gaussianity of the increments.
    blue   the m = 1 fit with both moments stretched by m^H; closeness to the red
           curve is self-similarity with exponent H."""
    asset_class, log_rv, _ = series[symbol]
    colour = CLASS_COLOURS[asset_class]
    h = estimate_hurst(log_rv, LAGS, k)

    base = fbms.simulate_fgn(log_rv, 1)
    base_mean, base_sd = base.mean(), base.std()
    # the x-range is shared across panels, otherwise the m^H widening is invisible
    span = 4.0 * fbms.simulate_fgn(log_rv, max(lags)).std()
    grid = np.linspace(-span, span, 400)
    bins = np.linspace(-span, span, 41)

    columns = 2
    rows = -(-len(lags) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.7 * rows),
                             squeeze=False, layout="constrained")
    fig.suptitle(rf"{display_name(symbol)}: increments of $\log RV$, $\hat H = {h:.3f}$",
                 fontsize=12)

    for idx, lag in enumerate(lags):
        ax = axes[idx // columns, idx % columns]
        incr = fbms.simulate_fgn(log_rv, lag)

        ax.hist(incr, bins=bins, density=True, color=colour, alpha=0.55,
                edgecolor="white", linewidth=0.4)
        fitted, = ax.plot(grid, norm.pdf(grid, incr.mean(), incr.std()),
                          color=REFERENCE_COLOUR, linewidth=1.3, label="fitted normal")
        scaled, = ax.plot(grid, norm.pdf(grid, base_mean * lag ** h, base_sd * lag ** h),
                          color=SCALED_COLOUR, linestyle="--", linewidth=1.3,
                          label=r"$m = 1$ fit scaled by $m^{\hat H}$")

        ax.set_title(rf"$m = {lag}$", fontsize=10)
        ax.set_ylabel("density", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if idx // columns == rows - 1:
            ax.set_xlabel(r"$X^{(m)}$", fontsize=9)
        if idx == 0:
            ax.legend(handles=[fitted, scaled], loc="upper right", fontsize=7.5,
                      framealpha=0.9)

    for idx in range(len(lags), rows * columns):  # blank any unused panel
        axes[idx // columns, idx % columns].axis("off")

    return fig


def main() -> None:
    series = add_weekday_variant(load_log_rv())
    table = hurst_table(series, LAGS, MOMENTS)
    table.to_csv(DATA_DIR / "hurst_estimates.csv")
    print(table.round(3).to_string())
    print("\nH_mono by class:")
    print(table.groupby("class")["H_mono"].describe()[["count", "mean", "std", "min", "max"]].round(3).to_string())

    FIGURE_DIR.mkdir(exist_ok=True)
    for symbol in PLOT_SYMBOLS:
        figures = {
            "log_log_scatter": log_log_scatter(series, symbol, LAGS, 2.0),
            "moment_stability": moment_stability(series, symbol, MOMENTS, LAGS),
            "increment_gaussianity": increment_gaussianity(series, symbol, GAUSSIAN_LAGS),
        }
        for name, fig in figures.items():
            fig.savefig(FIGURE_DIR / f"{name}_{symbol}.pdf", format="pdf", bbox_inches="tight")
            plt.close(fig)

    print(f"\nwrote {3 * len(PLOT_SYMBOLS)} figures to {FIGURE_DIR.name}/")


if __name__ == "__main__":
    main()