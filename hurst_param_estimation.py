import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import fbm_simulation as fbms
from pathlib import Path

# Hurst values in their canonical order. Colours are taken from a value's position
# here, not from its position in the list a figure was called with, so a plot of
# only a subset still colours them consistently.
H_PALETTE_ORDER = (0.1, 0.3, 0.5, 0.7, 0.9)

def h_colour(h: float) -> str:
    """Colour-cycle entry for a Hurst value. Values off the canonical grid fall
    back to the nearest one, which keeps them comparable to their neighbour but
    can make two such values share a colour."""
    idx = min(range(len(H_PALETTE_ORDER)), key=lambda i: abs(H_PALETTE_ORDER[i] - h))
    return f"C{idx}"

def k_mom(k: float, incr: np.ndarray) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    return sum(abs(incr)**k) / incr.shape[0]

def cross_stepsize_correlation(stepsize_1: int, stepsize_2: int, lag, h: float):
    """Correlation rho_{m,m'}(i) between the standardised increments
    X^{(m)}_t = B_H(t+m) - B_H(t) and X^{(m')}_{t+i} = B_H(t+i+m') - B_H(t+i),
    with m = stepsize_1, m' = stepsize_2, i = lag (in grid-step units).

    For fBm, Cov(X^{(m)}_0, X^{(m')}_i)
        = 1/2 [ |m-i|^{2H} + |i+m'|^{2H} - |m-i-m'|^{2H} - |i|^{2H} ],
    and Var(X^{(m)}) = m^{2H}, so dividing by m^H m'^H gives the correlation.
    `lag` may be a scalar or an array, in which case the result is broadcast."""
    if not (0 < h < 1):
        raise ValueError(f"H must be in (0, 1), got {h}")
    if stepsize_1 <= 0 or stepsize_2 <= 0:
        raise ValueError('increments must have positive stepsizes')
    two_h = 2 * h
    i = np.asarray(lag, dtype=float)
    cov = 0.5 * (np.abs(stepsize_1 - i) ** two_h + np.abs(i + stepsize_2) ** two_h
                 - np.abs(stepsize_1 - i - stepsize_2) ** two_h - np.abs(i) ** two_h)
    return cov / (stepsize_1 ** h * stepsize_2 ** h)

def asymptotic_variance(lags: list[int], h: float, max_corr_lag: int = 20000) -> float:
    """Asymptotic variance of the OLS Hurst estimator for k = 2, i.e. the limit of
    N * Var(H_hat_N) for H < 3/4:

        sigma^2_OLS = A^t G A / (4 ||A||^4),   g_{m,m'} = 2 * sum_i rho_{m,m'}(i)^2,

    with A_m = log(m) - mean_m log(m) the centred log-lags (the covariate is
    2*log m, hence the 4 = k^2). The Hermite expansion behind G collapses to a
    single term here, since |x|^2 - E|Z|^2 = H_2(x) gives c_2 = 2 and c_{2j} = 0
    for j >= 2.

    The i-sum converges iff H < 3/4 and is truncated at `max_corr_lag`. Its terms
    decay like i^{4H-4}, so that bound needs raising near H = 3/4."""
    if not 0 < h < 0.75:
        raise ValueError(f'asymptotic variance requires 0 < H < 3/4, got {h}')

    lags_arr = np.asarray(lags, dtype=float)
    a = np.log(lags_arr) - np.mean(np.log(lags_arr))

    i = np.arange(-max_corr_lag, max_corr_lag + 1)
    m_count = lags_arr.shape[0]
    g = np.zeros((m_count, m_count))
    for a_idx in range(m_count):
        for b_idx in range(a_idx, m_count):
            rho = cross_stepsize_correlation(lags_arr[a_idx], lags_arr[b_idx], i, h)
            g[a_idx, b_idx] = g[b_idx, a_idx] = 2.0 * np.sum(rho ** 2)

    return float((a @ g @ a) / (4.0 * (a @ a) ** 2))

def ols_estimation(k: float, fgns: dict[int, np.ndarray]) -> tuple[float, float]:
    rows = len(fgns.keys())
    ones = np.ones((rows,1))
    covariates = k * np.log(np.array(list(fgns.keys())).reshape(rows,1))
    design = np.hstack((ones, covariates))
    response = np.array([np.log(k_mom(k, fgns[i])) for i in fgns.keys()]).reshape(rows,1)

    estimates = np.linalg.inv(design.transpose() @ design) @ design.transpose() @ response

    return float(estimates[0, 0]), float(estimates[1, 0])

def plot_log_log_scatter(timepoints: np.ndarray, h_values: list[float], noise: np.ndarray, lags: list[int],
                         k: float):
    log_lags = np.log(lags)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for h in h_values:
        b_h = fbms.simulate_fbm(timepoints, h, noise)
        fgns = {}
        log_moments = []
        for lag in lags:
            fgns[lag] = fbms.simulate_fgn(b_h, lag)
            log_moments.append(np.log(k_mom(k, fgns[lag])))
        intercept, slope = ols_estimation(k, fgns)

        colour = h_colour(h)
        ax.scatter(log_lags, log_moments, s=12, alpha=0.8, color=colour, zorder=3)
        ax.axline((0, intercept), slope=slope * k, color=colour, linewidth=1.2,
                  label = rf"True $H={h:.2f}$, Estimated $\hat H={slope:.3f}$")

    ax.set_xlabel(r"$\log(m)$")
    ax.set_ylabel(rf"$\log(S_N({k},m))$")
    ax.set_title("Log-log scaling of absolute moments with OLS fits")
    ax.legend(loc="lower right", fontsize=6,
              title_fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig

def distribution_check_plot(simulation_count: int, k: float, h_values: list[float], rng, lags: list[int],
                            sample_size: int ):
    centered_estimates = {h: [] for h in h_values}
    timepoints = np.linspace(1/sample_size, 1, sample_size)
    cholesky_matrices = {h: np.linalg.cholesky(fbms.cov_matrix(timepoints, h)) for h in h_values}
    for _ in range(simulation_count):
        noise = rng.normal(0, 1, sample_size)
        for h in h_values:
            fbm = cholesky_matrices[h] @ noise
            fgns = {lag: fbms.simulate_fgn(fbm, lag) for lag in lags}
            _, slope = ols_estimation(k, fgns)
            centered_estimates[h].append(slope - h)

    fig, axes = plt.subplots(len(h_values), 2, figsize=(7.5, 2.6 * len(h_values)),
                             squeeze=False, layout="constrained")
    fig.suptitle("Distribution of the Centered Estimator", fontsize=16)

    z_scores = {h: np.asarray(centered_estimates[h]) / np.std(centered_estimates[h]) for h in h_values}
    # plotting positions centered in their 1/M slice, keeping the extremes finite
    plot_positions = (np.arange(1, simulation_count + 1) - 0.5) / simulation_count
    normal_quantiles = norm.ppf(plot_positions)
    grid = np.linspace(-4.0, 4.0, 400)
    bins = np.linspace(-4.0, 4.0, 41)

    # column 0: normal QQ plot
    for row_idx, h in enumerate(h_values):
        ax = axes[row_idx, 0]
        z = np.sort(z_scores[h])

        ax.scatter(normal_quantiles, z, s=8, alpha=0.7, color=h_colour(h), zorder=3)
        ref = ax.axline((0, 0), slope=1, color="#c0392b", linestyle="--", linewidth=1.2,
                        label="standard normal")
        ax.set_ylabel(rf"$H = {h}$" "\n" r"sample quantiles", fontsize=9)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if row_idx == len(h_values) - 1:
            ax.set_xlabel("theoretical quantiles", fontsize=9)
        if row_idx == 0:
            ax.legend(handles=[ref], loc="upper left", fontsize=8, framealpha=0.9)

    # column 1: histogram against the standard normal density
    for row_idx, h in enumerate(h_values):
        ax = axes[row_idx, 1]
        z = z_scores[h]

        ax.hist(z, bins=bins, density=True, color=h_colour(h), alpha=0.55,
                edgecolor="white", linewidth=0.4)
        ax.plot(grid, norm.pdf(grid), color="#c0392b", linestyle="--", linewidth=1.2)
        ax.set_ylabel("density", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if row_idx == len(h_values) - 1:
            ax.set_xlabel(r"standardized $\hat H - H$", fontsize=9)

    return fig


def consistency_boxplots(rng, h_values: list[float], lags: list[int], k: float, sample_sizes: list[int],
                         simulation_count: int):
    # estimates[h][n] holds the estimated slopes H-hat over all simulations
    estimates = {h: {n: [] for n in sample_sizes} for h in h_values}
    for n in sample_sizes:
        timepoints = np.linspace(1/n, 1, n)
        for _ in range(simulation_count):
            noise = rng.normal(0, 1, n)
            for h in h_values:
                fbm = fbms.simulate_fbm(timepoints, h, noise)
                fgns = {lag: fbms.simulate_fgn(fbm, lag) for lag in lags}
                _, slope = ols_estimation(k, fgns)
                estimates[h][n].append(slope)

    errors = {h: {n: [s - h for s in estimates[h][n]] for n in sample_sizes}
              for h in h_values}
    lim = 1.1 * max(abs(e) for h in h_values for n in sample_sizes for e in errors[h][n])

    fig = plt.figure(figsize=(7.5, 10.5), layout="constrained")
    fig.suptitle("Consistency Across Sample Sizes", fontsize=16)
    subfigs = np.atleast_1d(fig.subfigures(len(h_values), 1, hspace=0.07))

    box_style = dict(
        widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
        medianprops=dict(color="#c0392b", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                        markeredgecolor="black", alpha=0.6),
    )

    for row_idx, (subfig, h) in enumerate(zip(subfigs, h_values)):
        subfig.suptitle(rf"$H = {h}$", fontsize=14, fontweight="bold")
        axes = np.atleast_1d(subfig.subplots(1, len(sample_sizes), sharey=True))
        for ax, n in zip(axes, sample_sizes):
            ax.boxplot(errors[h][n], **box_style)
            ref = ax.axhline(0, color="#333333", linestyle="--", linewidth=1.0,
                             label="no bias ($\\hat H = H$)")
            ax.set_ylim(-lim, lim)
            ax.set_xticks([1])
            ax.set_xticklabels([f"$N = {n}$"])
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(r"estimation error $\hat H - H$")
        if row_idx == 0:
            axes[-1].legend(handles=[ref], loc="upper right", fontsize=8, framealpha=0.9)

    return fig

def var_vs_n_plot(min_n: int, max_n: int, rng, h_values:list[float], lags: list[int],
                  simulation_count: int, num_points: int = 25):
    # fixed to k = 2, which is what the asymptotic variance below is derived for
    k = 2.0
    if max_n <= min_n:
        raise ValueError(f"max_n = {max_n} is too small")
    if min_n <= max(lags):
        raise ValueError(f"min_n = {min_n} is not large enough for given lags")
    n_values = np.unique(np.geomspace(min_n, max_n, num=num_points).astype(int))
    timepoints = np.linspace(1/max_n, 1, max_n)
    estimates = {h: {n: [] for n in n_values} for h in h_values}
    for h in h_values:
        # factorise once outside the loop and only multiply by the noise vector inside
        chol = np.linalg.cholesky(fbms.cov_matrix(timepoints, h))
        for _ in range(simulation_count):
            fbm_full = chol @ rng.normal(0, 1, max_n)
            for n in n_values:
                fgns = {lag: fbms.simulate_fgn(fbm_full[:n], lag) for lag in lags}
                _, slope = ols_estimation(k, fgns)
                estimates[h][n].append(slope)
    rescaled_var = {h: [n * np.var(estimates[h][n]) for n in n_values] for h in h_values}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for h in h_values:
        ax.plot(n_values, rescaled_var[h], color=h_colour(h), marker="o", markersize=3,
                markeredgecolor="white", markeredgewidth=0.4, linewidth=1.3, alpha=0.9,
                label=rf"$H={h:.2f}$")
        # asymptotic variance sigma^2_OLS is only defined for H < 3/4
        if h < 0.75:
            ax.axhline(asymptotic_variance(lags, h), color=h_colour(h), linestyle="--",
                       linewidth=1.0, alpha=0.7)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"sample size $N$")
    ax.set_ylabel(r"$N\,\cdot\,$Var($\hat H$)")
    ax.set_title(r"CLT scaling of the OLS Hurst estimator")
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([], [], color="black", linestyle="--", linewidth=1.0,
            label=r"$\sigma^2_{\mathrm{OLS}}$ (theory)")
    ax.legend(title="true $H$", fontsize=8, title_fontsize=9, framealpha=0.9)
    fig.tight_layout()

    return fig

def moment_stability_plot(moments: list[float], h: float, rng, sample_size: int, lags: list[int]):
    timepoints = np.linspace(1/sample_size, 1, sample_size)
    noise = rng.normal(0, 1, sample_size)
    fbm = fbms.simulate_fbm(timepoints, h, noise)
    fgns = {lag: fbms.simulate_fgn(fbm, lag) for lag in lags}
    # scaling exponent of the k-th moment, which under self-similarity is k * H,
    # so the estimates have to fall on a line through the origin
    k_grid = np.array(moments)
    exponents = np.array([ols_estimation(k, fgns)[1] * k for k in moments])

    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    colour = h_colour(h)
    upper_k = float(k_grid.max()) * 1.04

    ax.plot([0, upper_k], [0, h * upper_k], color="#c0392b", linestyle="--", linewidth=1.0,
            zorder=2, label=rf"$kH$, $H = {h}$")
    ax.plot(k_grid, exponents, color=colour, linewidth=0.9, alpha=0.55, zorder=3)
    ax.scatter(k_grid, exponents, s=14, color=colour, zorder=4, label=r"$k\hat H$")

    ax.set_xlim(0, upper_k)
    ax.set_ylim(0, float(max(exponents.max(), h * upper_k)) * 1.06)
    ax.set_xticks([0.0, *k_grid])
    ax.set_xlabel("$k$", fontsize=10)
    ax.set_ylabel(r"$k\hat H$", fontsize=10)
    ax.set_title("Scaling exponent of the absolute moments", fontsize=12, pad=10)
    ax.tick_params(labelsize=9, length=3, width=0.8)
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.legend(loc="upper left", fontsize=9, frameon=False, handlelength=1.6)

    return fig

def main() -> None:
    rng = np.random.default_rng(404)
    timepoints = np.linspace(0.01, 1, 999)
    noise = rng.normal(0, 1, len(timepoints))

    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig3 = plot_log_log_scatter(timepoints, hurst_values, noise, list(range(1, 31)), 2)
    ols = Path(__file__).with_name("log_log_scatter.pdf")
    fig3.savefig(ols, format="pdf", bbox_inches="tight")
    fig4 = consistency_boxplots(rng, [0.1, 0.5, 0.9], list(range(1,31)), 2, [512, 1024, 2048],
                                1000)
    consistent = Path(__file__).with_name("consistency_boxplots.pdf")
    fig4.savefig(consistent, format="pdf", bbox_inches="tight")
    fig5 = var_vs_n_plot(6, 16000, rng, hurst_values, list(range(1,5)), 10000, num_points=35)
    var_vs_n = Path(__file__).with_name("var_vs_n.pdf")
    fig5.savefig(var_vs_n, format="pdf", bbox_inches="tight")
    fig6 = distribution_check_plot(1000, 2, [0.1, 0.5, 0.9], rng, list(range(1, 10)), 3000)
    distr = Path(__file__).with_name("distribution_check.pdf")
    fig6.savefig(distr, format="pdf", bbox_inches="tight")
    fig7 = moment_stability_plot([0.5, 1, 1.5, 2, 3], 0.2, rng, 3500, list(range(1,31)))
    stabl = Path(__file__).with_name("moment_stability_check.pdf")
    fig7.savefig(stabl, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()