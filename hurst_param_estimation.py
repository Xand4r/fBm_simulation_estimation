import numpy as np
import matplotlib.pyplot as plt
import fbm_simulation as fbms
from pathlib import Path
from math import gamma, factorial

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
    `lag` may be a scalar or an array (the result is broadcast accordingly)."""
    if not (0 < h < 1):
        raise ValueError(f"H must be in (0, 1), got {h}")
    if stepsize_1 <= 0 or stepsize_2 <= 0:
        raise ValueError('increments must have positive stepsizes')
    two_h = 2 * h
    i = np.asarray(lag, dtype=float)
    cov = 0.5 * (np.abs(stepsize_1 - i) ** two_h + np.abs(i + stepsize_2) ** two_h
                 - np.abs(stepsize_1 - i - stepsize_2) ** two_h - np.abs(i) ** two_h)
    return cov / (stepsize_1 ** h * stepsize_2 ** h)

def _gaussian_abs_moment(p: float) -> float:
    """E|Z|^p for Z ~ N(0,1): 2^{p/2} / sqrt(pi) * Gamma((p+1)/2)."""
    return 2.0 ** (p / 2.0) / np.sqrt(np.pi) * gamma((p + 1.0) / 2.0)

def hermite_abs_moment_coeff(k: float, two_j: int) -> float:
    """The (unnormalised) Hermite coefficient c_{2j} = E[|Z|^k H_{2j}(Z)] of the
    absolute-power function |x|^k, with H_n the probabilists' Hermite polynomial
    H_n(x) = n! * sum_{l=0}^{n/2} (-1)^l / (l! (n-2l)! 2^l) x^{n-2l}.
    Since n = 2j is even, |Z|^k x^{n-2l} = |Z|^{k+n-2l} under the expectation, so
    the coefficient is a finite sum of Gaussian absolute moments (exact)."""
    n = two_j
    total = 0.0
    for l in range(n // 2 + 1):
        weight = (-1) ** l * factorial(n) / (factorial(l) * factorial(n - 2 * l) * 2 ** l)
        total += weight * _gaussian_abs_moment(k + n - 2 * l)
    return total

def asymptotic_variance(k: float, lags: list[int], h: float,
                        max_hermite: int = 20, max_corr_lag: int = 20000) -> float:
    """Asymptotic variance sigma^2_OLS(k, M) of the OLS Hurst estimator, i.e. the
    limit of N * Var(H_hat_N) for H < 3/4 (Theorem):

        sqrt(N) (H_hat_N(k, M) - H) -> N(0, sigma^2_OLS(k, M)),
        sigma^2_OLS(k, M) = A^t G_M(k) A / (k^2 ||A||^4),

    where A_m = log(m) - mean_m log(m) are the centred log-lags (the regression
    covariate is k*log m, hence the k^2), and G_M(k) has entries

        g_{m,m'} = sum_{j>=1} c_{2j}^2 / (2j)! * sum_{i in Z} rho_{m,m'}(i)^{2j}.

    Here g_{m,m'} is the asymptotic covariance of sqrt(N) * log S_N(k, m); the
    delta-method 1/E|Z|^k factors are folded into the coefficients, i.e.
    c_{2j} = E[|Z|^k H_{2j}(Z)] / E|Z|^k. The sum_i rho^{2j} converges iff
    2j(2-2H) > 1; the binding term j=1 converges exactly for H < 3/4.

    The infinite sums are truncated at `max_hermite` Hermite orders and
    `max_corr_lag` correlation lags (the i-sum decays like i^{2j(2H-2)}, so it
    converges slowly as H approaches 3/4 -- raise `max_corr_lag` there)."""
    if k <= 0:
        raise ValueError('k must be positive')
    if not 0 < h < 0.75:
        raise ValueError(f'asymptotic variance requires 0 < H < 3/4, got {h}')

    lags_arr = np.asarray(lags, dtype=float)
    a = np.log(lags_arr) - np.mean(np.log(lags_arr))

    # Normalised Hermite coefficients c_{2j} and their weights c_{2j}^2 / (2j)!.
    mu_k = _gaussian_abs_moment(k)
    j_values = np.arange(1, max_hermite + 1)
    coeffs = np.array([hermite_abs_moment_coeff(k, 2 * j) / mu_k for j in j_values])
    weights = coeffs ** 2 / np.array([float(factorial(2 * j)) for j in j_values])

    i = np.arange(-max_corr_lag, max_corr_lag + 1)
    m_count = lags_arr.shape[0]
    g = np.zeros((m_count, m_count))
    for a_idx in range(m_count):
        for b_idx in range(a_idx, m_count):
            rho2 = cross_stepsize_correlation(lags_arr[a_idx], lags_arr[b_idx], i, h) ** 2
            power = np.ones_like(rho2)  # rho^0
            entry = 0.0
            for w in weights:
                power = power * rho2     # rho^{2j}
                entry += w * power.sum()
            g[a_idx, b_idx] = entry
            g[b_idx, a_idx] = entry

    return float((a @ g @ a) / (k ** 2 * (a @ a) ** 2))

def ols_estimation(k: float, fgns: dict[int, np.ndarray]) -> tuple[float, float]:
    rows = len(fgns.keys())
    ones = np.ones((rows,1))
    covariates = k * np.log(np.array(list(fgns.keys())).reshape(rows,1))
    design = np.hstack((ones, covariates))
    response = np.array([np.log(k_mom(k, fgns[i])) for i in fgns.keys()]).reshape(rows,1)

    estimates = np.linalg.inv(design.transpose() @ design) @ design.transpose() @ response

    return float(estimates[0, 0]), float(estimates[1, 0])

def plot_log_log_scatter(timepoints: np.ndarray, H_values: list[float], noise: np.ndarray, lags: list[int],
                         k: float):
    if k <= 0:
        raise ValueError('k must be positive')
    log_lags = np.log(lags)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for h in H_values:
        b_h = fbms.simulate_fbm(timepoints, h, noise)
        fgns = {}
        log_moments = []
        for lag in lags:
            fgns[lag] = fbms.simulate_fgn(b_h, lag)
            log_moments.append(np.log(k_mom(k, fgns[lag])))
        intercept, slope = ols_estimation(k, fgns)

        points = ax.scatter(log_lags, log_moments, s=12, alpha=0.8, zorder=3)
        colour = points.get_facecolor()[0]
        ax.axline((0, intercept), slope=slope * k, color=colour, linewidth=1.2,
                  label = rf"True $H={h:.2f}$, Estimated $\hat H={slope:.3f}$")

    ax.set_xlabel(r"$\log(m)$")
    ax.set_ylabel(rf"$\log(S_N({k},m))$")
    ax.set_title("Log-log scaling of absolute moments with OLS fits")
    ax.legend(title="true H (estimated slope)", loc="lower right", fontsize=6,
              title_fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig

def consistency_boxplots(rng, h_values: list[float], lags: list[int], k: float, sample_sizes: list[int],
                         simulation_count: int):
    # estimates[h][n] -> list of estimated slopes H-hat over the simulations
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

def var_vs_n_plot(min_n: int, max_n: int, rng, h_values:list[float], lags: list[int], k: float,
                  simulation_count: int, num_points: int = 25):
    if max_n <= min_n:
        raise ValueError(f"max_n = {max_n} is too small")
    if min_n <= max(lags):
        raise ValueError(f"min_n = {min_n} is not large enough for given lags")
    n_values = np.unique(np.geomspace(min_n, max_n, num=num_points).astype(int))
    timepoints = np.linspace(1/max_n, 1, max_n)
    estimates = {h: {n: [] for n in n_values} for h in h_values}
    for h in h_values:
        # calculate cholesky matrix here once for efficiency reasons and only then multiply with noise vector in loop
        chol = np.linalg.cholesky(fbms.cov_matrix(timepoints, h))
        for _ in range(simulation_count):
            fbm_full = chol @ rng.normal(0, 1, max_n)
            for n in n_values:
                fgns = {lag: fbms.simulate_fgn(fbm_full[:n], lag) for lag in lags}
                _, slope = ols_estimation(k, fgns)
                estimates[h][n].append(slope)
    rescaled_var = {h: [n * np.var(estimates[h][n]) for n in n_values] for h in h_values}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, h in enumerate(h_values):
        ax.plot(n_values, rescaled_var[h], color=f"C{i}", marker="o", markersize=3,
                markeredgecolor="white", markeredgewidth=0.4, linewidth=1.3, alpha=0.9,
                label=rf"$H={h:.2f}$")
        # asymptotic variance sigma^2_OLS is only defined for H < 3/4
        if h < 0.75:
            ax.axhline(asymptotic_variance(k, lags, h), color=f"C{i}", linestyle="--",
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
    fig5 = var_vs_n_plot(6, 5000, rng, hurst_values, list(range(1,5)), 2, 300)
    var_vs_n = Path(__file__).with_name("var_vs_n.pdf")
    fig5.savefig(var_vs_n, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()