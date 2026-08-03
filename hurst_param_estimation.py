import numpy as np
import matplotlib.pyplot as plt
import fbm_simulation as fbms
from pathlib import Path

def k_mom(k: float, incr: np.ndarray) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    return sum(abs(incr)**k) / incr.shape[0]

def OLS_estimation(k: float, fgns: dict[int, np.ndarray]) -> tuple[float, float]:
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
        b_H = fbms.simulate_fbm(timepoints, h, noise)
        fgns = {}
        log_moments = []
        for lag in lags:
            fgns[lag] = fbms.simulate_fgn(b_H, lag)
            log_moments.append(np.log(k_mom(k, fgns[lag])))
        intercept, slope = OLS_estimation(k, fgns)

        points = ax.scatter(log_lags, log_moments, s=12, alpha=0.8, zorder=3)
        colour = points.get_facecolor()[0]
        ax.axline((0, intercept), slope=slope * k, color=colour, linewidth=1.2,
                  label = rf"True $H={h:.2f}$, Estimated $\hat H={slope:.3f}$")

    ax.set_xlabel("log(m)")
    ax.set_ylabel(f"log(S_N({k},m))")
    ax.set_title("Log-log scaling of absolute moments with OLS fits")
    ax.legend(title="true H (estimated slope)", loc="lower right", fontsize=6,
              title_fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig

def consistency_boxplots(rng, H_values: list[float], lags: list[int], k: float, sample_sizes: list[int],
                         simulation_count: int):
    # estimates[h][n] -> list of estimated slopes (\hat H) over the simulations
    estimates = {h: {n: [] for n in sample_sizes} for h in H_values}
    for n in sample_sizes:
        timepoints = np.linspace(1/n, 1, n)
        for _ in range(simulation_count):
            noise = rng.normal(0, 1, n)
            for h in H_values:
                fbm = fbms.simulate_fbm(timepoints, h, noise)
                fgns = {lag: fbms.simulate_fgn(fbm, lag) for lag in lags}
                _, slope = OLS_estimation(k, fgns)
                estimates[h][n].append(slope)

    errors = {h: {n: [s - h for s in estimates[h][n]] for n in sample_sizes}
              for h in H_values}
    lim = 1.1 * max(abs(e) for h in H_values for n in sample_sizes for e in errors[h][n])

    fig = plt.figure(figsize=(7.5, 10.5))
    subfigs = np.atleast_1d(fig.subfigures(len(H_values), 1, hspace=0.07))

    box_style = dict(
        widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
        medianprops=dict(color="#c0392b", linewidth=1.8),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                        markeredgecolor="black", alpha=0.6),
    )

    for row_idx, (subfig, h) in enumerate(zip(subfigs, H_values)):
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

def main() -> None:
    rng = np.random.default_rng(404)
    timepoints = np.linspace(0.01, 1, 999)
    noise = rng.normal(0, 1, len(timepoints))

    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig3 = plot_log_log_scatter(timepoints, hurst_values, noise, list(range(1, 31)), 2)
    ols = Path(__file__).with_name("log_log_scatter.pdf")
    fig3.savefig(ols, format="pdf", bbox_inches="tight")
    plt.show()
    fig4 = consistency_boxplots(rng, [0.1, 0.5, 0.9], list(range(1,31)), 2, [512, 1024, 2048],
                                1000)
    consistent = Path(__file__).with_name("consistency_boxplots.pdf")
    fig4.savefig(consistent, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()