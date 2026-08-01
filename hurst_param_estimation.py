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

def main() -> None:
    rng = np.random.default_rng()
    timepoints = np.linspace(0.01, 1, 999)
    noise = rng.normal(0, 1, len(timepoints))

    hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig3 = plot_log_log_scatter(timepoints, hurst_values, noise, list(range(1, 31)), 2)
    # save as a vector PDF
    ols = Path(__file__).with_name("log_log_scatter.pdf")
    fig3.savefig(ols, format="pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()