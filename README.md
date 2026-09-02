# Simulating and Estimating Fractional Brownian Motion

This project accompanies a bachelor thesis on **fractional Brownian motion (fBm)**.
It is split into three parts:

1. **Simulation** — generating sample paths of fBm for different values of the
   Hurst parameter *H*.
2. **Estimation** — estimating *H* from an observed path with a log-log
   regression of the absolute moments, and checking that the estimator is
   consistent and asymptotically normal.
3. **Application** — applying the same estimator to financial time series,
   namely daily **realized variance** of crypto, equities and futures.

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

**Fractional Gaussian noise (fGn)** is the increment process of fBm sampled on
an equally spaced grid,

```
X_i = B_H(t_{i+1}) - B_H(t_i).
```

It is a stationary, centered Gaussian sequence. For `H = 0.5` the increments are
independent, while `H < 0.5` produces negatively correlated (anti-persistent)
and `H > 0.5` positively correlated (long-range dependent) noise.

## Simulation

`fbm_simulation.py` builds the covariance matrix on a grid of timepoints,
factorises it with a Cholesky decomposition, and multiplies by standard normal
noise to produce sample paths. The same noise vector is reused across all *H*
so that the differences in the plot are due to *H* alone.

The fGn paths are obtained by differencing each simulated fBm path, so the same
underlying noise vector drives both the fBm and its increments for every *H*.

Running the script produces figures showing, for each Hurst value
(`H = 0.1, 0.3, 0.5, 0.7, 0.9`), one sample fBm path and its corresponding fGn
increment series.

## Estimation

`hurst_param_estimation.py` estimates *H* from the scaling of the absolute
moments of the increments. For fBm the *k*-th absolute moment of the lag-*m*
increment satisfies

```
S_N(k, m) = 1/N sum_t |B_H(t+m) - B_H(t)|^k  ~  c_k * m^{kH},
```

so regressing `log S_N(k, m)` on `k log m` across lags *m* gives `H_hat` as the
slope. On simulated paths the script checks

- the log-log scaling and the fitted slope,
- consistency, as boxplots of `H_hat - H` over growing sample sizes,
- asymptotic normality, as QQ plots and histograms of the centered estimator,
- the CLT rate, comparing `N * Var(H_hat)` against the asymptotic variance
  `sigma^2_OLS` computed in closed form for `k = 2` and `H < 3/4`,
- monoscaling, i.e. that the estimated exponent `zeta(k)` is linear in *k*.

## Application to financial time series

`data_handling.py` runs the same estimator on daily **realized variance** (RV).
The quantity modelled is `log RV`: RV is a variance, bounded below by zero and
heavily right-skewed, so its increments are far from the stationary Gaussian
ones the estimator assumes, whereas `log RV` is what the rough-volatility
literature treats as fBm-like. The script reports `H_hat` per symbol at several
moment orders and reproduces the log-log, monoscaling and increment-Gaussianity
diagnostics on real data.

The estimates come out well below `0.5` across all three asset classes, which is
the "rough volatility" finding the thesis sets out to check.

### Data sources

Neither the raw downloads nor the built datasets are tracked in this repository
(see `.gitignore`); the scripts below regenerate them.

**Crypto** — BTCUSDT 5-minute klines scraped from the public
[Binance market data archive](https://data.binance.vision) by
`btc_data_scrape.py`, which pulls the monthly archive, tops it up with the daily
files for the still-open month, and combines everything into
`data/BTCUSDT_5m_full.csv`. `realized_variance.py` then aggregates those bars
into daily realized variance,

```
RV_t = sum of squared 5-minute log returns over day t (UTC),
```

written to `data/BTCUSDT_daily_rv.csv`. BTC trades continuously, so returns
across midnight are kept, while returns spanning an exchange outage are dropped
rather than squared into that day's sum.

**Equities and futures** — daily realized measures from the **VOLatility Archive
for Realized Estimates (VOLARE)** [1], placed by hand in
`data/realized_variance_stocks.csv` (40 DOW30 / S&P100 names) and
`data/realized_variance_futures.csv` (5 contracts: C, CL, ES, GC, NG). Both
carry the standard realized-measure columns; the one used here is `rv5`, the
5-minute realized variance, matching the estimator used for BTC.

One difference in indexing matters for the estimator: increments are taken by
row, so one row must be one time step. The VOLARE panels are indexed in trading
days, with weekends and holidays simply absent, whereas BTC is indexed in
calendar days. Because weekend activity on BTC is markedly lower, that
seasonality aliases into every lag that is a multiple of 7, so `data_handling.py`
also carries a weekend-free BTCUSDT variant for comparison.

## Usage

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python fbm_simulation.py          # simulated fBm and fGn paths
python hurst_param_estimation.py  # estimator diagnostics on simulated paths

python btc_data_scrape.py         # download and combine Binance klines
python realized_variance.py       # 5-minute bars -> daily RV
python data_handling.py           # Hurst estimates and figures on real data
```

The first two scripts run standalone. The last three need the datasets described
above: `btc_data_scrape.py` downloads roughly 140 MB of klines and builds a
~146 MB CSV, and the VOLARE files have to be supplied separately.

## Requirements

See `requirements.txt` (NumPy, SciPy, Matplotlib, pandas, and requests plus
`binance-bulk-downloader` for the scrape).

## References

[1] Cipollini, F., Cruciani, G., Gallo, G. M., Insana, A., Otranto, E., &
Spagnolo, F. (2026). *VOLatility Archive for Realized Estimates (VOLARE).*
https://doi.org/10.48550/arXiv.2602.19732