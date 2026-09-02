import numpy as np
import pandas as pd

import btc_data_scrape as btc

OUTPUT = btc.DATA_DIR / "BTCUSDT_daily_rv.csv"

STEP = pd.Timedelta("5min")
BARS_PER_DAY = 288  # 24h / 5min


def daily_rv(df: pd.DataFrame) -> pd.DataFrame:
    """Daily realized variance from 5-minute closes: RV_t = sum of squared
    5-minute log returns over day t (UTC).

    BTC trades continuously, so the return across midnight is an ordinary
    5-minute return and is kept, counted towards the day it ends in. Returns
    spanning an exchange outage are dropped, since a 7-hour move is not a
    5-minute return and squaring it would swamp that day's sum."""
    close = df.set_index("open_time")["close"]
    log_return = np.log(close).diff()
    log_return = log_return[close.index.to_series().diff() == STEP]

    squared = log_return ** 2
    daily = squared.groupby(squared.index.normalize()).agg(rv="sum", n_obs="size")
    daily["rvol"] = np.sqrt(daily["rv"])
    daily.index = daily.index.date
    daily.index.name = "date"

    return daily[["rv", "rvol", "n_obs"]]


def main() -> None:
    daily = daily_rv(btc.load())
    btc.DATA_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT)

    incomplete = daily[daily["n_obs"] < BARS_PER_DAY]
    print(f"{len(daily)} days, {daily.index[0]} to {daily.index[-1]}, wrote {OUTPUT.name}")
    print(f"incomplete days (< {BARS_PER_DAY} bars): {len(incomplete)}")
    print(f"rv: min {daily['rv'].min():.3e}, median {daily['rv'].median():.3e}, max {daily['rv'].max():.3e}")


if __name__ == "__main__":
    main()