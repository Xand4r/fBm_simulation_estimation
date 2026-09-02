import glob
import io
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from binance_bulk_downloader.downloader import BinanceBulkDownloader

PROJECT_DIR = Path(__file__).parent
RAW_DIR = PROJECT_DIR / "raw"
DATA_DIR = PROJECT_DIR / "data"

MONTHLY_DIR = RAW_DIR / "data/spot/monthly/klines/BTCUSDT/5m"
DAILY_DIR = RAW_DIR / "data/spot/daily/klines/BTCUSDT/5m"
DAILY_URL = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{d}.zip"
OUTPUT = DATA_DIR / "BTCUSDT_5m_full.csv"

# BTCUSDT starts here on Binance spot
FIRST_MONTH = "2017-08"

COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
]
PRICE_COLS = ["open", "high", "low", "close", "volume"]

# Binance switched candlestick timestamps from milliseconds to microseconds with the
# 2025-01 files, so a single `unit=` for the whole archive is wrong. Anything
# above this threshold is microseconds
US_THRESHOLD = 1e15


def to_utc(timestamps: pd.Series) -> pd.Series:
    """Binance epoch timestamps to UTC datetimes, in either ms or us."""
    ts = pd.to_numeric(timestamps, errors="coerce")
    return pd.to_datetime(ts.where(ts < US_THRESHOLD, ts // 1000), unit="ms", utc=True)


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    BinanceBulkDownloader(
        destination_dir=str(RAW_DIR),
        data_type="klines",
        data_frequency="5m",
        asset="spot",  # or 'um' for USDT-M Futures
        timeperiod_per_file="monthly",
        symbols="BTCUSDT",
    ).run_download()


def download_recent_days(lookback: int = 70) -> int:
    """Fill in the open month from the daily files, since the monthly archive only
    appears once a month has closed. The lookback overlaps the last complete
    month, but combine() drops the duplicate rows."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    fetched = 0
    for offset in range(lookback, -1, -1):
        day = today - timedelta(days=offset)
        target = DAILY_DIR / f"BTCUSDT-5m-{day}.csv"
        if target.exists():
            continue
        response = requests.get(DAILY_URL.format(d=day), timeout=30)
        if response.status_code == 404:  # not published yet, or before the pair existed
            continue
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(DAILY_DIR)
        fetched += 1
    return fetched


def combine() -> pd.DataFrame:
    csv_files = sorted(glob.glob(str(MONTHLY_DIR / "*.csv")))
    csv_files += sorted(glob.glob(str(DAILY_DIR / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"no klines found under {RAW_DIR}")

    df = pd.concat([pd.read_csv(f, names=COLS, header=None) for f in csv_files],
                   ignore_index=True)

    df["open_time"] = to_utc(df["open_time"])
    df["close_time"] = to_utc(df["close_time"])
    df[PRICE_COLS] = df[PRICE_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    # the ms/us mixup used to put rows in the year 56971, so check the range
    span = (df["open_time"].min(), df["open_time"].max())
    if span[0] < pd.Timestamp("2017-01-01", tz="UTC") or span[1] > pd.Timestamp.now(tz="UTC"):
        raise ValueError(f"timestamps outside the plausible range: {span[0]} to {span[1]}")

    return df


def load() -> pd.DataFrame:
    #Read the combined CSV back
    df = pd.read_csv(OUTPUT)
    for col in ("open_time", "close_time"):
        df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
    return df


def main() -> None:
    download()
    print(f"fetched {download_recent_days()} daily files for the open month")
    df = combine()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    steps = (df["open_time"].diff() == pd.Timedelta("5min")).sum()
    print(f"{len(df)} rows, {df['open_time'].min()} to {df['open_time'].max()}, wrote {OUTPUT.name}")
    print(f"5m steps: {steps}, other: {len(df) - 1 - steps}")


if __name__ == "__main__":
    main()