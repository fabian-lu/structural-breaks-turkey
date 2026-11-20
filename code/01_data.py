"""
01_data.py -- download and clean the series used in the paper.

Series (see Table 2 of the paper):
  tur_etf      iShares MSCI Turkey ETF (TUR), monthly close, USD        Yahoo Finance
  cpi_yoy      CPI inflation, year-on-year, %                            OECD SDMX (DF_PRICES_ALL)
  policy_rate  overnight interbank / call-money rate, %                  OECD SDMX (DF_FINMARK)
  cci          consumer confidence index, amplitude-adjusted             OECD SDMX (DF_CLI)
  daily/usdtry USD/TRY, daily close (daily analysis)                     Yahoo Finance

Usage
  python 01_data.py             # clean the raw files already in data/raw (the 27 May 2026 snapshot)
  python 01_data.py --download  # re-download everything, truncate at --cutoff, then clean

Cleaning (identical for every monthly series): drop missing values and duplicate months,
place the series on a continuous month-start grid, linearly interpolate internal gaps,
start the sample at the date given in PLAN, add the log level for price series.
Output: data/processed/<name>.csv with columns Date, level[, log_level] and _manifest.csv.
"""
import argparse, datetime as dt, io, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, PROC = os.path.join(HERE, "data", "raw"), os.path.join(HERE, "data", "processed")

OECD = "https://sdmx.oecd.org/public/rest/data/"
SOURCES = {
    "tur_etf":     ("yf", "TUR"),
    "cpi_yoy":     ("oecd", OECD + "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0/TUR.M.N.CPI.._T.N.GY?dimensionAtObservation=AllDimensions&format=csvfile"),
    "policy_rate": ("oecd", OECD + "OECD.SDD.STES,DSD_STES@DF_FINMARK,/TUR.M.IRSTCI......?format=csvfile"),
    "cci":         ("oecd", OECD + "OECD.SDD.STES,DSD_STES@DF_CLI,/TUR.M.CCICP......?format=csvfile"),
}
# analysis plan: transform and sample start (paper, Section 3.1)
PLAN = {
    "tur_etf":     dict(price=True,  start="2008-04", label="MSCI Turkey ETF (TUR), USD"),
    "cpi_yoy":     dict(price=False, start="2004-01", label="CPI inflation, year-on-year (%)"),
    "policy_rate": dict(price=False, start="2006-01", label="Overnight interbank rate (%)"),
    "cci":         dict(price=False, start="2005-01", label="Consumer confidence index"),
}

def fetch_yf(ticker, interval, start="2003-01-01", end=None, retries=4):
    import yfinance as yf
    err = None
    for i in range(retries):
        try:
            d = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
            if d is not None and len(d) > 10:
                c = d["Close"]; c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
                out = c.dropna().reset_index(); out.columns = ["Date", "value"]
                out["Date"] = pd.to_datetime(out["Date"])
                if interval == "1mo": out["Date"] = out["Date"].dt.to_period("M").dt.to_timestamp()
                return out
            err = "empty"
        except Exception as e: err = repr(e)
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"yfinance failed for {ticker}: {err}")

def fetch_oecd(url):
    import requests
    r = requests.get(url, timeout=60, headers={"User-Agent": "research-script"}); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    df["Date"] = pd.to_datetime(df["TIME_PERIOD"], format="%Y-%m").dt.to_period("M").dt.to_timestamp()
    return df[["Date", "OBS_VALUE"]].rename(columns={"OBS_VALUE": "value"}).sort_values("Date").reset_index(drop=True)

def download(cutoff):
    os.makedirs(os.path.join(RAW, "daily"), exist_ok=True)
    for name, (kind, ref) in SOURCES.items():
        df = fetch_yf(ref, "1mo", end=cutoff) if kind == "yf" else fetch_oecd(ref)
        df = df[df["Date"] <= cutoff]
        df.to_csv(os.path.join(RAW, f"{name}.csv"), index=False); print(f"[raw] {name:12s} {len(df):4d} rows  {df.Date.min().date()}..{df.Date.max().date()}")
    d = fetch_yf("TRY=X", "1d", start="2015-01-01", end=cutoff)
    d.to_csv(os.path.join(RAW, "daily", "usdtry.csv"), index=False); print(f"[raw] daily/usdtry {len(d)} rows")

def clean():
    os.makedirs(PROC, exist_ok=True); manifest = []
    for name, cfg in PLAN.items():
        df = pd.read_csv(os.path.join(RAW, f"{name}.csv"), parse_dates=["Date"])
        df = df.dropna(subset=["value"]).drop_duplicates("Date").sort_values("Date")
        grid = pd.date_range(df["Date"].min(), df["Date"].max(), freq="MS")
        df = df.set_index("Date").reindex(grid).rename_axis("Date").reset_index()
        n_gap = int(df["value"].isna().sum())
        df["value"] = df["value"].interpolate(method="linear", limit_direction="both")
        df = df[df["Date"] >= cfg["start"]].rename(columns={"value": "level"}).reset_index(drop=True)
        if cfg["price"]: df["log_level"] = np.log(df["level"])
        df.to_csv(os.path.join(PROC, f"{name}.csv"), index=False)
        manifest.append(dict(series=name, label=cfg["label"], transform="log level" if cfg["price"] else "level",
                             n=len(df), start=df.Date.min().date(), end=df.Date.max().date(), interpolated=n_gap))
        print(f"[proc] {name:12s} n={len(df):4d}  {df.Date.min().date()}..{df.Date.max().date()}  interpolated={n_gap}")
    d = pd.read_csv(os.path.join(RAW, "daily", "usdtry.csv"), parse_dates=["Date"]).dropna().drop_duplicates("Date").sort_values("Date")
    d = d.rename(columns={"value": "level"}); d["log_level"] = np.log(d["level"])
    d["rv21"] = d["log_level"].diff().rolling(21).std() * np.sqrt(252)
    d.to_csv(os.path.join(PROC, "usdtry_daily.csv"), index=False); print(f"[proc] usdtry_daily n={len(d)}  {d.Date.min().date()}..{d.Date.max().date()}")
    pd.DataFrame(manifest).to_csv(os.path.join(PROC, "_manifest.csv"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--download", action="store_true"); ap.add_argument("--cutoff", default="2026-05-31")
    a = ap.parse_args()
    if a.download: download(pd.Timestamp(a.cutoff))
    clean()
