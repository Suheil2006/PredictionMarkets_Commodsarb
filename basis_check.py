"""
MCX-COMEX silver basis check.
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import eikon as ek

KG_TO_TROY_OZ = 32.1507
START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "mcx_comex_basis.csv")

MCX_RIC = "MSIc1"
COMEX_RIC = "SIc1"
FX_RIC = "INR="


def connect():
    load_dotenv()
    ek.set_app_key(os.environ.get("EIKON_APP_KEY"))

    # Ensure Eikon session is open (requires Refinitiv Workspace / LSEG Workspace running in the background)
    try:
        from eikon.streaming_session.session import Session
        session = ek.get_desktop_session()
        if session.get_open_state() != Session.State.Open:
            raise RuntimeError(
                "Eikon session is not open. Please ensure Refinitiv Workspace (LSEG) is running "
                "in the background and logged in on this machine."
            )
    except Exception as e:
        raise RuntimeError(
            "Failed to connect to Eikon API Proxy. Please ensure that Refinitiv Workspace (LSEG) "
            "is open in the background and logged in."
        ) from e


def sanity_check_mcx_unit():
    """Print the raw snapshot value so you can eyeball the correct multiplier
    against a real MCX quote instead of inferring it from the basis itself."""
    df, err = ek.get_data(MCX_RIC, ["CF_CLOSE", "TR.InstrumentDescription", "CF_CURR"])
    print("\n=== MCX RAW SNAPSHOT SANITY CHECK ===")
    print(df)
    if err:
        print(f"Errors: {err}")
    raw = df["CF_CLOSE"].iloc[0]
    print(f"\nRaw MSIc1 CF_CLOSE = {raw}")
    print(f"If quoted per 10g:  implies INR/kg = {raw * 100:,.0f}")
    print(f"If quoted per 100g: implies INR/kg = {raw * 10:,.0f}")
    print("Compare both against a real MCX silver quote (e.g. Groww/Investing.com, ~INR/kg) "
          "and pick whichever is close. Do NOT pick based on which produces a 'nicer' basis.")
    return raw


def pull_prices(ric, label):
    df = ek.get_timeseries(ric, fields=["CLOSE"], start_date=START_DATE, end_date=END_DATE, interval="daily")
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {label} ({ric})")
    df = df.rename(columns={"CLOSE": label})
    df.index.name = "Date"
    return df[[label]]


def build_dataset():
    mcx = pull_prices(MCX_RIC, "MCX_RAW")
    comex = pull_prices(COMEX_RIC, "CME_USD_OZ")
    fx = pull_prices(FX_RIC, "USDINR")

    merged = mcx.join(comex, how="outer").join(fx, how="outer").sort_index()
    merged = merged.ffill(limit=2).dropna(how="any")
    return merged


def compute_basis(df, mcx_multiplier_to_kg):
    df = df.copy()
    df["MCX_INR_KG"] = df["MCX_RAW"] * mcx_multiplier_to_kg
    df["MCX_USD_OZ"] = (df["MCX_INR_KG"] / df["USDINR"]) / KG_TO_TROY_OZ
    df["BASIS_USD"] = df["MCX_USD_OZ"] - df["CME_USD_OZ"]
    df["BASIS_PCT"] = df["BASIS_USD"] / df["CME_USD_OZ"] * 100.0
    return df


def compute_vol_stats(df):
    cme_ret = df["CME_USD_OZ"].pct_change().dropna()
    cme_vol_ann = cme_ret.std() * np.sqrt(252)

    # convert BASIS_PCT (percentage-point scale, e.g. 20.6) to fraction scale
    # (0.206) BEFORE differencing, so it's on the same scale as pct_change()
    basis_frac = df["BASIS_PCT"] / 100.0
    basis_vol_ann = basis_frac.diff().dropna().std() * np.sqrt(252)

    ratio = basis_vol_ann / cme_vol_ann if cme_vol_ann != 0 else np.nan
    return cme_vol_ann, basis_vol_ann, ratio


def main():
    connect()
    raw = sanity_check_mcx_unit()

    # >>> SET THIS AFTER LOOKING AT THE SANITY CHECK OUTPUT ABOVE <
    # 100 if MSIc1 is per 10g, 10 if it's per 100g
    MCX_MULTIPLIER_TO_KG = 100  # <-- change if sanity check says otherwise

    df = build_dataset()
    df = compute_basis(df, MCX_MULTIPLIER_TO_KG)
    cme_vol_ann, basis_vol_ann, ratio = compute_vol_stats(df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)

    print("\n=== Summary ===")
    print(f"Date range: {df.index.min()} to {df.index.max()} ({len(df)} obs)")
    print(df[["MCX_USD_OZ", "CME_USD_OZ", "BASIS_USD", "BASIS_PCT"]].describe())
    print(f"\nAnnualized CME return volatility:   {cme_vol_ann:.4%}")
    print(f"Annualized basis volatility:         {basis_vol_ann:.4%}")
    print(f"Ratio (basis vol / CME vol):          {ratio:.3f}")
    print(f"\nSaved to: {OUTPUT_PATH}")

    test = ek.get_timeseries("SIc1", fields=["CLOSE"], start_date="2020-01-01", end_date=END_DATE, interval="daily")
    print(f"SIc1 (COMEX, control) rows: {len(test)}, range: {test.index.min()} to {test.index.max()}")

    test2 = ek.get_timeseries("MSIc1", fields=["CLOSE"], start_date="2020-01-01", end_date=END_DATE, interval="daily")
    print(f"MSIc1 (MCX) rows: {len(test2)}, range: {test2.index.min()} to {test2.index.max()}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)