import requests
import pandas as pd
import time

BASE_URL = "https://www.mcxindia.com/market-data/bhavcopy/GetCommoditywiseBhavCopy"

EXPIRIES = [
    "05MAR2027", "04DEC2026", "04SEP2026", "03JUL2026", "05MAY2026",
    "05MAR2026", "05DEC2025", "05SEP2025", "04JUL2025", "05MAY2025",
    "05MAR2025", "05DEC2024", "05SEP2024", "05JUL2024", "03MAY2024",
    "05MAR2024", "05DEC2023", "05SEP2023", "05JUL2023", "05MAY2023",
    "03MAR2023", "05DEC2022", "05SEP2022", "05JUL2022", "05MAY2022",
    "04MAR2022", "03DEC2021", "03SEP2021", "05JUL2021", "05MAY2021",
    "05MAR2021", "04DEC2020", "04SEP2020", "03JUL2020", "05MAY2020",
    "05MAR2020",
]

FROM_DATE = "01/01/2020"  # DD/MM/YYYY
TO_DATE = "01/07/2026"
HEADERS = {"User-Agent": "Mozilla/5.0"}

RAW_OUTPUT_PATH = "data/mcx_silver_all_expiries.csv"
CONTINUOUS_OUTPUT_PATH = "data/mcx_silver_continuous.csv"


def fetch_expiry(expiry):
    params = {
        "InstrumentName": "FUTCOM",
        "Symbol": "SILVER",
        "Expiry": expiry,
        "fromDate": FROM_DATE,
        "toDate": TO_DATE,
    }
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("IsSuccess"):
        print(f"  WARNING: {expiry} -> IsSuccess=False: {payload.get('Message')}")
        return pd.DataFrame()
    return pd.DataFrame(payload.get("Data", []))


def download_all():
    frames = []
    for expiry in EXPIRIES:
        print(f"Fetching {expiry}...")
        df = fetch_expiry(expiry)
        print(f"  -> {len(df)} rows")
        frames.append(df)
        time.sleep(0.5)

    combined = pd.concat(frames, ignore_index=True)
    combined["Symbol"] = combined["Symbol"].str.strip()
    combined["Date"] = pd.to_datetime(combined["Date"], format="%m/%d/%Y")
    combined["ExpiryDate"] = pd.to_datetime(combined["ExpiryDate"], format="%d%b%Y")
    combined = combined.drop_duplicates(subset=["Date", "ExpiryDate"]).sort_values(["Date", "ExpiryDate"])

    combined.to_csv(RAW_OUTPUT_PATH, index=False)
    print(f"\nRaw data: {len(combined)} rows, {combined['Date'].min()} to {combined['Date'].max()}")
    print(f"Saved to {RAW_OUTPUT_PATH}")
    return combined


def build_continuous(df):
    """For each date, pick the expiry with the highest traded Volume that day."""
    idx = df.groupby("Date")["Volume"].idxmax()
    front_month = df.loc[idx].sort_values("Date").reset_index(drop=True)

    front_month["IS_ROLL_DAY"] = front_month["ExpiryDate"].ne(front_month["ExpiryDate"].shift(1))

    n_rolls = front_month["IS_ROLL_DAY"].sum()
    print(f"\nContinuous series: {len(front_month)} days, {front_month['Date'].min()} to {front_month['Date'].max()}")
    print(f"Roll events: {n_rolls}")

    out = front_month[["Date", "ExpiryDate", "Close", "Volume", "OpenInterest", "IS_ROLL_DAY"]]
    out.to_csv(CONTINUOUS_OUTPUT_PATH, index=False)
    print(f"Saved to {CONTINUOUS_OUTPUT_PATH}")
    return out


def main():
    raw = download_all()
    continuous = build_continuous(raw)
    print("\nSample:")
    print(continuous.head(10))


if __name__ == "__main__":
    main()