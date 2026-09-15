"""
Clean the ecommerce dataset for Text-to-SQL use.

Usage:
  pip install -r requirements.txt
  python scripts/clean_ecommerce.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "archive" / "ecommerce_dataset_+1m.csv"
OUT_DIR = ROOT / "data" / "cleaned"
OUT_PARQUET = OUT_DIR / "ecommerce_cleaned.parquet"
OUT_CSV = OUT_DIR / "ecommerce_cleaned.csv"
PROFILE_PATH = OUT_DIR / "ecommerce_profile.txt"

# Yes/No style columns to normalize to boolean
BOOL_LIKE_COLS = [
    "is_weekend",
    "installment_plan",
    "coupon_used",
    "abandoned_cart_before",
    "support_ticket_created",
]

DATE_COLS = ["order_date", "account_creation_date"]

# Columns often empty for good reason (e.g. no return) — keep as nullable text
NULLABLE_TEXT = ["return_reason", "customer_feedback", "coupon_code"]

# Explicit labels for NULLABLE_TEXT columns, so "missing" is a real value
# instead of a blank cell. This keeps counts consistent across pandas,
# SQL, and Excel (COUNT/COUNTIF/COUNTA all treat blanks differently).
NULLABLE_TEXT_FILL = {
    "return_reason": "Nothing",
    "coupon_code": "No Coupon",
    "customer_feedback": "No Feedback",
}


def yes_no_to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "yes": True,
        "y": True,
        "true": True,
        "1": True,
        "no": False,
        "n": False,
        "false": False,
        "0": False,
    }
    # Preserve missing values: astype(str) turns pd.NA/None/NaN into
    # "<NA>"/"None"/"nan", which would miss the map and become float NaN.
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    mask = series.notna()
    if mask.any():
        normalized = series.loc[mask].astype(str).str.strip().str.lower()
        result.loc[mask] = normalized.map(mapping)
    return result


def profile(df: pd.DataFrame) -> str:
    n_rows = len(df)
    lines = [
        f"rows={n_rows:,}",
        f"cols={df.shape[1]}",
        f"duplicate_rows={df.duplicated().sum():,}",
        "",
        "--- null counts ---",
    ]
    nulls = df.isna().sum().sort_values(ascending=False)
    for col, n in nulls.items():
        if n > 0:
            pct = (n / n_rows) if n_rows else 0.0
            lines.append(f"{col}: {n:,} ({pct:.1%})")
    lines.append("")
    lines.append("--- dtypes ---")
    for col, dtype in df.dtypes.items():
        lines.append(f"{col}: {dtype}")
    return "\n".join(lines)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace on object columns (preserve missing values)
    for col in df.select_dtypes(include="object").columns:
        mask = df[col].notna()
        stripped = df.loc[mask, col].astype(str).str.strip()
        stripped = stripped.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
        df[col] = df[col].astype("object")
        df.loc[mask, col] = stripped
        # Ensure original nulls stay null (not the string "<NA>")
        df.loc[~mask, col] = pd.NA

    # Parse dates
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Normalize Yes/No -> boolean
    for col in BOOL_LIKE_COLS:
        if col in df.columns:
            df[col] = yes_no_to_bool(df[col])

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows")

    # Drop rows missing critical keys
    key_cols = [c for c in ["order_id", "customer_id", "product_id"] if c in df.columns]
    if key_cols:
        before = len(df)
        df = df.dropna(subset=key_cols)
        print(f"Dropped {before - len(df):,} rows missing keys {key_cols}")

    # Basic numeric sanity: non-negative money/qty where it makes sense
    for col in [
        "quantity",
        "unit_price_usd",
        "total_price_usd",
        "shipping_cost_usd",
        "age",
    ]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = pd.NA

    # Fill nullable text columns with explicit "not applicable" labels
    # instead of leaving them blank/NaN. Do this last so it only replaces
    # genuine "not applicable" gaps, not ones caused by earlier parsing.
    for col, fill_val in NULLABLE_TEXT_FILL.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_val)

    # Optional: drop derived date parts if you keep order_date (reduces redundancy for SQL)
    # Uncomment if you want a leaner table:
    # df = df.drop(columns=[c for c in ["order_year","order_month","order_day","order_hour","order_minute","order_second"] if c in df.columns])

    return df.reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_CSV} ...")
    # Use chunks if memory is tight; for ~1M rows full load is usually fine
    df = pd.read_csv(RAW_CSV, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols")

    PROFILE_PATH.write_text(profile(df), encoding="utf-8")
    print(f"Wrote profile -> {PROFILE_PATH}")

    cleaned = clean(df)
    PROFILE_PATH.with_name("ecommerce_profile_after.txt").write_text(
        profile(cleaned), encoding="utf-8"
    )

    # Prefer Parquet for speed + typed columns; also write CSV for easy peeking
    cleaned.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved {OUT_PARQUET}")

    cleaned.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

    print("Done.")


if __name__ == "__main__":
    main()
