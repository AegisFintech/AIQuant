"""
scripts/check_features.py
Diagnostic: run feature engineering on a small slice and report any issues.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from aiquant.features import build_full_feature_set

# Use the raw parquet
parquet = 'data/raw/BTCUSDT_1m_full.parquet'
print(f"Loading {parquet}...")
df_raw = pd.read_parquet(parquet).tail(5000)
print(f"Loaded {len(df_raw):,} bars, {len(df_raw.columns)} cols: {list(df_raw.columns)}")

print("\nRunning build_full_feature_set...")
df = build_full_feature_set(df_raw, verbose=True)

print(f"\nTotal cols after feature engineering: {len(df.columns)}")

# Check for duplicates
dups = df.columns[df.columns.duplicated(keep=False)].tolist()
if dups:
    print(f"\n*** DUPLICATE COLUMNS FOUND ({len(set(dups))} unique names) ***")
    for d in sorted(set(dups)):
        count = list(df.columns).count(d)
        print(f"  '{d}' appears {count} times")
else:
    print("\n✓ No duplicate columns")

# Check float32 cast works
float_cols = df.select_dtypes(include=[np.float64]).columns
print(f"\nFloat64 cols: {len(float_cols)}")
try:
    df[float_cols] = df[float_cols].astype(np.float32)
    print("✓ float32 cast succeeded")
except Exception as e:
    print(f"*** float32 cast FAILED: {e}")

print("\nDone.")
