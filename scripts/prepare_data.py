"""
scripts/prepare_data.py
========================
CLI wrapper for AIQuant data preparation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiquant.data.preparer import ensure_data_prepared

def main():
    parser = argparse.ArgumentParser(description='Prepare BTCUSDT 1m data for AIQuant')
    parser.add_argument('--days', type=int, default=1825,
                        help='Days of history to include (default: 1825 = 5 years)')
    args = parser.parse_args()
    
    try:
        ensure_data_prepared(days=args.days, verbose=True)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
