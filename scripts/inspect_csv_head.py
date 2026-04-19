import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=str)
    ap.add_argument("--nrows", type=int, default=3)
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--print-all-columns", action="store_true", default=False)
    args = ap.parse_args()

    p = Path(args.csv_path)
    df = pd.read_csv(p, nrows=args.nrows, low_memory=False)
    print("path:", str(p))
    print("shape(nrows_loaded, ncols_total):", (len(df), len(df.columns)))
    print("first_cols:", list(df.columns[: args.ncols]))
    if args.print_all_columns:
        print("all_cols:", list(df.columns))
    print(df.iloc[: args.nrows, : args.ncols].to_string(index=False))


if __name__ == "__main__":
    main()

