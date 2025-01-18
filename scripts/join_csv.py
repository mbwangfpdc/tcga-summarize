"""Join any number of CSV files on a given column"""
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str, nargs="+", help="Files you want to join")
    parser.add_argument("--suffixes", type=str, nargs="+", help="Column suffix for the non-index columns. Must be same length as --files argument - suffixes[i] is the suffix for columns from files[i]")
    parser.add_argument("--index_col", help="Index column which will also be joined on")
    parser.add_argument("--out", type=str, default="out.csv", help="File to output csv to")
    return parser.parse_args()

args = parse_args()
if len(args.files) != len(args.suffixes):
    print("ERROR: --files and --suffixes must be equal length")
    exit(1)
if len(args.files) == 0:
    print("No files provided, doing nothing")
    exit(0)
base_df = None
for filename, suffix in zip(args.files, args.suffixes):
    df = pd.read_csv(filename, index_col=args.index_col)
    df = df.add_prefix(prefix=suffix, axis='columns')
    if base_df is None:
        base_df = df
    else:
        base_df = base_df.join(df)

base_df.to_csv(args.out, sep="^")
