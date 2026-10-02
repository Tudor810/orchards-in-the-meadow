import argparse
from datetime import datetime
from pathlib import Path
import time
from urllib.error import ContentTooShortError

import pandas as pd
from tqdm import tqdm

import breizhcrops as bzh


def load_dataset_with_retries(region: str, level: str, retries: int, retry_sleep_s: int):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return bzh.BreizhCrops(region, level=level)
        except ContentTooShortError as e:
            last_err = e
            print(f"Download incomplete for {region} (attempt {attempt}/{retries}). Retrying in {retry_sleep_s}s...")
            time.sleep(retry_sleep_s)
    raise last_err


def count_labels(dataset, num_classes: int = 9):
    counts = [0] * num_classes
    for i in tqdm(range(len(dataset)), desc="Counting", leave=False):
        _, y, _ = dataset[i]
        y = int(y)
        if 0 <= y < num_classes:
            counts[y] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Count label distribution in BreizhCrops regions.")
    parser.add_argument("--level", default="L2A", help="Dataset level (e.g. L1C or L2A).")
    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--retries", type=int, default=3, help="Retries for interrupted downloads.")
    parser.add_argument("--retry-sleep", type=int, default=20, help="Seconds to wait between retries.")
    parser.add_argument("--skip-failed", action="store_true", help="Skip regions that still fail after retries.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    id_to_crop = {
        0: "barley",
        1: "wheat",
        2: "rapeseed",
        3: "corn",
        4: "sunflower",
        5: "orchards",
        6: "nuts",
        7: "permanent meadows",
        8: "temporary meadows",
    }

    rows = {}
    for region in args.regions:
        print(f"Loading {region} level={args.level} ...")
        try:
            ds = load_dataset_with_retries(
                region=region,
                level=args.level,
                retries=args.retries,
                retry_sleep_s=args.retry_sleep,
            )
        except ContentTooShortError as e:
            if args.skip_failed:
                print(f"Skipping {region} after {args.retries} failed download attempts: {e}")
                continue
            raise
        counts = count_labels(ds, num_classes=9)
        rows[region] = {f"{i} {id_to_crop[i]}": counts[i] for i in range(9)}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "region"

    # Add totals row (sum across regions)
    df.loc["TOTAL"] = df.sum(axis=0)

    out_path = args.output.strip()
    if not out_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = str(Path("Results") / f"train_class_counts_{args.level}_{ts}.xlsx")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing Excel to: {out_path}")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="class_counts")

    print("\nCounts:")
    print(df)


if __name__ == "__main__":
    main()

