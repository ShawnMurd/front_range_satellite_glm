#!/usr/bin/env python3
"""
download_glm_frontrange.py
==========================

Download GOES Geostationary Lightning Mapper (GLM) Level-2 LCFA data from
NOAA's public AWS S3 archive and keep only lightning over the Colorado
Front Range.

How it works
------------
1. Lists GLM-L2-LCFA files on S3 for the requested date/hour window.
2. Downloads files one at a time (stops before exceeding ``--max-gb``).
3. Subsets each file to the Front Range bounding box, keeping the
   flash -> group -> event hierarchy consistent:
     * flashes whose centroid falls inside the box are kept;
     * all groups belonging to those flashes are kept;
     * all events belonging to those groups are kept.
4. Writes the subset as a compressed netCDF file and deletes the raw file
   (unless ``--keep-raw``), so the retained dataset is typically only
   ~2-5% of the downloaded volume.

Examples
--------
# Estimate sizes only (no downloads)
python download_glm_frontrange.py --start 2024-06-01 --end 2024-06-30 --dry-run

# June 2024, GOES-16, all hours, stop at 50 GB downloaded
python download_glm_frontrange.py --start 2024-06-01 --end 2024-06-30

# Afternoon/evening convection only (18-23 UTC), GOES-19
python download_glm_frontrange.py --start 2025-06-01 --end 2025-06-30 \
    --satellite 19 --hours 18-23

Notes
-----
* Satellite choice: GOES-16 (GOES-East) through early 2025; GOES-19 is
  GOES-East from ~April 2025 onward. GOES-18 (GOES-West) also covers
  Colorado well. Both East and West view the Front Range.
* Requires: boto3, numpy, xarray, netCDF4
      conda install -c conda-forge boto3 numpy xarray netcdf4
* Re-running the script skips subsets that already exist (resumable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUCKETS = {
    16: "noaa-goes16",
    17: "noaa-goes17",
    18: "noaa-goes18",
    19: "noaa-goes19",
}
PRODUCT = "GLM-L2-LCFA"  # Lightning Cluster-Filter Algorithm (events/groups/flashes)

# Colorado Front Range corridor (roughly Trinidad -> Fort Collins, foothills
# to just east of I-25). Adjust with --lat-min/--lat-max/--lon-min/--lon-max.
FRONT_RANGE = {
    "lat_min": 37.0,
    "lat_max": 41.0,
    "lon_min": -106.0,
    "lon_max": -104.0,
}

GB = 1e9  # bytes per "GB" for the download cap


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
def make_client():
    """Anonymous (unsigned) client for the public NOAA GOES buckets."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def iter_prefixes(start: dt.date, end: dt.date, hours):
    """Yield 'YYYY/DDD/HH' S3 prefixes for each day/hour in the window."""
    day = start
    while day <= end:
        doy = day.timetuple().tm_yday
        for hour in hours:
            yield f"{day.year}/{doy:03d}/{hour:02d}"
        day += dt.timedelta(days=1)


def list_objects(client, bucket: str, prefix: str):
    """Yield (key, size_bytes) for every GLM file under one hour prefix."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{PRODUCT}/{prefix}/"):
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------
def subset_to_bbox(src_path: str, dst_path: Path, bbox: dict) -> bool:
    """Subset one GLM LCFA file to the bounding box.

    Returns True if any flashes were kept (subset written to ``dst_path``),
    False if the file contains no lightning inside the box.
    """
    # decode_times=False keeps original packed integer encodings intact,
    # which makes the re-write faithful and compact.
    with xr.open_dataset(src_path, engine="netcdf4", decode_times=False) as ds:
        in_box = (
            (ds.flash_lat >= bbox["lat_min"])
            & (ds.flash_lat <= bbox["lat_max"])
            & (ds.flash_lon >= bbox["lon_min"])
            & (ds.flash_lon <= bbox["lon_max"])
        )

        # 1. flashes whose centroid is inside the box
        f_idx = np.flatnonzero(in_box.values)
        if f_idx.size == 0:
            return False
        flash_ids = ds.flash_id.values[f_idx]

        # 2. all groups belonging to those flashes (keeps flashes whole,
        #    even if a group sits just outside the box)
        g_idx = np.flatnonzero(np.isin(ds.group_parent_flash_id.values, flash_ids))
        group_ids = ds.group_id.values[g_idx]

        # 3. all events belonging to those groups
        e_idx = np.flatnonzero(np.isin(ds.event_parent_group_id.values, group_ids))

        sub = ds.isel(
            number_of_flashes=f_idx,
            number_of_groups=g_idx,
            number_of_events=e_idx,
        )

        # Avoid the classic "_FillValue present in both attrs and encoding"
        # netCDF write conflict.
        for name in sub.variables:
            var = sub[name]
            if "_FillValue" in var.attrs and "_FillValue" in var.encoding:
                del var.attrs["_FillValue"]

        # Deflate all array variables to keep the subset small on disk.
        encoding = {
            name: {"zlib": True, "complevel": 4}
            for name, var in sub.data_vars.items()
            if var.ndim > 0
        }

        sub.attrs["subset_note"] = (
            f"Subset to Front Range box: lat {bbox['lat_min']}..{bbox['lat_max']}, "
            f"lon {bbox['lon_min']}..{bbox['lon_max']} (flash-centroid based)"
        )
        sub.attrs["history"] = (
            f"{dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ} subset by "
            "download_glm_frontrange.py"
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_netcdf(dst_path, engine="netcdf4", encoding=encoding)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_hours(text: str | None):
    """Parse --hours: '18-23' (inclusive range) or '18,20,22'. None = all."""
    if text is None:
        return list(range(24))
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download GOES GLM L2 data and subset to the CO Front Range.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", required=True, help="Start date, YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="End date, YYYY-MM-DD (inclusive)")
    p.add_argument("--satellite", type=int, choices=sorted(BUCKETS), default=16,
                   help="GOES satellite number (16=East<=2024, 19=East>=2025, 18=West)")
    p.add_argument("--hours", default=None,
                   help="UTC hours to include, e.g. '18-23' or '18,20,22'. Default: all")
    p.add_argument("--outdir", default="glm_front_range", help="Output directory")
    p.add_argument("--max-gb", type=float, default=50.0,
                   help="Stop before cumulative downloads exceed this many GB")
    p.add_argument("--keep-raw", action="store_true",
                   help="Also keep the full (unsubsetted) files under <outdir>/raw")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process subsets that already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="Only report the total size of matching files; download nothing")
    p.add_argument("--lat-min", type=float, default=FRONT_RANGE["lat_min"])
    p.add_argument("--lat-max", type=float, default=FRONT_RANGE["lat_max"])
    p.add_argument("--lon-min", type=float, default=FRONT_RANGE["lon_min"])
    p.add_argument("--lon-max", type=float, default=FRONT_RANGE["lon_max"])
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    args = parse_args(argv)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    hours = parse_hours(args.hours)
    bbox = {
        "lat_min": args.lat_min, "lat_max": args.lat_max,
        "lon_min": args.lon_min, "lon_max": args.lon_max,
    }
    bucket = BUCKETS[args.satellite]
    outdir = Path(args.outdir)
    cap_bytes = args.max_gb * GB

    client = make_client()

    print(f"Bucket   : s3://{bucket}/{PRODUCT}")
    print(f"Window   : {start} .. {end}, hours {hours[0]:02d}-{hours[-1]:02d} UTC")
    print(f"BBox     : lat {bbox['lat_min']}..{bbox['lat_max']}, "
          f"lon {bbox['lon_min']}..{bbox['lon_max']}")
    print(f"Cap      : {args.max_gb:g} GB downloaded")
    if args.dry_run:
        print("Mode     : DRY RUN (no downloads)\n")

    total_bytes = 0
    n_files = 0
    n_kept = 0
    n_empty = 0
    n_skipped_existing = 0
    kept_bytes = 0

    for prefix in iter_prefixes(start, end, hours):
        for key, size in list_objects(client, bucket, prefix):
            n_files += 1

            if args.dry_run:
                total_bytes += size
                continue

            # Destination mirrors the S3 layout: <outdir>/YYYY/DDD/HH/<file>
            dst = outdir / Path(key).parent.relative_to(PRODUCT) / Path(key).name
            if dst.exists() and not args.overwrite:
                n_skipped_existing += 1
                continue

            # Enforce the download cap *before* fetching the file.
            if total_bytes + size > cap_bytes:
                print(f"\nReached download cap of {args.max_gb:g} GB "
                      f"({total_bytes / GB:.2f} GB used). Stopping.")
                _print_summary(n_files, n_kept, n_empty, n_skipped_existing,
                               total_bytes, kept_bytes)
                return 0

            # --- download (to temp unless --keep-raw) ---------------------
            if args.keep_raw:
                raw_path = outdir / "raw" / Path(key).parent.relative_to(PRODUCT) / Path(key).name
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(raw_path))
                src, tmp = str(raw_path), None
            else:
                fd, tmp = tempfile.mkstemp(suffix=".nc")
                os.close(fd)
                client.download_file(bucket, key, tmp)
                src = tmp
            total_bytes += size

            # --- subset ----------------------------------------------------
            try:
                kept = subset_to_bbox(src, dst, bbox)
            except Exception as exc:  # corrupt file, read error, etc.
                print(f"  ! failed {Path(key).name}: {exc}")
                kept = False
            finally:
                if tmp is not None:
                    Path(tmp).unlink(missing_ok=True)

            if kept:
                n_kept += 1
                kept_bytes += dst.stat().st_size
            else:
                n_empty += 1

            if n_kept % 25 == 0 and n_kept > 0:
                print(f"  ... {total_bytes / GB:6.2f} GB downloaded, "
                      f"{n_kept} files with Front Range lightning "
                      f"({kept_bytes / GB:.3f} GB kept)")

        print(f"hour {prefix}: running total {total_bytes / GB:.2f} GB, "
              f"{n_kept} kept, {n_empty} empty")

    _print_summary(n_files, n_kept, n_empty, n_skipped_existing,
                   total_bytes, kept_bytes, dry_run=args.dry_run)
    return 0


def _print_summary(n_files, n_kept, n_empty, n_skipped, total_bytes, kept_bytes,
                   dry_run=False):
    print("\n" + "=" * 60)
    if dry_run:
        print(f"Matching files          : {n_files}")
        print(f"Total download size     : {total_bytes / GB:.2f} GB")
        print(f"Estimated kept subset   : ~{total_bytes / GB * 0.03:.1f} GB "
              "(typical subset is ~2-5% of raw)")
    else:
        print(f"Files examined          : {n_files}")
        print(f"Already on disk (skip)  : {n_skipped}")
        print(f"Files with FR lightning : {n_kept}")
        print(f"Files empty over FR     : {n_empty}")
        print(f"Downloaded volume       : {total_bytes / GB:.2f} GB")
        print(f"Retained subset volume  : {kept_bytes / GB:.3f} GB")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
