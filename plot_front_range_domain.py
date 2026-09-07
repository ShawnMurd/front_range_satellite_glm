#!/usr/bin/env python3
"""
plot_front_range_domain.py
==========================

Plot the FRONT_RANGE bounding box defined in ``download_glm_frontrange.py``
on a Cartopy map with rivers, county borders, and state borders.

The map uses Natural Earth features:
    * rivers        - 10m physical 'rivers_lake_centerlines'
    * county borders - 10m cultural 'admin_2_counties'
    * state borders - 10m cultural 'admin_1_states_provinces_lines'

Natural Earth shapefiles are downloaded automatically by Cartopy on first
use (internet required once; they are cached afterwards).

Examples
--------
# Save to the default PNG and open an interactive window
python plot_front_range_domain.py

# Save to a specific file, no interactive window
python plot_front_range_domain.py --out front_range_domain.png --no-show

Requires: cartopy, matplotlib
    conda install -c conda-forge cartopy matplotlib
"""

from __future__ import annotations

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from download_glm_frontrange import FRONT_RANGE

# Padding (degrees) added around the domain so the box doesn't touch the
# map edges.
PAD = 1.0


def build_map(ax, bbox: dict) -> None:
    """Draw base map + rivers, counties, states on a PlateCarree axes."""
    ax.set_extent(
        [
            bbox["lon_min"] - PAD,
            bbox["lon_max"] + PAD,
            bbox["lat_min"] - PAD,
            bbox["lat_max"] + PAD,
        ],
        crs=ccrs.PlateCarree(),
    )

    # Subtle land/ocean/lake background
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f5f2e8", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dcecf5", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="#dcecf5",
                   edgecolor="#7fa8c9", linewidth=0.4, zorder=1)

    # Rivers (10m detail for a regional-scale map)
    rivers = cfeature.NaturalEarthFeature(
        category="physical",
        name="rivers_lake_centerlines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(rivers, edgecolor="#4a90d9", linewidth=0.6, zorder=2)

    # County borders (thin, light) and state borders (thick, dark)
    counties = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_2_counties",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(counties, edgecolor="#b0b0b0", linewidth=0.4, zorder=3)

    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(states, edgecolor="#333333", linewidth=1.1, zorder=4)

    # National borders for context (domain straddles CO only, but cheap)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="#333333",
                   linewidth=1.1, zorder=4)

    # Gridlines with lat/lon labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888888",
                      linestyle="--", zorder=5)
    gl.top_labels = False
    gl.right_labels = False


def draw_domain_box(ax, bbox: dict) -> None:
    """Overlay the FRONT_RANGE bounding box and annotate its corners."""
    width = bbox["lon_max"] - bbox["lon_min"]
    height = bbox["lat_max"] - bbox["lat_min"]

    rect = Rectangle(
        (bbox["lon_min"], bbox["lat_min"]),
        width,
        height,
        transform=ccrs.PlateCarree(),
        facecolor="red",
        alpha=0.12,
        edgecolor="red",
        linewidth=2.0,
        zorder=6,
    )
    ax.add_patch(rect)

    # Label the box
    ax.text(
        bbox["lon_min"] + width / 2,
        bbox["lat_max"] + 0.08,
        "FRONT_RANGE",
        transform=ccrs.PlateCarree(),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="red",
        zorder=7,
    )

    # Corner coordinate annotations
    corners = [
        (bbox["lon_min"], bbox["lat_max"], "right", "bottom"),
        (bbox["lon_max"], bbox["lat_max"], "left", "bottom"),
        (bbox["lon_min"], bbox["lat_min"], "right", "top"),
        (bbox["lon_max"], bbox["lat_min"], "left", "top"),
    ]
    for lon, lat, ha, va in corners:
        ax.plot(lon, lat, marker="o", markersize=4, color="red",
                transform=ccrs.PlateCarree(), zorder=7)
        ax.text(
            lon, lat, f" {lat:.1f}, {lon:.1f} ",
            transform=ccrs.PlateCarree(),
            ha=ha, va=va, fontsize=7, color="red", zorder=7,
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Plot the FRONT_RANGE GLM domain with Cartopy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", default="front_range_domain.png",
                   help="Output image file (PNG/PDF/etc. by extension)")
    p.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open an interactive plot window")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    bbox = FRONT_RANGE

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        figsize=(8, 8), subplot_kw={"projection": proj}
    )

    build_map(ax, bbox)
    draw_domain_box(ax, bbox)

    ax.set_title(
        "GLM Front Range Domain\n"
        f"lat {bbox['lat_min']:.1f}..{bbox['lat_max']:.1f}, "
        f"lon {bbox['lon_min']:.1f}..{bbox['lon_max']:.1f}",
        fontsize=12,
    )

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.out}")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
