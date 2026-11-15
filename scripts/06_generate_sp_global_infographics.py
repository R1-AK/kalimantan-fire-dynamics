"""
06_generate_sp_global_infographics.py
-------------------------------------
Generates two flagship editorial cartographic infographic posters inspired by
S&P Global Energy / Commodity Insights cartographic design:

1. figures/sp_infographic_1_megafire_2015_legacy.png (Portrait Editorial Spread)
   - Crisp white poster canvas, crimson masthead bar, bold headline & lead deck
   - Left analytical column: 4 frameless mini-graphics (archipelago map, teal bar
     chart, 25-yr drought fire cycle & post-fire trajectory bars, peat-dome inset)
   - Frameless hero Kalimantan map: muted mint-seafoam-slate-amber land-use zones,
     662 coal concession block outlines, shaded Borneo interior relief, coastal
     bathymetric shelf contour, proportional crimson 2015 burn-scar circles,
     curved transition flow arrows, and circular regional inset with arc text.

2. figures/sp_infographic_2_oct2026_fire_reality.png (Landscape Editorial Spread)
   - Crisp white poster canvas, crimson masthead bar, bold headline & lead deck
   - Full-width shaded-relief parchment-stone Borneo map with layered glacial-blue
     coastal/wetland basins, terracotta burn-scar & concession corridors, active
     Oct 2026 fire clusters, 5 big bold metric callouts with leader lines, curved
     monsoon/frontier arrows, on-map historical timeline (1997-2026), and circular
     Ex-MRP peat-epicenter inset with curved arc title.
   - Bottom 3-column S&P Global analytical chart strip with editorial micro-copy.
"""

import os
import glob
import tempfile
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mpl_kal_sp"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import pyproj

os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation, binary_erosion
from shapely.geometry import box, LineString

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(REPO_ROOT, "data", "processed")
REF_DIR = os.path.join(REPO_ROOT, "data", "reference")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
RAW_MCD_DIR = os.path.join(RAW_DIR, "mcd64a1_2001_2025")
FIG_DIR = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# S&P Global Editorial Design Tokens
SP_RED = "#D61F2C"
SP_CRIMSON = "#C9184A"
SP_CRIMSON_DARK = "#7A0C2E"
SP_TEAL_DARK = "#1B6B7D"
SP_TEAL_MID = "#3E8E9E"
SP_TEAL_LIGHT = "#8EC1CC"
SP_TEAL_PALE = "#CFE5DF"
SP_AMBER = "#E88D39"
SP_TERRACOTTA = "#C85A17"
SP_INK = "#111827"
SP_SUB = "#4B5563"
SP_MUTED = "#6B7280"
SP_HAIRLINE = "#D1D5DB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "text.color": SP_INK,
})


def halo(lw=2.2, col="#FFFFFF"):
    return [pe.withStroke(linewidth=lw, foreground=col)]


def save_poster(fig, filename):
    p1 = os.path.join(FIG_DIR, filename)
    fig.savefig(p1, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    print(f"Saved infographic: {p1}")


def draw_curved_arc_text(ax, text, radius=1.06, center_angle_deg=90.0, char_step_deg=2.25, fontsize=7.6, color="#1F2937"):
    """Render curved title text hugging the top outer arc of a unit circle in square axes."""
    n = len(text)
    total_span = (n - 1) * char_step_deg
    start_ang = center_angle_deg + 0.5 * total_span
    for i, ch in enumerate(text):
        ang_deg = start_ang - i * char_step_deg
        ang_rad = np.deg2rad(ang_deg)
        x = radius * np.cos(ang_rad)
        y = radius * np.sin(ang_rad)
        ax.text(
            x, y, ch,
            fontsize=fontsize, fontweight="bold", color=color,
            ha="center", va="bottom", rotation=ang_deg - 90.0,
            rotation_mode="anchor", zorder=30,
        )


def build_borneo_hillshade(kal, mys_borneo, surr_idn, sea_ctry, idn_prov, hob, klhk_ov, klhk_bounds):
    """Synthesize smooth physical Borneo & surrounding archipelago shaded relief (0.01 deg)."""
    xmin, xmax, ymin, ymax = 104.5, 123.5, -5.1, 5.6
    res = 0.01
    W = int(round((xmax - xmin) / res))
    H = int(round((ymax - ymin) / res))
    trans = from_origin(xmin, ymax, res, res)

    borneo_all = list(kal.geometry) + list(mys_borneo.geometry) + list(surr_idn.geometry) + list(sea_ctry.geometry) + list(idn_prov.geometry)
    land_mask = rasterize([(g, 1) for g in borneo_all], out_shape=(H, W), transform=trans, fill=0, dtype=np.uint8)
    hob_mask = rasterize([(g, 1) for g in hob.geometry], out_shape=(H, W), transform=trans, fill=0, dtype=np.uint8)

    # Major Borneo & Sulawesi mountain spines
    spines = [
        LineString([(111.3, -1.2), (112.6, -0.7), (113.8, 0.2), (114.8, 1.3), (115.6, 2.8), (116.5, 4.4), (116.55, 5.5)]),
        LineString([(113.8, 0.2), (115.0, 0.8), (116.2, 1.4)]),
        LineString([(115.45, -3.45), (115.75, -2.75), (115.95, -1.95)]),  # Meratus
        LineString([(117.1, 0.9), (117.85, 1.45)]),  # Sankulirang
        LineString([(109.4, 0.8), (110.4, 1.15), (112.0, 1.35)]),  # Kapuas Hulu
        LineString([(119.3, -3.2), (119.9, -1.8), (120.3, -0.8), (120.8, 0.6), (122.5, 0.7)]),  # Sulawesi spine
    ]
    spine_mask = rasterize([(g.buffer(0.14), 1) for g in spines], out_shape=(H, W), transform=trans, fill=0, dtype=np.uint8)

    dist_coast = distance_transform_edt(land_mask > 0) * res
    elev = np.power(np.clip(dist_coast / 2.2, 0, 1.4), 1.15) * 450.0
    elev += gaussian_filter(hob_mask.astype(np.float32), sigma=9.0) * 850.0
    elev += gaussian_filter(spine_mask.astype(np.float32), sigma=5.5) * 950.0

    # Sample KLHK overview to flatten peat swamps/rivers and add upland forest texture
    ys = np.linspace(ymax - 0.5 * res, ymin + 0.5 * res, H)
    xs = np.linspace(xmin + 0.5 * res, xmax - 0.5 * res, W)
    kh, kw = klhk_ov.shape
    k_cols = np.clip(((xs - klhk_bounds.left) / (klhk_bounds.right - klhk_bounds.left) * kw).astype(np.int32), 0, kw - 1)
    k_rows = np.clip(((klhk_bounds.top - ys) / (klhk_bounds.top - klhk_bounds.bottom) * kh).astype(np.int32), 0, kh - 1)
    k_sub = klhk_ov[np.ix_(k_rows, k_cols)]
    in_k_box = (
        (xs[None, :] >= klhk_bounds.left) & (xs[None, :] <= klhk_bounds.right)
        & (ys[:, None] >= klhk_bounds.bottom) & (ys[:, None] <= klhk_bounds.top)
    )
    lowland = in_k_box & np.isin(k_sub, [2005, 20051, 2004, 20041, 20071, 50011, 20093, 20094, 5001])
    upland = in_k_box & (k_sub == 2001)
    river = in_k_box & (k_sub == 5001)

    elev[upland] += 260.0
    elev *= (1.0 - 0.85 * gaussian_filter(lowland.astype(np.float32), sigma=2.5))
    elev -= gaussian_filter(river.astype(np.float32), sigma=1.2) * 140.0

    # Deterministic fine dendritic ridge/valley micro-relief so hillshade looks crisp like SRTM
    rng = np.random.default_rng(2026)
    n1 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=1.6)
    n2 = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=4.2)
    ridge_tex = (np.abs(n1) * 85.0 + n2 * 120.0) * np.clip(elev / 500.0, 0.04, 1.2)
    elev = np.clip(gaussian_filter(elev, sigma=1.4) + ridge_tex, 0.0, None)
    elev[land_mask == 0] = 0.0

    ls = LightSource(azdeg=315, altdeg=42)
    hs = ls.hillshade(elev, vert_exag=0.018, dx=1.0, dy=1.0)
    return hs, land_mask, (xmin, xmax, ymin, ymax)


def compute_2015_burn_bubbles(b2015, trans, cell_deg=0.14):
    """Aggregate 500 m 2015 burned pixels into proportional cluster bubbles (lon, lat, ha)."""
    rows, cols = np.where(b2015)
    lons = trans.c + (cols + 0.5) * trans.a
    lats = trans.f + (rows + 0.5) * trans.e
    # Geodesic area per 500 m pixel (~21.4 ha near equator)
    pix_ha = (abs(trans.a) * 111.32) * (abs(trans.e) * 110.57) * np.cos(np.deg2rad(lats)) * 100.0

    gx = np.floor(lons / cell_deg).astype(np.int32)
    gy = np.floor(lats / cell_deg).astype(np.int32)
    df_b = pd.DataFrame({"gx": gx, "gy": gy, "lon": lons, "lat": lats, "ha": pix_ha})
    agg = df_b.groupby(["gx", "gy"], as_index=False).agg(
        lon=("lon", "mean"), lat=("lat", "mean"), burn_ha=("ha", "sum")
    )
    return agg[agg["burn_ha"] >= 180.0].sort_values("burn_ha", ascending=False)


CITIES_SP = [
    ("Pontianak", 109.342, -0.026, True, (-0.16, -0.18)),
    ("Ketapang", 109.972, -1.850, False, (-0.15, -0.15)),
    ("Sampit", 112.950, -2.533, False, (-0.12, 0.16)),
    ("Palangka Raya", 113.921, -2.214, True, (0.14, 0.12)),
    ("Banjarmasin", 114.591, -3.319, True, (-0.18, -0.22)),
    ("Balikpapan", 116.831, -1.265, True, (0.18, -0.12)),
    ("IKN Nusantara", 116.703, -0.973, True, (-0.18, 0.13)),
    ("Samarinda", 117.154, -0.502, True, (0.18, -0.04)),
    ("Tanjung Selor", 117.365, 2.838, True, (-0.18, -0.14)),
    ("Tarakan", 117.633, 3.300, False, (0.18, 0.08)),
]


# ===========================================================================
# POSTER 1: S&P GLOBAL PORTRAIT SPREAD (Modeled on MSGBC Basin Showcase)
# ===========================================================================
def make_sp_poster_1(kal, mys_borneo, surr_idn, sea_ctry, idn_prov, hob, coal_blk, klhk_ov, klhk_bounds, hs, hs_mask, hs_ext):
    FIG_W, FIG_H = 15.0, 18.8
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")

    # ---------------- 1. Editorial Masthead (Top of Page) ----------------
    ax_hdr = fig.add_axes([0.038, 0.875, 0.924, 0.105])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")

    # Red S&P top bar & brand lockup
    ax_hdr.plot([0.0, 0.135], [0.95, 0.95], color=SP_RED, lw=3.2, solid_capstyle="butt")
    ax_hdr.text(0.0, 0.78, "Kalimantan Spatial", fontsize=13.5, fontweight="bold", color=SP_RED, va="top")
    ax_hdr.text(0.0, 0.61, "Landscape Intelligence", fontsize=12.5, fontweight="normal", color="#374151", va="top")

    # Big Editorial Headline & Narrative Deck
    ax_hdr.text(
        0.0, 0.38,
        "Kalimantan’s peat-fire trap: What replaced the 2015 mega-fire scars",
        fontsize=20.5, fontweight="bold", color=SP_INK, va="top",
    )
    deck = (
        "A decade after the 2015 Super El Niño burned 973,093 hectares across Indonesian Borneo, passive natural forest recovery remains below 5%.\n"
        "Over two-fifths (41.6%) of the burned footprint is locked in chronically re-burning peat-swamp scrub, while oil palm estates, industrial timber\n"
        "plantations, and smallholder rice paddies reshaped another 43.2% of the post-fire landscape—anchoring recurrent October haze episodes."
    )
    ax_hdr.text(0.0, 0.16, deck, fontsize=9.5, color=SP_SUB, va="top", linespacing=1.42)

    # ---------------- 2. Main Frameless Hero Map (Right/Center) ----------------
    ax_map = fig.add_axes([0.275, 0.045, 0.705, 0.815])
    ax_map.set_facecolor("#FFFFFF")
    xlim_m, ylim_m = (108.35, 119.30), (-4.75, 4.95)

    # Coastal shelf bathymetric ribbon (like Atlantic shelf in MSGBC spread)
    all_land_u = gpd.GeoSeries(list(kal.geometry) + list(mys_borneo.geometry) + list(surr_idn.geometry), crs=4326).union_all()
    shelf_outer = all_land_u.buffer(0.42)
    shelf_inner = all_land_u.buffer(0.18)
    gpd.GeoSeries([shelf_outer], crs=4326).plot(ax=ax_map, color="#EEF4F7", edgecolor="#B8CBD6", linewidth=0.55, zorder=1)
    gpd.GeoSeries([shelf_inner], crs=4326).plot(ax=ax_map, color="#E1ECF2", edgecolor="none", zorder=1)

    # Neighboring land (Sarawak, Sabah, Sulawesi/Java) in whisper-soft warm alabaster + taupe boundary ribbon
    surr_idn.plot(ax=ax_map, color="#F3F1EC", edgecolor="#C8C2B8", linewidth=0.5, zorder=2)
    mys_borneo.plot(ax=ax_map, color="#F2EFE9", edgecolor="#BEB7AB", linewidth=0.55, zorder=2)

    # Taupe transboundary / upland transition ribbon along Sarawak-Sabah border (MSGBC signature touch!)
    border_rib = mys_borneo.union_all().buffer(0.12).intersection(kal.union_all())
    gpd.GeoSeries([border_rib], crs=4326).plot(ax=ax_map, color="#DFCBB8", edgecolor="none", alpha=0.65, zorder=4)

    # S&P Global MSGBC-style muted mint / seafoam / slate-teal / golden-ochre land-use zone raster
    sp1_palette = {
        2001:  "#D2E7DF",  # Primary upland forest -> Soft celadon-mint
        2002:  "#DFEFE9",  # Secondary dryland forest -> Pale sage-mint
        2005:  "#9BC7C9",  # Primary swamp forest -> Muted seafoam
        20051: "#B2D5D6",  # Secondary swamp forest -> Pale mist-teal
        2004:  "#7FB3B8",  # Primary mangrove -> Coastal seafoam-teal
        20041: "#95C2C6",  # Secondary mangrove
        2006:  "#A8C6D4",  # Timber plantation (HTI) -> Dusty powder-slate
        2010:  "#B7D3DC",  # Estate crop (Oil palm) -> Soft glacier-blue-mint
        20093: "#EEB462",  # Rice paddy (Sawah) -> S&P warm amber-gold (Guinea-Bissau accent!)
        20091: "#F3CE93",  # Dryland farming -> Soft sand-ochre
        20092: "#E6DFCE",  # Agro-shrub mosaic -> Warm stone-cream
        2007:  "#DCE3D8",  # Dryland shrub -> Muted lichen
        3000:  "#E9D9B9",  # Savanna
        20071: "#5893A8",  # Swamp shrub (peat fire-trap!) -> Deep S&P slate-cerulean teal
        50011: "#437C94",  # Swamp wetland -> Deep basin teal
        2014:  "#E4C5A8",  # Open land
        20141: "#7B8C9E",  # Mining -> Cool slate
        2012:  "#D97B6C",  # Urban / Settlement
        20121: "#C96A6A",
        20122: "#E39B88",
        5001:  "#78AFC8",  # Major rivers & lakes -> Crisp hydro blue
        20094: "#9AC4D8",  # Coastal tambak ponds
        2500:  "#EDF2F0",
    }
    max_c = max(sp1_palette.keys()) + 1
    lut = np.zeros((max_c, 4), dtype=np.float32)
    for k, hx in sp1_palette.items():
        lut[k, :3] = mcolors.to_rgb(hx)
        lut[k, 3] = 1.0
    rgba_zones = lut[np.clip(klhk_ov, 0, max_c - 1)]
    rgba_zones[klhk_ov <= 0, 3] = 0.0

    kal.plot(ax=ax_map, color="#E4EFEA", edgecolor="none", zorder=2)
    ax_map.imshow(
        rgba_zones,
        extent=[klhk_bounds.left, klhk_bounds.right, klhk_bounds.bottom, klhk_bounds.top],
        origin="upper", interpolation="bilinear", zorder=3, aspect="equal",
    )

    # Subtle shaded topographic relief overlay across Borneo (gives 3D S&P Global physical feel)
    hs_rgba = np.zeros((*hs.shape, 4), dtype=np.float32)
    shadow = np.clip((0.62 - hs) * 1.35, 0.0, 0.32)
    shadow[hs_mask == 0] = 0.0
    hs_rgba[..., :3] = 0.18
    hs_rgba[..., 3] = shadow
    ax_map.imshow(hs_rgba, extent=[hs_ext[0], hs_ext[1], hs_ext[2], hs_ext[3]], origin="upper", interpolation="bilinear", zorder=4, aspect="equal")

    # Architectural Concession Block Geometries (662 Kalimantan Coal Permits — like MSGBC license blocks!)
    coal_blk.plot(ax=ax_map, facecolor="#4A6B7C", alpha=0.10, edgecolor="#4A6B7C", linewidth=0.35, zorder=5)
    hob.boundary.plot(ax=ax_map, color="#8C7A6B", linewidth=0.75, linestyle=(0, (4, 3)), alpha=0.75, zorder=5)

    # Crisp white + thin slate province borders
    kal.boundary.plot(ax=ax_map, color="#FFFFFF", linewidth=1.6, alpha=0.85, zorder=6)
    kal.boundary.plot(ax=ax_map, color="#4B5563", linewidth=0.65, alpha=0.85, zorder=7)
    mys_borneo.boundary.plot(ax=ax_map, color="#9CA3AF", linewidth=0.6, linestyle="--", zorder=6)

    # Proportional Crimson-Magenta 2015 Burn-Scar Cluster Bubbles (MSGBC signature visual!)
    ann_tif = os.path.join(RAW_MCD_DIR, "mcd64a1_annual_2001_2026_500m_mosaic.tif")
    with rasterio.open(ann_tif) as src_a:
        b2015 = src_a.read(2015 - 2001 + 1) > 0
        atrans = src_a.transform
        abounds = src_a.bounds
    prov_r = rasterize([(g, 1) for g in kal.geometry], out_shape=b2015.shape, transform=atrans, fill=0, dtype=np.uint8)
    b2015 = b2015 & (prov_r > 0)

    # Underlying fine 500m 2015 scar blush so exact scar shapes are also visible inside bubbles
    scar_fine = np.zeros((*b2015.shape, 4), dtype=np.float32)
    scar_fine[b2015] = mcolors.to_rgba(SP_CRIMSON, 0.35)
    ax_map.imshow(scar_fine, extent=[abounds.left, abounds.right, abounds.bottom, abounds.top], origin="upper", interpolation="nearest", zorder=7, aspect="equal")

    bub = compute_2015_burn_bubbles(b2015, atrans, cell_deg=0.15)
    sizes = np.clip(bub["burn_ha"].values / 85.0, 8.0, 420.0)
    ax_map.scatter(
        bub["lon"], bub["lat"], s=sizes,
        facecolor=mcolors.to_rgba(SP_CRIMSON, 0.62),
        edgecolor=SP_CRIMSON_DARK, linewidth=0.55, zorder=8,
    )

    # Subtle Province Labels & Neighbor Labels
    prov_lbls = {
        "West Kalimantan": (110.95, 0.35),
        "Central Kalimantan": (113.55, -0.95),
        "South Kalimantan": (115.42, -2.68),
        "East Kalimantan": (116.25, 0.65),
        "North Kalimantan": (116.20, 3.05),
    }
    for pname, (px, py) in prov_lbls.items():
        ax_map.text(px, py, pname, fontsize=8.3, color="#374151", fontweight="bold", ha="center", va="center", path_effects=halo(2.4, "#FFFFFF"), zorder=9)
    ax_map.text(113.15, 2.45, "Sarawak\n(Malaysia)", fontsize=8.0, color="#78716C", ha="center", va="center", path_effects=halo(2.0, "#F2EFE9"), zorder=9)
    ax_map.text(116.95, 4.55, "Sabah", fontsize=8.0, color="#78716C", ha="center", va="center", path_effects=halo(2.0, "#F2EFE9"), zorder=9)
    ax_map.text(114.65, 1.35, "Heart of Borneo\nUpland Core", fontsize=7.0, style="italic", color="#6B5E53", ha="center", va="center", path_effects=halo(2.0, "#E4EFEA"), zorder=9)

    # Cities (Sleek S&P dot + crisp halo label)
    for cname, clon, clat, is_cap, (dx, dy) in CITIES_SP:
        ax_map.scatter([clon], [clat], s=26 if is_cap else 16, facecolor="#FFFFFF", edgecolor="#111827", linewidth=1.0, zorder=10)
        ax_map.scatter([clon], [clat], s=6 if is_cap else 4, facecolor=SP_RED if cname == "IKN Nusantara" else "#111827", zorder=11)
        ha = "left" if dx >= 0 else "right"
        ax_map.text(
            clon + dx, clat + dy, cname,
            fontsize=7.3 if is_cap else 6.6,
            fontweight="bold" if (is_cap or cname == "IKN Nusantara") else "normal",
            color=SP_CRIMSON_DARK if cname == "IKN Nusantara" else "#1F2937",
            ha=ha, va="center", path_effects=halo(2.2, "#FFFFFF"), zorder=11,
        )

    # S&P MSGBC-Style Curved Export/Transition Arrows on SW Ocean (Karimata Strait / Java Sea)
    ax_map.text(109.55, -3.12, "2015 Burn-Scar Fate\n(967 kha tracked to 2025)", fontsize=7.6, fontweight="bold", color=SP_INK, ha="center", va="bottom", path_effects=halo(2.4, "#FFFFFF"), zorder=12)
    flow_specs = [
        ((110.55, -3.48), (108.48, -3.42), "Peat-scrub trap: 41.6% (402 kha)", 0.14, SP_TEAL_DARK, 2.7),
        ((110.55, -3.84), (108.48, -3.88), "Oil palm & HTI: 21.8% (210 kha)", 0.06, SP_TEAL_MID, 2.1),
        ((110.55, -4.20), (108.52, -4.34), "Rice & farming: 21.5% (207 kha)", -0.10, SP_AMBER, 2.0),
    ]
    for p_start, p_end, flbl, rad_v, fcol, flw in flow_specs:
        arr = mpatches.FancyArrowPatch(
            p_start, p_end, connectionstyle=f"arc3,rad={rad_v}",
            arrowstyle="-|>,head_length=6,head_width=4", color=fcol, lw=flw, zorder=11,
        )
        ax_map.add_patch(arr)
        mx = 0.50 * p_start[0] + 0.50 * p_end[0]
        my = 0.50 * p_start[1] + 0.50 * p_end[1]
        ax_map.text(mx, my + 0.07, flbl, fontsize=6.4, fontweight="bold", color=fcol, ha="center", va="bottom", path_effects=halo(2.2, "#FFFFFF"), zorder=12)
    ax_map.text(109.55, -4.60, "Natural forest recovery: only 4.3% (41 kha)", fontsize=6.4, style="italic", color=SP_MUTED, ha="center", zorder=12)

    # S&P Multi-Column Matrix Legend Floating on Top-Left of Map (over ocean/white space)
    leg_ax = fig.add_axes([0.285, 0.745, 0.435, 0.115])
    leg_ax.set_xlim(0, 1)
    leg_ax.set_ylim(0, 1)
    leg_ax.axis("off")
    leg_ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, facecolor="#FFFFFF", alpha=0.88, edgecolor="none"))

    # Col 1: Land-Cover Zones
    leg_ax.text(0.0, 0.90, "KLHK 2025 Land Zones", fontsize=7.6, fontweight="bold", color=SP_INK)
    c1_items = [
        ("#D2E7DF", "Upland primary/sec. forest"),
        ("#9BC7C9", "Peat swamp & mangrove forest"),
        ("#5893A8", "Degraded peat / swamp scrub"),
        ("#B7D3DC", "Oil palm & HTI plantations"),
        ("#EEB462", "Rice paddy & smallholder agri."),
    ]
    for i, (col_hx, lbl_t) in enumerate(c1_items):
        yy = 0.72 - i * 0.155
        leg_ax.add_patch(mpatches.Rectangle((0.0, yy - 0.045), 0.042, 0.095, facecolor=col_hx, edgecolor="#6B7280", lw=0.35))
        leg_ax.text(0.058, yy, lbl_t, fontsize=6.7, color=SP_SUB, va="center")

    # Col 2: Boundaries & Concessions
    leg_ax.text(0.50, 0.90, "Boundaries & Burn Clusters", fontsize=7.6, fontweight="bold", color=SP_INK)
    leg_ax.add_patch(mpatches.Rectangle((0.50, 0.675), 0.042, 0.095, facecolor="#4A6B7C", alpha=0.18, edgecolor="#4A6B7C", lw=0.6))
    leg_ax.text(0.558, 0.72, "Coal concession block (n=662)", fontsize=6.7, color=SP_SUB, va="center")
    leg_ax.plot([0.50, 0.542], [0.565, 0.565], color="#8C7A6B", lw=1.0, ls="--")
    leg_ax.text(0.558, 0.565, "Heart of Borneo upland boundary", fontsize=6.7, color=SP_SUB, va="center")

    # Proportional bubble scale
    leg_ax.scatter([0.512, 0.526, 0.544], [0.36, 0.36, 0.36], s=[18, 55, 130], facecolor=mcolors.to_rgba(SP_CRIMSON, 0.62), edgecolor=SP_CRIMSON_DARK, lw=0.55)
    leg_ax.text(0.572, 0.36, "2015 scar cluster (2–35 kha)", fontsize=6.7, color=SP_SUB, va="center")
    leg_ax.scatter([0.52], [0.16], s=24, facecolor="#FFFFFF", edgecolor="#111827", lw=1.0)
    leg_ax.scatter([0.52], [0.16], s=6, facecolor=SP_RED)
    leg_ax.text(0.558, 0.16, "Provincial capital / IKN Nusantara", fontsize=6.7, color=SP_SUB, va="center")

    # Minimalist Scalebar at Bottom-Right of Map
    lat_m = -0.5
    deg_100km = 100.0 / (111.32 * np.cos(np.deg2rad(lat_m)))
    sb_x, sb_y = 116.45, -4.42
    ax_map.plot([sb_x, sb_x + 2 * deg_100km], [sb_y, sb_y], color="#374151", lw=1.2, zorder=12)
    for tx_off, t_lbl in [(0, "0"), (deg_100km, "100"), (2 * deg_100km, "200 km")]:
        ax_map.plot([sb_x + tx_off, sb_x + tx_off], [sb_y - 0.06, sb_y + 0.06], color="#374151", lw=1.0, zorder=12)
        ax_map.text(sb_x + tx_off, sb_y + 0.10, t_lbl, fontsize=6.6, color="#374151", ha="center", va="bottom", zorder=12)

    ax_map.set_xlim(*xlim_m)
    ax_map.set_ylim(*ylim_m)
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    # ---------------- 3. Top-Right Circular Globe/Regional Inset with Curved Text ----------------
    circ_w = 0.215
    circ_h = circ_w * (FIG_W / FIG_H)  # Exact physical 1:1 square in inches!
    ax_circ = fig.add_axes([0.748, 0.692, circ_w, circ_h])
    ax_circ.set_xlim(-1.22, 1.22)
    ax_circ.set_ylim(-1.22, 1.22)
    ax_circ.set_aspect("equal")
    ax_circ.axis("off")

    clip_circle = mpatches.Circle((0, 0), 1.0, transform=ax_circ.transData)
    ocean_bg = mpatches.Circle((0, 0), 1.0, facecolor="#F3F7FA", edgecolor="none", zorder=1)
    ax_circ.add_patch(ocean_bg)

    # Transform lon/lat around Borneo (cx=112.2, cy=0.2, span=15.2 deg) into unit circle
    cx_g, cy_g, span_g = 112.2, 0.2, 15.2
    for _, r_c in sea_ctry.iterrows():
        g_scaled = gpd.GeoSeries([r_c.geometry]).translate(xoff=-cx_g, yoff=-cy_g).scale(xfact=1.0 / span_g, yfact=1.0 / span_g, origin=(0, 0))
        for poly in g_scaled.explode(index_parts=False):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            ext_xy = np.column_stack(poly.exterior.xy)
            p_patch = mpatches.Polygon(ext_xy, closed=True, facecolor="#E6E4DF", edgecolor="#B8B4AC", lw=0.4, zorder=2)
            p_patch.set_clip_path(clip_circle)
            ax_circ.add_patch(p_patch)

    # Highlight Kalimantan in soft teal & 2015 scar hotspots in crimson inside the circular globe
    for _, r_k in kal.iterrows():
        g_sc = gpd.GeoSeries([r_k.geometry]).translate(xoff=-cx_g, yoff=-cy_g).scale(xfact=1.0 / span_g, yfact=1.0 / span_g, origin=(0, 0))
        for poly in g_sc.explode(index_parts=False):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            p_patch = mpatches.Polygon(np.column_stack(poly.exterior.xy), closed=True, facecolor="#9CC7C9", edgecolor="#2B6777", lw=0.45, zorder=3)
            p_patch.set_clip_path(clip_circle)
            ax_circ.add_patch(p_patch)

    bx_c = (bub["lon"].values - cx_g) / span_g
    by_c = (bub["lat"].values - cy_g) / span_g
    in_c = (bx_c**2 + by_c**2) < 0.95
    sc_c = ax_circ.scatter(bx_c[in_c], by_c[in_c], s=np.clip(bub["burn_ha"].values[in_c] / 450.0, 2.0, 35.0),
                           facecolor=SP_CRIMSON, alpha=0.75, linewidths=0, zorder=4)
    sc_c.set_clip_path(clip_circle)

    for lbl_g, lx, ly in [("Malaysia", 103.0, 3.8), ("Sumatra", 102.2, -1.2), ("Java", 110.2, -7.2), ("Sulawesi", 120.5, -1.8), ("Kalimantan", 114.0, 0.5)]:
        gx_t, gy_t = (lx - cx_g) / span_g, (ly - cy_g) / span_g
        if gx_t**2 + gy_t**2 < 0.82:
            ax_circ.text(gx_t, gy_t, lbl_g, fontsize=5.8, fontweight="bold" if lbl_g == "Kalimantan" else "normal",
                         color="#1F2937" if lbl_g == "Kalimantan" else "#6B7280", ha="center", va="center", zorder=5)

    # Crisp dark ring + Curved arc title along top rim!
    ax_circ.add_patch(mpatches.Circle((0, 0), 1.0, fill=False, edgecolor="#374151", lw=0.95, zorder=20))
    draw_curved_arc_text(ax_circ, "Maritime Continent & Borneo fire belt", radius=1.05, center_angle_deg=90.0, char_step_deg=2.55, fontsize=7.2)

    # ---------------- 4. Left Sidebar Column: 4 Frameless S&P Global Mini-Graphics ----------------
    # Block 1: Archipelago Provincial Context Map (like Africa Country Map in MSGBC spread)
    ax_s1_txt = fig.add_axes([0.038, 0.785, 0.225, 0.068])
    ax_s1_txt.axis("off")
    ax_s1_txt.text(0, 0.92, "The geography of Borneo’s fire belt", fontsize=8.8, fontweight="bold", color=SP_INK, va="top")
    ax_s1_txt.text(0, 0.58, "Kalimantan spans 53.3 million ha across five Indonesian\nprovinces, holding 51% of Indonesia’s tropical peatlands\nand the core of Southeast Asia’s El Niño fire regime.",
                   fontsize=6.6, color=SP_SUB, va="top", linespacing=1.35)

    ax_s1_map = fig.add_axes([0.038, 0.655, 0.225, 0.128])
    ax_s1_map.set_facecolor("#FFFFFF")
    kal_names = set(kal["shapeName"].unique())
    idn_other = idn_prov[~idn_prov["shapeName"].isin(kal_names)]
    idn_other.plot(ax=ax_s1_map, color="#E5E2DC", edgecolor="#FFFFFF", lw=0.35)
    s1_pcols = {
        "Central Kalimantan": "#C95B50",
        "South Kalimantan": "#E88D39",
        "West Kalimantan": "#EAC47C",
        "East Kalimantan": "#5893A8",
        "North Kalimantan": "#9BC7C9",
    }
    for _, rp in kal.iterrows():
        gpd.GeoSeries([rp.geometry], crs=4326).plot(ax=ax_s1_map, color=s1_pcols.get(rp["shapeName"], SP_TEAL_MID), edgecolor="#FFFFFF", lw=0.45)
    ax_s1_map.set_xlim(94.8, 141.2)
    ax_s1_map.set_ylim(-11.2, 6.5)
    ax_s1_map.set_aspect("equal")
    ax_s1_map.axis("off")
    ax_s1_map.text(114.0, -5.6, "Kalimantan (53.3 Mha)", fontsize=6.4, fontweight="bold", color=SP_INK, ha="center")

    # Block 2: Horizontal Teal Bar Chart — 2025 Land Cover on 2015 Burn Scars
    df_fate = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_summary.csv"))
    ax_s2_txt = fig.add_axes([0.038, 0.582, 0.225, 0.062])
    ax_s2_txt.axis("off")
    ax_s2_txt.text(0, 0.95, "Where 2015 burn scars stand in 2025", fontsize=8.8, fontweight="bold", color=SP_INK, va="top")
    ax_s2_txt.text(0, 0.56, "Unrecovered swamp shrub and wetland scrub occupy\n41.6% of all 2015 scars, ahead of oil palm and rice.",
                   fontsize=6.6, color=SP_SUB, va="top", linespacing=1.35)

    ax_s2 = fig.add_axes([0.115, 0.435, 0.145, 0.142])
    ax_s2.set_facecolor("#FFFFFF")
    top_f = df_fate.head(8).iloc[::-1]
    short_names = {
        "Swamp shrub": "Swamp shrub",
        "Estate crop": "Oil palm / estate",
        "Rice field": "Rice paddy",
        "Swamp": "Peat swamp",
        "Shrub": "Dryland shrub",
        "Dryland farming mixed with shrub": "Agro-shrub",
        "Plantation forest": "Timber (HTI)",
        "Dryland farming": "Dryland agri.",
    }
    yf = np.arange(len(top_f))
    bar_c = [SP_CRIMSON if c in (20071, 50011) else SP_TEAL_DARK for c in top_f["code"]]
    ax_s2.barh(yf, top_f["pct_of_2015_burn"], height=0.62, color=bar_c)
    ax_s2.set_yticks(yf)
    ax_s2.set_yticklabels([short_names.get(s, s) for s in top_f["class"]], fontsize=6.6, color=SP_SUB)
    ax_s2.set_xlim(0, 42)
    for idx_b, val_b in enumerate(top_f["pct_of_2015_burn"]):
        ax_s2.text(val_b + 0.8, idx_b, f"{val_b:.1f}%", va="center", fontsize=6.4, fontweight="bold", color=SP_INK)
    ax_s2.xaxis.grid(True, linestyle=":", color=SP_HAIRLINE, lw=0.6)
    ax_s2.set_axisbelow(True)
    for sp in ["top", "right", "bottom"]:
        ax_s2.spines[sp].set_visible(False)
    ax_s2.spines["left"].set_color(SP_HAIRLINE)
    ax_s2.tick_params(axis="x", labelsize=6.0, colors=SP_MUTED, length=0)
    ax_s2.tick_params(axis="y", length=0)

    # Block 3: Two Stacked Ultra-Clean Column Charts (25-yr Burned Area + Oil Palm Surge)
    df_ann_p = pd.read_csv(os.path.join(PROC_DIR, "mcd64a1_kalimantan_annual_burned_by_province_2001_2026.csv"))
    df_ann_p = df_ann_p[df_ann_p["year"] != "2026_JanSep"].copy()
    df_ann_p["yr"] = df_ann_p["year"].astype(int)
    df_ann = df_ann_p.groupby("yr", as_index=False)["burned_area_ha"].sum()
    df_ann["kha"] = df_ann["burned_area_ha"] / 1e3

    ax_s3_txt = fig.add_axes([0.038, 0.358, 0.225, 0.058])
    ax_s3_txt.axis("off")
    ax_s3_txt.text(0, 0.95, "Drought fire spikes & plantation shift", fontsize=8.8, fontweight="bold", color=SP_INK, va="top")
    ax_s3_txt.text(0, 0.54, "El Niño episodes trigger ~1 Mha burn spikes (top), followed\nby steady oil palm expansion onto cleared scars (bottom).",
                   fontsize=6.6, color=SP_SUB, va="top", linespacing=1.35)

    ax_s3a = fig.add_axes([0.055, 0.268, 0.205, 0.082])
    ax_s3a.set_facecolor("#FFFFFF")
    c_bars = [SP_CRIMSON if y == 2015 else (SP_AMBER if y in (2002, 2004, 2006, 2009, 2014, 2019, 2023) else "#CBD5E1") for y in df_ann["yr"]]
    ax_s3a.bar(df_ann["yr"], df_ann["kha"], width=0.72, color=c_bars)
    ax_s3a.text(2015, 1030, "2015\n973 kha", fontsize=6.0, fontweight="bold", color=SP_CRIMSON, ha="center", va="bottom")
    ax_s3a.set_ylim(0, 1650)
    ax_s3a.set_xticks([2001, 2006, 2015, 2019, 2025])
    ax_s3a.tick_params(labelsize=6.0, colors=SP_MUTED, length=2)
    ax_s3a.set_ylabel("kha/yr", fontsize=6.2, color=SP_MUTED)
    ax_s3a.yaxis.grid(True, linestyle=":", color=SP_HAIRLINE, lw=0.6)
    for sp in ["top", "right"]:
        ax_s3a.spines[sp].set_visible(False)
    ax_s3a.spines["left"].set_visible(False)
    ax_s3a.spines["bottom"].set_color(SP_HAIRLINE)

    # Mini post-fire Oil Palm vs Peat Swamp Forest % on 2015 scars (2014 -> 2024)
    df_mb = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_mapbiomas_trajectory_2014_to_2024.csv"))
    piv_mb = df_mb.pivot(index="mb_code", columns="epoch", values="pct")
    eps = ["2014_pre_fire", "2016_post_fire_1yr", "2020_post_fire_5yr", "2024_post_fire_9yr"]
    op_vals = piv_mb.loc[35, eps].values
    pf_vals = piv_mb.loc[76, eps].values

    ax_s3b = fig.add_axes([0.055, 0.175, 0.205, 0.072])
    ax_s3b.set_facecolor("#FFFFFF")
    xe = np.arange(4)
    ax_s3b.bar(xe - 0.16, op_vals, width=0.30, color=SP_TEAL_DARK, label="Oil palm (4.5%→9.9%)")
    ax_s3b.bar(xe + 0.16, pf_vals, width=0.30, color=SP_AMBER, label="Peat forest (3.7%→2.7%)")
    ax_s3b.set_xticks(xe)
    ax_s3b.set_xticklabels(["2014", "2016", "2020", "2024"], fontsize=6.2, color=SP_MUTED)
    ax_s3b.set_ylabel("% of scar", fontsize=6.2, color=SP_MUTED)
    ax_s3b.set_ylim(0, 12.5)
    ax_s3b.legend(loc="upper left", frameon=False, fontsize=5.9)
    ax_s3b.yaxis.grid(True, linestyle=":", color=SP_HAIRLINE, lw=0.6)
    for sp in ["top", "right", "left"]:
        ax_s3b.spines[sp].set_visible(False)
    ax_s3b.spines["bottom"].set_color(SP_HAIRLINE)
    ax_s3b.tick_params(axis="y", labelsize=6.0, colors=SP_MUTED, length=0)

    # Block 4: Bottom-Left Chronic Peat-Dome Core Locator (like bottom-left grey-teal Africa map in MSGBC)
    ax_s4_txt = fig.add_axes([0.038, 0.118, 0.225, 0.042])
    ax_s4_txt.axis("off")
    ax_s4_txt.text(0, 0.95, "Chronic repeat-burn peatland core", fontsize=8.6, fontweight="bold", color=SP_INK, va="top")
    ax_s4_txt.text(0, 0.45, "78.3% of 2015 swamp-shrub scars burned >=2 times in 2001–25.", fontsize=6.4, color=SP_SUB, va="top")

    ax_s4_map = fig.add_axes([0.038, 0.025, 0.225, 0.092])
    ax_s4_map.set_facecolor("#FFFFFF")
    mys_borneo.plot(ax=ax_s4_map, color="#EAE8E3", edgecolor="#FFFFFF", lw=0.3)
    kal.plot(ax=ax_s4_map, color="#D8D6D0", edgecolor="#FFFFFF", lw=0.4)
    # Highlight southern peat dome corridor in S&P teal + crimson chronic core
    peat_box = box(113.1, -3.6, 115.3, -1.8)
    gpd.GeoSeries([kal.union_all().intersection(peat_box)], crs=4326).plot(ax=ax_s4_map, color=SP_TEAL_MID, alpha=0.55)
    ax_s4_map.scatter(bub.head(45)["lon"], bub.head(45)["lat"], s=8, color=SP_CRIMSON, alpha=0.8)
    ax_s4_map.set_xlim(108.5, 119.0)
    ax_s4_map.set_ylim(-4.5, 4.6)
    ax_s4_map.set_aspect("equal")
    ax_s4_map.axis("off")
    ax_s4_map.text(114.2, -4.15, "Ex-MRP Peat Dome Focus", fontsize=6.0, fontweight="bold", color=SP_TEAL_DARK, ha="center")

    # Source Footer
    fig.text(
        0.28, 0.014,
        "Sources: Ministry of Forestry KLHK Penutupan Lahan 2025 (28 m); NASA MODIS MCD64A1 v061 (500 m, 2001–2025); MapBiomas Indonesia C4 (2014–2024); ESDM Coal Permits.",
        fontsize=6.5, color=SP_MUTED,
    )

    save_poster(fig, "sp_infographic_1_megafire_2015_legacy.png")


# ===========================================================================
# POSTER 2: S&P GLOBAL LANDSCAPE SPREAD (Modeled on Caspian Oil & Gas Showcase)
# ===========================================================================
def make_sp_poster_2(kal, mys_borneo, surr_idn, sea_ctry, idn_prov, hob, coal_blk, klhk_ov, klhk_bounds, hs, hs_mask, hs_ext):
    FIG_W, FIG_H = 18.4, 14.8
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")

    # ---------------- 1. Top Editorial Masthead ----------------
    ax_hdr = fig.add_axes([0.036, 0.855, 0.928, 0.120])
    ax_hdr.set_xlim(0, 1)
    ax_hdr.set_ylim(0, 1)
    ax_hdr.axis("off")

    ax_hdr.plot([0.0, 0.115], [0.95, 0.95], color=SP_RED, lw=3.2, solid_capstyle="butt")
    ax_hdr.text(0.0, 0.78, "Kalimantan Fire Monitor", fontsize=13.5, fontweight="bold", color=SP_RED, va="top")
    ax_hdr.text(0.0, 0.60, "Near-Real-Time Spatial Insights", fontsize=12.5, fontweight="normal", color="#374151", va="top")

    ax_hdr.text(
        0.0, 0.36,
        "October 2026 Kalimantan burning: Adapting to chronic scar lock-in",
        fontsize=21.5, fontweight="bold", color=SP_INK, va="top",
    )
    deck2 = (
        "Satellite active-fire detections (VIIRS 375 m) in October 2026 confirm that degraded peat-swamp scrub burns at 4.42× its landscape share, while one-third (33.7%) of all\n"
        "active burning re-ignites historical 2001–2025 burn scars. Breaking Kalimantan’s seasonal haze cycle requires hydrological peat restoration and targeted dry-season enforcement."
    )
    ax_hdr.text(0.0, 0.12, deck2, fontsize=9.6, color=SP_SUB, va="top", linespacing=1.40)

    # ---------------- 2. Central Full-Width Shaded-Relief Caspian-Style Map ----------------
    ax_m = fig.add_axes([0.036, 0.252, 0.928, 0.590])
    # Layered Caspian-style glacial-cerulean ocean background spanning full 1.95:1 width!
    ax_m.set_facecolor("#D6E6EF")
    xlim_m, ylim_m = (104.80, 123.15), (-4.65, 4.75)

    all_land_u = gpd.GeoSeries(
        list(kal.geometry) + list(mys_borneo.geometry) + list(surr_idn.geometry) + list(sea_ctry.geometry) + list(idn_prov.geometry), crs=4326
    ).union_all()
    for b_deg, c_sea in [(0.55, "#E0EEF5"), (0.26, "#E9F3F8"), (0.09, "#F2F8FB")]:
        gpd.GeoSeries([all_land_u.buffer(b_deg)], crs=4326).plot(ax=ax_m, color=c_sea, edgecolor="none", zorder=1)

    # Warm Caspian parchment-stone base land (#EAE6DF) across surrounding Maritime Continent
    sea_ctry.plot(ax=ax_m, color="#E8E5DF", edgecolor="#C5C0B7", linewidth=0.45, zorder=2)
    idn_prov.plot(ax=ax_m, color="#E8E5DF", edgecolor="#CFCAC0", linewidth=0.40, zorder=2)
    mys_borneo.plot(ax=ax_m, color="#E8E5DF", edgecolor="#C5C0B7", linewidth=0.55, zorder=2)
    kal.plot(ax=ax_m, color="#ECE9E2", edgecolor="none", zorder=2)

    # Subtle hydrological wetland & forest-basin tinting (Caspian soft blue-sage wash on wetlands & rivers)
    casp_lut = np.zeros((50012, 4), dtype=np.float32)
    for code_w, col_w, al_w in [
        (2001,  "#D4DFD5", 0.55),  # Upland forest whisper-sage
        (2002,  "#DFE5DC", 0.45),
        (2005,  "#B4D2D6", 0.72),  # Peat swamp forest -> Caspian seafoam-blue
        (20051, "#C3DCE0", 0.72),
        (2004,  "#9EC5CE", 0.80),  # Mangrove -> Caspian coastal teal
        (20041, "#B0D0D8", 0.75),
        (20071, "#D8C2B8", 0.78),  # Degraded swamp shrub -> Warm terracotta-blush basin
        (50011, "#A8CAD6", 0.80),  # Swamp wetland -> Glacial wetland blue
        (20093, "#E6D5A8", 0.65),  # Rice paddy -> Soft parchment-gold
        (2010,  "#DFDDD4", 0.50),
        (5001,  "#6CA6C1", 0.95),  # Major rivers (Kapuas, Barito, Mahakam, Kayan) -> Crisp Caspian blue!
        (20094, "#8CB8CE", 0.85),
    ]:
        casp_lut[code_w, :3] = mcolors.to_rgb(col_w)
        casp_lut[code_w, 3] = al_w

    casp_rgba = casp_lut[np.clip(klhk_ov, 0, 50011)]
    casp_rgba[klhk_ov <= 0, 3] = 0.0
    ax_m.imshow(
        casp_rgba,
        extent=[klhk_bounds.left, klhk_bounds.right, klhk_bounds.bottom, klhk_bounds.top],
        origin="upper", interpolation="bilinear", zorder=3, aspect="equal",
    )

    # Rich Topographic Shaded Relief (Caspian signature grey-taupe mountain relief!)
    hs_rgba = np.zeros((*hs.shape, 4), dtype=np.float32)
    dark_sh = np.clip((0.66 - hs) * 1.55, 0.0, 0.42)
    dark_sh[hs_mask == 0] = 0.0
    hs_rgba[..., :3] = np.array(mcolors.to_rgb("#57534E"))
    hs_rgba[..., 3] = dark_sh
    ax_m.imshow(hs_rgba, extent=[hs_ext[0], hs_ext[1], hs_ext[2], hs_ext[3]], origin="upper", interpolation="bilinear", zorder=4, aspect="equal")

    # Crisp white + thin grey borders (exact Caspian country border styling!)
    kal.boundary.plot(ax=ax_m, color="#FFFFFF", linewidth=1.8, alpha=0.9, zorder=5)
    kal.boundary.plot(ax=ax_m, color="#78716C", linewidth=0.65, zorder=6)
    mys_borneo.boundary.plot(ax=ax_m, color="#FFFFFF", linewidth=1.4, zorder=5)
    mys_borneo.boundary.plot(ax=ax_m, color="#8A837C", linewidth=0.6, linestyle="--", zorder=6)

    # Historical 2001–2025 Burn Scars as Soft Terracotta-Sienna Corridors + Coal Concession Outlines
    ann_tif = os.path.join(RAW_MCD_DIR, "mcd64a1_annual_2001_2026_500m_mosaic.tif")
    with rasterio.open(ann_tif) as src_a:
        ever_b = src_a.read()[:25].sum(axis=0)
        atrans = src_a.transform
        abounds = src_a.bounds
    prov_r = rasterize([(g, 1) for g in kal.geometry], out_shape=ever_b.shape, transform=atrans, fill=0, dtype=np.uint8)
    ever_b[prov_r == 0] = 0

    scar_casp = np.zeros((*ever_b.shape, 4), dtype=np.float32)
    scar_casp[ever_b == 1] = mcolors.to_rgba("#D9824C", 0.38)
    scar_casp[ever_b >= 2] = mcolors.to_rgba("#B84A27", 0.58)
    ax_m.imshow(scar_casp, extent=[abounds.left, abounds.right, abounds.bottom, abounds.top], origin="upper", interpolation="nearest", zorder=6, aspect="equal")

    coal_blk.boundary.plot(ax=ax_m, color="#2B6777", linewidth=0.32, alpha=0.55, zorder=6)

    # October 2026 VIIRS 375m Active Fire Clusters
    pts = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_viirs_375m_hotspots.csv"))
    pts = pts[pts["klhk_code"] > 0].copy()
    acols = np.clip(np.floor((pts["lon"].values - atrans.c) / atrans.a).astype(np.int32), 0, ever_b.shape[1] - 1)
    arows = np.clip(np.floor((pts["lat"].values - atrans.f) / atrans.e).astype(np.int32), 0, ever_b.shape[0] - 1)
    pts["reburn"] = ever_b[arows, acols] > 0

    p_new = pts[~pts["reburn"]]
    p_reb = pts[pts["reburn"]]
    p_per = pts[pts["n_nom_hi"] >= 25]

    ax_m.scatter(p_new["lon"], p_new["lat"], s=1.4, c="#E67E22", alpha=0.42, linewidths=0, zorder=7)
    ax_m.scatter(p_reb["lon"], p_reb["lat"], s=2.2, c=SP_CRIMSON, alpha=0.68, linewidths=0, zorder=8)
    ax_m.scatter(p_per["lon"], p_per["lat"], s=20, facecolor="#1B6B7D", edgecolor="#FFFFFF", linewidth=0.45, marker="D", zorder=9)

    # Cities & White Province / Surrounding Region Watermarks (exact Caspian country & sea label look!)
    for pname, (px, py), fsz in [
        ("West Kalimantan", (110.95, 0.35), 8.8),
        ("Central Kalimantan", (113.45, -0.85), 8.8),
        ("South Kalimantan", (115.45, -2.65), 8.8),
        ("East Kalimantan", (116.15, 0.75), 8.8),
        ("North Kalimantan", (116.10, 3.05), 8.8),
        ("Sarawak (Malaysia)", (113.30, 2.45), 8.6),
        ("Sabah", (117.00, 4.45), 8.5),
        ("Sulawesi", (120.25, -1.85), 8.6),
        ("Bangka–Belitung", (106.85, -2.55), 7.6),
    ]:
        ax_m.text(px, py, pname, fontsize=fsz, fontweight="bold", color="#FFFFFF", ha="center", va="center",
                  path_effects=halo(2.2, "#8C857B"), zorder=9)

    for sname, (sx, sy) in [
        ("South China Sea", (107.6, 2.15)),
        ("Karimata Strait", (108.35, -1.95)),
        ("Java Sea", (112.5, -4.32)),
        ("Makassar Strait", (118.45, -1.85)),
    ]:
        ax_m.text(sx, sy, sname, fontsize=8.0, style="italic", color="#5B8296", ha="center", va="center", zorder=5)

    for cname, clon, clat, is_cap, (dx, dy) in CITIES_SP:
        ax_m.scatter([clon], [clat], s=24 if is_cap else 14, facecolor="#FFFFFF", edgecolor="#1F2937", lw=1.0, zorder=10)
        ax_m.scatter([clon], [clat], s=5, facecolor=SP_RED if cname == "IKN Nusantara" else "#1F2937", zorder=11)
        ax_m.text(clon + dx, clat + dy, cname, fontsize=7.0 if is_cap else 6.4,
                  fontweight="bold" if (is_cap or cname == "IKN Nusantara") else "normal",
                  color=SP_CRIMSON_DARK if cname == "IKN Nusantara" else "#1F2937",
                  ha="left" if dx >= 0 else "right", va="center", path_effects=halo(2.0, "#F2EFE9"), zorder=11)

    # 5 Big Bold S&P Callout Metrics with Crisp Leader Lines & Anchor Dots (zero collisions!)
    callouts = [
        (114.02, -2.88, 112.15, -3.72, "51.4%", "of Oct 2026 fire", "Central Kalimantan epicenter\n(118,914 active VIIRS cells)", "right"),
        (114.52, -3.15, 116.10, -4.02, "4.42×", "selection ratio", "Swamp shrub (25.1% of fire\non just 5.7% of land area)", "left"),
        (116.85, -0.35, 118.35, -0.35, "19.9%", "of Oct 2026 fire", "East Kalimantan agro-coal\n& Kutai-Mahakam corridor", "left"),
        (115.15, -2.95, 117.15, -2.85, "16.4%", "of Oct 2026 fire", "South Kalimantan Barito\nsedimentary & rice basin", "left"),
        (110.22, -1.95, 108.35, -0.95, "33.7%", "scar re-ignition", "One-third of Oct 2026 fires\nburn on 2001–25 MODIS scars", "right"),
    ]
    for ax_x, ax_y, tx_x, tx_y, b_num, b_unit, b_desc, ha_c in callouts:
        ax_m.scatter([ax_x], [ax_y], s=18, color="#111827", zorder=13)
        mid_x = 0.5 * (ax_x + tx_x)
        ax_m.plot([ax_x, mid_x, tx_x], [ax_y, tx_y, tx_y], color="#374151", lw=0.75, zorder=12)
        ax_m.text(tx_x, tx_y + 0.06, f"{b_num} ", fontsize=12.5, fontweight="bold", color=SP_INK,
                  ha=ha_c, va="bottom", path_effects=halo(2.6, "#EAF3F8"), zorder=14)
        ax_m.text(tx_x, tx_y - 0.04, f"{b_unit}\n{b_desc}", fontsize=6.5, color="#374151",
                  ha=ha_c, va="top", linespacing=1.25, path_effects=halo(2.2, "#EAF3F8"), zorder=14)

    # Curved Caspian-Style Monsoon & Transboundary Haze Arrows
    haze_arrows = [
        ((112.45, -3.25), (110.35, -2.55), "SE Monsoon Haze Plume", 0.16, SP_TERRACOTTA),
        ((109.75, -0.35), (107.65, 1.45), "Transboundary Smoke Drift", 0.15, SP_TERRACOTTA),
        ((117.45, 1.45), (119.25, 2.45), "Makassar Strait Outflow", -0.15, SP_TEAL_DARK),
    ]
    for p0, p1, albl, rad_a, acol in haze_arrows:
        ax_m.add_patch(mpatches.FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad_a}",
            arrowstyle="-|>,head_length=6.5,head_width=4.2", color=acol, lw=2.2, alpha=0.85, zorder=11,
        ))
        mx, my = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
        ax_m.text(mx, my + 0.14, albl, fontsize=6.4, fontweight="bold", color=acol, ha="center", path_effects=halo(2.2, "#EAF3F8"), zorder=12)

    # On-Map Horizontal Historical Fire-Regime Timeline at Lower-Left (placed cleanly on SW sea!)
    tl_x0, tl_x1, tl_y = 105.15, 109.65, -3.65
    ax_m.add_patch(mpatches.FancyBboxPatch(
        (tl_x0 - 0.18, tl_y - 0.82), (tl_x1 - tl_x0) + 0.36, 1.42,
        boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="none", alpha=0.82, zorder=11,
    ))
    ax_m.text(tl_x0, tl_y + 0.40, "Major Kalimantan Drought-Fire Episodes (2001–2026)", fontsize=7.1, fontweight="bold", color=SP_INK, zorder=12)
    ax_m.plot([tl_x0, tl_x1], [tl_y, tl_y], color="#374151", lw=0.85, zorder=12)
    milestones = [
        (0.00, "2002", "1.20 Mha\nEl Niño"),
        (0.20, "2006", "1.50 Mha\nEl Niño/IOD+"),
        (0.41, "2015", "973 kha\nSuper El Niño"),
        (0.62, "2019", "953 kha\nStrong IOD+"),
        (0.81, "2023", "464 kha\nEl Niño"),
        (1.00, "Oct 2026", "33.7% on\nold scars"),
    ]
    for frac, yr_s, desc_s in milestones:
        mx = tl_x0 + frac * (tl_x1 - tl_x0)
        is_hi = yr_s in ("2015", "Oct 2026")
        ax_m.scatter([mx], [tl_y], s=22 if is_hi else 14, color=SP_CRIMSON if is_hi else "#1F2937", zorder=13)
        ax_m.text(mx, tl_y + 0.10, yr_s, fontsize=6.5, fontweight="bold", color=SP_CRIMSON if is_hi else SP_INK, ha="center", va="bottom", zorder=13)
        ax_m.text(mx, tl_y - 0.12, desc_s, fontsize=5.7, color=SP_SUB, ha="center", va="top", linespacing=1.18, zorder=13)

    # Top-Left Floating Transparent Multi-Column Legend inside Map (over South China Sea)
    lx0, ly0 = 105.15, 4.48
    ax_m.add_patch(mpatches.FancyBboxPatch((lx0 - 0.14, ly0 - 1.50), 5.85, 1.72, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="none", alpha=0.84, zorder=11))
    ax_m.text(lx0, ly0, "Active Burning (Oct 2026)", fontsize=7.4, fontweight="bold", color=SP_INK, va="top", zorder=12)
    ax_m.scatter([lx0 + 0.08], [ly0 - 0.35], s=18, color=SP_CRIMSON, zorder=12)
    ax_m.text(lx0 + 0.24, ly0 - 0.35, "Oct 2026 re-burn on 2001–25 scar", fontsize=6.4, color=SP_SUB, va="center", zorder=12)
    ax_m.scatter([lx0 + 0.08], [ly0 - 0.68], s=14, color="#E67E22", zorder=12)
    ax_m.text(lx0 + 0.24, ly0 - 0.68, "Oct 2026 fire outside old scars", fontsize=6.4, color=SP_SUB, va="center", zorder=12)
    ax_m.scatter([lx0 + 0.08], [ly0 - 1.02], s=22, facecolor=SP_TEAL_DARK, edgecolor="#FFFFFF", lw=0.5, marker="D", zorder=12)
    ax_m.text(lx0 + 0.24, ly0 - 1.02, "Persistent thermal source (>=25 det.)", fontsize=6.4, color=SP_SUB, va="center", zorder=12)

    lx2 = lx0 + 2.95
    ax_m.text(lx2, ly0, "Landscape & Scars", fontsize=7.4, fontweight="bold", color=SP_INK, va="top", zorder=12)
    ax_m.add_patch(mpatches.Rectangle((lx2, ly0 - 0.42), 0.20, 0.14, facecolor="#B84A27", alpha=0.65, zorder=12))
    ax_m.text(lx2 + 0.28, ly0 - 0.35, "Historical repeat burn scar (2001–25)", fontsize=6.4, color=SP_SUB, va="center", zorder=12)
    ax_m.add_patch(mpatches.Rectangle((lx2, ly0 - 0.75), 0.20, 0.14, facecolor="#B4D2D6", alpha=0.85, zorder=12))
    ax_m.text(lx2 + 0.28, ly0 - 0.68, "Peat swamp & coastal wetland basin", fontsize=6.4, color=SP_SUB, va="center", zorder=12)
    ax_m.add_patch(mpatches.Rectangle((lx2, ly0 - 1.09), 0.20, 0.14, fill=False, edgecolor="#2B6777", lw=0.7, zorder=12))
    ax_m.text(lx2 + 0.28, ly0 - 1.02, "Coal mining concession block", fontsize=6.4, color=SP_SUB, va="center", zorder=12)

    ax_m.set_xlim(*xlim_m)
    ax_m.set_ylim(*ylim_m)
    ax_m.set_aspect("equal")
    ax_m.axis("off")

    # ---------------- 3. Top-Right Circular Zoom Inset (Ex-MRP Peat Epicenter) ----------------
    cw2 = 0.185
    ch2 = cw2 * (FIG_W / FIG_H)
    ax_c2 = fig.add_axes([0.792, 0.632, cw2, ch2])
    ax_c2.set_xlim(-1.22, 1.22)
    ax_c2.set_ylim(-1.22, 1.22)
    ax_c2.set_aspect("equal")
    ax_c2.axis("off")

    clip_c2 = mpatches.Circle((0, 0), 1.0, transform=ax_c2.transData)
    ax_c2.add_patch(mpatches.Circle((0, 0), 1.0, facecolor="#DCEAF2", edgecolor="none", zorder=1))

    # Map Palangka Raya - Pulang Pisau - Banjarmasin (cx=114.25, cy=-2.80, span=0.95 deg) into unit circle
    cx_z, cy_z, sp_z = 114.25, -2.80, 0.92
    for _, rk in kal.iterrows():
        gs_z = gpd.GeoSeries([rk.geometry]).translate(xoff=-cx_z, yoff=-cy_z).scale(xfact=1.0 / sp_z, yfact=1.0 / sp_z, origin=(0, 0))
        for poly in gs_z.explode(index_parts=False):
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            p_z = mpatches.Polygon(np.column_stack(poly.exterior.xy), closed=True, facecolor="#EAE6DF", edgecolor="#78716C", lw=0.6, zorder=2)
            p_z.set_clip_path(clip_c2)
            ax_c2.add_patch(p_z)

    # Render KLHK wetland/rice/scrub + scars inside the circle
    c0_k = max(0, int((cx_z - sp_z - klhk_bounds.left) / 0.002))
    c1_k = min(klhk_ov.shape[1], int((cx_z + sp_z - klhk_bounds.left) / 0.002))
    r0_k = max(0, int((klhk_bounds.top - (cy_z + sp_z)) / 0.002))
    r1_k = min(klhk_ov.shape[0], int((klhk_bounds.top - (cy_z - sp_z)) / 0.002))
    sub_casp = casp_rgba[r0_k:r1_k, c0_k:c1_k]
    ex0 = (klhk_bounds.left + c0_k * 0.002 - cx_z) / sp_z
    ex1 = (klhk_bounds.left + c1_k * 0.002 - cx_z) / sp_z
    ey1 = (klhk_bounds.top - r0_k * 0.002 - cy_z) / sp_z
    ey0 = (klhk_bounds.top - r1_k * 0.002 - cy_z) / sp_z
    im_z = ax_c2.imshow(sub_casp, extent=[ex0, ex1, ey0, ey1], origin="upper", interpolation="bilinear", zorder=3)
    im_z.set_clip_path(clip_c2)

    sub_pts = pts[(np.abs(pts["lon"] - cx_z) < sp_z) & (np.abs(pts["lat"] - cy_z) < sp_z)]
    sc_z = ax_c2.scatter(
        (sub_pts["lon"] - cx_z) / sp_z, (sub_pts["lat"] - cy_z) / sp_z,
        s=1.5, c=np.where(sub_pts["reburn"], SP_CRIMSON, "#E67E22"), alpha=0.55, linewidths=0, zorder=4,
    )
    sc_z.set_clip_path(clip_c2)

    for cn_z, lx_z, ly_z in [("Palangka Raya", 113.921, -2.214), ("Banjarmasin", 114.591, -3.319)]:
        gx_z, gy_z = (lx_z - cx_z) / sp_z, (ly_z - cy_z) / sp_z
        ax_c2.scatter([gx_z], [gy_z], s=18, facecolor="#FFFFFF", edgecolor="#111827", lw=0.9, zorder=6)
        ax_c2.text(gx_z, gy_z + 0.11, cn_z, fontsize=6.0, fontweight="bold", color=SP_INK, ha="center", path_effects=halo(1.8, "#FFFFFF"), zorder=7)

    ax_c2.add_patch(mpatches.Circle((0, 0), 1.0, fill=False, edgecolor="#374151", lw=0.95, zorder=20))
    draw_curved_arc_text(ax_c2, "Southern peat-dome epicenter", radius=1.05, center_angle_deg=90.0, char_step_deg=2.85, fontsize=7.4)

    # ---------------- 4. Bottom 3-Column S&P Global Editorial Chart Strip ----------------
    df_oct = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_fire_by_klhk2025_complete.csv"))
    df_oct_p = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_fire_by_province_summary.csv"))
    df_f15_p = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_by_province.csv"))
    df_fate = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_summary.csv"))

    # --- Bottom Left Chart: Fire Share vs Land Share ---
    ax_b1_t = fig.add_axes([0.036, 0.178, 0.285, 0.058])
    ax_b1_t.axis("off")
    ax_b1_t.text(0, 0.95, "Degraded peat scrub dominates active burning", fontsize=9.2, fontweight="bold", color=SP_INK, va="top")
    ax_b1_t.text(0, 0.52, "Swamp shrub (25.1%) and swamp wetland (5.9%) burn at 4.42× their baseline\nKalimantan land share, whereas primary forest accounts for just 0.3%.",
                 fontsize=6.8, color=SP_SUB, va="top", linespacing=1.35)

    ax_b1 = fig.add_axes([0.052, 0.048, 0.268, 0.122])
    ax_b1.set_facecolor("#FFFFFF")
    top6 = df_oct.head(7)
    lbl6 = ["Swamp\nshrub", "Sec. dry\nforest", "Agro-\nshrub", "Sec. peat\nforest", "Dryland\nshrub", "Oil palm\nestate", "Swamp\nwetland"]
    x6 = np.arange(len(top6))
    ax_b1.bar(x6 - 0.17, top6["pct_viirs_nom_hi"], width=0.32, color=SP_TEAL_DARK, label="% of Oct 2026 fire")
    ax_b1.bar(x6 + 0.17, top6["pct_kalimantan_land"], width=0.32, color=SP_TEAL_LIGHT, label="% of Kalimantan land")
    ax_b1.set_xticks(x6)
    ax_b1.set_xticklabels(lbl6, fontsize=6.4, color=SP_SUB)
    ax_b1.set_ylim(0, 30)
    ax_b1.set_ylabel("Share (%)", fontsize=6.8, color=SP_MUTED)
    ax_b1.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax_b1.yaxis.grid(True, linestyle=":", color=SP_HAIRLINE, lw=0.6)
    for sp in ["top", "right", "left"]:
        ax_b1.spines[sp].set_visible(False)
    ax_b1.spines["bottom"].set_color(SP_HAIRLINE)
    ax_b1.tick_params(axis="y", labelsize=6.4, colors=SP_MUTED, length=0)

    # --- Bottom Center Chart: Re-ignition & Chronic Repeat Rate by Land Cover ---
    ax_b2_t = fig.add_axes([0.358, 0.178, 0.285, 0.058])
    ax_b2_t.axis("off")
    ax_b2_t.text(0, 0.95, "Unconverted scars suffer chronic re-ignition", fontsize=9.2, fontweight="bold", color=SP_INK, va="top")
    ax_b2_t.text(0, 0.52, "Among 2015 burn scars, swamp wetlands (51.4%) and swamp shrub (42.4%)\nre-ignited in Oct 2026—over double the rate of oil palm estates (17.4%).",
                 fontsize=6.8, color=SP_SUB, va="top", linespacing=1.35)

    ax_b2 = fig.add_axes([0.375, 0.048, 0.268, 0.122])
    ax_b2.set_facecolor("#FFFFFF")
    sel_codes = [50011, 20071, 20093, 20051, 20091, 2006, 2010, 20141]
    sel_lbls = ["Swamp", "Swamp\nshrub", "Rice\npaddy", "Sec. peat\nforest", "Dryland\nagri.", "Timber\n(HTI)", "Oil palm", "Mining"]
    f_sub = df_fate.set_index("code").loc[sel_codes]
    x8 = np.arange(len(sel_codes))
    ax_b2.plot(x8, f_sub["pct_repeat_within_2015_scar"], "o-", color=SP_TEAL_DARK, lw=1.9, markersize=4.5, label="Repeat burn >=2 yrs (2001–25)")
    ax_b2.plot(x8, f_sub["pct_oct2026_reignited_within_2015_scar"], "s-", color=SP_TERRACOTTA, lw=1.9, markersize=4.5, label="Re-ignited in Oct 2026 (%)")
    ax_b2.set_xticks(x8)
    ax_b2.set_xticklabels(sel_lbls, fontsize=6.3, color=SP_SUB)
    ax_b2.set_ylim(0, 96)
    ax_b2.set_ylabel("% of 2015 scar", fontsize=6.8, color=SP_MUTED)
    ax_b2.legend(loc="upper right", frameon=False, fontsize=6.3)
    ax_b2.yaxis.grid(True, linestyle=":", color=SP_HAIRLINE, lw=0.6)
    for sp in ["top", "right", "left"]:
        ax_b2.spines[sp].set_visible(False)
    ax_b2.spines["bottom"].set_color(SP_HAIRLINE)
    ax_b2.tick_params(axis="y", labelsize=6.4, colors=SP_MUTED, length=0)

    # --- Bottom Right Chart: Multi-Episode Provincial Risk Concentration ---
    ax_b3_t = fig.add_axes([0.680, 0.178, 0.284, 0.058])
    ax_b3_t.axis("off")
    ax_b3_t.text(0, 0.95, "Central Kalimantan anchors regional fire risk", fontsize=9.2, fontweight="bold", color=SP_INK, va="top")
    ax_b3_t.text(0, 0.52, "Across major fire episodes (2015, 2019, and October 2026), Central Kalimantan\nconsistently generates 43%–54% of Kalimantan’s total burning footprint.",
                 fontsize=6.8, color=SP_SUB, va="top", linespacing=1.35)

    ax_b3 = fig.add_axes([0.696, 0.048, 0.268, 0.122])
    ax_b3.set_facecolor("#FFFFFF")
    p15 = df_f15_p.groupby("province")[["burn2015_ha", "burn2019_ha"]].sum()
    p15["p15"] = 100.0 * p15["burn2015_ha"] / p15["burn2015_ha"].sum()
    p15["p19"] = 100.0 * p15["burn2019_ha"] / p15["burn2019_ha"].sum()
    pm = p15.join(df_oct_p.set_index("province")[["pct_of_kal_burning"]]).sort_values("pct_of_kal_burning", ascending=False)
    xp = np.arange(len(pm))
    ax_b3.bar(xp - 0.22, pm["p15"], width=0.21, color=SP_TEAL_LIGHT, label="2015 Mega-Fire")
    ax_b3.bar(xp, pm["p19"], width=0.21, color=SP_TEAL_MID, label="2019 Drought")
    ax_b3.bar(xp + 0.22, pm["pct_of_kal_burning"], width=0.21, color=SP_CRIMSON, label="Oct 2026 Active")
    ax_b3.set_xticks(xp)
    ax_b3.set_xticklabels([p.replace(" Kalimantan", "\nKal.") for p in pm.index], fontsize=6.5, color=SP_SUB)
    ax_b3.set_ylim(0, 62)
    ax_b3.set_ylabel("% of total", fontsize=6.8, color=SP_MUTED)
    ax_b3.legend(loc="upper right", frameon=False, fontsize=6.3)
    ax_b3.yaxis.grid(True, linestyle=":", color=SP_HAIRLINE, lw=0.6)
    for sp in ["top", "right", "left"]:
        ax_b3.spines[sp].set_visible(False)
    ax_b3.spines["bottom"].set_color(SP_HAIRLINE)
    ax_b3.tick_params(axis="y", labelsize=6.4, colors=SP_MUTED, length=0)

    fig.text(
        0.036, 0.012,
        "Sources: NASA/LANCE S-NPP & NOAA-20 VIIRS 375 m Active Fire (Oct 1–31, 2026); MODIS MCD64A1 v061 (2001–2025); Ministry of Forestry KLHK Penutupan Lahan 2025 (28 m).",
        fontsize=6.6, color=SP_MUTED,
    )

    save_poster(fig, "sp_infographic_2_oct2026_fire_reality.png")


def main():
    kal = gpd.read_file(os.path.join(REF_DIR, "kalimantan_adm1.geojson"))
    mys_borneo = gpd.read_file(os.path.join(REF_DIR, "borneo_malaysia_adm1.geojson"))
    surr_idn = gpd.read_file(os.path.join(REF_DIR, "surrounding_indonesia_land.geojson"))
    sea_ctry = gpd.read_file(os.path.join(REF_DIR, "sea_countries.geojson"))
    idn_prov = gpd.read_file(os.path.join(REF_DIR, "idn_provinces_simpl.geojson"))
    hob = gpd.read_file(os.path.join(REF_DIR, "heart_of_borneo.geojson"))
    coal_blk = gpd.read_file(os.path.join(REF_DIR, "kalimantan_coal_blocks.geojson"))

    cache_tif = os.path.join(RAW_DIR, "klhk2025_kalimantan_002deg_cache.tif")
    with rasterio.open(cache_tif) as s:
        klhk_ov = s.read(1)
        klhk_bounds = s.bounds

    print("Synthesizing Borneo shaded topographic relief...")
    hs, hs_mask, hs_ext = build_borneo_hillshade(kal, mys_borneo, surr_idn, sea_ctry, idn_prov, hob, klhk_ov, klhk_bounds)

    print("Rendering S&P Global Poster 1 (MSGBC Portrait Style)...")
    make_sp_poster_1(kal, mys_borneo, surr_idn, sea_ctry, idn_prov, hob, coal_blk, klhk_ov, klhk_bounds, hs, hs_mask, hs_ext)

    print("Rendering S&P Global Poster 2 (Caspian Landscape Style)...")
    make_sp_poster_2(kal, mys_borneo, surr_idn, sea_ctry, idn_prov, hob, coal_blk, klhk_ov, klhk_bounds, hs, hs_mask, hs_ext)

    print("Both S&P Global editorial infographics completed successfully!")


if __name__ == "__main__":
    main()
