"""
05_generate_publication_figures.py
----------------------------------
Generates four cohesive, soft urban-atlas style publication figures (300 DPI):
  - figures/fig4_kalimantan_soft_landcover_fire_atlas.png  (Hero Multi-Scale Soft Pastel Land-Use & Fire Atlas)
  - figures/fig1_kalimantan_25yr_fire_history_and_scars.png
  - figures/fig2_2015_megafire_postfire_fate_2025.png
  - figures/fig3_oct2026_active_burning_and_fire_memory.png
"""
import glob
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.environ.get("TEMP", "."), "mpl_cache_kal_fire"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import pyproj
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
os.environ["PROJ_DATA"] = pyproj.datadir.get_data_dir()

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation, binary_erosion
from shapely.geometry import box

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(REPO_DIR, "data", "raw")
RAW_MCD_DIR = os.path.join(RAW_DIR, "mcd64a1_2001_2025")
RAW_VIIRS_DIR = os.path.join(RAW_DIR, "viirs_oct2026")
REF_DIR = os.path.join(REPO_DIR, "data", "reference")
PROC_DIR = os.path.join(REPO_DIR, "data", "processed")
FIG_DIR = os.path.join(REPO_DIR, "figures")
KLHK_TILES_DIR = os.environ.get(
    "KLHK_PL2025_TILES_DIR",
    os.path.join(REPO_DIR, "data", "raw", "klhk_pl2025_national_tiles"),
)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Curated Editorial / Soft Urban-Regional Planning Design System
# ---------------------------------------------------------------------------
CANVAS_BG = "#F6F4EE"      # Warm gallery alabaster paper
CARD_BG = "#FCFBF8"        # Crisp warm ivory card
SEA_BG = "#D7E6EF"         # Soft Nordic powder-cerulean sea
COAST_GLOW_1 = "#E0EDF5"
COAST_GLOW_2 = "#E9F3F8"
COAST_GLOW_3 = "#F2F8FB"
NEIGHBOR_LAND = "#EAE6DD"  # Muted warm stone-linen for Sarawak/Sabah/Sulawesi/Java
NEIGHBOR_EDGE = "#C2BCB0"
PROV_BORDER = "#6E6A63"
INK_DARK = "#26292E"
INK_MUTED = "#5A6069"

# 14-tone Soft Pastel Land-Cover Palette (KLHK Penutupan Lahan 2025)
KLHK_SOFT_PALETTE = {
    2001:  "#9EC5A4",  # Primary dryland forest -> Soft sage-eucalyptus
    2002:  "#C5DEC2",  # Secondary dryland forest -> Soft celadon-mint
    2005:  "#8CC0BA",  # Primary swamp forest -> Soft seafoam-jade
    20051: "#A7D2CB",  # Secondary swamp forest -> Mist seafoam
    2004:  "#78AFB2",  # Primary mangrove -> Dusty teal-aqua
    20041: "#93C2C4",  # Secondary mangrove -> Soft coastal aqua
    2006:  "#B7C7E2",  # Plantation forest (HTI) -> Soft periwinkle-blue
    2010:  "#D5C2E5",  # Estate crop (Oil palm) -> Soft pastel lavender-lilac
    20093: "#F1E394",  # Rice field (Sawah) -> Soft buttercream yellow
    20091: "#F5D9B4",  # Dryland farming -> Warm peach-cream
    20092: "#F3DFC4",  # Dryland farming + shrub -> Soft almond-apricot
    2007:  "#EEC6A5",  # Shrub -> Soft terracotta-apricot
    3000:  "#EBD0A7",  # Savanna / grassland -> Warm sand-ochre
    20071: "#E7BFC7",  # Swamp shrub -> Soft dusty rose-blush (peat scrub)
    50011: "#D8BDC8",  # Swamp -> Soft mauve-mist wetland
    2014:  "#E4B39C",  # Open land -> Soft clay-blush
    20141: "#A794B8",  # Mining -> Soft dusty slate-plum
    2012:  "#DF7876",  # Settlement / Urban -> Architectural warm coral-rose
    20121: "#C96A7B",  # Airport / port -> Dusty crimson-rose
    20122: "#E69896",  # Transmigration -> Soft salmon-coral
    5001:  "#92C2DE",  # Water body (rivers/lakes) -> Soft sky-cerulean
    20094: "#ABD2E6",  # Fish pond (Tambak) -> Pale coastal blue
    2500:  "#F2EFE9",  # Cloud -> Warm ivory
}

CITIES = [
    ("Pontianak", 109.342, -0.026, True, (0.14, -0.16)),
    ("Ketapang", 109.972, -1.850, False, (-0.14, -0.18)),
    ("Sampit", 112.950, -2.533, False, (-0.12, 0.16)),
    ("Palangka Raya", 113.921, -2.214, True, (0.0, 0.20)),
    ("Banjarmasin", 114.591, -3.319, True, (-0.22, -0.22)),
    ("Balikpapan", 116.831, -1.265, True, (0.20, -0.14)),
    ("IKN Nusantara", 116.703, -0.973, True, (-0.25, 0.14)),
    ("Samarinda", 117.154, -0.502, True, (0.22, -0.05)),
    ("Tanjung Selor", 117.365, 2.838, True, (-0.22, -0.16)),
    ("Tarakan", 117.633, 3.300, False, (0.20, 0.10)),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#CCC7BC",
    "axes.linewidth": 0.8,
    "text.color": INK_DARK,
    "axes.labelcolor": INK_DARK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


def text_halo(lw=2.4, color="#FAF8F3"):
    return [pe.withStroke(linewidth=lw, foreground=color)]


def build_lut_rgb(desaturate=0.0):
    """Build fast Lookup Table (max_code+1, 4) uint8 RGBA for KLHK codes."""
    max_code = max(KLHK_SOFT_PALETTE.keys()) + 1
    lut = np.zeros((max_code, 4), dtype=np.float32)
    bg_rgb = np.array(mcolors.to_rgb(CARD_BG), dtype=np.float32)
    for code, hex_c in KLHK_SOFT_PALETTE.items():
        rgb = np.array(mcolors.to_rgb(hex_c), dtype=np.float32)
        if desaturate > 0:
            rgb = (1.0 - desaturate) * rgb + desaturate * bg_rgb
        lut[code, :3] = rgb
        lut[code, 3] = 1.0
    return lut


def load_or_build_klhk_overview(kal):
    """Build or load cached 0.002-deg (~220 m) modal KLHK 2025 raster over Kalimantan."""
    cache_tif = os.path.join(RAW_DIR, "klhk2025_kalimantan_002deg_cache.tif")
    if os.path.exists(cache_tif):
        with rasterio.open(cache_tif) as s:
            return s.read(1), s.bounds, s.transform

    print("Building cached 220 m Kalimantan KLHK 2025 overview mosaic...")
    kal_union = kal.union_all()
    tiles = sorted(glob.glob(os.path.join(KLHK_TILES_DIR, "*.tif")))
    kal_tiles = []
    for fn in tiles:
        with rasterio.open(fn) as s:
            if box(*s.bounds).intersects(kal_union):
                kal_tiles.append((fn, s.bounds))

    minx = min(b.left for _, b in kal_tiles)
    miny = min(b.bottom for _, b in kal_tiles)
    maxx = max(b.right for _, b in kal_tiles)
    maxy = max(b.top for _, b in kal_tiles)
    out_res = 0.002  # 8x downsample of 0.00025 deg -> exact 512x512 per 1.024-deg tile!
    W = int(round((maxx - minx) / out_res))
    H = int(round((maxy - miny) / out_res))
    mosaic_arr = np.zeros((H, W), dtype=np.int32)
    out_trans = from_origin(minx, maxy, out_res, out_res)

    for fn, b in kal_tiles:
        with rasterio.open(fn) as s:
            sub = s.read(1, out_shape=(512, 512), resampling=Resampling.mode)
        c0 = int(round((b.left - minx) / out_res))
        r0 = int(round((maxy - b.top) / out_res))
        mosaic_arr[r0:r0 + 512, c0:c0 + 512] = sub

    in_kal = rasterize([(kal_union, 1)], out_shape=(H, W), transform=out_trans, fill=0, dtype=np.uint8)
    mosaic_arr[in_kal == 0] = 0

    with rasterio.open(
        cache_tif, "w", driver="GTiff", width=W, height=H, count=1, dtype="int32",
        crs="EPSG:4326", transform=out_trans, nodata=0, compress="deflate",
    ) as dst:
        dst.write(mosaic_arr, 1)

    with rasterio.open(cache_tif) as s:
        return s.read(1), s.bounds, s.transform


def extract_klhk_highres_window(xmin, ymin, xmax, ymax, out_res=0.0005):
    """Extract ~55 m modal KLHK 2025 cutout for high-detail urban-regional zoom windows."""
    W = int(round((xmax - xmin) / out_res))
    H = int(round((ymax - ymin) / out_res))
    out_arr = np.zeros((H, W), dtype=np.int32)
    w_box = box(xmin, ymin, xmax, ymax)

    for fn in sorted(glob.glob(os.path.join(KLHK_TILES_DIR, "*.tif"))):
        with rasterio.open(fn) as s:
            tb = box(*s.bounds)
            if not tb.intersects(w_box):
                continue
            inter = tb.intersection(w_box)
            ix0, iy0, ix1, iy1 = inter.bounds
            win = from_bounds(ix0, iy0, ix1, iy1, s.transform)
            sub_h = max(1, int(round((iy1 - iy0) / out_res)))
            sub_w = max(1, int(round((ix1 - ix0) / out_res)))
            data = s.read(1, window=win, out_shape=(sub_h, sub_w), resampling=Resampling.mode)
            c0 = max(0, int(round((ix0 - xmin) / out_res)))
            r0 = max(0, int(round((ymax - iy1) / out_res)))
            h_fit = min(sub_h, H - r0)
            w_fit = min(sub_w, W - c0)
            if h_fit > 0 and w_fit > 0:
                valid = data[:h_fit, :w_fit] > 0
                out_arr[r0:r0 + h_fit, c0:c0 + w_fit][valid] = data[:h_fit, :w_fit][valid]
    return out_arr, from_origin(xmin, ymax, out_res, out_res)


def render_klhk_rgba(arr, desaturate=0.0):
    lut = build_lut_rgb(desaturate=desaturate)
    clipped = np.clip(arr, 0, len(lut) - 1)
    rgba = lut[clipped]
    rgba[arr <= 0, 3] = 0.0
    return rgba


def draw_soft_basemap(ax, kal, mys_borneo, surr_idn, klhk_rgba, klhk_bounds, xlim, ylim, show_neighbors_lbl=True):
    """Render soft powder-blue sea, 3-ring bathymetric coastal glow, neighbor stone-linen land, and pastel KLHK."""
    ax.set_facecolor(SEA_BG)

    # Combine landmass for coastal bathymetric glow
    all_land = gpd.GeoSeries(
        list(kal.geometry) + list(mys_borneo.geometry) + list(surr_idn.geometry), crs=4326
    ).union_all()
    for buf_deg, col_g in [(0.34, COAST_GLOW_1), (0.18, COAST_GLOW_2), (0.07, COAST_GLOW_3)]:
        gpd.GeoSeries([all_land.buffer(buf_deg)], crs=4326).plot(ax=ax, color=col_g, edgecolor="none", zorder=1)

    # Neighboring land (Malaysia Borneo + Sulawesi/Java/Belitung)
    surr_idn.plot(ax=ax, color=NEIGHBOR_LAND, edgecolor=NEIGHBOR_EDGE, linewidth=0.55, zorder=2)
    mys_borneo.plot(ax=ax, color=NEIGHBOR_LAND, edgecolor=NEIGHBOR_EDGE, linewidth=0.60, zorder=2)
    kal.plot(ax=ax, color=CARD_BG, edgecolor="none", zorder=2)

    # Soft Pastel KLHK 2025 Land Cover Mosaic
    ax.imshow(
        klhk_rgba,
        extent=[klhk_bounds.left, klhk_bounds.right, klhk_bounds.bottom, klhk_bounds.top],
        origin="upper", interpolation="bilinear", zorder=3, aspect="equal",
    )

    # Crisp subtle province borders & international boundary
    kal.boundary.plot(ax=ax, color=PROV_BORDER, linewidth=0.65, alpha=0.82, zorder=5)
    mys_borneo.boundary.plot(ax=ax, color="#9E978B", linewidth=0.65, linestyle=(0, (4, 3)), zorder=5)

    if show_neighbors_lbl:
        if xlim[0] < 114.0 and ylim[1] > 2.2:
            ax.text(113.15, 2.45, "SARAWAK\n(MALAYSIA)", fontsize=7.6, color="#857F74", fontweight="bold",
                    ha="center", va="center", style="italic", path_effects=text_halo(2.0, NEIGHBOR_LAND), zorder=6)
        if xlim[1] > 117.0 and ylim[1] > 4.2:
            ax.text(117.05, 4.55, "SABAH", fontsize=7.5, color="#857F74", fontweight="bold",
                    ha="center", va="center", style="italic", path_effects=text_halo(2.0, NEIGHBOR_LAND), zorder=6)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle=(0, (2, 4)), linewidth=0.45, alpha=0.42, color="#7B8FA1", zorder=0)


def draw_cities_and_provinces(ax, kal, xlim, ylim, show_provinces=True, show_cities=True, zoom_mode=False):
    if show_provinces and not zoom_mode:
        prov_pos = {
            "Central Kalimantan": (113.55, -0.95),
            "South Kalimantan": (115.42, -2.72),
            "West Kalimantan": (111.08, 0.28),
            "East Kalimantan": (116.32, 0.62),
            "North Kalimantan": (116.28, 3.08),
        }
        for _, row in kal.iterrows():
            px, py = prov_pos.get(row["shapeName"], (row.geometry.representative_point().x, row.geometry.representative_point().y))
            if xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]:
                ax.text(
                    px, py, row["shapeName"].upper().replace(" KALIMANTAN", "\nKALIMANTAN"),
                    fontsize=7.6, ha="center", va="center", fontweight="bold", color="#3F444C",
                    alpha=0.90, path_effects=text_halo(2.8, "#FAF8F3"), zorder=8,
                )

    zoom_offsets = {
        "Palangka Raya": (0.045, -0.015, "left"),
        "Banjarmasin": (-0.045, 0.025, "right"),
        "IKN Nusantara": (0.035, 0.035, "left"),
        "Balikpapan": (0.035, 0.028, "left"),
        "Samarinda": (0.035, 0.0, "left"),
        "Ketapang": (0.035, -0.025, "left"),
    }

    if show_cities:
        for name, lon, lat, is_cap, (dx, dy) in CITIES:
            if not (xlim[0] + 0.04 <= lon <= xlim[1] - 0.04 and ylim[0] + 0.04 <= lat <= ylim[1] - 0.04):
                continue
            sz_outer = 44 if (is_cap or zoom_mode) else 26
            sz_inner = 14 if (is_cap or zoom_mode) else 8
            ax.scatter([lon], [lat], s=sz_outer, facecolor="#FAF8F3", edgecolor="#2B2D42", linewidth=1.15, zorder=9)
            ax.scatter([lon], [lat], s=sz_inner, facecolor="#D95D5B" if name == "IKN Nusantara" else "#2B2D42", edgecolor="none", zorder=10)

            if zoom_mode and name in zoom_offsets:
                zdx, zdy, ha = zoom_offsets[name]
                tx, ty = lon + zdx, lat + zdy
            else:
                scale_off = 0.22 if zoom_mode else 1.0
                tx = lon + dx * scale_off
                ty = lat + dy * scale_off
                ha = "left" if dx > 0.05 else ("right" if dx < -0.05 else "center")
            ax.text(
                tx, ty, name,
                fontsize=7.8 if zoom_mode else (7.3 if is_cap else 6.7),
                fontweight="bold" if (is_cap or name == "IKN Nusantara") else "medium",
                color="#9E2A2B" if name == "IKN Nusantara" else "#1F242B",
                ha=ha, va="center", path_effects=text_halo(2.4, "#FAF8F3"), zorder=10,
            )


def draw_scalebar_and_north(ax, xlim, ylim, km_total=200, loc="bottom_left", north_loc="top_right"):
    """Draw sleek minimalist architectural scalebar & compass arrow."""
    lat_mid = 0.5 * (ylim[0] + ylim[1])
    deg_per_km = 1.0 / (111.32 * np.cos(np.deg2rad(lat_mid)))
    bar_deg = km_total * deg_per_km
    half_deg = 0.5 * bar_deg

    w_span = xlim[1] - xlim[0]
    h_span = ylim[1] - ylim[0]
    x0 = xlim[0] + (0.048 * w_span if "left" in loc else 0.68 * w_span)
    y0 = ylim[0] + 0.048 * h_span
    bar_h = 0.011 * h_span

    pad_x, pad_y = 0.024 * w_span, 0.026 * h_span
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0 - pad_x * 0.85, y0 - pad_y * 0.65), bar_deg + 2.55 * pad_x, bar_h + 2.35 * pad_y,
        boxstyle="round,pad=0.01", facecolor="#FAF8F3", edgecolor="#CFCAC0", lw=0.6, alpha=0.90, zorder=11,
    ))
    ax.add_patch(mpatches.Rectangle((x0, y0), half_deg, bar_h, facecolor="#3A3F47", edgecolor="#3A3F47", lw=0.6, zorder=12))
    ax.add_patch(mpatches.Rectangle((x0 + half_deg, y0), half_deg, bar_h, facecolor="#FAF8F3", edgecolor="#3A3F47", lw=0.6, zorder=12))
    ax.text(x0, y0 + bar_h * 1.45, "0", fontsize=6.7, ha="center", va="bottom", color="#3A3F47", fontweight="bold", zorder=12)
    ax.text(x0 + half_deg, y0 + bar_h * 1.45, f"{km_total//2}", fontsize=6.7, ha="center", va="bottom", color="#3A3F47", fontweight="bold", zorder=12)
    ax.text(x0 + bar_deg, y0 + bar_h * 1.45, f"{km_total} km", fontsize=6.7, ha="center", va="bottom", color="#3A3F47", fontweight="bold", zorder=12)

    nx = (xlim[1] - 0.052 * w_span) if "right" in north_loc else (xlim[0] + 0.052 * w_span)
    ny = ylim[1] - 0.072 * h_span
    ax.annotate(
        "N", xy=(nx, ny + 0.038 * h_span), xytext=(nx, ny),
        ha="center", va="center", fontsize=8.0, fontweight="bold", color="#3A3F47",
        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color="#3A3F47", lw=1.2),
        path_effects=text_halo(2.2, "#FAF8F3"), zorder=12,
    )


def save_both(fig, filename):
    p1 = os.path.join(FIG_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved figure: {p1}")


# ===========================================================================
# FIGURE 4 (HERO ATLAS): Multi-Scale Soft Pastel Land-Cover & Fire Legacy Atlas
# ===========================================================================
def make_fig4_urban_atlas(kal, mys_borneo, surr_idn, klhk_ov, klhk_bounds):
    ann_tif = os.path.join(RAW_MCD_DIR, "mcd64a1_annual_2001_2026_500m_mosaic.tif")
    with rasterio.open(ann_tif) as src_a:
        b2015_full = src_a.read(2015 - 2001 + 1) > 0
        atrans = src_a.transform
        abounds = src_a.bounds

    pts = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_viirs_375m_hotspots.csv"))
    pts = pts[pts["klhk_code"] > 0].copy()

    fig = plt.figure(figsize=(16.8, 12.2), facecolor=CANVAS_BG)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.32, 1.0], height_ratios=[1.0, 1.0, 1.0], hspace=0.23, wspace=0.06)

    # ---------------- Left Panel (a): Full Kalimantan Soft Urban-Atlas Map ----------------
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_anchor("E")
    klhk_rgba_full = render_klhk_rgba(klhk_ov, desaturate=0.0)
    xlim_m, ylim_m = (108.45, 119.15), (-4.55, 4.75)
    draw_soft_basemap(ax_main, kal, mys_borneo, surr_idn, klhk_rgba_full, klhk_bounds, xlim_m, ylim_m)

    # Subtle translucent berry-coral rim & tint for 2015 Mega-Fire Scars on overview
    b15_edge = binary_dilation(b2015_full, iterations=1) ^ binary_erosion(b2015_full, iterations=1)
    scar_ov = np.zeros((*b2015_full.shape, 4), dtype=np.float32)
    scar_ov[b2015_full] = mcolors.to_rgba("#C84B58", 0.24)
    scar_ov[b15_edge] = mcolors.to_rgba("#9E2A3B", 0.80)
    ax_main.imshow(
        scar_ov, extent=[abounds.left, abounds.right, abounds.bottom, abounds.top],
        origin="upper", interpolation="nearest", zorder=4, aspect="equal",
    )

    draw_cities_and_provinces(ax_main, kal, xlim_m, ylim_m, show_provinces=True, show_cities=True)
    draw_scalebar_and_north(ax_main, xlim_m, ylim_m, km_total=200, loc="bottom_left")

    # Define 3 Regional Urban-Corridor Zoom Windows
    zooms = [
        ("b", "1", "Palangka Raya – Pulang Pisau – Banjarmasin", (113.70, -3.52, 114.92, -2.08), 40),
        ("c", "2", "Balikpapan – IKN Nusantara – Samarinda", (116.56, -1.38, 117.44, -0.36), 30),
        ("d", "3", "Ketapang – Pawan Delta Agro-Peat Corridor", (109.86, -2.36, 110.84, -1.38), 30),
    ]
    for p_ltr, num_lbl, z_title, (zx0, zy0, zx1, zy1), _ in zooms:
        rect = mpatches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, edgecolor="#2B2D42", lw=1.4, linestyle="--", zorder=11)
        ax_main.add_patch(rect)
        bx = zx1 if num_lbl == "1" else zx0
        bha = "right" if num_lbl == "1" else "left"
        ax_main.text(
            bx, zy1 + 0.08, f"Window {num_lbl}", fontsize=7.6, fontweight="bold", color="#FAF8F3", ha=bha,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#2B2D42", edgecolor="none", alpha=0.90), zorder=12,
        )

    ax_main.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="semibold")
    ax_main.set_ylabel("Latitude (°N)", fontsize=9.5, fontweight="semibold")
    ax_main.set_title(
        "a   Kalimantan Soft Land-Cover Mosaic (KLHK 2025) & 2015 Mega-Fire Footprint",
        loc="left", fontsize=12.4, fontweight="bold", color=INK_DARK, pad=10,
    )

    # Curated Soft Pastel Land-Cover Legend Card at top-left of main map
    legend_items = [
        ("#9EC5A4", "Primary Dryland Forest (2001)"),
        ("#C5DEC2", "Secondary Dryland Forest (2002)"),
        ("#9CCBC4", "Peat / Swamp & Mangrove Forest"),
        ("#D5C2E5", "Oil Palm / Estate Crop (2010)"),
        ("#B7C7E2", "Timber Plantation / HTI (2006)"),
        ("#F1E394", "Rice Paddy / Sawah (20093)"),
        ("#F3DFC4", "Smallholder Dryland Farming"),
        ("#E7BFC7", "Swamp Shrub / Degraded Peat"),
        ("#EEC6A5", "Dryland Shrub & Open Land"),
        ("#A794B8", "Coal & Mineral Mining (20141)"),
        ("#DF7876", "Urban / Settlement (2012)"),
        ("#92C2DE", "Rivers, Lakes & Coastal Ponds"),
    ]
    handles = [mpatches.Patch(facecolor=c, edgecolor="#8E8A82", linewidth=0.4, label=l) for c, l in legend_items]
    handles.append(mpatches.Patch(facecolor=mcolors.to_rgba("#C84B58", 0.30), edgecolor="#9E2A3B", linewidth=1.1, label="2015 Mega-Fire Scar Perimeter"))
    handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E05A47", markeredgecolor="#FAF8F3", markersize=6, label="Oct 2026 Active Fire Cluster"))

    leg = ax_main.legend(
        handles=handles, loc="upper left", ncol=1, frameon=True,
        facecolor="#FAF9F5", edgecolor="#CFCAC0", framealpha=0.93,
        fontsize=7.55, title="KLHK 2025 Land-Use & Fire Layers", title_fontsize=8.0,
        borderpad=0.65, labelspacing=0.36,
    )
    leg.get_title().set_fontweight("bold")

    # ---------------- Right Panels (b, c, d): 3 High-Res (~55 m) Urban-Regional Windows ----------------
    for idx_z, (p_ltr, num_lbl, z_title, (zx0, zy0, zx1, zy1), sb_km) in enumerate(zooms):
        ax_z = fig.add_subplot(gs[idx_z, 1])
        ax_z.set_anchor("W")
        ax_z.set_facecolor(SEA_BG)

        # High-resolution (~55 m) modal KLHK 2025 extraction
        sub_klhk, _ = extract_klhk_highres_window(zx0, zy0, zx1, zy1, out_res=0.0005)
        sub_rgba = render_klhk_rgba(sub_klhk, desaturate=0.0)

        kal.plot(ax=ax_z, color=CARD_BG, edgecolor="none", zorder=1)
        ax_z.imshow(sub_rgba, extent=[zx0, zx1, zy0, zy1], origin="upper", interpolation="bilinear", zorder=2, aspect="equal")
        kal.boundary.plot(ax=ax_z, color=PROV_BORDER, linewidth=0.8, alpha=0.85, zorder=4)

        # Overlay crisp 2015 Burn Scar Contour & delicate glaze
        c0_a = max(0, int(np.floor((zx0 - atrans.c) / atrans.a)))
        c1_a = min(b2015_full.shape[1], int(np.ceil((zx1 - atrans.c) / atrans.a)))
        r0_a = max(0, int(np.floor((zy1 - atrans.f) / atrans.e)))
        r1_a = min(b2015_full.shape[0], int(np.ceil((zy0 - atrans.f) / atrans.e)))
        if r1_a > r0_a and c1_a > c0_a:
            sub_b15 = b2015_full[r0_a:r1_a, c0_a:c1_a]
            edge_15 = binary_dilation(sub_b15, iterations=1) ^ binary_erosion(sub_b15, iterations=1)
            ov15 = np.zeros((*sub_b15.shape, 4), dtype=np.float32)
            ov15[sub_b15] = mcolors.to_rgba("#C84B58", 0.14)
            ov15[edge_15] = mcolors.to_rgba("#9E2A3B", 0.88)
            sx0 = atrans.c + c0_a * atrans.a
            sx1 = atrans.c + c1_a * atrans.a
            sy1 = atrans.f + r0_a * atrans.e
            sy0 = atrans.f + r1_a * atrans.e
            ax_z.imshow(ov15, extent=[sx0, sx1, sy0, sy1], origin="upper", interpolation="nearest", zorder=3, aspect="equal")

        # Overlay Oct 2026 VIIRS active fire points delicately so pastel land-use parcels remain visible
        sub_pts = pts[(pts["lon"] >= zx0) & (pts["lon"] <= zx1) & (pts["lat"] >= zy0) & (pts["lat"] <= zy1)]
        if len(sub_pts) > 0:
            pt_sz = 1.15 if num_lbl == "1" else 2.2
            pt_al = 0.34 if num_lbl == "1" else 0.48
            ax_z.scatter(sub_pts["lon"], sub_pts["lat"], s=pt_sz, c="#D94E3B", alpha=pt_al, linewidths=0, zorder=5)

        draw_cities_and_provinces(ax_z, kal, (zx0, zx1), (zy0, zy1), show_provinces=False, show_cities=True, zoom_mode=True)
        draw_scalebar_and_north(ax_z, (zx0, zx1), (zy0, zy1), km_total=sb_km, loc="bottom_left")

        ax_z.set_xlim(zx0, zx1)
        ax_z.set_ylim(zy0, zy1)
        ax_z.tick_params(labelsize=7.8)
        ax_z.set_title(f"{p_ltr}   Window {num_lbl}: {z_title}", loc="left", fontsize=9.8, fontweight="bold", color=INK_DARK, pad=5)
        ax_z.grid(True, linestyle=(0, (2, 4)), linewidth=0.4, alpha=0.38, color="#64748B", zorder=0)

    save_both(fig, "fig4_kalimantan_soft_landcover_fire_atlas.png")


# ===========================================================================
# FIGURE 1: 25-Year Fire Regime + Soft-Pastel Fire Recurrence & Epicenter Zoom
# ===========================================================================
def make_fig1(kal, mys_borneo, surr_idn, prov_shapes, klhk_ov, klhk_bounds):
    df_prov = pd.read_csv(os.path.join(PROC_DIR, "mcd64a1_kalimantan_annual_burned_by_province_2001_2026.csv"))
    df_prov = df_prov[df_prov["year"] != "2026_JanSep"].copy()
    df_prov["year_int"] = df_prov["year"].astype(int)
    df_prov["kha"] = df_prov["burned_area_ha"] / 1e3
    df_mo = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_monthly_seasonality.csv"))

    ann_tif = os.path.join(RAW_MCD_DIR, "mcd64a1_annual_2001_2026_500m_mosaic.tif")
    with rasterio.open(ann_tif) as src:
        ann = src.read()[:25]
        trans = src.transform
        h, w = src.height, src.width
        bounds = src.bounds

    prov_mask = rasterize(prov_shapes, out_shape=(h, w), transform=trans, fill=0, dtype=np.uint8)
    n_burns = ann.sum(axis=0).astype(np.uint8)
    n_burns[prov_mask == 0] = 0
    b2015 = (ann[2015 - 2001] > 0) & (prov_mask > 0)

    fig = plt.figure(figsize=(17.0, 12.0), facecolor=CANVAS_BG)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.88, 1.28], width_ratios=[1.08, 1.0], hspace=0.25, wspace=0.16)

    # ---------------- Panel A: Soft-Editorial 25-Year Stacked Bar ----------------
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.set_facecolor(CARD_BG)

    prov_order = ["Central Kalimantan", "South Kalimantan", "West Kalimantan", "East Kalimantan", "North Kalimantan"]
    prov_cols = {
        "Central Kalimantan": "#C95B50",  # Warm terracotta-coral
        "South Kalimantan":   "#E69560",  # Soft apricot-tangerine
        "West Kalimantan":    "#EED18A",  # Soft warm butter-ochre
        "East Kalimantan":    "#84B5AA",  # Soft seafoam-sage
        "North Kalimantan":   "#647C90",  # Dusty Nordic slate
    }

    piv = df_prov.pivot(index="year_int", columns="province", values="kha").reindex(columns=prov_order).fillna(0.0)
    yrs = piv.index.values
    bottom = np.zeros(len(yrs))
    for p in prov_order:
        vals = piv[p].values
        ax_ts.bar(yrs, vals, bottom=bottom, width=0.72, color=prov_cols[p], edgecolor=CARD_BG, linewidth=0.5, label=p)
        bottom += vals

    max_tot = bottom.max()
    ax_ts.set_ylim(0, max_tot * 1.25)
    ax_ts.set_xlim(2000.3, 2025.7)
    ax_ts.set_xticks(yrs)
    ax_ts.set_xticklabels([str(y) for y in yrs], fontsize=8.8)
    ax_ts.set_ylabel("Burned Area (thousand ha / yr)", fontsize=10.2, fontweight="bold")
    ax_ts.set_title(
        "a   25-Year Kalimantan Spectral Burned Area Regime (2001–2025, MODIS MCD64A1 v061, 500 m)",
        loc="left", fontsize=12.4, fontweight="bold", color=INK_DARK, pad=10,
    )
    ax_ts.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45, color="#B8B2A6")
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    peaks = [
        (2002, "2002 El Niño", -0.30, 0.055, False),
        (2004, "2004 El Niño", 0.0, 0.055, False),
        (2006, "2006 El Niño / IOD+", 0.20, 0.055, False),
        (2009, "2009 El Niño", 0.0, 0.055, False),
        (2015, "2015 MEGA-FIRE\n(Super El Niño + IOD+)", -0.15, 0.105, True),
        (2019, "2019 Strong IOD+", -0.75, 0.065, False),
        (2023, "2023 El Niño", 0.0, 0.045, False),
    ]
    for y_pk, lbl_pk, dx_pk, dy_frac, is_15 in peaks:
        tot_y = float(piv.loc[y_pk].sum())
        ax_ts.annotate(
            f"{lbl_pk}\n{tot_y:,.0f} kha",
            xy=(y_pk, tot_y),
            xytext=(y_pk + dx_pk, tot_y + max_tot * dy_frac),
            ha="center", va="bottom",
            fontsize=8.1 if not is_15 else 8.8,
            fontweight="bold" if is_15 else "semibold",
            color="#9E2A3B" if is_15 else "#3F444C",
            arrowprops=dict(arrowstyle="-", color="#A8A298", lw=0.75),
        )

    ax_ts.legend(loc="upper right", bbox_to_anchor=(0.745, 0.99), ncol=2, frameon=True, facecolor=CARD_BG, edgecolor="#DED9CE", fontsize=8.3)

    ins = ax_ts.inset_axes([0.79, 0.53, 0.198, 0.40])
    ins.set_facecolor("#F7F5F0")
    mo_cols = ["#C95B50" if v >= 10 else "#BFB9AE" for v in df_mo["pct_2015_burn"]]
    ins.bar(np.arange(12), df_mo["pct_2015_burn"], color=mo_cols, width=0.72)
    ins.set_xticks(np.arange(12))
    ins.set_xticklabels([m[0] for m in df_mo["month"]], fontsize=7.2)
    ins.set_ylabel("% of 2015", fontsize=7.5)
    ins.set_title("2015 Burn Timing (Aug–Nov = {:.0f}%)".format(df_mo.iloc[7:11]["pct_2015_burn"].sum()), fontsize=8.0, fontweight="bold", color="#8C2D3C")
    ins.spines["top"].set_visible(False)
    ins.spines["right"].set_visible(False)
    ins.tick_params(axis="y", labelsize=7)

    # ---------------- Panel B: Soft-Pastel Kalimantan Fire Recurrence Map ----------------
    ax_map = fig.add_subplot(gs[1, 0])
    klhk_soft_bg = render_klhk_rgba(klhk_ov, desaturate=0.30)
    xlim_m, ylim_m = (108.45, 119.15), (-4.55, 4.65)
    draw_soft_basemap(ax_map, kal, mys_borneo, surr_idn, klhk_soft_bg, klhk_bounds, xlim_m, ylim_m)

    rec_rgba = np.zeros((h, w, 4), dtype=np.float32)
    rec_rgba[n_burns == 1] = mcolors.to_rgba("#F2A65A", 0.85)  # Warm honey-apricot
    rec_rgba[n_burns == 2] = mcolors.to_rgba("#E06D53", 0.90)  # Soft coral-terracotta
    rec_rgba[n_burns == 3] = mcolors.to_rgba("#B83B5E", 0.94)  # Rich berry-rose
    rec_rgba[n_burns >= 4] = mcolors.to_rgba("#5C2751", 0.98)  # Deep plum-aubergine

    ax_map.imshow(rec_rgba, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin="upper", interpolation="nearest", zorder=4, aspect="equal")
    draw_cities_and_provinces(ax_map, kal, xlim_m, ylim_m, show_provinces=True, show_cities=True)
    draw_scalebar_and_north(ax_map, xlim_m, ylim_m, km_total=200, loc="bottom_left")

    zx0, zx1, zy0, zy1 = 113.25, 115.35, -3.65, -1.75
    rect = mpatches.Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, edgecolor="#2B2D42", lw=1.5, linestyle="--", zorder=11)
    ax_map.add_patch(rect)
    ax_map.text(zx1 - 0.06, zy1 - 0.10, "Inset c", fontsize=8.0, fontweight="bold", color="#FAF8F3", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="#2B2D42", edgecolor="none", alpha=0.88), zorder=12)

    ax_map.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="semibold")
    ax_map.set_ylabel("Latitude (°N)", fontsize=9.5, fontweight="semibold")
    ax_map.set_title("b   25-Year Fire Recurrence Over Soft KLHK 2025 Landscape", loc="left", fontsize=11.6, fontweight="bold", color=INK_DARK, pad=8)

    leg_patches = [
        mpatches.Patch(color="#F2A65A", label="Burned 1 year (single event)"),
        mpatches.Patch(color="#E06D53", label="Burned 2 years"),
        mpatches.Patch(color="#B83B5E", label="Burned 3 years"),
        mpatches.Patch(color="#5C2751", label="Chronic fire trap (>= 4 yrs)"),
    ]
    ax_map.legend(handles=leg_patches, loc="upper left", frameon=True, facecolor="#FAF9F5", edgecolor="#CFCAC0", framealpha=0.93, fontsize=8.0, title="Fire Recurrence (2001–2025)", title_fontsize=8.2)

    # ---------------- Panel C: High-Detail Soft-Pastel Epicenter Zoom ----------------
    ax_zm = fig.add_subplot(gs[1, 1])
    ax_zm.set_facecolor(SEA_BG)
    sub_klhk, _ = extract_klhk_highres_window(zx0, zy0, zx1, zy1, out_res=0.0006)
    sub_rgba = render_klhk_rgba(sub_klhk, desaturate=0.18)

    kal.plot(ax=ax_zm, color=CARD_BG, edgecolor="none", zorder=1)
    ax_zm.imshow(sub_rgba, extent=[zx0, zx1, zy0, zy1], origin="upper", interpolation="bilinear", zorder=2, aspect="equal")
    kal.boundary.plot(ax=ax_zm, color=PROV_BORDER, linewidth=0.9, zorder=5)

    zm_rgba = np.zeros((h, w, 4), dtype=np.float32)
    other_yr = (n_burns >= 1) & (~b2015)
    b15_single = b2015 & (n_burns == 1)
    b15_repeat = b2015 & (n_burns >= 2)

    zm_rgba[other_yr] = mcolors.to_rgba("#7C8B99", 0.36)
    zm_rgba[b15_single] = mcolors.to_rgba("#E87A54", 0.78)
    zm_rgba[b15_repeat] = mcolors.to_rgba("#9E2A47", 0.86)

    ax_zm.imshow(zm_rgba, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top], origin="upper", interpolation="nearest", zorder=4, aspect="equal")
    draw_cities_and_provinces(ax_zm, kal, (zx0, zx1), (zy0, zy1), show_provinces=False, show_cities=True, zoom_mode=True)
    draw_scalebar_and_north(ax_zm, (zx0, zx1), (zy0, zy1), km_total=40, loc="bottom_right", north_loc="top_left")

    ax_zm.text(114.18, -2.58, "Ex-Mega Rice Project\nPeat Dome", fontsize=8.2, fontweight="bold",
               color="#4A1C2A", ha="center", path_effects=text_halo(2.6, "#FAF8F3"), zorder=11)

    ax_zm.set_xlim(zx0, zx1)
    ax_zm.set_ylim(zy0, zy1)
    ax_zm.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="semibold")
    ax_zm.set_ylabel("Latitude (°N)", fontsize=9.5, fontweight="semibold")
    ax_zm.set_title("c   Southern Peatland Epicenter: 2015 Scars & Repeat-Burn Trap", loc="left", fontsize=11.6, fontweight="bold", color=INK_DARK, pad=8)
    ax_zm.grid(True, linestyle=(0, (2, 4)), linewidth=0.45, alpha=0.45, color="#64748B", zorder=0)

    zm_leg = [
        mpatches.Patch(color="#E87A54", label="2015 Mega-Fire scar (burned once)"),
        mpatches.Patch(color="#9E2A47", label="2015 Mega-Fire + Repeat burn (>=2 yrs)"),
        mpatches.Patch(color=mcolors.to_rgba("#7C8B99", 0.55), label="Other fire years (non-2015)"),
    ]
    ax_zm.legend(handles=zm_leg, loc="upper right", frameon=True, facecolor="#FAF9F5", edgecolor="#CFCAC0", framealpha=0.93, fontsize=8.0)

    save_both(fig, "fig1_kalimantan_25yr_fire_history_and_scars.png")


# ===========================================================================
# FIGURE 2: Soft-Pastel Alluvial/Sankey + 2025 KLHK Fate & Trajectory
# ===========================================================================
def draw_sankey(ax, df_flows):
    ax.set_facecolor(CARD_BG)
    ax.set_xlim(-0.52, 1.76)
    ax.set_ylim(-0.04, 1.08)
    ax.axis("off")

    left_tot = df_flows.groupby("mb2014_broad")["pct_of_2015_burn"].sum().sort_values(ascending=True)
    right_tot = df_flows.groupby("klhk2025_group")["pct_of_2015_burn"].sum().sort_values(ascending=True)
    df_f = df_flows.copy()
    tiny_r = right_tot[right_tot < 0.45].index.tolist()
    if tiny_r:
        df_f["klhk2025_group"] = df_f["klhk2025_group"].replace({k: "Other (Primary/Built/Water)" for k in tiny_r})
        right_tot = df_f.groupby("klhk2025_group")["pct_of_2015_burn"].sum().sort_values(ascending=True)

    df_f = df_f.groupby(["mb2014_broad", "klhk2025_group"], as_index=False)["pct_of_2015_burn"].sum()

    left_colors = {
        "Peat Swamp Forest": "#76B0A8",
        "Dryland / Mangrove Forest": "#8CB893",
        "Shrub / Non-Forest Wetland": "#E5A472",
        "Smallholder Agriculture & Rice": "#E5CA6E",
        "Oil Palm & Timber Plantation": "#B599CE",
        "Open Land / Mining / Other": "#9AA5B1",
    }
    right_colors = {
        "Degraded Peat / Swamp Scrub": "#D47484",
        "Dryland Shrub & Grassland": "#E5A472",
        "Estate Crop / Oil Palm (Perkebunan)": "#B599CE",
        "Industrial Timber Plantation (HTI)": "#98B1D4",
        "Smallholder Agriculture & Rice": "#E5CA6E",
        "Secondary Swamp Forest": "#76B0A8",
        "Secondary Dryland / Mangrove Forest": "#8CB893",
        "Open / Cleared Land": "#DA9B82",
        "Mining": "#9E8BB0",
        "Settlement / Infrastructure / Pond": "#D97C7A",
        "Other (Primary/Built/Water)": "#A8B2BC",
    }

    gap = 0.022
    scale = min(1.0 - gap * max(0, len(left_tot) - 1), 1.0 - gap * max(0, len(right_tot) - 1)) / 100.0

    l_pos = {}
    cur_y = (1.0 - (100.0 * scale + gap * (len(left_tot) - 1))) / 2.0
    for name, val in left_tot.items():
        h_n = val * scale
        l_pos[name] = [cur_y, cur_y + h_n, cur_y]
        cur_y += h_n + gap

    r_pos = {}
    cur_y = (1.0 - (100.0 * scale + gap * (len(right_tot) - 1))) / 2.0
    for name, val in right_tot.items():
        h_n = val * scale
        r_pos[name] = [cur_y, cur_y + h_n, cur_y]
        cur_y += h_n + gap

    bar_w, x_l, x_r = 0.055, 0.08, 0.92
    df_f["l_ord"] = df_f["mb2014_broad"].map({k: i for i, k in enumerate(left_tot.index)})
    df_f["r_ord"] = df_f["klhk2025_group"].map({k: i for i, k in enumerate(right_tot.index)})
    df_f = df_f.sort_values(["l_ord", "r_ord"])

    for _, row in df_f.iterrows():
        pct = row["pct_of_2015_burn"]
        if pct < 0.08:
            continue
        h_f = pct * scale
        ln, rn = row["mb2014_broad"], row["klhk2025_group"]
        yl0, yl1 = l_pos[ln][2], l_pos[ln][2] + h_f
        l_pos[ln][2] = yl1
        yr0, yr1 = r_pos[rn][2], r_pos[rn][2] + h_f
        r_pos[rn][2] = yr1

        x0, x1 = x_l + bar_w, x_r
        xm = 0.5 * (x0 + x1)
        verts = [(x0, yl0), (xm, yl0), (xm, yr0), (x1, yr0), (x1, yr1), (xm, yr1), (xm, yl1), (x0, yl1), (x0, yl0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=left_colors.get(ln, "#A8B2BC"), alpha=0.48, edgecolor="none", zorder=2))

    for name, (y0, y1, _) in l_pos.items():
        val = left_tot[name]
        ax.add_patch(mpatches.Rectangle((x_l, y0), bar_w, max(0.004, y1 - y0), facecolor=left_colors.get(name, "#64748B"), edgecolor=CARD_BG, lw=0.9, zorder=4))
        short_l = name.replace("Smallholder Agriculture & Rice", "Smallholder Agri. & Rice").replace("Shrub / Non-Forest Wetland", "Shrub / Non-Forest")
        ax.text(x_l - 0.02, 0.5 * (y0 + y1), f"{short_l}\n({val:.1f}%)", ha="right", va="center", fontsize=8.3, fontweight="bold", color=INK_DARK)

    for name, (y0, y1, _) in r_pos.items():
        val = right_tot[name]
        ax.add_patch(mpatches.Rectangle((x_r, y0), bar_w, max(0.004, y1 - y0), facecolor=right_colors.get(name, "#64748B"), edgecolor=CARD_BG, lw=0.9, zorder=4))
        short_r = (
            name.replace("Estate Crop / Oil Palm (Perkebunan)", "Estate Crop / Oil Palm")
            .replace("Industrial Timber Plantation (HTI)", "Timber Plantation (HTI)")
            .replace("Secondary Dryland / Mangrove Forest", "Secondary Dryland Forest")
            .replace("Smallholder Agriculture & Rice", "Smallholder Agri. & Rice")
            .replace("Settlement / Infrastructure / Pond", "Settlement / Pond")
            .replace("Other (Primary/Built/Water)", "Other (Primary/Water)")
        )
        ax.text(x_r + bar_w + 0.02, 0.5 * (y0 + y1), f"{short_r} ({val:.1f}%)", ha="left", va="center", fontsize=8.1, fontweight="bold", color=INK_DARK)

    ax.text(x_l, 1.03, "Pre-Fire (2014)\nMapBiomas C4", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=INK_DARK)
    ax.text(0.50, 1.035, "→ 2015 Fire →", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#B83B5E")
    ax.text(x_r + bar_w, 1.03, "Current Fate (2025)\nKLHK 28 m", ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=INK_DARK)


def make_fig2():
    df_sankey = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_sankey_flows_2014_to_2025.csv"))
    df_fate = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_summary.csv"))
    df_mb = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_mapbiomas_trajectory_2014_to_2024.csv"))

    fig = plt.figure(figsize=(19.2, 10.8), facecolor=CANVAS_BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0], height_ratios=[1.18, 0.90], hspace=0.30, wspace=0.56)

    ax_sk = fig.add_subplot(gs[:, 0])
    draw_sankey(ax_sk, df_sankey)
    ax_sk.set_title("a   Decade-Later Land-Cover Transformation of 2015 Burn Scars (2014 → 2025)", loc="left", fontsize=12.2, fontweight="bold", color=INK_DARK, pad=14)

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_facecolor(CARD_BG)
    tot15 = df_fate["burn2015_ha"].sum()
    top = df_fate[df_fate["pct_of_2015_burn"] >= 0.45].head(12).copy().iloc[::-1]
    pct_single = 100.0 * top["burn2015_single_only_ha"] / tot15
    pct_repeat = 100.0 * top["burn2015_repeat_2plus_ha"] / tot15

    y = np.arange(len(top))
    ax_bar.barh(y, pct_single, height=0.66, color="#84A9C0", label="Burned once (2015 only — stabilized / converted)")
    ax_bar.barh(y, pct_repeat, left=pct_single, height=0.66, color="#D46A6A", label="Chronic repeat fire (>=2 burn yrs in 2001–2025)")

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(
        [f"{r['class'].replace('Dryland farming mixed with shrub', 'Dryland farming + shrub')} ({int(r['code'])})" for _, r in top.iterrows()],
        fontsize=8.7, fontweight="semibold",
    )
    ax_bar.set_xlabel("% of Total 2015 Kalimantan Burned Area", fontsize=9.5, fontweight="bold")
    ax_bar.set_title("b   2025 KLHK Categories on 2015 Scars & Repeat-Fire Trap", loc="left", fontsize=11.5, fontweight="bold", color=INK_DARK, pad=8)

    xmax = (pct_single + pct_repeat).max() * 1.24
    ax_bar.set_xlim(0, xmax)
    for idx_i, (_, r) in enumerate(top.iterrows()):
        tot_p = r["pct_of_2015_burn"]
        ax_bar.text(tot_p + xmax * 0.012, idx_i, f"{tot_p:.1f}%  ({r['pct_repeat_within_2015_scar']:.0f}% repeat)",
                    va="center", ha="left", fontsize=7.9, fontweight="bold", color=INK_DARK)

    ax_bar.legend(loc="lower right", frameon=True, facecolor=CARD_BG, edgecolor="#DED9CE", fontsize=8.1)
    ax_bar.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.45, color="#B8B2A6")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    ax_tr = fig.add_subplot(gs[1, 1])
    ax_tr.set_facecolor(CARD_BG)
    epochs = ["2014_pre_fire", "2016_post_fire_1yr", "2020_post_fire_5yr", "2024_post_fire_9yr"]
    ep_labels = ["2014\n(Pre-Fire)", "2016\n(+1 yr)", "2020\n(+5 yrs)", "2024\n(+9 yrs)"]
    piv_mb = df_mb.pivot(index="mb_code", columns="epoch", values="pct").reindex(columns=epochs).fillna(0.0)

    series_cfg = [
        ([13], "Shrub / Non-forest wetland (13)", "#D97F54", "o-", 2.4),
        ([35], "Oil palm plantation (35)", "#9B72AA", "s-", 2.5),
        ([21, 40], "Agriculture & Rice paddy (21+40)", "#C8A951", "^-", 2.1),
        ([76], "Peat swamp forest (76)", "#5E9D96", "D--", 2.1),
        ([3, 5], "Dryland & Mangrove forest (3+5)", "#6E9E75", "v--", 2.0),
        ([9], "Pulpwood plantation / HTI (9)", "#6B8EB8", "P-", 2.0),
    ]
    x_ep = np.arange(len(epochs))
    ax_tr.axhline(0.0, color="#8C877E", linestyle="-", linewidth=1.0, alpha=0.7, zorder=1)
    for codes, lbl, col, style, lw in series_cfg:
        present = [c for c in codes if c in piv_mb.index]
        vals = piv_mb.loc[present].sum(axis=0).values if present else np.zeros(len(epochs))
        delta = vals - vals[0]
        sign = "+" if delta[-1] >= 0 else ""
        ax_tr.plot(x_ep, delta, style, color=col, linewidth=lw, markersize=6.2, zorder=3,
                   label=f"{lbl}: {vals[0]:.1f}%→{vals[-1]:.1f}% ({sign}{delta[-1]:.1f} pp)")

    ymin_t, ymax_t = ax_tr.get_ylim()
    ax_tr.set_ylim(-8.6, ymax_t + 1.2)
    ax_tr.axvline(0.5, color="#C95B50", linestyle=":", linewidth=1.5, alpha=0.85, zorder=2)
    ax_tr.text(0.53, ymax_t + 0.85, "2015 Fire", color="#A63A3A", fontsize=8.3, fontweight="bold", va="top")

    ax_tr.set_xticks(x_ep)
    ax_tr.set_xticklabels(ep_labels, fontsize=8.9, fontweight="semibold")
    ax_tr.set_ylabel("Net Change vs. 2014 (pct. points)", fontsize=9.2, fontweight="bold")
    ax_tr.set_title("c   Post-Fire Land-Cover Shift vs. 2014 Baseline (MapBiomas C4)", loc="left", fontsize=11.5, fontweight="bold", color=INK_DARK, pad=8)
    ax_tr.legend(loc="lower left", ncol=2, columnspacing=0.9, frameon=True, facecolor=CARD_BG, edgecolor="#DED9CE", framealpha=0.95, fontsize=7.0)
    ax_tr.grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color="#B8B2A6")
    ax_tr.spines["top"].set_visible(False)
    ax_tr.spines["right"].set_visible(False)

    save_both(fig, "fig2_2015_megafire_postfire_fate_2025.png")


# ===========================================================================
# FIGURE 3: October 2026 Active Burning Over Soft Pastel Map & Diagnostics
# ===========================================================================
def make_fig3(kal, mys_borneo, surr_idn, klhk_ov, klhk_bounds):
    pts = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_viirs_375m_hotspots.csv"))
    df_oct = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_fire_by_klhk2025_complete.csv"))
    df_oct_p = pd.read_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_fire_by_province_summary.csv"))
    df_f15_p = pd.read_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_by_province.csv"))

    ann_tif = os.path.join(RAW_MCD_DIR, "mcd64a1_annual_2001_2026_500m_mosaic.tif")
    with rasterio.open(ann_tif) as src_a:
        ever_burned = src_a.read()[:25].sum(axis=0) > 0
        atrans = src_a.transform
        ah, aw = src_a.height, src_a.width

    valid_pts = pts[pts["klhk_code"] > 0].copy()
    acols = np.clip(np.floor((valid_pts["lon"].values - atrans.c) / atrans.a).astype(np.int32), 0, aw - 1)
    arows = np.clip(np.floor((valid_pts["lat"].values - atrans.f) / atrans.e).astype(np.int32), 0, ah - 1)
    valid_pts["on_hist_scar"] = ever_burned[arows, acols]

    fig = plt.figure(figsize=(18.6, 10.6), facecolor=CANVAS_BG)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.05, 1.15], height_ratios=[1.22, 0.85], hspace=0.30, wspace=0.48)

    ax_m = fig.add_subplot(gs[:, 0])
    klhk_soft_bg = render_klhk_rgba(klhk_ov, desaturate=0.26)
    xlim_m, ylim_m = (108.45, 119.15), (-4.55, 4.65)
    draw_soft_basemap(ax_m, kal, mys_borneo, surr_idn, klhk_soft_bg, klhk_bounds, xlim_m, ylim_m)

    p_new = valid_pts[(valid_pts["n_nom_hi"] < 25) & (~valid_pts["on_hist_scar"])]
    p_reb = valid_pts[(valid_pts["n_nom_hi"] < 25) & (valid_pts["on_hist_scar"])]
    p_sta = valid_pts[valid_pts["n_nom_hi"] >= 25]

    ax_m.scatter(p_new["lon"], p_new["lat"], s=1.6, c="#EB9448", alpha=0.42, linewidths=0, zorder=4,
                 label=f"Oct 2026 fire outside 2001–25 scars (n={len(p_new):,})")
    ax_m.scatter(p_reb["lon"], p_reb["lat"], s=2.2, c="#C43D53", alpha=0.58, linewidths=0, zorder=5,
                 label=f"Oct 2026 re-burn on 2001–25 scars (n={len(p_reb):,})")
    ax_m.scatter(p_sta["lon"], p_sta["lat"], s=24, c="#3E7CB1", marker="^", edgecolors="#FAF8F3", linewidths=0.4, zorder=6,
                 label=f"Persistent thermal source >=25 det. (n={len(p_sta):,})")

    draw_cities_and_provinces(ax_m, kal, xlim_m, ylim_m, show_provinces=True, show_cities=True)
    draw_scalebar_and_north(ax_m, xlim_m, ylim_m, km_total=200, loc="bottom_left")

    ax_m.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="semibold")
    ax_m.set_ylabel("Latitude (°N)", fontsize=9.5, fontweight="semibold")
    ax_m.set_title(
        "a   October 2026 Active Fires & Scar Re-Ignition\n     Over Soft KLHK 2025 Landscape (VIIRS 375 m)",
        loc="left", fontsize=11.4, fontweight="bold", color=INK_DARK, pad=8,
    )
    m_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#EB9448", markersize=6.5, label=f"Oct 2026 fire outside 2001–25 scars (n={len(p_new):,})"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C43D53", markersize=6.5, label=f"Oct 2026 re-burn on 2001–25 scars (n={len(p_reb):,})"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#3E7CB1", markeredgecolor="#FAF8F3", markersize=8.0, label=f"Persistent thermal source >=25 det. (n={len(p_sta):,})"),
    ]
    ax_m.legend(handles=m_handles, loc="upper left", frameon=True, facecolor="#FAF9F5", edgecolor="#CFCAC0", framealpha=0.93, fontsize=8.0)

    # ---------------- Panel B: Soft Editorial Dumbbell Chart ----------------
    ax_db = fig.add_subplot(gs[0, 1])
    ax_db.set_facecolor(CARD_BG)

    top_o = df_oct[df_oct["pct_viirs_nom_hi"] >= 0.45].head(12).copy().iloc[::-1]
    yo = np.arange(len(top_o))

    for i_r, (_, r) in enumerate(top_o.iterrows()):
        x_land, x_fire = r["pct_kalimantan_land"], r["pct_viirs_nom_hi"]
        col_line = "#D46A6A" if x_fire > x_land else "#9AA5B1"
        ax_db.plot([x_land, x_fire], [i_r, i_r], color=col_line, linewidth=2.3, alpha=0.80, zorder=2)

    ax_db.scatter(top_o["pct_kalimantan_land"], yo, s=68, color="#A8C6DF", edgecolors="#4B749F", linewidths=1.1, zorder=3, label="% of Kalimantan Baseline Land (KLHK 2025)")
    ax_db.scatter(top_o["pct_viirs_nom_hi"], yo, s=78, color="#D45D68", edgecolors="#7A2838", linewidths=1.0, zorder=4, label="% of Oct 2026 Burning (VIIRS 375 m)")

    ax_db.set_yticks(yo)
    ax_db.set_yticklabels([f"{r['class']} ({int(r['code'])})" for _, r in top_o.iterrows()], fontsize=8.7, fontweight="semibold")
    ax_db.set_xlabel("Share (%)", fontsize=9.5, fontweight="bold")
    ax_db.set_title("b   October 2026 Burning Share vs. Land Share & Selection Ratio", loc="left", fontsize=11.5, fontweight="bold", color=INK_DARK, pad=8)

    max_v = max(top_o["pct_viirs_nom_hi"].max(), top_o["pct_kalimantan_land"].max()) * 1.22
    ax_db.set_xlim(-0.5, max_v)
    for i_r, (_, r) in enumerate(top_o.iterrows()):
        xr = max(r["pct_viirs_nom_hi"], r["pct_kalimantan_land"])
        sr = r["selection_ratio"]
        ax_db.text(xr + 0.55, i_r, f"{r['pct_viirs_nom_hi']:.1f}% ({sr:.2f}×)", va="center", ha="left", fontsize=8.0, fontweight="bold",
                   color="#9E2A3B" if sr >= 1.4 else INK_DARK)

    ax_db.legend(loc="lower right", frameon=True, facecolor=CARD_BG, edgecolor="#DED9CE", fontsize=8.1)
    ax_db.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.45, color="#B8B2A6")
    ax_db.spines["top"].set_visible(False)
    ax_db.spines["right"].set_visible(False)

    # ---------------- Panel C: Soft Editorial Provincial Share ----------------
    ax_pr = fig.add_subplot(gs[1, 1])
    ax_pr.set_facecolor(CARD_BG)

    p15 = df_f15_p.groupby("province")[["burn2015_ha", "burn2019_ha"]].sum()
    p15["pct_2015"] = 100.0 * p15["burn2015_ha"] / p15["burn2015_ha"].sum()
    p15["pct_2019"] = 100.0 * p15["burn2019_ha"] / p15["burn2019_ha"].sum()
    poct = df_oct_p.set_index("province")[["pct_of_kal_burning"]]
    pm = p15.join(poct).sort_values("pct_of_kal_burning", ascending=False)

    xp = np.arange(len(pm))
    bw = 0.26
    ax_pr.bar(xp - bw, pm["pct_2015"], width=bw, color="#E89963", edgecolor=CARD_BG, lw=0.5, label="2015 Mega-Fire (MCD64A1)")
    ax_pr.bar(xp, pm["pct_2019"], width=bw, color="#989AC8", edgecolor=CARD_BG, lw=0.5, label="2019 Fire Episode (MCD64A1)")
    ax_pr.bar(xp + bw, pm["pct_of_kal_burning"], width=bw, color="#D45D68", edgecolor=CARD_BG, lw=0.5, label="Oct 2026 Active Burning (VIIRS)")

    ax_pr.set_xticks(xp)
    ax_pr.set_xticklabels([p.replace(" Kalimantan", "\nKalimantan") for p in pm.index], fontsize=8.8, fontweight="semibold")
    ax_pr.set_ylabel("% of Kalimantan Total", fontsize=9.5, fontweight="bold")
    ax_pr.set_title("c   Provincial Share Across Fire Episodes (2015 vs. 2019 vs. Oct 2026)", loc="left", fontsize=11.5, fontweight="bold", color=INK_DARK, pad=8)
    ax_pr.set_ylim(0, max(pm["pct_2015"].max(), pm["pct_of_kal_burning"].max()) * 1.18)
    ax_pr.legend(loc="upper right", frameon=True, facecolor=CARD_BG, edgecolor="#DED9CE", fontsize=8.1)
    ax_pr.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.45, color="#B8B2A6")
    ax_pr.spines["top"].set_visible(False)
    ax_pr.spines["right"].set_visible(False)

    save_both(fig, "fig3_oct2026_active_burning_and_fire_memory.png")


def main():
    kal = gpd.read_file(os.path.join(REF_DIR, "kalimantan_adm1.geojson"))
    mys_borneo = gpd.read_file(os.path.join(REF_DIR, "borneo_malaysia_adm1.geojson"))
    surr_idn = gpd.read_file(os.path.join(REF_DIR, "surrounding_indonesia_land.geojson"))
    kal["prov_idx"] = np.arange(1, len(kal) + 1, dtype=np.uint8)
    prov_shapes = list(zip(kal.geometry, kal["prov_idx"]))

    klhk_ov, klhk_bounds, _ = load_or_build_klhk_overview(kal)

    make_fig4_urban_atlas(kal, mys_borneo, surr_idn, klhk_ov, klhk_bounds)
    make_fig1(kal, mys_borneo, surr_idn, prov_shapes, klhk_ov, klhk_bounds)
    make_fig2()
    make_fig3(kal, mys_borneo, surr_idn, klhk_ov, klhk_bounds)
    print("All 4 soft urban-atlas publication figures generated successfully!")


if __name__ == "__main__":
    main()
