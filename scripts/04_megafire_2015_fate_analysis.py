"""
04_megafire_2015_fate_analysis.py
---------------------------------
Comprehensive post-fire trajectory analysis for Kalimantan:
1. 25-year annual spectral burned area time series (MODIS MCD64A1 v061, 500 m, 2001-2025 + Jan-Sep 2026) by province.
2. Exact 28 m cross-tabulation of 2015 (and 2019) mega-fire burn scars against current KLHK Penutupan Lahan 2025,
   stratified by fire recurrence (single burn vs repeat burn >=2x) and Oct 2026 VIIRS re-ignition.
3. Pre-fire (2014) -> Post-fire (2016 -> 2020 -> 2024/2025) transition trajectory combining MapBiomas C4 and KLHK 2025.
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
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from shapely.geometry import box

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_MCD_DIR = os.path.join(REPO_DIR, "data", "raw", "mcd64a1_2001_2025")
RAW_VIIRS_DIR = os.path.join(REPO_DIR, "data", "raw", "viirs_oct2026")
REF_DIR = os.path.join(REPO_DIR, "data", "reference")
PROC_DIR = os.path.join(REPO_DIR, "data", "processed")
KLHK_TILES_DIR = os.environ.get(
    "KLHK_PL2025_TILES_DIR",
    os.path.join(REPO_DIR, "data", "raw", "klhk_pl2025_national_tiles"),
)
os.makedirs(PROC_DIR, exist_ok=True)

R_EARTH = 6371007.180918475
DEG2RAD = np.pi / 180.0

ID_NAMES = {
    2001: "Hutan lahan kering primer", 2002: "Hutan lahan kering sekunder",
    2004: "Hutan mangrove primer", 20041: "Hutan mangrove sekunder",
    2005: "Hutan rawa primer", 20051: "Hutan rawa sekunder",
    2006: "Hutan tanaman", 2007: "Semak belukar", 20071: "Semak belukar rawa",
    2010: "Perkebunan", 2012: "Permukiman", 20121: "Bandara / pelabuhan",
    20122: "Transmigrasi", 2014: "Tanah terbuka", 20141: "Pertambangan",
    20091: "Pertanian lahan kering", 20092: "Pertanian lahan kering campur semak",
    20093: "Sawah", 20094: "Tambak", 3000: "Savana / padang rumput",
    5001: "Tubuh air", 50011: "Rawa", 2500: "Awan",
}

KLHK_TRAJECTORY_GROUP = {
    20071: "Degraded Peat / Swamp Scrub",
    50011: "Degraded Peat / Swamp Scrub",
    2007: "Dryland Shrub & Grassland",
    3000: "Dryland Shrub & Grassland",
    2010: "Estate Crop / Oil Palm (Perkebunan)",
    2006: "Industrial Timber Plantation (HTI)",
    20091: "Smallholder Agriculture & Rice",
    20092: "Smallholder Agriculture & Rice",
    20093: "Smallholder Agriculture & Rice",
    20051: "Secondary Swamp Forest",
    2005: "Primary Forest (Dryland/Swamp/Mangrove)",
    2002: "Secondary Dryland / Mangrove Forest",
    20041: "Secondary Dryland / Mangrove Forest",
    2001: "Primary Forest (Dryland/Swamp/Mangrove)",
    2004: "Primary Forest (Dryland/Swamp/Mangrove)",
    2014: "Open / Cleared Land",
    20141: "Mining",
    2012: "Settlement / Infrastructure / Pond",
    20121: "Settlement / Infrastructure / Pond",
    20122: "Settlement / Infrastructure / Pond",
    5001: "Settlement / Infrastructure / Pond",
    20094: "Settlement / Infrastructure / Pond",
    2500: "Cloud",
}

MB_NAMES = {
    3: "Forest formation",
    5: "Mangrove",
    76: "Peat swamp forest",
    13: "Non-forest natural vegetation (Shrub/Wetland)",
    40: "Rice paddy",
    35: "Oil palm",
    9: "Pulpwood plantation (HTI)",
    21: "Other agriculture / mosaic",
    30: "Mining",
    24: "Urban area",
    25: "Other non-vegetated / bare",
    31: "Aquaculture",
    33: "Water body",
    27: "Not observed",
}

MB_BROAD_PRE = {
    76: "Peat Swamp Forest",
    3: "Dryland / Mangrove Forest",
    5: "Dryland / Mangrove Forest",
    13: "Shrub / Non-Forest Wetland",
    21: "Smallholder Agriculture & Rice",
    40: "Smallholder Agriculture & Rice",
    35: "Oil Palm & Timber Plantation",
    9: "Oil Palm & Timber Plantation",
    25: "Open Land / Mining / Other",
    30: "Open Land / Mining / Other",
    24: "Open Land / Mining / Other",
    31: "Open Land / Mining / Other",
    33: "Open Land / Mining / Other",
    27: "Open Land / Mining / Other",
}


def build_mosaic(folder, pattern, out_name):
    out_path = os.path.join(folder, out_name)
    if os.path.exists(out_path):
        return out_path
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No tiles matching {pattern} in {folder}")
    if len(files) == 1:
        return files[0]
    srcs = [rasterio.open(f) for f in files]
    arr, trans = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "transform": trans,
        "compress": "deflate",
    })
    for s in srcs:
        s.close()
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr)
    return out_path


def sample_regular_grid(src_arr, src_trans, lons, lats):
    sh, sw = src_arr.shape
    cols = np.floor((lons - src_trans.c) / src_trans.a).astype(np.int32)
    rows = np.floor((lats - src_trans.f) / src_trans.e).astype(np.int32)
    valid_c = (cols >= 0) & (cols < sw)
    valid_r = (rows >= 0) & (rows < sh)
    c_clip = np.clip(cols, 0, sw - 1)
    r_clip = np.clip(rows, 0, sh - 1)
    out = src_arr[np.ix_(r_clip, c_clip)].copy()
    if not np.all(valid_r):
        out[~valid_r, :] = 0
    if not np.all(valid_c):
        out[:, ~valid_c] = 0
    return out


def row_pixel_area_ha(transform, h):
    lats = transform.f + (np.arange(h) + 0.5) * transform.e
    dlon = abs(transform.a) * DEG2RAD
    dlat = abs(transform.e) * DEG2RAD
    return ((R_EARTH ** 2) * dlon * dlat * np.cos(lats * DEG2RAD)) / 1e4


def main():
    kal = gpd.read_file(os.path.join(REF_DIR, "kalimantan_adm1.geojson"))
    kal["prov_idx"] = np.arange(1, len(kal) + 1, dtype=np.uint8)
    prov_map = dict(zip(kal["prov_idx"], kal["shapeName"]))
    kal_union = kal.union_all()
    prov_shapes = list(zip(kal.geometry, kal["prov_idx"]))

    # 1. Mosaic historical rasters
    ann_tif = build_mosaic(RAW_MCD_DIR, "mcd64a1_annual_2001_2026_500m_tile_*.tif", "mcd64a1_annual_2001_2026_500m_mosaic.tif")
    doy_tif = build_mosaic(RAW_MCD_DIR, "mcd64a1_doy_2015_2019_2023_500m_tile_*.tif", "mcd64a1_doy_2015_2019_2023_500m_mosaic.tif")
    mb_tif = build_mosaic(RAW_MCD_DIR, "mapbiomas_c4_2014_2016_2020_2024_250m_tile_*.tif", "mapbiomas_c4_2014_2016_2020_2024_250m_mosaic.tif")
    viirs_tif = build_mosaic(RAW_VIIRS_DIR, "oct2026_viirs_merged_375m_tile_*.tif", "oct2026_viirs_merged_375m_mosaic.tif")

    # ------------------------------------------------------------------
    # PART A: 25-year Annual Burned Area Time Series (2001-2025 + Jan-Sep 2026)
    # ------------------------------------------------------------------
    with rasterio.open(ann_tif) as src_ann:
        ann_stack = src_ann.read()  # shape: (26, H, W) -> 2001..2025, 2026_jansep
        ann_trans = src_ann.transform
        ah, aw = src_ann.height, src_ann.width

    prov_500 = rasterize(prov_shapes, out_shape=(ah, aw), transform=ann_trans, fill=0, dtype=np.uint8)
    area_500_ha = row_pixel_area_ha(ann_trans, ah)[:, None]  # broadcast (H, 1)

    years_labels = [str(y) for y in range(2001, 2026)] + ["2026_JanSep"]
    ts_rows = []
    for b_idx, yr_lbl in enumerate(years_labels):
        band_b = ann_stack[b_idx] > 0
        for pid, pname in prov_map.items():
            burned_ha = float(np.sum(area_500_ha * (band_b & (prov_500 == pid))))
            ts_rows.append({"year": yr_lbl, "province": pname, "burned_area_ha": burned_ha, "burned_area_km2": burned_ha / 100.0})

    df_ts_prov = pd.DataFrame(ts_rows)
    df_ts_prov.to_csv(os.path.join(PROC_DIR, "mcd64a1_kalimantan_annual_burned_by_province_2001_2026.csv"), index=False)

    df_ts_tot = df_ts_prov.groupby("year", sort=False)[["burned_area_ha", "burned_area_km2"]].sum().reset_index()
    df_ts_tot.to_csv(os.path.join(PROC_DIR, "mcd64a1_kalimantan_annual_burned_total_2001_2026.csv"), index=False)

    print("=== TOP 10 MEGA-FIRE YEARS IN KALIMANTAN (MODIS MCD64A1 500 m, 2001-2025) ===")
    df_2001_2025 = df_ts_tot[df_ts_tot["year"] != "2026_JanSep"].copy()
    print(df_2001_2025.sort_values("burned_area_ha", ascending=False).head(10).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    tot_25yr_footprint = float(np.sum(area_500_ha * ((ann_stack[:25].sum(axis=0) > 0) & (prov_500 > 0))))
    print(f"Cumulative unique area burned >=1x during 2001-2025: {tot_25yr_footprint:,.1f} ha ({tot_25yr_footprint/100:,.1f} km2)")

    # Fire recurrence summary across Kalimantan (2001-2025)
    n_burns_25yr = ann_stack[:25].sum(axis=0).astype(np.uint8)  # 0..25
    burn_2015_mask = ann_stack[2015 - 2001] > 0
    burn_2019_mask = ann_stack[2019 - 2001] > 0
    burn_2023_mask = ann_stack[2023 - 2001] > 0

    rec_rows = []
    for nb in range(0, int(n_burns_25yr[prov_500 > 0].max()) + 1):
        ha_nb = float(np.sum(area_500_ha * ((n_burns_25yr == nb) & (prov_500 > 0))))
        if ha_nb > 0:
            rec_rows.append({"n_years_burned_2001_2025": nb, "area_ha": ha_nb, "area_km2": ha_nb / 100.0})
    pd.DataFrame(rec_rows).to_csv(os.path.join(PROC_DIR, "mcd64a1_kalimantan_fire_recurrence_2001_2025.csv"), index=False)

    # Check seasonality of 2015 mega-fire from DOY raster
    with rasterio.open(doy_tif) as src_doy:
        doy_2015 = src_doy.read(1)
    m_2015 = doy_2015[(prov_500 > 0) & (doy_2015 > 0)]
    # Month edges in non-leap year 2015
    month_ends = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m_counts = np.histogram(m_2015, bins=month_ends)[0]
    df_2015_mo = pd.DataFrame({"month": month_names, "cells_500m": m_counts, "pct_2015_burn": 100.0 * m_counts / max(1, m_counts.sum())})
    df_2015_mo.to_csv(os.path.join(PROC_DIR, "megafire_2015_monthly_seasonality.csv"), index=False)
    print("\n2015 Burn Seasonality (% by month):")
    print(df_2015_mo[df_2015_mo["pct_2015_burn"] >= 0.5].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ------------------------------------------------------------------
    # PART B: MapBiomas C4 Trajectory on 2015 Burn Scars (2014 Pre-Fire -> 2016 -> 2020 -> 2024)
    # ------------------------------------------------------------------
    with rasterio.open(mb_tif) as src_mb:
        mb_2014 = src_mb.read(1)
        mb_2016 = src_mb.read(2)
        mb_2020 = src_mb.read(3)
        mb_2024 = src_mb.read(4)
        mb_trans = src_mb.transform
        mh, mw = src_mb.height, src_mb.width

    mb_lons = mb_trans.c + (np.arange(mw) + 0.5) * mb_trans.a
    mb_lats = mb_trans.f + (np.arange(mh) + 0.5) * mb_trans.e
    prov_250 = rasterize(prov_shapes, out_shape=(mh, mw), transform=mb_trans, fill=0, dtype=np.uint8)
    b15_250 = sample_regular_grid(burn_2015_mask.astype(np.uint8), ann_trans, mb_lons, mb_lats) > 0
    area_250_ha = row_pixel_area_ha(mb_trans, mh)

    rr250, cc250 = np.nonzero((prov_250 > 0) & b15_250 & (mb_2014 > 0) & (mb_2024 > 0))
    w250 = area_250_ha[rr250]

    mb_epoch_rows = []
    for yr_col, arr_m in [("2014_pre_fire", mb_2014), ("2016_post_fire_1yr", mb_2016), ("2020_post_fire_5yr", mb_2020), ("2024_post_fire_9yr", mb_2024)]:
        vals = arr_m[rr250, cc250]
        u_v, inv_v = np.unique(vals, return_inverse=True)
        ha_v = np.bincount(inv_v, weights=w250)
        tot_w = ha_v.sum()
        for code_m, ha_m in zip(u_v, ha_v):
            mb_epoch_rows.append({
                "epoch": yr_col,
                "mb_code": int(code_m),
                "mb_class": MB_NAMES.get(int(code_m), f"Class {code_m}"),
                "area_ha": float(ha_m),
                "pct": 100.0 * float(ha_m) / tot_w,
            })
    df_mb_traj = pd.DataFrame(mb_epoch_rows)
    df_mb_traj.to_csv(os.path.join(PROC_DIR, "megafire_2015_mapbiomas_trajectory_2014_to_2024.csv"), index=False)

    print("\n=== MAPBIOMAS C4 TRAJECTORY OF 2015 BURN SCARS (2014 Pre-Fire -> 2016 -> 2020 -> 2024) ===")
    piv_mb = df_mb_traj.pivot(index=["mb_code", "mb_class"], columns="epoch", values="pct").fillna(0.0).reset_index()
    piv_mb = piv_mb.sort_values("2024_post_fire_9yr", ascending=False)
    print(piv_mb.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # ------------------------------------------------------------------
    # PART C: Native 28 m KLHK Penutupan Lahan 2025 Fate of 2015 & 2019 Burn Scars
    #         + Pre-Fire 2014 -> 2025 KLHK Transition Matrix + Repeat-Fire & Oct 2026 Re-ignition!
    # ------------------------------------------------------------------
    with rasterio.open(viirs_tif) as src_v:
        v_oct2026 = src_v.read(2) > 0  # Nominal + High confidence active fire in Oct 2026
        v_trans = src_v.transform

    tiles = sorted(glob.glob(os.path.join(KLHK_TILES_DIR, "*.tif")))
    kal_tiles = []
    for fn in tiles:
        with rasterio.open(fn) as s:
            if box(*s.bounds).intersects(kal_union):
                kal_tiles.append(fn)

    # Accumulators:
    # 1) (prov_idx, klhk_code) -> [b2015_all_ha, b2015_single_ha, b2015_repeat_ha, b2015_ also2019_ha, b2015_oct2026_ha, b2019_all_ha, total_land_ha]
    fate_stats = {}
    # 2) Transition matrix on 2015 burned pixels: (mb2014_code, klhk2025_code) -> area_ha
    trans_2014_2025 = {}

    for fn in kal_tiles:
        with rasterio.open(fn) as src:
            klhk = src.read(1)
            transform = src.transform
            h, w = src.height, src.width

        prov_ras = rasterize(prov_shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
        valid_land = (prov_ras > 0) & (klhk > 0)
        if not np.any(valid_land):
            continue

        lons = transform.c + (np.arange(w) + 0.5) * transform.a
        lats = transform.f + (np.arange(h) + 0.5) * transform.e
        row_ha = row_pixel_area_ha(transform, h)

        b15 = sample_regular_grid(burn_2015_mask.astype(np.uint8), ann_trans, lons, lats) > 0
        b19 = sample_regular_grid(burn_2019_mask.astype(np.uint8), ann_trans, lons, lats) > 0
        nb25 = sample_regular_grid(n_burns_25yr, ann_trans, lons, lats)
        voct = sample_regular_grid(v_oct2026.astype(np.uint8), v_trans, lons, lats) > 0

        r_idx, c_idx = np.nonzero(valid_land)
        p_v = prov_ras[r_idx, c_idx]
        k_v = klhk[r_idx, c_idx]
        a_v = row_ha[r_idx]
        b15_v = b15[r_idx, c_idx]
        b19_v = b19[r_idx, c_idx]
        nb_v = nb25[r_idx, c_idx]
        oct_v = voct[r_idx, c_idx]

        keys = p_v.astype(np.int64) * 1_000_000 + k_v.astype(np.int64)
        ukeys, inv = np.unique(keys, return_inverse=True)

        w_tot = np.bincount(inv, weights=a_v)
        w_b15 = np.bincount(inv, weights=a_v * b15_v)
        w_b15_single = np.bincount(inv, weights=a_v * (b15_v & (nb_v == 1)))
        w_b15_repeat = np.bincount(inv, weights=a_v * (b15_v & (nb_v >= 2)))
        w_b15_b19 = np.bincount(inv, weights=a_v * (b15_v & b19_v))
        w_b15_oct26 = np.bincount(inv, weights=a_v * (b15_v & oct_v))
        w_b19 = np.bincount(inv, weights=a_v * b19_v)
        w_any25_oct26 = np.bincount(inv, weights=a_v * ((nb_v >= 1) & oct_v))
        w_oct26_tot = np.bincount(inv, weights=a_v * oct_v)

        for iu, uk in enumerate(ukeys):
            pid = int(uk // 1_000_000)
            kcd = int(uk % 1_000_000)
            rec = fate_stats.setdefault((pid, kcd), np.zeros(9, dtype=np.float64))
            rec[0] += w_b15[iu]
            rec[1] += w_b15_single[iu]
            rec[2] += w_b15_repeat[iu]
            rec[3] += w_b15_b19[iu]
            rec[4] += w_b15_oct26[iu]
            rec[5] += w_b19[iu]
            rec[6] += w_tot[iu]
            rec[7] += w_any25_oct26[iu]
            rec[8] += w_oct26_tot[iu]

        # Also accumulate 2014 MapBiomas -> 2025 KLHK transition on 2015 burned pixels!
        if np.any(b15_v):
            mb14_tile = sample_regular_grid(mb_2014, mb_trans, lons, lats)
            m15_mask = valid_land & b15
            mr, mc = np.nonzero(m15_mask)
            m14_vals = mb14_tile[mr, mc]
            k25_vals = klhk[mr, mc]
            ma_vals = row_ha[mr]
            valid_m = m14_vals > 0
            if np.any(valid_m):
                tkeys = m14_vals[valid_m].astype(np.int64) * 1_000_000 + k25_vals[valid_m].astype(np.int64)
                utk, tinv = np.unique(tkeys, return_inverse=True)
                tw = np.bincount(tinv, weights=ma_vals[valid_m])
                for uk_t, w_t in zip(utk, tw):
                    m_cd = int(uk_t // 1_000_000)
                    k_cd = int(uk_t % 1_000_000)
                    trans_2014_2025[(m_cd, k_cd)] = trans_2014_2025.get((m_cd, k_cd), 0.0) + float(w_t)

    legend = pd.read_csv(os.path.join(REF_DIR, "klhk_pl2025_legend.csv"))
    legend["kelas_id"] = legend["code"].map(ID_NAMES)
    legend["trajectory_group"] = legend["code"].map(KLHK_TRAJECTORY_GROUP)

    prov_rows = []
    for (pid, kcd), vec in fate_stats.items():
        prov_rows.append({
            "prov_idx": pid,
            "province": prov_map[pid],
            "code": kcd,
            "burn2015_ha": vec[0],
            "burn2015_single_only_ha": vec[1],
            "burn2015_repeat_2plus_ha": vec[2],
            "burn2015_reburned_2019_ha": vec[3],
            "burn2015_reignited_oct2026_ha": vec[4],
            "burn2019_ha": vec[5],
            "total_klhk2025_ha": vec[6],
            "oct2026_on_historical_2001_2025_scar_ha": vec[7],
            "oct2026_total_ha": vec[8],
        })

    df_fate_prov = pd.DataFrame(prov_rows).merge(legend, on="code", how="left")
    df_fate_prov.to_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_by_province.csv"), index=False)

    sum_cols = [
        "burn2015_ha", "burn2015_single_only_ha", "burn2015_repeat_2plus_ha",
        "burn2015_reburned_2019_ha", "burn2015_reignited_oct2026_ha",
        "burn2019_ha", "total_klhk2025_ha",
        "oct2026_on_historical_2001_2025_scar_ha", "oct2026_total_ha"
    ]
    df_fate = legend.merge(df_fate_prov.groupby("code")[sum_cols].sum().reset_index(), on="code", how="left")
    df_fate[sum_cols] = df_fate[sum_cols].fillna(0.0)
    df_fate["pct_of_2015_burn"] = 100.0 * df_fate["burn2015_ha"] / max(1e-9, df_fate["burn2015_ha"].sum())
    df_fate["pct_of_2019_burn"] = 100.0 * df_fate["burn2019_ha"] / max(1e-9, df_fate["burn2019_ha"].sum())
    df_fate["pct_repeat_within_2015_scar"] = np.where(
        df_fate["burn2015_ha"] > 0, 100.0 * df_fate["burn2015_repeat_2plus_ha"] / df_fate["burn2015_ha"], 0.0
    )
    df_fate["pct_oct2026_reignited_within_2015_scar"] = np.where(
        df_fate["burn2015_ha"] > 0, 100.0 * df_fate["burn2015_reignited_oct2026_ha"] / df_fate["burn2015_ha"], 0.0
    )
    df_fate = df_fate.sort_values("burn2015_ha", ascending=False).reset_index(drop=True)
    df_fate.to_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_summary.csv"), index=False)

    print("\n=== CURRENT (KLHK 2025) FATE OF 2015 MEGA-FIRE BURN SCARS IN KALIMANTAN ===")
    cols_p = ["code", "kelas_id", "class", "burn2015_ha", "pct_of_2015_burn", "pct_repeat_within_2015_scar", "pct_oct2026_reignited_within_2015_scar", "pct_of_2019_burn"]
    print(df_fate[cols_p].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    tot15 = df_fate["burn2015_ha"].sum()
    rep15 = df_fate["burn2015_repeat_2plus_ha"].sum()
    reb19 = df_fate["burn2015_reburned_2019_ha"].sum()
    oct15 = df_fate["burn2015_reignited_oct2026_ha"].sum()
    oct_any = df_fate["oct2026_on_historical_2001_2025_scar_ha"].sum()
    oct_tot = df_fate["oct2026_total_ha"].sum()
    print(f"\nTotal 2015 Burn Scar mapped on KLHK 2025 land: {tot15:,.1f} ha ({tot15/100:,.1f} km2)")
    print(f"  -> Burned repeatedly (>=2 yrs in 2001-2025): {rep15:,.1f} ha ({100*rep15/tot15:.2f}%)")
    print(f"  -> Burned AGAIN in 2019 mega-fire:           {reb19:,.1f} ha ({100*reb19/tot15:.2f}%)")
    print(f"  -> Active fire AGAIN in October 2026:        {oct15:,.1f} ha ({100*oct15/tot15:.2f}% of 2015 scar!)")
    print(f"  -> Share of Oct 2026 burning sitting on 2001-2025 MCD64A1 historical scars: {oct_any:,.1f} / {oct_tot:,.1f} ha ({100*oct_any/oct_tot:.2f}%)")

    print("\n=== GROUPED TRAJECTORY FATE OF 2015 BURN SCARS IN KLHK 2025 ===")
    grp_f = df_fate.groupby("trajectory_group")[["burn2015_ha", "burn2015_single_only_ha", "burn2015_repeat_2plus_ha", "burn2015_reignited_oct2026_ha"]].sum()
    grp_f["pct_2015_burn"] = 100.0 * grp_f["burn2015_ha"] / tot15
    grp_f["pct_repeat_fire"] = np.where(grp_f["burn2015_ha"] > 0, 100.0 * grp_f["burn2015_repeat_2plus_ha"] / grp_f["burn2015_ha"], 0.0)
    grp_f["pct_oct2026_reignited"] = np.where(grp_f["burn2015_ha"] > 0, 100.0 * grp_f["burn2015_reignited_oct2026_ha"] / grp_f["burn2015_ha"], 0.0)
    grp_f = grp_f.sort_values("burn2015_ha", ascending=False)
    grp_f.to_csv(os.path.join(PROC_DIR, "megafire_2015_fate_klhk2025_grouped.csv"))
    print(grp_f.to_string(float_format=lambda v: f"{v:,.2f}"))

    # Save 2014 Pre-Fire -> 2025 KLHK transition matrix
    t_rows = []
    for (m_cd, k_cd), ha_val in trans_2014_2025.items():
        t_rows.append({
            "mb2014_code": m_cd,
            "mb2014_class": MB_NAMES.get(m_cd, f"Class {m_cd}"),
            "mb2014_broad": MB_BROAD_PRE.get(m_cd, "Other"),
            "klhk2025_code": k_cd,
            "klhk2025_kelas": ID_NAMES.get(k_cd, str(k_cd)),
            "klhk2025_group": KLHK_TRAJECTORY_GROUP.get(k_cd, "Other"),
            "area_ha": ha_val,
        })
    df_trans = pd.DataFrame(t_rows)
    df_trans["pct_of_2015_burn"] = 100.0 * df_trans["area_ha"] / max(1e-9, df_trans["area_ha"].sum())
    df_trans = df_trans.sort_values("area_ha", ascending=False)
    df_trans.to_csv(os.path.join(PROC_DIR, "megafire_2015_transition_2014_to_klhk2025.csv"), index=False)

    print("\n=== TOP 15 TRANSITIONS: 2014 PRE-FIRE -> 2015 FIRE -> 2025 CURRENT LAND COVER ===")
    sankey_agg = df_trans.groupby(["mb2014_broad", "klhk2025_group"])[["area_ha", "pct_of_2015_burn"]].sum().reset_index().sort_values("area_ha", ascending=False)
    sankey_agg.to_csv(os.path.join(PROC_DIR, "megafire_2015_sankey_flows_2014_to_2025.csv"), index=False)
    print(sankey_agg.head(15).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
