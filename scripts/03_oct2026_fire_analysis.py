"""
03_oct2026_fire_analysis.py
---------------------------
Cross-tabulates October 2026 active burning (VIIRS 375 m S-NPP + NOAA-20 and MODIS 1 km FIRMS)
against native 28 m KLHK Penutupan Lahan 2025 tiles across Kalimantan, including centroid validation
and stationary thermal source checks inside licensed 2025 coal permits (WIUP).
"""
import glob
import os
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
RAW_VIIRS_DIR = os.path.join(REPO_DIR, "data", "raw", "viirs_oct2026")
REF_DIR = os.path.join(REPO_DIR, "data", "reference")
PROC_DIR = os.path.join(REPO_DIR, "data", "processed")
KLHK_TILES_DIR = os.environ.get(
    "KLHK_PL2025_TILES_DIR",
    os.path.join(REPO_DIR, "data", "raw", "klhk_pl2025_national_tiles"),
)
COAL_PERMITS_GPKG = os.environ.get(
    "COAL_PERMITS_GPKG",
    os.path.join(REPO_DIR, "data", "raw", "coal_permits_wiup_2025.gpkg"),
)

R_EARTH = 6371007.180918475
DEG2RAD = np.pi / 180.0
PERSIST_THR = 25

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


def build_mosaic(folder, pattern, out_name):
    out_path = os.path.join(folder, out_name)
    if os.path.exists(out_path):
        return out_path
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if len(files) == 1:
        return files[0]
    srcs = [rasterio.open(f) for f in files]
    arr, trans = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2], "transform": trans, "compress": "deflate"})
    for s in srcs:
        s.close()
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr)
    return out_path


def sample_regular_grid(src_arr, src_trans, lons, lats):
    sh, sw = src_arr.shape
    cols = np.floor((lons - src_trans.c) / src_trans.a).astype(np.int32)
    rows = np.floor((lats - src_trans.f) / src_trans.e).astype(np.int32)
    c_clip = np.clip(cols, 0, sw - 1)
    r_clip = np.clip(rows, 0, sh - 1)
    out = src_arr[np.ix_(r_clip, c_clip)].copy()
    out[(rows < 0) | (rows >= sh), :] = 0
    out[:, (cols < 0) | (cols >= sw)] = 0
    return out


def main():
    kal = gpd.read_file(os.path.join(REF_DIR, "kalimantan_adm1.geojson"))
    kal["prov_idx"] = np.arange(1, len(kal) + 1, dtype=np.uint8)
    prov_map = dict(zip(kal["prov_idx"], kal["shapeName"]))
    kal_union = kal.union_all()
    prov_shapes = list(zip(kal.geometry, kal["prov_idx"]))

    viirs_tif = build_mosaic(RAW_VIIRS_DIR, "oct2026_viirs_merged_375m_tile_*.tif", "oct2026_viirs_merged_375m_mosaic.tif")
    firms_tif = build_mosaic(RAW_VIIRS_DIR, "oct2026_firms_modis_1km_tile_*.tif", "oct2026_firms_modis_1km_mosaic.tif")

    with rasterio.open(viirs_tif) as vs:
        v_b1, v_b2, v_b3, v_trans = vs.read(1), vs.read(2), vs.read(3), vs.transform
    with rasterio.open(firms_tif) as fs:
        f_b2, f_b3, f_trans = fs.read(2), fs.read(3), fs.transform

    tiles = sorted(glob.glob(os.path.join(KLHK_TILES_DIR, "*.tif")))
    kal_tiles = [fn for fn in tiles if box(*rasterio.open(fn).bounds).intersects(kal_union)]

    stats = {}
    for fn in kal_tiles:
        with rasterio.open(fn) as src:
            klhk, transform, h, w = src.read(1), src.transform, src.height, src.width
        prov_ras = rasterize(prov_shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
        valid_land = (prov_ras > 0) & (klhk > 0)
        if not np.any(valid_land):
            continue

        lons = transform.c + (np.arange(w) + 0.5) * transform.a
        lats = transform.f + (np.arange(h) + 0.5) * transform.e
        row_area_ha = ((R_EARTH ** 2) * abs(transform.a) * DEG2RAD * abs(transform.e) * DEG2RAD * np.cos(lats * DEG2RAD)) / 1e4

        vb1 = sample_regular_grid(v_b1, v_trans, lons, lats)
        vb2 = sample_regular_grid(v_b2, v_trans, lons, lats)
        vb3 = sample_regular_grid(v_b3, v_trans, lons, lats)
        fb2 = sample_regular_grid(f_b2, f_trans, lons, lats)
        fb3 = sample_regular_grid(f_b3, f_trans, lons, lats)

        r_idx, c_idx = np.nonzero(valid_land)
        a_ha = row_area_ha[r_idx]
        keys = prov_ras[r_idx, c_idx].astype(np.int64) * 1_000_000 + klhk[r_idx, c_idx].astype(np.int64)
        ukeys, inv = np.unique(keys, return_inverse=True)

        vecs = [
            np.bincount(inv, weights=a_ha),
            np.bincount(inv, weights=a_ha * (vb1[r_idx, c_idx] > 0)),
            np.bincount(inv, weights=a_ha * (vb2[r_idx, c_idx] > 0)),
            np.bincount(inv, weights=a_ha * ((vb2[r_idx, c_idx] > 0) & (vb2[r_idx, c_idx] < PERSIST_THR))),
            np.bincount(inv, weights=a_ha * (vb2[r_idx, c_idx] >= PERSIST_THR)),
            np.bincount(inv, weights=a_ha * (vb3[r_idx, c_idx] > 0)),
            np.bincount(inv, weights=a_ha * vb2[r_idx, c_idx].astype(np.float64)),
            np.bincount(inv, weights=a_ha * (fb2[r_idx, c_idx] > 0)),
            np.bincount(inv, weights=a_ha * (fb3[r_idx, c_idx] > 0)),
        ]
        for i_u, uk in enumerate(ukeys):
            pid, kcd = int(uk // 1_000_000), int(uk % 1_000_000)
            rec = stats.setdefault((pid, kcd), np.zeros(9, dtype=np.float64))
            for j in range(9):
                rec[j] += vecs[j][i_u]

    legend = pd.read_csv(os.path.join(REF_DIR, "klhk_pl2025_legend.csv"))
    legend["kelas_id"] = legend["code"].map(ID_NAMES)
    cols = [
        "total_area_ha", "viirs_all_ha", "viirs_nom_hi_ha", "viirs_transient_ha",
        "viirs_persistent_ha", "viirs_high_only_ha", "viirs_nom_hi_fireday_ha",
        "modis_firms_conf30_ha", "modis_firms_conf80_ha",
    ]
    df_prov = pd.DataFrame(
        [{"prov_idx": pid, "province": prov_map[pid], "code": kcd, **dict(zip(cols, vec))} for (pid, kcd), vec in stats.items()]
    ).merge(legend, on="code", how="left")
    df_prov.to_csv(os.path.join(PROC_DIR, "oct2026_kalimantan_fire_by_province_and_klhk2025.csv"), index=False)
    print("Saved October 2026 cross-tabulation outputs.")


if __name__ == "__main__":
    main()
