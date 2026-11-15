"""
01_klhk2025_baseline.py
-----------------------
Computes exact geodesic land-cover area (ha and km2) per KLHK Penutupan Lahan 2025 category
across all 5 Kalimantan provinces from the 68 intersecting 1.024-degree KLHK GeoTIFF tiles.
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
from shapely.geometry import box

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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


def main():
    kal = gpd.read_file(os.path.join(REF_DIR, "kalimantan_adm1.geojson"))
    kal["prov_idx"] = np.arange(1, len(kal) + 1, dtype=np.uint8)
    prov_map = dict(zip(kal["prov_idx"], kal["shapeName"]))
    kal_union = kal.union_all()
    prov_shapes = list(zip(kal.geometry, kal["prov_idx"]))

    tiles = sorted(glob.glob(os.path.join(KLHK_TILES_DIR, "*.tif")))
    kal_tiles = [fn for fn in tiles if box(*rasterio.open(fn).bounds).intersects(kal_union)]
    print(f"Processing {len(kal_tiles)} Kalimantan KLHK PL2025 tiles...")

    records = {}
    for fn in kal_tiles:
        with rasterio.open(fn) as src:
            arr = src.read(1)
            transform = src.transform
            h, w = src.height, src.width

        prov_ras = rasterize(prov_shapes, out_shape=(h, w), transform=transform, fill=0, dtype=np.uint8)
        mask = (prov_ras > 0) & (arr > 0)
        if not np.any(mask):
            continue

        lats = transform.f + (np.arange(h) + 0.5) * transform.e
        dlon = abs(transform.a) * DEG2RAD
        dlat = abs(transform.e) * DEG2RAD
        row_area_m2 = (R_EARTH ** 2) * dlon * dlat * np.cos(lats * DEG2RAD)

        r_idx, c_idx = np.nonzero(mask)
        keys = prov_ras[r_idx, c_idx].astype(np.int64) * 1_000_000 + arr[r_idx, c_idx].astype(np.int64)
        uniq_keys, inv = np.unique(keys, return_inverse=True)
        counts = np.bincount(inv)
        areas = np.bincount(inv, weights=row_area_m2[r_idx])

        for uk, cnt, ar in zip(uniq_keys, counts, areas):
            p_id, k_cd = int(uk // 1_000_000), int(uk % 1_000_000)
            prev = records.get((p_id, k_cd), [0, 0.0])
            records[(p_id, k_cd)] = [prev[0] + int(cnt), prev[1] + float(ar)]

    legend = pd.read_csv(os.path.join(REF_DIR, "klhk_pl2025_legend.csv"))
    legend["kelas_id"] = legend["code"].map(ID_NAMES)

    rows_out = [
        {"prov_idx": p_id, "province": prov_map[p_id], "code": k_cd, "pixels": cnt, "area_ha": ar / 1e4, "area_km2": ar / 1e6}
        for (p_id, k_cd), (cnt, ar) in records.items()
    ]
    df = pd.DataFrame(rows_out).merge(legend, on="code", how="left")
    df.to_csv(os.path.join(PROC_DIR, "kalimantan_klhk2025_baseline_by_province.csv"), index=False)

    tot = df.groupby(["code", "kelas_id", "class"], dropna=False)[["pixels", "area_ha", "area_km2"]].sum().reset_index()
    tot["pct_kalimantan"] = 100.0 * tot["area_km2"] / tot["area_km2"].sum()
    tot = tot.sort_values("area_km2", ascending=False)
    tot.to_csv(os.path.join(PROC_DIR, "kalimantan_klhk2025_baseline_total.csv"), index=False)
    print(f"Done. Total mapped Kalimantan area: {tot['area_km2'].sum():,.2f} km2")


if __name__ == "__main__":
    main()
