"""
02_fetch_gee_datasets.py
------------------------
Downloads satellite fire & historical land-cover rasters over Kalimantan from Google Earth Engine:
  1. October 2026 VIIRS 375 m active fires (S-NPP + NOAA-20 Collection 2) & MODIS 1 km FIRMS
  2. Annual MODIS MCD64A1 v061 500 m spectral burned-area stack (2001-2025 + Jan-Sep 2026)
  3. Burn Day-of-Year (DOY) rasters for mega-fire years (2015, 2019, 2023)
  4. MapBiomas Indonesia Collection 4 pre/post-fire epochs (2014, 2016, 2020, 2024 at 250 m)

Authentication uses environment variables (see .env.example) or standard `earthengine authenticate`.
"""
import io
import os
import time
import zipfile
import ee
import numpy as np
import requests

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIIRS_DIR = os.path.join(REPO_DIR, "data", "raw", "viirs_oct2026")
MCD_DIR = os.path.join(REPO_DIR, "data", "raw", "mcd64a1_2001_2025")
os.makedirs(VIIRS_DIR, exist_ok=True)
os.makedirs(MCD_DIR, exist_ok=True)

MINX, MINY, MAXX, MAXY = 108.5, -5.2, 119.0, 4.5
M_PER_DEG = 111319.49079327358
RES_250M = 250.0 / M_PER_DEG
RES_375M = 375.0 / M_PER_DEG
RES_500M = 500.0 / M_PER_DEG
RES_1KM = 1000.0 / M_PER_DEG


def init_gee():
    project = os.environ.get("GEE_PROJECT")
    sa = os.environ.get("GEE_SERVICE_ACCOUNT")
    key_path = os.environ.get("GEE_PRIVATE_KEY_PATH")
    if sa and key_path and os.path.exists(key_path):
        creds = ee.ServiceAccountCredentials(sa, key_path)
        ee.Initialize(creds, project=project)
    elif project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()
    print("Google Earth Engine initialized.")


def download_geotiff_tiled(img, out_dir, prefix, res_deg, nx=2, ny=2):
    total_w = int(np.ceil((MAXX - MINX) / res_deg))
    total_h = int(np.ceil((MAXY - MINY) / res_deg))
    tile_w = int(np.ceil(total_w / nx))
    tile_h = int(np.ceil(total_h / ny))
    for iy in range(ny):
        for ix in range(nx):
            x0 = MINX + ix * tile_w * res_deg
            y_top = MAXY - iy * tile_h * res_deg
            w = min(tile_w, total_w - ix * tile_w)
            h = min(tile_h, total_h - iy * tile_h)
            if w <= 0 or h <= 0:
                continue
            x1 = x0 + w * res_deg
            y_bot = y_top - h * res_deg
            fn = os.path.join(out_dir, f"{prefix}_tile_{ix}_{iy}.tif")
            if os.path.exists(fn) and os.path.getsize(fn) > 1000:
                continue
            for attempt in range(4):
                try:
                    url = img.getDownloadURL({
                        "region": ee.Geometry.Rectangle([x0, y_bot, x1, y_top], None, False),
                        "crs": "EPSG:4326",
                        "crsTransform": [res_deg, 0, x0, 0, -res_deg, y_top],
                        "dimensions": f"{w}x{h}",
                        "format": "GEO_TIFF",
                    })
                    r = requests.get(url, timeout=300)
                    r.raise_for_status()
                    b = r.content
                    if b[:2] == b"PK":
                        z = zipfile.ZipFile(io.BytesIO(b))
                        b = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
                    with open(fn, "wb") as f:
                        f.write(b)
                    print(f"Saved {os.path.basename(fn)} ({w}x{h})")
                    break
                except Exception as e:
                    print(f"Retry {attempt} for {fn}: {e}")
                    time.sleep(5 * (attempt + 1))


def main():
    init_gee()
    kal_rect = ee.Geometry.Rectangle([MINX, MINY, MAXX, MAXY], None, False)

    # 1. October 2026 VIIRS 375m (S-NPP + NOAA-20)
    viirs_prepped = []
    for cid, tag in [("NASA/LANCE/SNPP_VIIRS/C2", "snpp_viirs"), ("NASA/LANCE/NOAA20_VIIRS/C2", "noaa20_viirs")]:
        col = ee.ImageCollection(cid).filterDate("2026-10-01", "2026-11-01").filterBounds(kal_rect)

        def prep_v(im):
            t4 = im.select("Bright_ti4")
            conf = im.select("confidence")
            valid = t4.gt(0)
            nom_hi = valid.And(conf.gte(1))
            hi = valid.And(conf.gte(2))
            return ee.Image.cat([
                valid.toUint16().rename("n_all"),
                nom_hi.toUint16().rename("n_nom_hi"),
                hi.toUint16().rename("n_hi"),
                nom_hi.toUint16().rename("n_veg_nom_hi"),
                conf.toUint16().updateMask(valid).rename("max_conf"),
                t4.toFloat().updateMask(valid).rename("max_ti4"),
            ])

        p = col.map(prep_v)
        viirs_prepped.append(p)
        comp = ee.Image.cat([
            p.select(["n_all", "n_nom_hi", "n_hi", "n_veg_nom_hi"]).sum().unmask(0).toUint16(),
            p.select("max_conf").max().add(1).unmask(0).toUint16().rename("max_conf_plus1"),
            p.select("max_ti4").max().unmask(0).round().toUint16().rename("max_ti4_k"),
        ])
        download_geotiff_tiled(comp, VIIRS_DIR, f"oct2026_{tag}_375m", RES_375M, nx=2, ny=2)

    merged_v = viirs_prepped[0].merge(viirs_prepped[1])
    comp_m = ee.Image.cat([
        merged_v.select(["n_all", "n_nom_hi", "n_hi", "n_veg_nom_hi"]).sum().unmask(0).toUint16(),
        merged_v.select("max_conf").max().add(1).unmask(0).toUint16().rename("max_conf_plus1"),
        merged_v.select("max_ti4").max().unmask(0).round().toUint16().rename("max_ti4_k"),
    ])
    download_geotiff_tiled(comp_m, VIIRS_DIR, "oct2026_viirs_merged_375m", RES_375M, nx=2, ny=2)

    # 2. October 2026 MODIS FIRMS 1 km
    firms = ee.ImageCollection("FIRMS").filterDate("2026-10-01", "2026-11-01").filterBounds(kal_rect)
    pf = firms.map(lambda im: ee.Image.cat([
        im.select("T21").gt(0).toUint16().rename("n_all"),
        im.select("T21").gt(0).And(im.select("confidence").gte(30)).toUint16().rename("n_conf30"),
        im.select("T21").gt(0).And(im.select("confidence").gte(80)).toUint16().rename("n_conf80"),
        im.select("confidence").toUint16().updateMask(im.select("T21").gt(0)).rename("max_conf"),
    ]))
    comp_f = ee.Image.cat([
        pf.select(["n_all", "n_conf30", "n_conf80"]).sum().unmask(0).toUint16(),
        pf.select("max_conf").max().unmask(0).toUint16().rename("max_conf"),
    ])
    download_geotiff_tiled(comp_f, VIIRS_DIR, "oct2026_firms_modis_1km", RES_1KM, nx=1, ny=1)

    # 3. MODIS MCD64A1 v061 Annual Burned Area (2001-2026) & Key Mega-Fire DOY (2015, 2019, 2023)
    mcd = ee.ImageCollection("MODIS/061/MCD64A1")
    yr_bands = [
        mcd.filterDate(f"{y}-01-01", f"{y+1}-01-01").select("BurnDate").max().gt(0).unmask(0).toUint8().rename(f"b{y}")
        for y in range(2001, 2026)
    ]
    yr_bands.append(mcd.filterDate("2026-01-01", "2026-10-01").select("BurnDate").max().gt(0).unmask(0).toUint8().rename("b2026_jansep"))
    download_geotiff_tiled(ee.Image.cat(yr_bands), MCD_DIR, "mcd64a1_annual_2001_2026_500m", RES_500M, nx=3, ny=2)

    doy_stack = ee.Image.cat([
        mcd.filterDate(f"{y}-01-01", f"{y+1}-01-01").select("BurnDate").max().clamp(0, 366).unmask(0).toUint16().rename(f"doy_{y}")
        for y in (2015, 2019, 2023)
    ])
    download_geotiff_tiled(doy_stack, MCD_DIR, "mcd64a1_doy_2015_2019_2023_500m", RES_500M, nx=2, ny=2)

    # 4. MapBiomas Collection 4 (2014, 2016, 2020, 2024) at 250 m
    mb = ee.Image("projects/mapbiomas-public/assets/indonesia/lulc/collection4/mapbiomas_indonesia_collection4_coverage_v2")
    mb_stack = mb.select(["classification_2014", "classification_2016", "classification_2020", "classification_2024"]).toUint8()
    download_geotiff_tiled(mb_stack, MCD_DIR, "mapbiomas_c4_2014_2016_2020_2024_250m", RES_250M, nx=3, ny=2)


if __name__ == "__main__":
    main()
