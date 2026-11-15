# Raw Satellite Data Directory (`data/raw/`)

Raw satellite GeoTIFF tiles are excluded from Git tracking via `.gitignore` to keep the repository lightweight.
To reproduce all raw GeoTIFFs in `data/raw/viirs_oct2026/` and `data/raw/mcd64a1_2001_2025/`, configure your Earth Engine credentials (see `.env.example`) and run:

```bash
python scripts/02_fetch_gee_datasets.py
```

### Datasets Retrieved
1. **`NASA/LANCE/SNPP_VIIRS/C2` & `NASA/LANCE/NOAA20_VIIRS/C2`** (`375 m`, Oct 1–31, 2026): Active fire detections (`Bright_ti4`, `confidence`, `frp`).
2. **`FIRMS`** (`1 km` MODIS Terra/Aqua, Oct 1–31, 2026): Cross-sensor active fire verification.
3. **`MODIS/061/MCD64A1`** (`500 m`, Jan 2001 – Sep 2026): Monthly global spectral Burned Area product (`BurnDate > 0`), used for the 25-year Kalimantan fire regime and 2015 / 2019 mega-fire burn scars.
4. **`MapBiomas Indonesia Collection 4 v2`** (`250 m` subset over Kalimantan, epochs `2014`, `2016`, `2020`, `2024`): Pre-fire and multi-step post-fire land-cover transitions.
5. **`KLHK Penutupan Lahan 2025`** (`~28 m`, 68 Kalimantan tiles): Official Ministry of Forestry 23-class 1:50,000 land-cover map (`KLHK_PL2025_TILES_DIR`).
