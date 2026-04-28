# NDVI Analysis — Mallorca (Summer vs Winter)

## Project Overview

This project analyzes vegetation health on Mallorca by comparing NDVI (Normalized Difference Vegetation Index) between summer and winter using Landsat 8 Collection 2 Level-2 satellite imagery. It automates the complete geospatial workflow—mosaicking, clipping, calculating, classifying, visualizing, and exporting NDVI data for the island—providing clear, reproducible outputs for environmental monitoring, land management, and research.

## Why This Matters

2026 has been marked as a particularly wet year across the Mediterranean region, with above-average precipitation during winter months. This project captures how that moisture translates into vegetation patterns. The **summer vs. winter NDVI difference** reveals:

- **Seasonal Vegetation Dynamics**: Areas that green-up dramatically in summer show high photosynthetic activity when conditions are optimal (warm + available moisture from winter rains)
- **Water Stress Indicators**: Regions with minimal seasonal variation may indicate chronic water scarcity or Mediterranean scrubland that doesn't respond strongly to seasonal rainfall
- **Agricultural Productivity**: Cropland and managed forests show pronounced seasonal swings; natural vegetation is more stable
- **Urbanization & Land Use**: Built-up areas consistently show low NDVI regardless of season
- **Climate Resilience**: In a wet year like 2026, we see how vegetation responds to abundance—providing a baseline to compare against drought years

The difference maps highlight which ecosystems are most responsive to seasonal water availability, crucial for understanding climate impacts and planning water management.

## Data Source

- **Satellite**: Landsat 8 Collection 2 Level-2 Surface Reflectance
- **Bands Used**: Red (B4) and Near-Infrared (B5)
- **Source**: [USGS EarthExplorer](https://earthexplorer.usgs.gov)
- **Study Area**: Mallorca administrative boundaries (municipis)

## Outputs

The script generates five key outputs in `data/output/`:

| File                  | Description                                                                             |
| --------------------- | --------------------------------------------------------------------------------------- |
| `ndvi_summer.tif`     | Continuous NDVI raster for summer (GeoTIFF, float32)                                    |
| `ndvi_winter.tif`     | Continuous NDVI raster for winter (GeoTIFF, float32)                                    |
| `ndvi_difference.tif` | Seasonal difference map (Summer − Winter)                                               |
| `ndvi_comparison.png` | Side-by-side visualization: Summer, Winter, Difference (high-res)                       |
| `ndvi_classified.png` | Classified vegetation categories (5 classes)                                            |
| `ndvi_mallorca.gpkg`  | GeoPackage with 3 layers: summer classified, winter classified, municipality boundaries |

## Classification Schema

NDVI values are binned into five vegetation categories:

| Class         | NDVI Range  | Color        | Meaning                              |
| ------------- | ----------- | ------------ | ------------------------------------ |
| Water/Barren  | −1.0 to 0.0 | Blue         | Water bodies, bare rock, urban       |
| Sparse/Urban  | 0.0 to 0.2  | Light Yellow | Sparse vegetation, built areas       |
| Moderate Veg. | 0.2 to 0.4  | Light Green  | Grassland, shrubland                 |
| Healthy Veg.  | 0.4 to 0.6  | Green        | Forests, dense crops                 |
| Dense Veg.    | 0.6 to 1.0  | Dark Green   | Very healthy forests, riparian zones |

## Installation & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Prepare Data

1. Download summer and winter Landsat 8 C2 L2 scenes from [USGS EarthExplorer](https://earthexplorer.usgs.gov)
   - Search for Mallorca (coordinates ~39.5°N, 3°E)
   - Select scenes from May–October (summer) and November–March (winter)
   - Download the `_SR_B4.TIF` (Red) and `_SR_B5.TIF` (NIR) bands

2. Place all band files in `data/raw/`

3. Ensure `data/raw/municipis.json` contains the island's administrative boundaries (GeoJSON)

### Run the Analysis

```bash
python ndviPalma.py
```

The script will:

- Auto-detect band files and split them by season
- Load and align CRS of boundaries and rasters
- Mosaic and clip bands to the island outline
- Compute NDVI for each season
- Calculate the seasonal difference
- Generate classified maps and visualizations
- Export rasters (GeoTIFF) and vectors (GeoPackage)

**Example repo structure for portfolio:**

```
NDVIpalma/
├── README.md (this file)
├── ndviPalma.py
├── requirements.txt
├── data/
│   ├── raw/
│   └── output/
├── notebooks/
│   └── ndvi_analysis_explained.ipynb
├── docs/
│   └── portfolio_visualization.html
└── LICENSE
```

- **Technical Skills**: Remote sensing, geospatial analysis, raster/vector data handling
- **Domain Knowledge**: NDVI calculation, land use classification, seasonal vegetation dynamics
- **Data Visualization**: Multi-panel plots, classified maps, difference maps
- **Reproducibility**: Automated end-to-end workflow, clear documentation
- **Business Value**: Environmental monitoring, land management insights, climate resilience assessment
