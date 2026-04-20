# NDVI Analysis — Mallorca (Summer vs Winter)
# Compares vegetation health between seasons using Landsat 8 C2 L2 surface reflectance.
# Place *_SR_B4.TIF (Red) and *_SR_B5.TIF (NIR) band files in data/raw/.

import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import rasterio
import rasterio.mask
from rasterio.features import shapes
from rasterio.merge import merge
from rasterio.io import MemoryFile
import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union, polygonize

# Suppress noisy rasterio/GDAL warnings (CRS comparisons, etc.)
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_OUT = os.path.join(BASE_DIR, 'data', 'output')
os.makedirs(DATA_OUT, exist_ok=True)

MUNICIPIS_JSON = os.path.join(DATA_RAW, 'municipis.json')

# Landsat C2 L2: convert uint16 DN to surface reflectance
L2_SCALE  =  0.0000275
L2_OFFSET = -0.2

# NDVI classification bins and labels (from water/barren to dense vegetation)
CLASS_BINS   = [-1.0, 0.0, 0.2, 0.4, 0.6, 1.01]
CLASS_LABELS = ['Water/Barren', 'Sparse/Urban', 'Moderate Veg.', 'Healthy Veg.', 'Dense Veg.']
CLASS_COLORS = ['#4575b4', '#ffffbf', '#a6d96a', '#66bd63', '#1a7837']


# --- Helper functions ---

def auto_detect_scenes(raw_dir):
    """Scan raw_dir for SR_B4/B5 files and split them into winter/summer by acquisition month."""
    b4_files = sorted(
        glob.glob(os.path.join(raw_dir, '*_SR_B4.TIF'))
        + glob.glob(os.path.join(raw_dir, '*_SR_B4.tif'))
    )
    if len(b4_files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 *_SR_B4.TIF files in {raw_dir}, found {len(b4_files)}."
        )

    # Landsat filename field 3 (0-indexed) is YYYYMMDD
    def acq_date(path):
        parts = os.path.basename(path).split('_')
        return parts[3] if len(parts) > 3 else ''

    summer_b4, summer_b5 = [], []
    winter_b4, winter_b5 = [], []
    winter_date, summer_date = None, None

    for path in b4_files:
        date_str = acq_date(path)
        if len(date_str) < 6:
            continue
        try:
            month = int(date_str[4:6])
        except ValueError:
            continue

        # Derive the NIR band path from the Red band path
        b5_path = path.replace('_SR_B4', '_SR_B5')
        is_summer = 5 <= month <= 10

        if is_summer:
            summer_b4.append(path)
            summer_b5.append(b5_path)
            if summer_date is None:
                summer_date = date_str
        else:
            winter_b4.append(path)
            winter_b5.append(b5_path)
            if winter_date is None:
                winter_date = date_str

    return winter_date, summer_date, summer_b4, summer_b5, winter_b4, winter_b5


def mosaic_and_clip_band(tif_paths, clip_geojson):
    """Merge multiple single-band TIFs into one mosaic and clip to the island outline."""
    datasets = [rasterio.open(p) for p in tif_paths]
    try:
        mosaic_arr, mosaic_transform = merge(datasets)
        src_crs = datasets[0].crs
        src_dtype = datasets[0].dtypes[0]

        # rasterio.mask needs a file-like source, so write mosaic to a MemoryFile
        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic_arr.shape[1],
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,
            count=1,
        )
        with MemoryFile() as memfile:
            with memfile.open(**profile) as mem:
                mem.write(mosaic_arr)
                clipped, clipped_transform = rasterio.mask.mask(
                    mem, clip_geojson, crop=True, nodata=0
                )
    finally:
        for ds in datasets:
            ds.close()

    band = clipped[0].astype(np.float32)
    # Apply Landsat C2 L2 scale factor if the source is raw uint16
    if src_dtype == 'uint16':
        band = band * L2_SCALE + L2_OFFSET
    return band, clipped_transform, src_crs


def compute_ndvi(red_paths, nir_paths, clip_geojson, label):
    """Load Red and NIR bands, clip to the island, and return NDVI array."""
    print(f"   [{label}] Processing {len(red_paths)} tile(s)...")
    red, transform, crs = mosaic_and_clip_band(red_paths, clip_geojson)
    nir, _, _ = mosaic_and_clip_band(nir_paths, clip_geojson)

    # Mask out invalid pixels (zero or negative reflectance)
    invalid = (red <= 0) | (nir <= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(invalid, np.nan, (nir - red) / (nir + red))
    ndvi = np.clip(ndvi, -1.0, 1.0)

    print(f"     mean={np.nanmean(ndvi):.3f}  std={np.nanstd(ndvi):.3f}  "
          f"valid_px={int(np.sum(~np.isnan(ndvi))):,}")
    return ndvi, transform, crs


def classify_ndvi(ndvi_arr):
    """Assign each pixel to a vegetation class based on CLASS_BINS thresholds."""
    classified = np.full(ndvi_arr.shape, np.nan)
    for i in range(len(CLASS_BINS) - 1):
        mask = (~np.isnan(ndvi_arr)) & (ndvi_arr >= CLASS_BINS[i]) & (ndvi_arr < CLASS_BINS[i + 1])
        classified[mask] = i
    return classified


def save_raster(array, transform, crs, path, nodata=-9999.0):
    """Write a float32 array to a compressed GeoTIFF."""
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'height': array.shape[0], 'width': array.shape[1],
        'crs': crs, 'transform': transform,
        'nodata': nodata, 'compress': 'lzw',
    }
    out = np.where(np.isnan(array), nodata, array).astype('float32')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(out, 1)
    print(f"   Saved: {os.path.basename(path)}")


def vectorize_ndvi_classes(ndvi_arr, transform, crs):
    """Convert classified NDVI raster to polygons (GeoDataFrame)."""
    # Classify into uint8 (1-based; 0 = nodata)
    classified = np.zeros(ndvi_arr.shape, dtype=np.uint8)
    for i in range(len(CLASS_BINS) - 1):
        mask = (~np.isnan(ndvi_arr)) & (ndvi_arr >= CLASS_BINS[i]) & (ndvi_arr < CLASS_BINS[i + 1])
        classified[mask] = i + 1

    records = []
    for geom, val in shapes(classified, mask=(classified > 0), transform=transform):
        idx = int(val) - 1
        if 0 <= idx < len(CLASS_LABELS):
            records.append({
                'geometry': shapely_shape(geom),
                'class_id': int(val),
                'class_label': CLASS_LABELS[idx],
            })
    if not records:
        return None
    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf[gdf.geometry.is_valid].copy()


# === STEP 1: Detect Landsat band files ===
print("─" * 60)
print("1. Detecting band files in data/raw/ ...")
WINTER_DATE, SUMMER_DATE, SUMMER_B4, SUMMER_B5, WINTER_B4, WINTER_B5 = auto_detect_scenes(DATA_RAW)
print(f"   Winter tiles: {len(WINTER_B4)}  |  Summer tiles: {len(SUMMER_B4)}")


# === STEP 2: Load municipality boundaries ===
print("\n2. Loading municipality boundaries...")
municipis = gpd.read_file(MUNICIPIS_JSON)
print(f"   CRS: {municipis.crs}  |  Features: {len(municipis)}")


# === STEP 3: Reproject boundaries to match the raster CRS ===
print("\n3. Aligning CRS...")
with rasterio.open(SUMMER_B4[0]) as src:
    raster_crs = src.crs

if municipis.crs is None:
    municipis = municipis.set_crs("EPSG:4326")

if municipis.crs != raster_crs:
    municipis = municipis.to_crs(raster_crs)
    print(f"   Reprojected municipis → {raster_crs}")
else:
    print(f"   CRS already matches ({raster_crs})")


# === STEP 4: Build island mask polygon from municipality geometries ===
print("\n4. Building island mask polygon...")
geom_types = municipis.geometry.geom_type.unique()

if any('Line' in g for g in geom_types):
    # Municipality data is line boundaries — convert to polygons
    merged = unary_union(municipis.geometry)
    polys = list(polygonize(merged))
    island_geom = unary_union(polys) if polys else merged.convex_hull
else:
    # Municipality data is already polygons — dissolve into one shape
    island_geom = unary_union(municipis.geometry)

island_geojson = [island_geom.__geo_interface__]
print(f"   Island mask ready ({island_geom.geom_type})")


# === STEP 5: Compute NDVI for each season ===
print("\n5. Computing NDVI...")
ndvi_summer, t_summer, crs_out = compute_ndvi(SUMMER_B4, SUMMER_B5, island_geojson, 'Summer')
ndvi_winter, t_winter, _ = compute_ndvi(WINTER_B4, WINTER_B5, island_geojson, 'Winter')


# === STEP 6: Seasonal difference (Summer − Winter) ===
print("\n6. Computing NDVI difference (Summer − Winter)...")
# Trim to matching dimensions in case mosaics differ slightly
rows = min(ndvi_summer.shape[0], ndvi_winter.shape[0])
cols = min(ndvi_summer.shape[1], ndvi_winter.shape[1])
ndvi_diff = ndvi_summer[:rows, :cols] - ndvi_winter[:rows, :cols]
print(f"   mean={np.nanmean(ndvi_diff):.3f}  std={np.nanstd(ndvi_diff):.3f}  "
      f"greener_in_summer={int(np.nansum(ndvi_diff > 0)):,} px")


# === STEP 7: Classify NDVI into vegetation categories ===
cls_summer = classify_ndvi(ndvi_summer)
cls_winter = classify_ndvi(ndvi_winter)


# === STEP 8: Plot continuous NDVI maps (summer, winter, diff) ===
print("\n7. Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(22, 8))
fig.suptitle('NDVI Analysis — Mallorca | Landsat 8 C2 L2', fontsize=16, fontweight='bold')

plot_specs = [
    (ndvi_summer[:rows, :cols], f'Summer NDVI\n({SUMMER_DATE})', plt.cm.RdYlGn, -0.2, 0.8, 'NDVI'),
    (ndvi_winter[:rows, :cols], f'Winter NDVI\n({WINTER_DATE})', plt.cm.RdYlGn, -0.2, 0.8, 'NDVI'),
    (ndvi_diff, 'Difference\n(Summer − Winter)', plt.cm.RdBu, -0.4, 0.4, 'ΔNDVI'),
]
for ax, (data, title, cmap, vmin, vmax, cbar_label) in zip(axes, plot_specs):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_title(title, fontsize=13, pad=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label, shrink=0.85)

plt.tight_layout()
fig.savefig(os.path.join(DATA_OUT, 'ndvi_comparison.png'), dpi=200, bbox_inches='tight')
print("   Saved: ndvi_comparison.png")


# === STEP 9: Plot classified NDVI maps ===
cmap_cls = mcolors.ListedColormap(CLASS_COLORS)
norm_cls = mcolors.BoundaryNorm(range(len(CLASS_LABELS) + 1), cmap_cls.N)
legend_patches = [Patch(facecolor=c, label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle('NDVI Classification — Mallorca', fontsize=16, fontweight='bold')

for ax, data, title in zip(
    axes2,
    [cls_summer[:rows, :cols], cls_winter[:rows, :cols]],
    [f'Summer ({SUMMER_DATE})', f'Winter ({WINTER_DATE})'],
):
    ax.imshow(data, cmap=cmap_cls, norm=norm_cls, interpolation='nearest')
    ax.set_title(title, fontsize=13)
    ax.axis('off')
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
              framealpha=0.9, title='NDVI Class')

plt.tight_layout()
fig2.savefig(os.path.join(DATA_OUT, 'ndvi_classified.png'), dpi=200, bbox_inches='tight')
print("   Saved: ndvi_classified.png")


# === STEP 10: Export NDVI rasters as GeoTIFF ===
print("\n8. Exporting GeoTIFF rasters...")
save_raster(ndvi_summer, t_summer, crs_out, os.path.join(DATA_OUT, 'ndvi_summer.tif'))
save_raster(ndvi_winter, t_winter, crs_out, os.path.join(DATA_OUT, 'ndvi_winter.tif'))
save_raster(ndvi_diff, t_summer, crs_out, os.path.join(DATA_OUT, 'ndvi_difference.tif'))


# === STEP 11: Vectorize classified NDVI and export as GeoPackage ===
print("\n9. Vectorizing to GeoPackage...")
gpkg_path = os.path.join(DATA_OUT, 'ndvi_mallorca.gpkg')

gdf_summer = vectorize_ndvi_classes(ndvi_summer, t_summer, crs_out)
gdf_winter = vectorize_ndvi_classes(ndvi_winter, t_winter, crs_out)

if gdf_summer is not None:
    gdf_summer.to_file(gpkg_path, layer='ndvi_summer_classified', driver='GPKG')
    print(f"   Layer: ndvi_summer_classified  ({len(gdf_summer):,} polygons)")
if gdf_winter is not None:
    gdf_winter.to_file(gpkg_path, layer='ndvi_winter_classified', driver='GPKG')
    print(f"   Layer: ndvi_winter_classified  ({len(gdf_winter):,} polygons)")

municipis.to_file(gpkg_path, layer='municipis_boundary', driver='GPKG')
print("   Layer: municipis_boundary")

print(f"\n{'─' * 60}")
print(f"All outputs saved to: {DATA_OUT}")
print("Done!")

