# NDVI Analysis — Mallorca (Summer 2025 vs Winter 2026)
# Compares vegetation changes between seasons using Landsat 8 C2 L2 surface reflectance from USGS Earth Explorer (website: https://earthexplorer.usgs.gov/): *_SR_B4.TIF (Red) and *_SR_B5.TIF (NIR) band files in data/raw/

import os # File system operations (paths, directories)
import glob # File pattern matching (find all .TIF files in a directory, because Landsat data comes in multiple tiles)

import warnings # Suppress non-critical warnings that clutter console output

import numpy as np # NumPy: Numerical array operations (handle satellite imagery as arrays of numbers): Satellite bands are rasters = 2D arrays; NumPy lets us do math on them efficiently

import matplotlib.pyplot as plt # Visualizations (maps, charts, legends).We need to visualize NDVI results as colored maps for analysis and reporting
import matplotlib.colors as mcolors # Matplotlib.colors: Custom color scales (e.g., red-yellow-green for NDVI)
from matplotlib.patches import Patch #Legend elements (colored boxes for vegetation classes)

import rasterio # Read/write geospatial raster data (GeoTIFF, bands, CRS info). Landsat bands are GeoTIFFs; rasterio handles the georeferencing metadata
import rasterio.mask #Clip raster arrays to polygon boundaries (clip to island)
from rasterio.features import shapes # Convert raster pixels to vector polygons -> Turn classified NDVI raster into vector shapes for GeoPackage export
from rasterio.merge import merge # Combine multiple raster tiles into one seamless mosaic -> Multiple Landsat scenes cover the island; merge them into single image
from rasterio.io import MemoryFile # Create temporary in-memory raster files. Why: Faster than disk I/O; used for intermediate processing steps

import geopandas as gpd # For vector geospatial data (shp, GeoJSON, GeoPackage). Municipality are vector polygons; GeoPandas manages them with geometry

from shapely.geometry import shape as shapely_shape # to convert GeoJSON dicts to Shapely geometry objects (used in vectorization step)

from shapely.ops import unary_union, polygonize # to merge multiple municipality polygons into a single island polygon (faster clipping)

################## warnings ##################

# Suppress noisy rasterio/GDAL warnings about CRS comparisons and metadata validation since these can clutter the console; we handle CRS issues explicitly
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')

# ============================================================================
# CONFIGURATION: Paths and Constants
# ============================================================================

# Set up directory structure relative to this script location to be able to run it from any directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw') 
DATA_OUT = os.path.join(BASE_DIR, 'data', 'output') 
os.makedirs(DATA_OUT, exist_ok=True)  # Create output folder if it doesn't exist

# Input file with municipality boundaries (GeoJSON format)
MUNICIPIS_JSON = os.path.join(DATA_RAW, 'municipis.json')

# === Landsat 8 Collection 2 Level-2 Calibration ===
#Landsat provides satellite data as raw digital numbers (DN) in uint16 format (0-65535). We need to convert these to physically meaningful "surface reflectance" values (0-1) to compare vegetation. You find these in the metadata MTL.txt file.
#Formula: reflectance = DN * SCALE + OFFSET
L2_SCALE  = 0.0000275   # Multiplication factor from Landsat metadata
L2_OFFSET = -0.2        # Offset from Landsat metadata

# === NDVI Classification Scheme ===
# Bin continuous NDVI values (-1 to +1) into discrete vegetation categories.
CLASS_BINS = [-1.0, 0.0, 0.2, 0.4, 0.6, 1.01]
# Human-readable names for each vegetation type
CLASS_LABELS = ['Water/Barren', 'Sparse/Urban', 'Moderate Veg', 'Healthy Veg.', 'Dense Veg.']
# Color codes for visualization (Red-Yellow-Green colormap for vegetation)
CLASS_COLORS = ['#4575b4', '#ffffbf', '#a6d96a', '#66bd63', '#1a7837']


# --- Helper functions ---

def auto_detect_scenes(raw_dir):
    """PROBLEM: We have multiple Landsat scenes in raw_dir. How do we know which
    are from summer vs winter?
    
    SOLUTION: Extract the acquisition date (YYYYMMDD) from each filename and
    check the month (characters 4-6 of the date). Then split into two groups."""
    # Find all Red band files (SR_B4) in the directory
    b4_files = sorted(
        glob.glob(os.path.join(raw_dir, '*_SR_B4.TIF'))
        + glob.glob(os.path.join(raw_dir, '*_SR_B4.tif'))
    )
    #At least 1 summer scene + 1 winter scene = 2 minimum
    if len(b4_files) < 2:
        raise FileNotFoundError(
            f"Need at least 2 *_SR_B4.TIF files in {raw_dir}, found {len(b4_files)}."
        )

    # Helper function: Extract YYYYMMDD from a Landsat filename
    # Landsat filenames are structured: LC08_LXTP_PPPRRR_YYYYMMDD_...
    # Field 3 (0-indexed, separated by underscores) is the acquisition date
    def acq_date(path):
        parts = os.path.basename(path).split('_')
        return parts[3] if len(parts) > 3 else ''

    # Separate files into summer (May-Oct) and winter (Nov-Apr) groups
    summer_b4, summer_b5 = [], []
    winter_b4, winter_b5 = [], []
    winter_date, summer_date = None, None

    for path in b4_files:
        date_str = acq_date(path)
        if len(date_str) < 6:
            continue  # Skip if we can't extract a proper date
        try:
            month = int(date_str[4:6])  # Extract month from YYYYMMDD
        except ValueError:
            continue  # Skip if month is not a valid number

        # The NIR band has the same filename but with _SR_B5 instead of _SR_B4
        b5_path = path.replace('_SR_B4', '_SR_B5')
        is_summer = 5 <= month <= 10  # May=5 through October=10

        if is_summer:
            summer_b4.append(path)
            summer_b5.append(b5_path)
            if summer_date is None:
                summer_date = date_str  # Remember the first summer date for reporting
        else:
            winter_b4.append(path)
            winter_b5.append(b5_path)
            if winter_date is None:
                winter_date = date_str  # Remember the first winter date for reporting

    return winter_date, summer_date, summer_b4, summer_b5, winter_b4, winter_b5


def mosaic_and_clip_band(tif_paths, clip_geojson):
    """PROBLEM: We may have multiple Landsat tiles covering the study area. How do we combine them into one image and focus only on the island?
    
    SOLUTION: 1. MOSAIC: Merge overlapping tiles using rasterio.merge (aligns overlaps perfectly)
    2. CLIP: Cut out only the island region using the municipality boundary polygon
    3. SCALE: Convert raw uint16 digital numbers to physical surface reflectance."""
    # Open all band files in the list (e.g., all Red band tiles for summer), p is short for "path"
    datasets = [rasterio.open(p) for p in tif_paths]
    try:
        # STEP 1: Merge overlapping tiles into a single raster
        # The merge() function handles coordinate system alignment automatically
        mosaic_arr, mosaic_transform = merge(datasets)
        src_crs = datasets[0].crs  # Get coordinate system from first file
        src_dtype = datasets[0].dtypes[0]  # Get data type (usually uint16 for Landsat)

        # STEP 2: Prepare the mosaic for clipping
        # rasterio.mask requires a file-like object, so we can't pass the array directly.
        # We write the mosaic to a temporary in-memory file, then clip from that.
        profile = datasets[0].profile.copy()  # Copy metadata (CRS, resolution, etc.)
        profile.update(
            height=mosaic_arr.shape[1],  # Update dimensions to match merged array
            width=mosaic_arr.shape[2],
            transform=mosaic_transform,  # Update geospatial transform
            count=1,  # We have 1 band (either Red or NIR, not multi-band)
        )
        with MemoryFile() as memfile:  # Temporary file in RAM (fast, not on disk)
            with memfile.open(**profile) as mem:
                mem.write(mosaic_arr)  # Write merged array to temp file
                # STEP 3: Clip to island boundary polygon
                # Only keep pixels inside the island_geojson geometry
                clipped, clipped_transform = rasterio.mask.mask(
                    mem, clip_geojson, crop=True, nodata=0
                )
    finally:
        # Always close rasterio datasets, even if an error occurs (prevents file locks)
        for ds in datasets:
            ds.close()

    # STEP 4: Convert to float32 and apply Landsat scale factor
    # Why float32? It's efficient for mathematical operations and can store
    # reflectance values (0-1) with enough precision
    band = clipped[0].astype(np.float32)
    
    # STEP 5: Apply the Landsat Collection 2 Level-2 scale factor. Landsat provides raw digital numbers (uint16, 0-65535). We need to convert to surface reflectance (float, 0-1) for NDVI calculations.
    # Formula: reflectance = DN * L2_SCALE + L2_OFFSET
    if src_dtype == 'uint16':
        band = band * L2_SCALE + L2_OFFSET
    
    return band, clipped_transform, src_crs


def compute_ndvi(red_paths, nir_paths, clip_geojson, label):
    """Use the NDVI formula: NDVI = (NIR - Red) / (NIR + Red)
    - NIR: How much near-infrared light the vegetation reflects (healthy leaves reflect a lot)
    - Red: How much red light the vegetation absorbs (leaves use red for photosynthesis)    
    """
    print(f"   [{label}] Processing {len(red_paths)} tile(s)...")
    
    # Step 1: Load and clip the Red band (Landsat B4)
    red, transform, crs = mosaic_and_clip_band(red_paths, clip_geojson)
    # Step 2: Load and clip the NIR band (Landsat B5)
    # When processing Red: We save transform and crs because we need them later for export
    # When processing NIR: We already have transform and crs from Red, so we only need the nir array. _ ignores the value.
    nir, _, _ = mosaic_and_clip_band(nir_paths, clip_geojson)

    # Step 3: Identify invalid pixels
    # Water and dark surfaces have low Red AND NIR reflectance (negative NDVI isn't meaningful)
    # Clouds also have low values. We mask these out to avoid division errors.
    invalid = (red <= 0) | (nir <= 0)
    
    # Step 4: Calculate NDVI with error handling
    # Why np.errstate? Division by (NIR + Red) could cause warnings for zero values.
    # We handle this by setting those pixels to NaN explicitly.
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where(invalid, np.nan, (nir - red) / (nir + red))
    
    # Step 5: Clip NDVI to valid range
    # Theory: NDVI should be in [-1, +1]. Numerical errors might push values outside.
    ndvi = np.clip(ndvi, -1.0, 1.0)

    # Step 6: Print statistics to console
    # Why? To verify the calculation worked (reasonable mean NDVI ~0.3-0.5 for vegetated areas)
    print(f"     mean={np.nanmean(ndvi):.3f}  std={np.nanstd(ndvi):.3f}  "
          f"valid_px={int(np.sum(~np.isnan(ndvi))):,}")
    
    return ndvi, transform, crs


def classify_ndvi(ndvi_arr):
    """
    PROBLEM: We have continuous NDVI values (-1 to +1). How do we group them
    into meaningful vegetation categories?
    
    SOLUTION: Use predefined CLASS_BINS thresholds to bin the values.
    For example, NDVI < 0 = water/barren, 0-0.2 = sparse vegetation, etc.
    
    Why this matters: Continuous values are hard to interpret visually.
    Classification makes it easy to see at a glance: "this pixel is dense forest"
    versus "this pixel is grassland."
    """
    # Create an empty array with same shape, filled with NaN (missing values)
    classified = np.full(ndvi_arr.shape, np.nan)
    
    # Loop through each class (e.g., class 0 = Water/Barren, class 1 = Sparse/Urban, etc.)
    for i in range(len(CLASS_BINS) - 1):
        # Create a boolean mask: pixels where:
        # - Not NaN (valid data)
        # - NDVI >= lower bin boundary
        # - NDVI < upper bin boundary
        mask = (
            (~np.isnan(ndvi_arr)) & 
            (ndvi_arr >= CLASS_BINS[i]) & 
            (ndvi_arr < CLASS_BINS[i + 1])
        )
        # Assign class index i to all pixels that match this condition
        classified[mask] = i
    
    return classified


def save_raster(array, transform, crs, path, nodata=-9999.0):
    """
    PROBLEM: We have NDVI arrays in NumPy. How do we save them as GeoTIFF files
    that other GIS software can read?
    
    SOLUTION: Use rasterio to write the array with geospatial metadata:
    - Coordinate Reference System (CRS): Where on Earth it is
    - Transform: How pixels map to geographic coordinates
    - Compression: Make files smaller (LZW compression)
    - NoData value: Pixels with this value are "missing" (e.g., ocean, clouds)
    
    Why this matters: GeoTIFF is the standard format for geospatial rasters.
    Other tools (QGIS, ArcGIS, Google Earth Engine) can read and process GeoTIFFs.
    """
    # Set up the GeoTIFF metadata
    profile = {
        'driver': 'GTiff',  # GeoTIFF format
        'dtype': 'float32',  # 32-bit floating point (good for continuous NDVI values)
        'count': 1,  # Single band (not RGB multi-band)
        'height': array.shape[0],  # Rows
        'width': array.shape[1],   # Columns
        'crs': crs,  # Coordinate reference system (e.g., UTM zone 31N)
        'transform': transform,  # Geo-transform (pixel-to-coordinate mapping)
        'nodata': nodata,  # Value representing "no data" (e.g., ocean = -9999)
        'compress': 'lzw',  # LZW compression (lossless, reduces file size ~50%)
    }
    
    # Convert NaN (NumPy's missing value) to our nodata value (-9999)
    # Why? GeoTIFF format doesn't understand NaN; we use -9999 instead.
    out = np.where(np.isnan(array), nodata, array).astype('float32')
    
    # Write the array to a GeoTIFF file with all metadata
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(out, 1)  # Write to band 1 (single band)
    
    print(f"   Saved: {os.path.basename(path)}")


def vectorize_ndvi_classes(ndvi_arr, transform, crs):
    """
    PROBLEM: We have classified NDVI as a raster (grid of pixels).
    How do we convert this to vector polygons for GeoPackage export?
    
    SOLUTION: Use rasterio.features.shapes() to trace polygon boundaries
    around pixels with the same classification value. This creates vector shapes
    that can be queried, styled, and analyzed in GIS software.
    
    Why this matters: Vectors are more efficient for analysis and editing.
    GIS software can calculate polygon areas, overlap with other boundaries, etc.
    """
    # Step 1: Convert NDVI from float (-1 to +1) to classified uint8 (0-5)
    # Why uint8? It's efficient and matches the rasterio.features.shapes() requirement
    classified = np.zeros(ndvi_arr.shape, dtype=np.uint8)
    
    # Assign class IDs (1-based; 0 = nodata/no polygon)
    for i in range(len(CLASS_BINS) - 1):
        mask = (
            (~np.isnan(ndvi_arr)) & 
            (ndvi_arr >= CLASS_BINS[i]) & 
            (ndvi_arr < CLASS_BINS[i + 1])
        )
        classified[mask] = i + 1  # 1-based indexing

    # Step 2: Trace polygon boundaries around each class
    # rasterio.features.shapes() returns (geojson_dict, value) pairs
    records = []
    for geom, val in shapes(classified, mask=(classified > 0), transform=transform):
        # Step 3: Look up the class label for this polygon
        idx = int(val) - 1  # Convert 1-based to 0-based for CLASS_LABELS indexing
        if 0 <= idx < len(CLASS_LABELS):
            records.append({
                'geometry': shapely_shape(geom),  # Convert GeoJSON dict to Shapely geometry
                'class_id': int(val),
                'class_label': CLASS_LABELS[idx],
            })
    
    # Step 4: Return empty if no polygons (shouldn't happen, but defensive)
    if not records:
        return None
    
    # Step 5: Create a GeoDataFrame (like a SQL table with geometry column)
    gdf = gpd.GeoDataFrame(records, crs=crs)
    
    # Step 6: Remove any invalid geometries (self-intersecting, etc.)
    return gdf[gdf.geometry.is_valid].copy()

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================
# The workflow follows a logical sequence: detect data → load boundaries →
# align coordinates → compute NDVI → visualize → export

# === STEP 1: Detect Landsat band files ===
# Why: Automate finding which files are summer vs winter without manual input
print("─" * 60)
print("1. Detecting band files in data/raw/ ...")
WINTER_DATE, SUMMER_DATE, SUMMER_B4, SUMMER_B5, WINTER_B4, WINTER_B5 = auto_detect_scenes(DATA_RAW)
print(f"   Winter tiles: {len(WINTER_B4)}  |  Summer tiles: {len(SUMMER_B4)}")


# === STEP 2: Load municipality boundaries ===
# Why: We need to know which pixels are inside Mallorca vs ocean.
# Mallorca's municipalities polygon is our "study area mask."
print("\n2. Loading municipality boundaries...")
municipis = gpd.read_file(MUNICIPIS_JSON)
print(f"   CRS: {municipis.crs}  |  Features: {len(municipis)}")


# === STEP 3: Reproject boundaries to match the raster CRS ===
# PROBLEM: Municipality boundaries (from GeoJSON) might be in EPSG:4326 (lat/lon).
# Landsat bands are in a projected CRS (e.g., UTM 31N).
# SOLUTION: Check if they match. If not, reproject to the raster's CRS.
# Why? To clip the raster, coordinates MUST match or we get misalignment errors.
print("\n3. Aligning CRS...")
with rasterio.open(SUMMER_B4[0]) as src:
    raster_crs = src.crs

if municipis.crs is None:
    # GeoJSON defaults to EPSG:4326 if no CRS is specified
    municipis = municipis.set_crs("EPSG:4326")

if municipis.crs != raster_crs:
    # Reproject municipality polygons to match raster's coordinate system
    municipis = municipis.to_crs(raster_crs)
    print(f"   Reprojected municipis → {raster_crs}")
else:
    print(f"   CRS already matches ({raster_crs})")


# === STEP 4: Build island mask polygon from municipality geometries ===
# Why: Instead of clipping to 53 separate municipality polygons (slower),
# dissolve them into a single "island" polygon (faster clipping).
print("\n4. Building island mask polygon...")
geom_types = municipis.geometry.geom_type.unique()

if any('Line' in g for g in geom_types):
    # If boundaries are lines (rare), convert them to polygons
    merged = unary_union(municipis.geometry)
    polys = list(polygonize(merged))
    island_geom = unary_union(polys) if polys else merged.convex_hull
else:
    # Most common: municipalities are already polygons
    # Merge them into a single polygon (dissolve operation)
    # unary_union handles overlaps and gaps automatically
    island_geom = unary_union(municipis.geometry)

# Convert Shapely geometry to GeoJSON dict (format rasterio.mask expects)
island_geojson = [island_geom.__geo_interface__]
print(f"   Island mask ready ({island_geom.geom_type})")


# === STEP 5: Compute NDVI for each season ===
# Why: Calculate the vegetation index separately for summer and winter
# to compare seasonal differences
print("\n5. Computing NDVI...")
ndvi_summer, t_summer, crs_out = compute_ndvi(SUMMER_B4, SUMMER_B5, island_geojson, 'Summer')
ndvi_winter, t_winter, _ = compute_ndvi(WINTER_B4, WINTER_B5, island_geojson, 'Winter')


# === STEP 6: Seasonal difference (Summer − Winter) ===
# Why: Highlight areas where vegetation changes dramatically between seasons.
# High values = responsive to seasonal water (agricultural land, deciduous trees).
# Low values = stable year-round (evergreen forests, urban, bare rock).
# This difference is KEY to understanding 2026's wet year impact.
print("\n6. Computing NDVI difference (Summer − Winter)...")
# Trim to matching dimensions in case mosaics differ slightly by 1-2 pixels
rows = min(ndvi_summer.shape[0], ndvi_winter.shape[0])
cols = min(ndvi_summer.shape[1], ndvi_winter.shape[1])
ndvi_diff = ndvi_summer[:rows, :cols] - ndvi_winter[:rows, :cols]
print(f"   mean={np.nanmean(ndvi_diff):.3f}  std={np.nanstd(ndvi_diff):.3f}  "
      f"greener_in_summer={int(np.nansum(ndvi_diff > 0)):,} px")


# === STEP 7: Classify NDVI into vegetation categories ===
# Why: Convert continuous NDVI (-1 to +1) into discrete classes for interpretation
cls_summer = classify_ndvi(ndvi_summer)
cls_winter = classify_ndvi(ndvi_winter)


# === STEP 8: Plot continuous NDVI maps (summer, winter, diff) ===
# Why: Visualize results for presentation and quality checking
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
# Why: Show vegetation categories (water, sparse, healthy, dense) for visual interpretation
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
# Why: GeoTIFF is the standard format for geospatial rasters.
# Other GIS software (QGIS, ArcGIS, R, Google Earth Engine) can read and analyze them.
print("\n8. Exporting GeoTIFF rasters...")
save_raster(ndvi_summer, t_summer, crs_out, os.path.join(DATA_OUT, 'ndvi_summer.tif'))
save_raster(ndvi_winter, t_winter, crs_out, os.path.join(DATA_OUT, 'ndvi_winter.tif'))
save_raster(ndvi_diff, t_summer, crs_out, os.path.join(DATA_OUT, 'ndvi_difference.tif'))


# === STEP 11: Vectorize classified NDVI and export as GeoPackage ===
# Why: Convert raster pixels to vector polygons.
# This allows: querying by attribute, calculating polygon areas, overlay with other boundaries.
# GeoPackage is SQLite-based, portable across all GIS software.
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

# Also export the municipality boundaries for reference
municipis.to_file(gpkg_path, layer='municipis_boundary', driver='GPKG')
print("   Layer: municipis_boundary")

print(f"\n{'─' * 60}")
print(f"All outputs saved to: {DATA_OUT}")
print("Done!")

