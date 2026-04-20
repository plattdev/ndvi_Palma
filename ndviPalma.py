# LIBRARIES
import rasterio as ras # To handle raster data (satellite images)
import geopandas as gpd # To handle geographic vector data (shapefiles or geojson) of Palma
import numpy as np # To perform numerical operations on the raster data, such as calculating the NDVI
import matplotlib.pyplot as plt # To visualize the NDVI results

#DATA ADQUISITION
# Load the satellite image (assuming it's a multi-band image with Red and NIR bands)