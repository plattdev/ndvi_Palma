# LIBRARIES
import rasterio as ras # To handle raster data (satellite images)
import geopandas as gpd # To handle geographic vector data (shapefiles or geojson) of Palma
import numpy as np # To perform numerical operations on the raster data, such as calculating the NDVI
import matplotlib.pyplot as plt # To visualize the NDVI results

#DATA ADQUISITION https://earthexplorer.usgs.gov
satellite_image_path = '/data/raw/landsatSummer.tif'
satellite_image = ras.open(satellite_image_path)

