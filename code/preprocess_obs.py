#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri Mar 14 09:01:04 2025

@author: wyan0065
'''

import numpy as np
import xarray as xr
import geopandas as gpd
from scipy.interpolate import NearestNDInterpolator

import rioxarray
from rasterio.features import geometry_mask
import rasterio.transform

work_dir = '/Users/wyan0065/Desktop/icepack-master/Heard_Island/Heard_Island/'

bathymetry_lagoon = rioxarray.open_rasterio(work_dir + 'Bathymetry_Lagoons.tif', masked=True)
bathymetry_sexton = rioxarray.open_rasterio(work_dir + 'Bathymetry_Sexton.tif', masked=True)
dem               = rioxarray.open_rasterio(work_dir + 'Pleiades_2019.tif', masked=True)
thk_farinotti     = rioxarray.open_rasterio(work_dir + 'Farinotti_Icethickness.tif', masked=True)
thk_millan        = rioxarray.open_rasterio(work_dir + 'Millan_Icethickness.tif', masked=True)
vx                = rioxarray.open_rasterio(work_dir + 'Millan_VX.tif', masked=True)
vy                = rioxarray.open_rasterio(work_dir + 'Millan_VY.tif', masked=True)

heard_island = gpd.read_file(work_dir + 'Heard_Island.shp')
glacier_2019 = gpd.read_file(work_dir + 'Glacier_2019.shp')
lagoon_2019 = gpd.read_file(work_dir + 'Lagoons_2019.shp')

bathymetry_lagoon = bathymetry_lagoon.squeeze()
bathymetry_sexton = bathymetry_sexton.squeeze()
dem               = dem.squeeze()
thk_farinotti     = thk_farinotti.squeeze()
thk_millan        = thk_millan.squeeze()
vx                = vx.squeeze()
vy                = vy.squeeze()

bathymetry_lagoon.values = np.where(bathymetry_lagoon.values < -3.4028235e+30, 0, bathymetry_lagoon.values)
bathymetry_sexton.values = np.where(bathymetry_sexton.values == -32768, 0, bathymetry_sexton.values)
dem.values               = np.where(dem.values < -3.4028235e+30, 0, dem.values)
thk_farinotti.values     = np.where(thk_farinotti.values < -3.4028235e+30, np.nan, thk_farinotti.values)
thk_millan.values        = np.where(thk_millan.values < -3.4028235e+30, np.nan, thk_millan.values)
vx.values                = np.where(vx.values < -3.4028235e+30, np.nan, vx.values)
vy.values                = np.where(vy.values < -3.4028235e+30, np.nan, vy.values)

# thk_farinotti
mask = geometry_mask(
    [geom for geom in glacier_2019.geometry],
    transform=thk_farinotti.rio.transform(),
    invert=True,
    out_shape=thk_farinotti.shape
)
thk_farinotti.values[~mask] = 0
thk_farinotti.values =np.where(thk_farinotti.values > np.nanpercentile(thk_farinotti.values, 99.5), np.nan, thk_farinotti.values)

data = thk_farinotti.values.copy()
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
mask = ~np.isnan(data)
interp = NearestNDInterpolator((x[mask], y[mask]), data[mask])
data_filled = interp(x, y)
thk_farinotti.values = data_filled

# thk_millan
mask = geometry_mask(
    [geom for geom in glacier_2019.geometry],
    transform=thk_millan.rio.transform(),
    invert=True,
    out_shape=thk_millan.shape
)
thk_millan.values[~mask] = 0
thk_millan.values =np.where(thk_millan.values > np.nanpercentile(thk_millan.values, 99.5), np.nan, thk_millan.values)

data = thk_millan.values.copy()
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
mask = ~np.isnan(data)
interp = NearestNDInterpolator((x[mask], y[mask]), data[mask])
data_filled = interp(x, y)
thk_millan.values = data_filled

# vx_millan
mask = geometry_mask(
    [geom for geom in glacier_2019.geometry],
    transform=vx.rio.transform(),
    invert=True,
    out_shape=vx.shape
)
vx.values[~mask] = 0
vx.values =np.where((vx.values > np.nanpercentile(vx.values, 99.5))+(vx.values < np.nanpercentile(vx.values, 0.5))==1, np.nan, vx.values)

data = vx.values.copy()
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
mask = ~np.isnan(data)
interp = NearestNDInterpolator((x[mask], y[mask]), data[mask])
data_filled = interp(x, y)
vx.values = data_filled

# vy_millan
mask = geometry_mask(
    [geom for geom in glacier_2019.geometry],
    transform=vy.rio.transform(),
    invert=True,
    out_shape=vy.shape
)
vy.values[~mask] = 0
vy.values =np.where((vy.values > np.nanpercentile(vy.values, 99.5))+(vy.values < np.nanpercentile(vy.values, 0.5))==1, np.nan, vy.values)

data = vy.values.copy()
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
mask = ~np.isnan(data)
interp = NearestNDInterpolator((x[mask], y[mask]), data[mask])
data_filled = interp(x, y)
vy.values = data_filled

longitude = np.arange(-18000,18050,50);
latitude  = np.arange(18000,-18050,-50);
transform = rasterio.transform.from_origin(
    longitude.min(),
    latitude.max(),
    (longitude[1] - longitude[0]),
    (latitude[0] - latitude[1])
)

grid      = {'y': latitude, 'x': longitude};
longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)

island_mask  = geometry_mask(heard_island.geometry, transform=transform, invert=True, out_shape=(len(latitude), len(longitude)))
glacier_mask = geometry_mask(glacier_2019.geometry, transform=transform, invert=True, out_shape=(len(latitude), len(longitude)))
lagoon_mask  = geometry_mask(lagoon_2019.geometry, transform=transform, invert=True, out_shape=(len(latitude), len(longitude)))

#%%

ds = xr.Dataset(coords={'y': latitude, 'x': longitude})

ds.attrs['projection'] = bathymetry_lagoon.rio.crs.to_string()

ds['x'].attrs['long_name'] = 'Longitude'
ds['x'].attrs['units'] = 'm'
ds['x'].attrs['standard_name'] = 'x coordinate of projection'

ds['y'].attrs['long_name'] = 'Latitude'
ds['y'].attrs['units'] = 'm'
ds['y'].attrs['standard_name'] = 'y coordinate of projection'

ds['bathymetry_lagoon'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['bathymetry_lagoon'].attrs['description'] = 'Lagoon depth data in 2003 (Donoghue, 2009)'
ds['bathymetry_lagoon'].attrs['units'] = 'm'

ds['bathymetry_sexton'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['bathymetry_sexton'].attrs['description'] = 'Ocean depth data in 2005 (Sexton et al., 2007)'
ds['bathymetry_sexton'].attrs['units'] = 'm'

ds['icethickness_farinotti'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['icethickness_farinotti'].attrs['description'] = 'Ice thickness data in RGI dates (Farinotti et al., 2019)'
ds['icethickness_farinotti'].attrs['units'] = 'm'

ds['icethickness_millan'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['icethickness_millan'].attrs['description'] = 'Ice thickness data during 2017-2018 (Millan et al., 2022)'
ds['icethickness_millan'].attrs['units'] = 'm'

ds['vx_millan'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['vx_millan'].attrs['description'] = 'Ice velocity in map x direction during 2017-2018 (Millan et al., 2022)'
ds['vx_millan'].attrs['units'] = 'm/year'

ds['vy_millan'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['vy_millan'].attrs['description'] = 'Ice velocity in map y direction during 2017-2018 (Millan et al., 2022)'
ds['vy_millan'].attrs['units'] = 'm/years'

ds['dem'] = (['y', 'x'], np.full((len(latitude), len(longitude)), np.nan))
ds['dem'].attrs['description'] = 'Ice surface elevation in 2019 (Pléiades images from LEGOS)'
ds['dem'].attrs['units'] = 'm'

#%%

ds['bathymetry_lagoon'].values = bathymetry_lagoon.interp(grid, method='nearest')
ds['bathymetry_lagoon'].values = np.where(np.isnan(ds['bathymetry_lagoon'].values), 0, ds['bathymetry_lagoon'].values)
ds['bathymetry_lagoon'].values = np.where(ds['bathymetry_lagoon'].values>0, 0, ds['bathymetry_lagoon'].values)
ds['bathymetry_lagoon'].values[~lagoon_mask] = 0

ds['bathymetry_sexton'].values = bathymetry_sexton.interp(grid, method='nearest')
ds['bathymetry_sexton'].values = np.where(ds['bathymetry_sexton'].values>0, 0, ds['bathymetry_sexton'].values)
ds['bathymetry_sexton'].values[island_mask] = 0

ds['dem'].values = dem.interp(grid, method='nearest')
ds['dem'].values = np.where(np.isnan(ds['dem'].values), 0, ds['dem'].values)
ds['dem'].values[~island_mask] = 0
ds['dem'].values[lagoon_mask]  = 0

ds['icethickness_farinotti'].values = thk_farinotti.interp(grid, method='nearest')
ds['icethickness_farinotti'].values = np.where(np.isnan(ds['icethickness_farinotti'].values), 0, ds['icethickness_farinotti'].values)
ds['icethickness_farinotti'].values = np.where(ds['icethickness_farinotti'].values<0, 0, ds['icethickness_farinotti'].values)
ds['icethickness_farinotti'].values[~glacier_mask] = 0

ds['icethickness_millan'].values = thk_millan.interp(grid, method='nearest')
ds['icethickness_millan'].values = np.where(np.isnan(ds['icethickness_millan'].values), 0, ds['icethickness_millan'].values)
ds['icethickness_millan'].values = np.where(ds['icethickness_millan'].values<0, 0, ds['icethickness_millan'].values)
ds['icethickness_millan'].values[~glacier_mask] = 0

ds['vx_millan'].values = vx.interp(grid, method='nearest')
ds['vx_millan'].values = np.where(np.isnan(ds['vx_millan'].values), 0, ds['vx_millan'].values)
ds['vx_millan'].values[~glacier_mask] = 0

ds['vy_millan'].values = vy.interp(grid, method='nearest')
ds['vy_millan'].values = np.where(np.isnan(ds['vy_millan'].values), 0, ds['vy_millan'].values)
ds['vy_millan'].values[~glacier_mask] = 0

ds.to_netcdf(work_dir + 'heard_island_obs.nc');