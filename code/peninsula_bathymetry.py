import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import xarray as xr

center = (0, 0)
outer_radius  = 16e3
inner_radius  = 5e3
num_points = 1000

theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)

outer_points = [Point(center[0] + outer_radius*np.cos(t),
                      center[1] + outer_radius*np.sin(t))
                for t in theta]

inner_points = [Point(center[0] + inner_radius*np.cos(t),
                      center[1] + inner_radius*np.sin(t))
                for t in theta]

outer_values = []
for pt in outer_points:
    if pt.x > 0:
        value = -200 * (pt.x / outer_radius)
    else:
        value = 0.0
    outer_values.append(value)

#outer_values = [-200] * num_points
inner_values = [0.0] * num_points

# Combine all points and values
all_points = np.array([[pt.x, pt.y] for pt in (inner_points + outer_points)])
all_values = np.array(inner_values + outer_values)

grid_res = 200
x_lin = np.linspace(-outer_radius, outer_radius, grid_res)
y_lin = np.linspace(-outer_radius, outer_radius, grid_res)
grid_x, grid_y = np.meshgrid(x_lin, y_lin)
grid_points = np.vstack((grid_x.ravel(), grid_y.ravel())).T

# Perform single interpolation using all points
grid_z = griddata(all_points, all_values, grid_points, method='linear')
grid_z = grid_z.reshape(grid_x.shape)

r_grid = np.sqrt(grid_x**2 + grid_y**2)
grid_z[r_grid < inner_radius] = 0

plt.figure(figsize=(6,5))
plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
plt.colorbar(label='Value')
#plt.scatter(all_points[:,0], all_points[:,1], c=all_values, edgecolor='k')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

ds = xr.Dataset(
    {'bathymetry': (('y', 'x'), grid_z)
    },coords={'x': x_lin,'y': y_lin})

#output_filename = '/Users/wyan0065/Desktop/icepack-master/Heard_Island/codes/island_bathymetry.nc'
output_filename = '/Users/wyan0065/Desktop/icepack-master/Heard_Island/codes/peninsula_bathymetry.nc'
ds.to_netcdf(output_filename)
