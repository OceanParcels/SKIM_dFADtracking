from parcels import FieldSet
import numpy as np
import xarray
from progressbar import ProgressBar


def set_skimulator_fieldset():
    filenames = '/Users/erik/Desktop/scisoc_trpac?.nc'
    variables = {'U': 'Ux_skim', 'V': 'Uy_skim'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    return FieldSet.from_netcdf(filenames, variables, dimensions)


def set_cmems_50m_fieldset():
    fnames = '/Users/erik/Desktop/CMEMSdownload/cmems2018*'
    filenames = {'U': fnames + 'uo.nc', 'V': fnames + 'vo.nc'}
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time', 'depth': 'depth'}
    indices = {'depth': [17]}
    return FieldSet.from_netcdf(filenames, variables, dimensions, indices)


def isocean(u, v):
    return True if u == 0 and v == 0 else False


def landborders(u, v, J, I, nx):
    mask = np.ones((3, 3), dtype=bool)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            mask[j+1, i+1] = isocean(u[j+J, (i+I) % nx], v[ j+J, (i+I) % nx])
    return mask


fset = set_cmems_50m_fieldset()
fset.computeTimeChunk(0, 1)
nx = fset.U.lon.size
ny = fset.U.lat.size

mask_uvel = np.zeros((ny, nx))
mask_vvel = np.zeros((ny, nx))
pbar = ProgressBar()
for i in pbar(range(0, nx)):
    for j in range(1, ny-1):
        if isocean(fset.U.data[0, j, i], fset.V.data[0, j, i]):
            mask = landborders(fset.U.data[0, :, :], fset.V.data[0, :, :], j, i, nx)
            if not mask.all():
                mask_uvel[j, i] = sum(mask[:, 2]) - sum(mask[:, 0])
                mask_vvel[j, i] = sum(mask[2, :]) - sum(mask[0, :])

dir = ''
coords = [('Latitude', fset.U.lat), ('Longitude', fset.U.lon)]
uvel = xarray.DataArray(mask_uvel, coords=coords)
vvel = xarray.DataArray(mask_vvel, coords=coords)
dcoo = {'Longitude':  fset.U.lon, 'Latitude': fset.U.lat}
dset = xarray.Dataset({'MaskUvel': uvel, 'MaskVvel': vvel}, coords=dcoo)
dset.to_netcdf(dir+"cmems_boundaryvelocities.nc")
