from parcels import (FieldSet, AdvectionRK4, BrownianMotion2D, Field,
                     ParticleSet, JITParticle, Variable, ErrorCode)
import numpy as np
from datetime import timedelta as delta
from matplotlib.dates import strpdate2num, num2date
import math


def OutOfBounds(particle, fieldset, time, dt):
    particle.delete()


def DriftTime(particle, fieldset, time, dt):
    particle.drift_time = particle.drift_time + math.fabs(dt)
    if particle.drift_time > fieldset.max_drift_time:
        particle.delete()


def SampleEEZ(particle, fieldset, time, dt):
    plon = particle.lon
    if plon < 100:
        plon = plon + 360
    particle.eez = fieldset.EEZ[0, plon, particle.lat, 0]


def WrapLon(particle, fieldset, time, dt):
    if particle.lon > 180.:
        particle.lon = particle.lon - 360.
    if particle.lon < -180.:
        particle.lon = particle.lon + 360.

def AntiBeaching(particle, fieldset, time, dt):
    dx = fieldset.MaskUvel[0, particle.lon, particle.lat, 0] * dt
    dy = fieldset.MaskVvel[0, particle.lon, particle.lat, 0] * dt
    particle.lon = particle.lon + dx
    particle.lat = particle.lat + dy


def set_cmems_surface_fieldset():
    fnames = '/projects/0/topios/hydrodynamic_data/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/cmems*'
    filenames = {'U': fnames + 'uo.nc', 'V': fnames + 'vo.nc'}
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    return FieldSet.from_netcdf(filenames, variables, dimensions)


def set_cmems_50m_fieldset():
    fnames = '/projects/0/topios/hydrodynamic_data/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/cmems*'
    filenames = {'U': fnames + 'uo.nc', 'V': fnames + 'vo.nc'}
    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time', 'depth': 'depth'}
    indices = {'depth': range(0, 18)}
    fset = FieldSet.from_netcdf(filenames, variables, dimensions, indices)
    fset.add_field(Field.from_netcdf('cmems_boundaryvelocities.nc', 'MaskUvel',
                                     {'lon': 'Longitude', 'lat': 'Latitude'}, fieldtype='U'))
    fset.add_field(Field.from_netcdf('cmems_boundaryvelocities.nc', 'MaskVvel',
                                     {'lon': 'Longitude', 'lat': 'Latitude'}, fieldtype='V'))
    return fset


def set_skimulator_fieldset():
    filenames = '/projects/0/topios/hydrodynamic_data/SKIMulator/scisoc_trpac?.nc'
    variables = {'U': 'Ux_skim', 'V': 'Uy_skim'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    fset = FieldSet.from_netcdf(filenames, variables, dimensions)
    fset.add_field(Field.from_netcdf('skimulator_boundaryvelocities.nc', 'MaskUvel',
                                     {'lon': 'Longitude', 'lat': 'Latitude'}, fieldtype='U'))
    fset.add_field(Field.from_netcdf('skimulator_boundaryvelocities.nc', 'MaskVvel',
                                     {'lon': 'Longitude', 'lat': 'Latitude'}, fieldtype='V'))
    return fset


def createFADset(fieldset, datafile, nperid):
    # Read in all FAD set locations/times/ids
    conv = lambda x: num2date(strpdate2num('%Y-%m-%d %H:%M:%S')(x))
    fads = np.loadtxt(datafile, delimiter=',', converters={1: conv},
                      dtype={'names': ('id', 'date', 'lon', 'lat'),
                             'formats': (np.int, 'datetime64[D]', np.float, np.float)})

    timerange = [np.timedelta64(int(fieldset.U.grid.time_full[0]), 's') + fieldset.U.grid.time_origin,
                 np.timedelta64(int(fieldset.U.grid.time_full[-1]), 's') + fieldset.U.grid.time_origin]

    fids = [f[0] for f in fads if timerange[0] < f[1] < timerange[-1]]
    date = [f[1] for f in fads if timerange[0] < f[1] < timerange[-1]]
    lons = [f[2] for f in fads if timerange[0] < f[1] < timerange[-1]]
    lats = [f[3] for f in fads if timerange[0] < f[1] < timerange[-1]]

    class FADParticle(JITParticle):
        drift_time = Variable('drift_time', dtype=np.float32, to_write=True)
        ID = Variable('ID', dtype=np.int32, to_write='once')
        eez = Variable('eez', dtype=np.int32, to_write=True)

    return ParticleSet(fieldset, pclass=FADParticle,
                       lon=np.repeat(lons, nperid),
                       lat=np.repeat(lats, nperid),
                       time=np.repeat(date, nperid),
                       ID=np.repeat(fids, nperid))


def run_particles(fieldsetname):
    if fieldsetname == 'cmems_surface':
        fieldset = set_cmems_surface_fieldset()
    elif fieldsetname == 'cmems_50m':
        fieldset = set_cmems_50m_fieldset()
        dz = np.gradient(fieldset.U.depth)
        DZ = np.moveaxis(np.tile(dz, (fieldset.U.grid.ydim, fieldset.U.grid.xdim+10, 1)), [0, 1, 2], [1, 2, 0])

        def compute(fieldset):
            # Calculating vertical weighted average
            for f in [fieldset.U, fieldset.V]:
                for tind in f.loaded_time_indices:
                    f.data[tind, :] = np.sum(f.data[tind, :] * DZ, axis=0) / sum(dz)

        fieldset.compute_on_defer = compute

    elif fieldsetname == 'skimulator':
        # requires ncatted -a units,time,o,c,"days since 2016-01-01 00:00:00" scisoc_trpac_compressed.nc for decode_cf
        fieldset = set_skimulator_fieldset()
    else:
        raise NotImplementedError('FieldSet %s not implemented')

    fieldset.add_constant('max_drift_time', delta(days=180).total_seconds())
    size2D = (fieldset.U.grid.ydim, fieldset.U.grid.xdim)
    fieldset.add_field(Field('Kh_zonal', data=10*np.ones(size2D),
                             lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                             mesh='spherical', allow_time_extrapolation=True))
    fieldset.add_field(Field('Kh_meridional', data=10*np.ones(size2D),
                             lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                             mesh='spherical', allow_time_extrapolation=True))

    fieldset.add_periodic_halo(zonal=True, meridional=False, halosize=5)

    fieldset.add_field(Field.from_netcdf('EEZ_Field.nc', 'EEZ',
                                         {'lon': 'lon', 'lat': 'lat'},
                                         allow_time_extrapolation=True))

    pset = createFADset(fieldset, 'dFADsets.txt', nperid=10)
    pset.execute(SampleEEZ, dt=-1, runtime=0)  # setting the EEZ

    ofile = pset.ParticleFile(name='fadtracks_antibeaching_%s' % fieldsetname,
                              outputdt=delta(days=5))

    kernels = WrapLon + pset.Kernel(AdvectionRK4) + BrownianMotion2D + AntiBeaching + \
              SampleEEZ + DriftTime
    pset.execute(kernels, dt=delta(minutes=-10), output_file=ofile,
                 recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})


if __name__ == "__main__":
    run_particles('cmems_50m')
