# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
from os.path import join

# acoular imports
import acoular as ac

ac.config.global_caching = 'none'  # to make sure that nothing is cached

# load exampledata
datafile = join('..', 'data', 'example_data.h5')
micgeofile = join('..', '..', 'acoular', 'xml', 'array_56.xml')

# values from example 1
t1 = ac.MaskedTimeSamples(name=datafile)
t1.start = 0  # first sample, default
t1.stop = 16000  # last valid sample = 15999
m = ac.MicGeom(file=micgeofile)
num = 1024
freq = 800
st = ac.SteeringVector(mics=m, steer_type='classic')
bounds = [(-0.6, 0.6), (-0.3, 0.3), (0.68, 0.68), (0.0, 1.0)]
ps = ac.PowerSpectra(source=t1, window='Hanning', block_size=num)


def test_beamformerea():
    ac.BeamformerEA(steer=st, freq_data=ps, bounds=bounds).synthetic(freq, 0)
