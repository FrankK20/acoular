# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# acoular imports
from os.path import join

from acoular import ( BeamformerEA, PowerSpectra, Environment, MaskedTimeSamples,
                      MicGeom, RectGrid, SteeringVector, config)

config.global_caching = 'none'  # to make sure that nothing is cached


# load exampledata
datafile = join('..', 'example_data.h5')
micgeofile = join('..', '..', 'acoular', 'xml', 'array_56.xml')

# values from example 1
t1 = MaskedTimeSamples(name=datafile)
t1.start = 0  # first sample, default
t1.stop = 16000  # last valid sample = 15999
m = MicGeom(from_file=micgeofile)
num = 1024
freq = 800
st = SteeringVector(mics=m, steer_type='classic')
bounds = [(-0.6, 0.6),(-0.3, 0.3), (0.68, 0.68), (0.,1.)]
ps = PowerSpectra(time_data=t1, window='Hanning', block_size=num)

def test_beamformerea():
    BeamformerEA(steer=st, freq_data=ps, bounds=bounds).calculate(freq)
