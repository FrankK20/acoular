# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2020, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements processing in the time domain.

.. autosummary::
    :toctree: generated/

    SpaceModesTransformer
    VirtualRotator
    VirtualRotatorAngle
    AngleTracker
    TrajectoryAnglesFromTrigger
    VirtualRotatorModal
    VirtualRotatorSpatial
    RotationalSpeedDetector
    RotationalSpeedDetector2
    RotationalSpeedDetector3
    RotationalSpeed

"""
from warnings import warn

# imports from other packages
from numpy import (absolute, angle, arange, argsort, array, ceil, empty, exp,
                   fft, hstack, integer, isscalar, linspace, matrix, median,
                   newaxis, pad, pi, roll, sinc, sqrt, sum, zeros)
from scipy.interpolate import interp1d
from scipy.signal import butter, decimate, filtfilt
from traits.api import (Any, Bool, Float, Instance, Int, List,
                        Property, Trait, Tuple, cached_property)

from acoular.base import TimeOut
from acoular.grids import Grid
from acoular.internal import digest

from .microphones import MicGeomCirc
from .trajectory import AngleTrajectory, TrajectoryAnglesFromTrigger
from .trigger import Trigger


class SpaceModesTransformer ( TimeOut ):
    """
    Class to transform mic signals into mode domain or vice versa via spatial fft.
    """
    
    #: Channels to be used and their order, so that neighboring mics are 
    #: channel neighbors as well (last is neigbor to first). 
    #: If left emtpy, mics are just used in their current order. 
    channel_order = List([],
        desc="list of mic channels in correct order")

    #: Direction of transformation:
    #: If 'False' (default), transform from spatial to mode domain. 
    #: If 'True', transform from mode to spatial domain
    inverse = Bool(False,
                   desc = 'direction of transform')

    #: Array of modes that are calculated. Read-only.
    #: If source input is from mode domain, order is assumed to be 
    #: [0, 1, 2, ... , M/2-1, -M/2, ..., -2, -1]
    modes = Property( depends_on = ['source.digest', 'num_channels'])

    #: Number of channels/modes, is set automatically.
    num_channels = Property(depends_on = ['channel_order', \
        'source.sigest'], desc="number of valid channels")

    # internal identifier
    digest = Property( depends_on = ['source.digest', 'channel_order'])

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_num_channels( self ):
        if self.channel_order:
            return len(self.channel_order)
        else:
            return self.source.num_channels


    @cached_property
    def _get_modes( self ):
        if not self.inverse:
            nc = self.num_channels
        else:
            nc = self.source.num_channels
        return fft.fftfreq(nc, 1/nc).astype(int)
        
    def result(self, num):
        """ 
        Python generator: yields transformed result.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`num_channels`). 
            The last block may be shorter than num.
        """
        
        # set up fft or ifft      
        if not self.inverse: # transform space -> modes
            fft_func = fft.fft
            
            # index slice: all channels
            channels_out = slice(None, None, None)
            if self.channel_order: # use index array for resorting neighbors
                channels_in = array(self.channel_order)
            else: # otherwise use all channels in that order
                channels_in = slice(None, None, None)
        
        else: # transform modes -> space
            fft_func = fft.ifft
            # index slice: all modes 
            channels_in = slice(None, None, None)
            if self.channel_order: # resort output channels to original order
                channels_out = argsort(array(self.channel_order))
            else: # otherwise use all channels in current order
                channels_out = slice(None, None, None)
            
        # the actual generator
        for block in self.source.result(num):
            yield fft_func(block[:,channels_in], None, axis=1, norm="ortho")[:,channels_out]





class VirtualRotator ( TimeOut ):
    """
    Class that provides the facilities to virtually rotate a microphone 
    array depending on a given constant rpm.
    
    The measured time data is recalculated using interpolation 
    between samples such that the resulting data
    represents that of a virtually rotating array.
    """

    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'mpos.digest', 'rpm','interpolation', '__class__' ])
    
    # basename for cache file
    basename = Property( depends_on = 'source.digest', 
        desc="basename for cache file" )
    
    #: Type of interpolation between microphones to be used.
    interpolation = Trait('linear','sinc',
                          desc='type of interpolation between mics')
    
    #: :class:`~.microphones.MicGeomCirc` object that provides the 
    #: microphone locations
    mpos = Trait(MicGeomCirc, 
        desc="circular microphone geometry")
        
    #: Number of revolutions per minute; use negative values 
    #: for clockwise rotation, defaults to 0.
    rpm = Float(0.0,
        desc="revolutions per minute of the virtual array, negative values for clockwise rotation")
    # maybe change to rad/s ?

    @cached_property
    def _get_basename( self ):
        if 'basename' in self.source.all_trait_names():
            return self.source.basename
        else: 
            return self.source.__class__.__name__ + self.source.digest
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, num_channels). 
            The last block may be shorter than num.
        """
        if self.rpm == 0.0: # if it doesn't rotate, do nothing...
            print("No virtual array rotation!")
            for block in self.source.result(num):
            # effectively no processing
                yield block
                
        else:

            mpos = self.mpos        
            
            num_rings = len(mpos.ringlist) 
            num_rmics = [len(mpos.ringlist[a].mics) for a in range(num_rings)]
    
            j = 0
            
            # number of mics in ring of corresponding mic indices
            num_rmics_long = []

            # start mic for every ring, needed for later ringwise re-sorting
            mics_base = []

            # relative indices for mics in rings
            ring_base = []
            
            imic = 0

            for nmics in num_rmics: 
                num_rmics_long += [nmics]*nmics
                mics_base += [imic]*nmics
                ring_base += list(range(nmics))
                imic += nmics
                
            mics_base = array(mics_base)
            ring_base = array(ring_base)
            
            # predefine constants and indices for faster use in loop...
            rpm2imic = self.rpm * 6.0 / self.sample_freq
            mm_index = mpos.mic_index[:]
            mmm_index = mpos.indices[mm_index]

            newdata = empty((num, len(mpos.mic_index))) 
            
            for block in self.source.result(num):
                ns = block.shape[0] # number of samples in block
                
                isamples = arange(j, j+ns)
                
                
                
                if self.interpolation == 'sinc':
                    # this implementation has to be sped up, current one is inefficient...
                    # walk through the rings:
                    for iring in range(num_rings):
                        N = num_rmics[iring]
                        ringmics = mpos.ringlist[iring].mics # indices of mics in current ring
                        rel_ind = arange(N)-N//2 # how should the relative weighting be done 
                        
                        imic0 = ((rpm2imic * isamples)%360.0 ) * N / 360.0
                        imic0int  = imic0.astype(integer) 
                        imic0frac = imic0 - imic0int
                        
                        # kernel contains weights for neighboring mics (use all mics in ring)
                        # weights depend on fraction between mics
                        kernel = sinc(imic0frac[:,newaxis] - rel_ind[newaxis, : ]) 
                        # roll the kernel by full mics according to angle:
                        for i in range(ns):
                            kernel[i]=roll(kernel[i],imic0int[i]-N//2)
                        
                        # get data of this ring from block
                        p = block[:,ringmics]
                        
                        # go through all mics, roll kernel accordingly, sum weighted signals
                        for n in range(N):
                            # fill newdata with sinc interpolated data
                            newdata[:ns,ringmics[n]] = sum( p * roll(kernel, n, axis=1), axis=1)
                
                else: #if self.interpolation == 'linear':
                    # calculate how many mics further the rotating array is now...
                    imic0 = array(( ( rpm2imic * matrix(isamples) )%360.0 ).T * \
                                                    matrix(num_rmics_long) / 360.0)
                    imic0int  = imic0.astype(integer) 
                    imic0frac = imic0 - imic0int# = imic0 % 1 # fractional part
    
                    # calculate indices of mics between which to interpolate...    
                    rmi1 = mics_base + (ring_base+imic0int) % array(num_rmics_long)
                    rmi2 = mics_base+(ring_base+imic0int+1) % array(num_rmics_long)
    
                    ns_index = arange(ns)[:, newaxis] 
                    
                    # fill newdata with linearly interpolated data...
                    """
                    newdata[:, mmm_index] = \
                             (1 - imic0frac) * block[ ns_index, mm_index[rmi1] ] \
                               + (imic0frac) * block[ ns_index, mm_index[rmi2] ]
                    """
                    newdata[:ns, mmm_index] = \
                             (1 - imic0frac) * block[ ns_index, mmm_index[rmi1] ] \
                               + (imic0frac) * block[ ns_index, mmm_index[rmi2] ]
                    #"""
                j += ns
                
                yield newdata[:ns]




class VirtualRotatorAngle ( VirtualRotator ):
    """
    Class that provides the facilities to virtually rotate a microphone 
    array depending on sampled angles.
    
    The measured time data is recalculated using interpolation 
    between samples such that the resulting data
    represents that of a virtually rotating array.
    
    The following assumptions are made:
       * The revolutions per minute vary little enough too use their average
         for calculating the retarded time at the microphones
       * the center of the focus grid can be used as reference for calculating 
         the retarded time (instead of each grid point)
    """

    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 
                      'angle.digest',
                      'mpos.digest',
                      'grid.z',
                      'phi_trigger',
                      'c',
                      'interpolation',
                      '__class__' ])
    
    #: Grid-derived object that provides the grid locations.
    grid = Trait(Grid, 
        desc="beamforming grid")   

    #: Delay for rotation. If set, overrides delay clac'ed from grid position. 
    #TODO: Make this ring-specific (as array or similar)    
    delay = Any(-1.)

    #: Type of interpolation between microphones to be used.
    interpolation = Trait('linear','sinc',
                          desc='type of interpolation between mics')
    
    #: Speed of sound, defaults to 343 m/s.
    c = Float(343., 
        desc="speed of sound")

    #: Angular position of the trigger (if only one trigger is used);
    #: defaults to 0. Use this parameter to align the sources to the
    #: rotating geometry.
    phi_trigger = Float(0.,
                        desc = 'angle of first trigger')
 
    #: :class:`AngleTrajectory` instance, that delivers the angles
    #: for virtual rotation.
    angle = Trait(AngleTrajectory,
                  desc = 'interpolated angles')

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, num_channels). 
            The last block may be shorter than num.
        """

        self.rpm = self.angle.rpm
        if self.rpm == 0.0: # if it doesn't rotate, do nothing...
            print("No virtual array rotation!")
            for block in self.source.result(num):
            # effectively no processing
                yield block
                
        else:
            
            mpos = self.mpos  
            
            # speed of sound in sample units (meter/sample)
            c = self.c/self.sample_freq
            
            # number of mic rings
            num_rings = len(mpos.ringlist) 
            
            # list of number of mics in rings
            num_rmics = [len(mpos.ringlist[a].mics) for a in range(num_rings)]
    
            # list of distances btw mic rings and grid center
            r_rings = array([mpos.ringlist[a].r for a in range(num_rings)])
            z_rings = array([mpos.ringlist[a].z for a in range(num_rings)])
            
            
            if self.delay == -1:            
                dr_ring = sqrt( r_rings**2 + (self.grid.z - z_rings)**2 ) 
                #dr_ring *= 0 # no delay between vent. rot. and array response
                # sound travel time from grid center to ring
                dt_ring = dr_ring / c 
            else:
                dt_ring = array(self.delay).reshape(-1,)*self.sample_freq
            
            
            # starting angles of rings, also adjust for trigger position
            phi_start = self.angle.location(self.angle.startind - dt_ring)
            phi_start -= self.phi_trigger
            
            
            # number of mics in ring of corresponding mic indices
            num_rmics_long = []

            # start mic for every ring, needed for later ringwise re-sorting
            mics_base = []

            # relative indices for mics in rings
            ring_base = []
            
            # id of ring each mic belongs to            
            ring_id_long = []
            
            # calculate all "help" arrays needed
            imic = 0
            iring = 0
            for nmics in num_rmics: 
                num_rmics_long += [nmics]*nmics
                mics_base += [imic]*nmics
                ring_base += list(range(nmics))
                ring_id_long += [iring]*nmics
                imic += nmics
                iring += 1
                
                
            mics_base = array(mics_base)
            ring_base = array(ring_base)
            num_rmics_long = array(num_rmics_long)
            
            # predefine constants and indices for faster use in loop...
            mm_index = mpos.mic_index[:]
            mmm_index = mpos.indices[mm_index]

            #adata = self.angle.result().next()
            iangle = arange(num)
            
            newdata = empty((num, len(mpos.mic_index))) 

            for block in self.source.result(num):
                
                ns = block.shape[0] # number of samples in block
                
                # get the angles at the retarded time for all mic rings                
                new_angles = (array([self.angle.location(iangle[:ns]-dt) for dt in dt_ring])  \
                                    - phi_start.reshape(num_rings,1) ).T % 360.0
                
                iangle += ns
                
                if self.interpolation == 'sinc':
                    # this implementation has to be sped up, current one is inefficient...
                    # walk through the rings:
                    for iring in range(num_rings):
                        N = num_rmics[iring]
                        ringmics = mpos.ringlist[iring].mics # indices of mics in current ring
                        rel_ind = arange(N)-N//2 # how should the relative weighting be done 
                        
                        imic0 = new_angles[:,iring] / 360.0 * N
                        imic0int  = imic0.astype(integer) 
                        imic0frac = imic0 - imic0int
                        
                        # kernel contains weights for neighboring mics (use all mics in ring)
                        # weights depend on fraction between mics
                        kernel = sinc(imic0frac[:,newaxis] - rel_ind[newaxis, : ]) 
                        # roll the kernel by full mics according to angle:
                        for i in range(ns):
                            kernel[i]=roll(kernel[i],imic0int[i]-N//2)
                        
                        # get data of this ring from block
                        p = block[:,ringmics]
                        
                        # go through all mics, roll kernel accordingly, sum weighted signals
                        for n in range(N):
                            # fill newdata with sinc interpolated data
                            #TODO: roll here is probably buggy, as it spills / shifts one time sample
                            newdata[:ns,ringmics[n]] = sum( p * roll(kernel, n, axis=1), axis=1)

                else: #if self.interpolation == 'linear':
                    # calculate how many mics further the rotating array is now...
                    # ring_id_long to put  new_angles in block form
                    imic0 = new_angles[:,ring_id_long] / 360.0 * num_rmics_long
                    imic0int  = imic0.astype(integer) 
                    imic0frac = imic0 - imic0int# = imic0 % 1 # fractional part
    
                    # calculate indices of mics between which to interpolate...    
                    rmi1 = mics_base + (ring_base+imic0int) % num_rmics_long
                    rmi2 = mics_base+(ring_base+imic0int+1) % num_rmics_long
        
                    # fill newdata with linearly interpolated data...                
                    newdata[:ns, mmm_index] = \
                             (1 - imic0frac) * \
                             array([block[a,:][mmm_index[rmi1[a,:]]] for a in range(ns)]) \
                             + (imic0frac) * \
                             array([block[a,:][mmm_index[rmi2[a,:]]] for a in range(ns)])
                
                yield newdata[:ns]




class VirtualRotatorModal ( TimeOut ):
    """
    Class that provides the facilities to virtually rotate modes
    depending on a given constant rps.
    
    The measured time data is recalculated using modal decomposition.
    """
    
    # Data source; :class:`~acoular.tprocess.Time` or derived object.
    #source = Trait(SpaceModesTransformer)
    # source can be any TimeInOut, but is expected to be in Spatial FFT domain and order
        
    #: Rotational speed in rps. Positive, if rotation is around positive z-axis sense,
    #: which means from x to y axis.
    rotational_speed = Any(0.0)

    
    #: Average time delay between grid and mic planes
    delay = Float(0.0)
    
    
    # internal identifier
    rotspeed_digest = Property(depends_on = ['rotational_speed'])
    
    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'rotspeed_digest', 'delay', '__class__' ])
    
    @cached_property
    def _get_rotspeed_digest( self ):
        if isscalar(self.rotational_speed):
            return self.rotational_speed
        else:
            return self.rotational_speed.digest
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, num_channels). 
            The last block may be shorter than num.
        """
        
        if not self.rotational_speed: # rps==0
            for block in self.source.result(num):
                # effectively no processing
                yield block
        else:
            
            # list of modes
            modes = fft.fftfreq(self.num_channels, 1/self.num_channels).astype(int)
            
          
            # constant rotation 
            if isscalar(self.rotational_speed):
                # initialize time vector
                tvec = arange(num)
                # calculate rotational speed in revs per sample
                revs_per_sample = self.rotational_speed / self.sample_freq
                            
                for block in self.source.result(num):
                    # get actual size of block
                    bsize = block.shape[0]
                    
                    # phase shift of time signal
                    phase_shift = exp(2j*pi * revs_per_sample * 
                                      modes[newaxis, :] * 
                                      tvec[:bsize, newaxis])
                    
                    yield block * phase_shift
                    tvec += num
           
            elif isinstance(self.rotational_speed, TrajectoryAnglesFromTrigger): 
                sample_delay = self.delay*self.sample_freq
                tvec = arange(num)
                theta0 = self.rotational_speed.location(self.rotational_speed.first_trigger-sample_delay)#-self.rotational_speed.location(0)
                for block in self.source.result(num):
                    # get actual size of block
                    bsize = block.shape[0]
                    # get rotation angles of array
                    theta = self.rotational_speed.location(tvec[:bsize]-sample_delay) - theta0
                    phase_shift = exp(1j * theta[:,newaxis] * modes[newaxis, :] )
                    yield block * phase_shift
                    tvec += num
           
            elif isinstance(self.rotational_speed, TimeOut):
                
                for theta, block in zip(self.rotational_speed.result(num), self.source.result(num)):
                    phase_shift = exp(1j * theta[:,newaxis] * modes[newaxis, :] )
                    yield block * phase_shift
            else:
                raise Exception('Unexpected data type (not constant or TimeInOut!')
            


class VirtualRotatorSpatial ( TimeOut ):
    """
    Class that provides the facilities to virtually rotate a microphone 
    array depending on sampled angles.
    
    The measured time data is recalculated using interpolation 
    between samples such that the resulting data
    represents that of a virtually rotating array.
    
    The following assumptions are made:
       * The revolutions per minute vary little enough too use their average
         for calculating the retarded time at the microphones
       * the center of the focus grid can be used as reference for calculating 
         the retarded time (instead of each grid point)
    """

    channel_order = List([],
        desc="list of mic channels in correct order")

    #: Rotational speed in rps. Positive, if rotation is around positive z-axis sense,
    #: which means from x to y axis.
    rotational_speed = Any(0.0)

    
    #: Average time delay between grid and mic planes
    delay = Float(0.0)
    
    
    # internal identifier
    rotspeed_digest = Property()
    
    # internal identifier
    digest = Property( 
        depends_on = ['source.digest', 'rotspeed_digest', 'delay', 'channel_order', '__class__' ])
    
    
    def _get_rotspeed_digest( self ):
        if isscalar(self.rotational_speed):
            return self.rotational_speed
        else:
            return self.rotational_speed.digest
    
    @cached_property
    def _get_digest( self ):
        return digest(self)    
    
    def result(self, num):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        Returns
        -------
        Samples in blocks of shape (num, num_channels). 
            The last block may be shorter than num.
        """

        if not self.rotational_speed: # rps==0
            for block in self.source.result(num):
                # effectively no processing
                yield block
        else:
            channel_order = self.channel_order
            num_mics = len(channel_order)
            
            # constant rotation 
            # if isscalar(self.rotational_speed):
            #     # initialize time vector
            #     tvec = arange(num)
            #     # calculate rotational speed in revs per sample
            #     revs_per_sample = self.rotational_speed / self.sample_freq
                            
            #     for block in self.source.result(num):
            #         # get actual size of block
            #         bsize = block.shape[0]
                    
            #         # phase shift of time signal
            #         phase_shift = exp(2j*pi * revs_per_sample * 
            #                           modes[newaxis, :] * 
            #                           tvec[:bsize, newaxis])
                    
            #         yield block * phase_shift
            #         tvec += num
           
            if isinstance(self.rotational_speed, TrajectoryAnglesFromTrigger): 
                sample_delay = self.delay*self.sample_freq
                tvec = arange(num)
                theta0 = self.rotational_speed.location(self.rotational_speed.first_trigger-sample_delay)
                prot = zeros((num,num_mics))
                for block in self.source.result(num):
                    # get actual size of block
                    bsize = block.shape[0]
                    
                    # get rotation angles of array
                    theta = self.rotational_speed.location(tvec[:bsize]-sample_delay) - theta0
                    
                    relpos = theta/2/pi*num_mics
                    i = 0
                    for rel,irel in zip(relpos,relpos.astype(int)):
                        mics_lower = roll(channel_order,-irel)
                        mics_upper = roll(channel_order,-irel-1)
                    
                        w_upper = rel-irel
                        w_lower = 1-w_upper
                        prot[i,channel_order] = w_lower*block[i,mics_lower]+\
                                                w_upper*block[i,mics_upper]
                        i += 1
                    yield prot[:bsize]
                    tvec += num
           
            # elif isinstance(self.rotational_speed, TimeInOut):
                
            #     for theta, block in zip(self.rotational_speed.result(num), self.source.result(num)):
            #         phase_shift = exp(1j * theta[:,newaxis] * modes[newaxis, :] )
            #         yield block * phase_shift
            else:
                raise Exception('Unexpected data type (not constant or TimeOut!')
                


class RotationalSpeedDetector( TimeOut ):
    """
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    """
    
    #: Maximum delta f of returned spectra in Hz
    df = Float(0.1)
    
    #: number of samples in one block for rotation detection
    block_size = Int(10000)
    
    #: Upper rotational frequency limit that should be resolved
    fmax = Float(100.0)
    
    #: Tuple with (lower, upper) frequency for band pass prior to cross correlation
    band_pass = Tuple((2500.0, 0.0))
    
    #: Result will be yielded for every `sample_step`-th sample. 
    #: If >1, this will result in a decreased `sample_freq`.
    sample_step = Int(10000)

    #: List of delta m (=b-a) to consider
    dm_list = List( [1,3,5,7,9] )#
    
    #: maximum cross mode to consider (max. nmodes/2 )
    m_max = Int(26)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Property( depends_on = ['source.sample_freq', 
                                          'sample_step'])
    
    #: Discrete resolved rotational frequencies
    rps_freqs = Property( depends_on = ['source.sample_freq', 'fmax',
                                        '_ns_new'])
    
    # number of samples in intermediate fft blocks
    _ns_new = Property( depends_on = ['source.sample_freq', 'df',
                                      'block_size', 'dm_list'])
    
    # max. index of data to consider (corresponding to fmax)
    _maxfreqind = Property( depends_on = ['source.sample_freq', 
                                          'fmax', '_ns_new'])
    
    @cached_property 
    def _get_sample_freq( self ):
        return self.source.sample_freq/self.sample_step
    
    @cached_property 
    def _get__ns_new( self ):
        # number of samples
        return max( int(self.source.sample_freq/self.df)+1, 
                        self.block_size * max(self.dm_list) )

    @cached_property 
    def _get__maxfreqind( self ):
        return int(self.fmax * self._ns_new / self.source.sample_freq)
    
    @cached_property 
    def _get_rps_freqs( self ):
        fftfreqs = fft.fftfreq(self._ns_new, 1./self.source.sample_freq)
        return hstack((fftfreqs[-self._maxfreqind:], fftfreqs[:self._maxfreqind]))     
        
    
    
    def result( self, num ):
        """
        see "12_frot_detection_weight_and_combine.py" for tests
        assumes that input is in mode order
        """

        bs = self.block_size
        step = self.sample_step
        modelist = self.source.modes
        nmodes = len(modelist)
        
        f_low, f_high = self.band_pass
                                ### band pass filter ##############################
        if f_low>0:  
            b, a = butter(8, f_low/self.source.sample_freq*2, btype='highpass')
        #TODO: f_high currently not very nicely implemented
        if f_high>0:
            bh, ah = butter(8, f_high/self.source.sample_freq*2, btype='lowpass')
        #    data = filtfilt(b, a, data, axis=0)
        
        mf_ind = self._maxfreqind
        
        # allocate arrays that hold the needed data
        pmfpart = zeros((2*mf_ind, self.m_max+1), dtype=complex)
        result_full = zeros((num, 2*mf_ind))
        
        result_step = zeros(2*mf_ind)
        
        pmt = zeros((self.block_size+num,nmodes), dtype=complex)
        
        ind = 0
        ind_out = 0
        
        for block in self.source.result( num ):
            #print(ind,end=',')
            ns = block.shape[0]
            ind1 = ind+ns
            
            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]
            
            ind = ind1
                        
            if ind >= bs:
                # highpass
                if f_low>0:  
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high>0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for dm in self.dm_list:
            
                    # calculate cross-mode product, only need half of the modes, since everything is mirrored
                    # shift left/right, so that all mode combinations are present (and none doubled) in the resulting array
                    ind_m_b = roll(modelist[::-1], int(0.5-dm/2))[:nmodes//2]
                    ind_m_a = roll(modelist      , int(0.0+dm/2))[:nmodes//2]
                    pmtprod = pmt_temp[:,ind_m_a] * pmt_temp[:,ind_m_b]
                
                    # ns_full = 2**int(round(log2(self._ns_new/dm))) # base 2 numbers too "rough" for interpolation between modes, otherwise, fft would be about 3 times as fast!
                    ns_full = int(self._ns_new/dm)
                    pmf = fft.fft(pad(pmtprod,[(0, ns_full-bs),(0, 0)]), None, 0, norm="ortho")
                
                    # fill array with only part of the data
                    pmfpart[:mf_ind, :] = pmf[-mf_ind:, :self.m_max+1]
                    pmfpart[mf_ind:, :] = pmf[ :mf_ind, :self.m_max+1]
                
                    # weighting array
                    # derivative of angle -> small if rotational velocity is found
                    pmfanglediff=sum(abs(angle(pmfpart[:,1:] * pmfpart[:,:-1].conjugate())), axis=1)
                    
                    #get result for this mode
                    apmf = absolute(pmfpart)
                    result_dm = 1/pmfanglediff * sum(apmf/median(apmf,axis=0), axis=1)
                    #result_dm = 1/pmfanglediff * sum(absolute(pmfpart), axis=1)
                    
                    # average to get overall result
                    #result_step *= result_dm/result_dm.max()
                    result_step += result_dm#/result_dm.max()
                result_full[ind_out,:] = result_step#/result_step.max()
                result_step[:] = zeros(2*mf_ind)
                ind_out += 1
                if ind_out >= num:
                    yield result_full
                    ind_out = 0
            
                # shift rest of data according to step size
                ind -= step#self.block_size
                if ind > 0:
                    pmt[:ind] = pmt[step:step+ind]
                
        if ind_out > 0:
            yield result_full[:ind_out]

import numpy as np
from scipy.signal import butter, filtfilt
from numpy import angle, absolute, median, pad, roll, zeros, hstack
from scipy import fft
import h5py

class RotationalSpeedDetectorPairs(RotationalSpeedDetector):
    """
    Extension of RotationalSpeedDetector that also exposes per-pair
    rotational spectra for all available mode pairs.

    Notes
    -----
    pair_spectra contains one spectrum per (dm, pair_index).
    By default this stores normalized magnitudes:
        abs(pmfpart) / median(abs(pmfpart), axis=0)
    which is usually best for separate inspection.
    """

    eps = 1e-12

    def pair_result(self, num, store="norm"):
        """
        Yield dictionaries with:
            total_spectrum : (n_out, n_freq)
            pair_spectra   : (n_out, n_dm, n_freq, n_pairs)
            pair_modes     : (n_dm, n_pairs, 2)
            rps_freqs      : (n_freq,)
            dm_list        : (n_dm,)

        Parameters
        ----------
        num : int
            Number of output steps per yielded block.
        store : {"norm", "abs", "complex"}
            What to store for each pair:
            - "norm": abs(pmfpart) / median(abs(pmfpart), axis=0)
            - "abs":  abs(pmfpart)
            - "complex": complex pmfpart itself
        """
        bs = self.block_size
        step = self.sample_step
        modelist = np.asarray(self.source.modes)
        nmodes = len(modelist)

        f_low, f_high = self.band_pass
        if f_low > 0:
            b, a = butter(8, f_low / self.source.sample_freq * 2, btype="highpass")
        if f_high > 0:
            bh, ah = butter(8, f_high / self.source.sample_freq * 2, btype="lowpass")

        mf_ind = self._maxfreqind
        n_freq = 2 * mf_ind
        n_dm = len(self.dm_list)

        # all available unique pairs
        n_pairs = nmodes // 2

        # metadata: pair labels for each dm
        # pair_modes[k, j] = [m_a, m_b] in paper-style notation
        # internally the multiplied columns are (m_a, -m_b)
        pair_modes = np.zeros((n_dm, n_pairs, 2), dtype=int)

        for k, dm in enumerate(self.dm_list):
            ind_m_b = roll(modelist[::-1], int(0.5 - dm / 2))[:n_pairs]
            ind_m_a = roll(modelist,       int(0.0 + dm / 2))[:n_pairs]

            pair_modes[k, :, 0] = ind_m_a
            pair_modes[k, :, 1] = -ind_m_b  # more intuitive labeling, e.g. (0,5)

        # allocate output buffers
        result_full = np.zeros((num, n_freq), dtype=float)

        if store == "complex":
            pair_spectra_full = np.zeros((num, n_dm, n_freq, n_pairs), dtype=complex)
        else:
            pair_spectra_full = np.zeros((num, n_dm, n_freq, n_pairs), dtype=float)

        result_step = np.zeros(n_freq, dtype=float)
        pmt = np.zeros((self.block_size + num, nmodes), dtype=complex)

        ind = 0
        ind_out = 0

        for block in self.source.result(num):
            ns = block.shape[0]
            ind1 = ind + ns

            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]

            ind = ind1

            if ind >= bs:
                if f_low > 0:
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high > 0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for k, dm in enumerate(self.dm_list):
                    ind_m_b = roll(modelist[::-1], int(0.5 - dm / 2))[:n_pairs]
                    ind_m_a = roll(modelist,       int(0.0 + dm / 2))[:n_pairs]

                    # one product signal per mode pair
                    pmtprod = pmt_temp[:, ind_m_a] * pmt_temp[:, ind_m_b]

                    # common rotational-frequency grid
                    ns_full = int(self._ns_new / dm)
                    pmf = fft.fft(
                        pad(pmtprod, [(0, ns_full - bs), (0, 0)]),
                        None,
                        axis=0,
                        norm="ortho",
                    )

                    pmfpart = np.zeros((n_freq, n_pairs), dtype=complex)
                    pmfpart[:mf_ind, :] = pmf[-mf_ind:, :n_pairs]
                    pmfpart[mf_ind:, :] = pmf[:mf_ind, :n_pairs]

                    apmf = np.abs(pmfpart)
                    med = np.maximum(np.median(apmf, axis=0, keepdims=True), self.eps)
                    pair_norm = apmf / med

                    # save per-pair spectra
                    if store == "complex":
                        pair_spectra_full[ind_out, k] = pmfpart
                    elif store == "abs":
                        pair_spectra_full[ind_out, k] = apmf
                    else:
                        pair_spectra_full[ind_out, k] = pair_norm

                    # original detector score, unchanged
                    pmfanglediff = np.sum(
                        np.abs(np.angle(pmfpart[:, 1:] * pmfpart[:, :-1].conjugate())),
                        axis=1,
                    )
                    pmfanglediff = np.maximum(pmfanglediff, self.eps)

                    result_dm = (1.0 / pmfanglediff) * np.sum(pair_norm, axis=1)
                    result_step += result_dm

                result_full[ind_out, :] = result_step
                result_step[:] = 0.0
                ind_out += 1

                if ind_out >= num:
                    yield {
                        "total_spectrum": result_full.copy(),
                        "pair_spectra": pair_spectra_full.copy(),
                        "pair_modes": pair_modes.copy(),
                        "rps_freqs": self.rps_freqs.copy(),
                        "dm_list": np.asarray(self.dm_list),
                    }
                    ind_out = 0
                    result_full[:] = 0.0
                    pair_spectra_full[:] = 0.0

                ind -= step
                if ind > 0:
                    pmt[:ind] = pmt[step:step + ind]

        if ind_out > 0:
            yield {
                "total_spectrum": result_full[:ind_out].copy(),
                "pair_spectra": pair_spectra_full[:ind_out].copy(),
                "pair_modes": pair_modes.copy(),
                "rps_freqs": self.rps_freqs.copy(),
                "dm_list": np.asarray(self.dm_list),
            }

    def save_pair_spectra_h5(self, filename, num=1, store="norm", compression="gzip"):
        """
        Stream all total and per-pair spectra to an HDF5 file.

        Output datasets:
            /total_spectrum : (n_time, n_freq)
            /pair_spectra   : (n_time, n_dm, n_freq, n_pairs)
            /pair_modes     : (n_dm, n_pairs, 2)
            /rps_freqs      : (n_freq,)
            /dm_list        : (n_dm,)
        """
        with h5py.File(filename, "w") as h5:
            first = True
            n_written = 0

            for out in self.pair_result(num=num, store=store):
                total = out["total_spectrum"]
                pairs = out["pair_spectra"]

                if first:
                    h5.create_dataset("rps_freqs", data=out["rps_freqs"])
                    h5.create_dataset("dm_list", data=out["dm_list"])
                    h5.create_dataset("pair_modes", data=out["pair_modes"])

                    h5.create_dataset(
                        "total_spectrum",
                        shape=(0, total.shape[1]),
                        maxshape=(None, total.shape[1]),
                        chunks=(1, total.shape[1]),
                        compression=compression,
                        dtype=total.dtype,
                    )

                    h5.create_dataset(
                        "pair_spectra",
                        shape=(0, pairs.shape[1], pairs.shape[2], pairs.shape[3]),
                        maxshape=(None, pairs.shape[1], pairs.shape[2], pairs.shape[3]),
                        chunks=(1, pairs.shape[1], pairs.shape[2], pairs.shape[3]),
                        compression=compression,
                        dtype=pairs.dtype,
                    )
                    first = False

                n_new = total.shape[0]

                h5["total_spectrum"].resize(n_written + n_new, axis=0)
                h5["total_spectrum"][n_written:n_written + n_new] = total

                h5["pair_spectra"].resize(n_written + n_new, axis=0)
                h5["pair_spectra"][n_written:n_written + n_new] = pairs

                n_written += n_new


import numpy as np
import h5py
from scipy.signal import butter, filtfilt
from scipy import fft

# If you paste this into tprocess.py, RotationalSpeedDetector already exists.
# This class reuses its parameters (df, block_size, fmax, band_pass, sample_step, etc.).

class RotationalSpeedDetectorAllPairs(RotationalSpeedDetector):
    """
    Compute rotational spectra for all available azimuthal mode pairs (a, b).

    Pair definition
    ---------------
    For a labeled pair (a, b), the time-domain product is formed as

        p_a(t) * p_{-b}(t)

    exactly in the spirit of the original method.

    Default behavior
    ----------------
    positive_dm_only=True:
        keep only pairs with dm = b-a > 0
        (recommended; avoids mirrored duplicates)

    include_zero_dm=False:
        skip dm = 0 because it does not encode rotational shift

    max_abs_dm=0:
        use all available |dm| from the available modes
        if > 0, restrict to |dm| <= max_abs_dm
    """

    #: Keep only pairs with dm = b-a > 0
    positive_dm_only = Bool(
        True,
        desc="if True, keep only non-redundant pairs with positive delta m"
    )

    #: Include dm = 0 pairs
    include_zero_dm = Bool(
        False,
        desc="if True, also include mode pairs with delta m = 0"
    )

    #: Maximum absolute delta m to include; 0 means use all available
    max_abs_dm = Int(
        0,
        desc="maximum absolute delta m to include; 0 means all available"
    )

    #: Small numerical protection constant
    eps = Float(
        1e-12,
        desc="small epsilon to avoid division by zero"
    )

    def _build_all_pair_groups(self):
        """
        Build all requested mode pairs and group them by |dm|.

        Returns
        -------
        groups : list of dict
            One entry per |dm| group. Each dict contains:
                dm_abs
                pair_slice
                a_cols
                negb_cols
                dm_signed
        mode_pairs : ndarray, shape (n_pairs_total, 2)
            Labels (a, b)
        pair_dm : ndarray, shape (n_pairs_total,)
            Signed dm = b-a for each pair
        """
        modelist = np.asarray(self.source.modes, dtype=int)
        mode_to_col = {int(m): i for i, m in enumerate(modelist)}

        # collect entries grouped by |dm|
        pairs_by_absdm = {}

        for a in modelist:
            a = int(a)
            for b in modelist:
                b = int(b)

                # we need the mode -b to exist because we multiply p_a with p_-b
                if -b not in mode_to_col:
                    continue

                dm = b - a
                if dm == 0 and not self.include_zero_dm:
                    continue
                if self.positive_dm_only and dm < 0:
                    continue
                if self.max_abs_dm > 0 and abs(dm) > self.max_abs_dm:
                    continue

                dm_abs = abs(dm)
                entry = (a, b, mode_to_col[a], mode_to_col[-b], dm)
                pairs_by_absdm.setdefault(dm_abs, []).append(entry)

        if not pairs_by_absdm:
            raise ValueError("No valid mode pairs found for the chosen settings.")

        groups = []
        mode_pairs_flat = []
        pair_dm_flat = []
        start = 0

        for dm_abs in sorted(pairs_by_absdm):
            entries = pairs_by_absdm[dm_abs]

            # sort pairs to get a stable and interpretable order
            entries = sorted(entries, key=lambda x: (x[0], x[1]))

            n_this = len(entries)
            stop = start + n_this

            groups.append({
                "dm_abs": dm_abs,
                "pair_slice": slice(start, stop),
                "a_cols": np.array([e[2] for e in entries], dtype=int),
                "negb_cols": np.array([e[3] for e in entries], dtype=int),
                "dm_signed": np.array([e[4] for e in entries], dtype=int),
            })

            mode_pairs_flat.extend([(e[0], e[1]) for e in entries])
            pair_dm_flat.extend([e[4] for e in entries])

            start = stop

        mode_pairs = np.asarray(mode_pairs_flat, dtype=int)
        pair_dm = np.asarray(pair_dm_flat, dtype=int)
        return groups, mode_pairs, pair_dm

    def _max_abs_dm_available(self):
        """Largest |dm| that will actually be used."""
        _, _, pair_dm = self._build_all_pair_groups()
        return int(np.max(np.abs(pair_dm)))

    @property
    def rps_freqs_allpairs(self):
        """
        Common rotational-frequency axis for the all-pairs computation.
        """
        fs = self.source.sample_freq
        max_abs_dm = self._max_abs_dm_available()
        ns_new = max(int(fs / self.df) + 1, self.block_size * max_abs_dm)
        maxfreqind = int(self.fmax * ns_new / fs)
        fftfreqs = fft.fftfreq(ns_new, 1.0 / fs)
        return np.hstack((fftfreqs[-maxfreqind:], fftfreqs[:maxfreqind]))

    def pair_result(self, num, store="norm"):
        """
        Yield blocks of all pair spectra.

        Parameters
        ----------
        num : int
            Number of output time steps per yielded block.
        store : {"norm", "abs", "complex"}
            What to store for each pair:
              - "norm":  |P_ab| / median(|P_ab|)
              - "abs":   |P_ab|
              - "complex": complex spectrum itself

        Yields
        ------
        dict with:
            pair_spectra : (n_out, n_pairs_total, n_freq)
            mode_pairs   : (n_pairs_total, 2)
            pair_dm      : (n_pairs_total,)
            rps_freqs    : (n_freq,)
        """
        if store not in {"norm", "abs", "complex"}:
            raise ValueError("store must be one of: 'norm', 'abs', 'complex'")

        bs = self.block_size
        step = self.sample_step
        fs = self.source.sample_freq
        modelist = np.asarray(self.source.modes, dtype=int)
        nmodes = len(modelist)

        groups, mode_pairs, pair_dm = self._build_all_pair_groups()
        n_pairs_total = mode_pairs.shape[0]

        max_abs_dm = int(np.max(np.abs(pair_dm)))
        ns_new = max(int(fs / self.df) + 1, bs * max_abs_dm)
        maxfreqind = int(self.fmax * ns_new / fs)
        n_freq = 2 * maxfreqind

        fftfreqs = fft.fftfreq(ns_new, 1.0 / fs)
        rps_freqs = np.hstack((fftfreqs[-maxfreqind:], fftfreqs[:maxfreqind]))

        # band-pass
        f_low, f_high = self.band_pass
        if f_low > 0:
            b, a = butter(8, f_low / fs * 2, btype="highpass")
        if f_high > 0:
            bh, ah = butter(8, f_high / fs * 2, btype="lowpass")

        # output buffers
        if store == "complex":
            pair_spectra_full = np.zeros((num, n_pairs_total, n_freq), dtype=np.complex64)
        else:
            pair_spectra_full = np.zeros((num, n_pairs_total, n_freq), dtype=np.float32)

        # rolling input buffer
        pmt = np.zeros((bs + num, nmodes), dtype=np.complex64)

        ind = 0
        ind_out = 0

        for block in self.source.result(num):
            ns = block.shape[0]
            ind1 = ind + ns

            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]

            ind = ind1

            if ind >= bs:
                if f_low > 0:
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high > 0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                # compute one spectrum per pair, grouped by |dm|
                for grp in groups:
                    dm_abs = grp["dm_abs"]
                    sl = grp["pair_slice"]
                    a_cols = grp["a_cols"]
                    negb_cols = grp["negb_cols"]
                    dm_signed = grp["dm_signed"]

                    # p_a(t) * p_-b(t)
                    pmtprod = pmt_temp[:, a_cols] * pmt_temp[:, negb_cols]

                    # common rotational-frequency mapping, same idea as original RSD
                    ns_full = int(ns_new / dm_abs)
                    pmf = fft.fft(
                        np.pad(pmtprod, [(0, ns_full - bs), (0, 0)]),
                        None,
                        axis=0,
                        norm="ortho",
                    )

                    pmfpart = np.zeros((n_freq, pmtprod.shape[1]), dtype=np.complex64)
                    pmfpart[:maxfreqind, :] = pmf[-maxfreqind:, :]
                    pmfpart[maxfreqind:, :] = pmf[:maxfreqind, :]

                    # if negative dm are included, they are mirrored on the common rps axis
                    if not self.positive_dm_only:
                        neg_mask = dm_signed < 0
                        if np.any(neg_mask):
                            pmfpart[:, neg_mask] = pmfpart[::-1, neg_mask]

                    apmf = np.abs(pmfpart)
                    med = np.maximum(np.median(apmf, axis=0, keepdims=True), self.eps)

                    if store == "complex":
                        pair_spectra_full[ind_out, sl, :] = pmfpart.T
                    elif store == "abs":
                        pair_spectra_full[ind_out, sl, :] = apmf.T.astype(np.float32)
                    else:
                        pair_spectra_full[ind_out, sl, :] = (apmf / med).T.astype(np.float32)

                ind_out += 1

                if ind_out >= num:
                    yield {
                        "pair_spectra": pair_spectra_full.copy(),
                        "mode_pairs": mode_pairs.copy(),
                        "pair_dm": pair_dm.copy(),
                        "rps_freqs": rps_freqs.copy(),
                    }
                    ind_out = 0
                    pair_spectra_full[:] = 0

                # shift rolling buffer
                ind -= step
                if ind > 0:
                    pmt[:ind] = pmt[step:step + ind]

        if ind_out > 0:
            yield {
                "pair_spectra": pair_spectra_full[:ind_out].copy(),
                "mode_pairs": mode_pairs.copy(),
                "pair_dm": pair_dm.copy(),
                "rps_freqs": rps_freqs.copy(),
            }

    def save_pair_spectra_h5(
        self,
        filename,
        num=1,
        store="norm",
        compression="gzip",
        compression_opts=4,
    ):
        """
        Save all pair spectra to HDF5.

        Datasets
        --------
        pair_spectra : (n_time, n_pairs_total, n_freq)
        mode_pairs   : (n_pairs_total, 2)
        pair_dm      : (n_pairs_total,)
        rps_freqs    : (n_freq,)
        """
        first = True
        n_written = 0

        with h5py.File(filename, "w") as h5:
            for out in self.pair_result(num=num, store=store):
                ps = out["pair_spectra"]

                if first:
                    h5.create_dataset("mode_pairs", data=out["mode_pairs"])
                    h5.create_dataset("pair_dm", data=out["pair_dm"])
                    h5.create_dataset("rps_freqs", data=out["rps_freqs"])

                    h5.create_dataset(
                        "pair_spectra",
                        shape=(0, ps.shape[1], ps.shape[2]),
                        maxshape=(None, ps.shape[1], ps.shape[2]),
                        chunks=(1, ps.shape[1], ps.shape[2]),
                        compression=compression,
                        compression_opts=compression_opts,
                        dtype=ps.dtype,
                    )
                    first = False

                n_new = ps.shape[0]
                h5["pair_spectra"].resize(n_written + n_new, axis=0)
                h5["pair_spectra"][n_written:n_written + n_new] = ps
                n_written += n_new


import numpy as np
import h5py
from scipy.signal import butter, filtfilt
from scipy import fft
from traits.api import Float, Int, Property, cached_property


class RotationalSpeedDetectorAutoAllPairsGrouped(RotationalSpeedDetector):
    """
    Automatically build all available positive dm groups from self.source.modes,
    compute rotational spectra for all valid mode pairs, and store them in a
    grouped HDF5 layout compatible with the grouped pair format.

    Output layout
    -------------
    total_spectrum : (n_out, n_freq)
    pair_spectra   : (n_out, n_dm, n_freq, n_pairs_max)
    pair_modes     : (n_dm, n_pairs_max, 2)
    rps_freqs      : (n_freq,)
    dm_list        : (n_dm,)
    pair_counts    : (n_dm,)   # extra metadata for padded last axis

    Notes
    -----
    - For a labeled pair (a, b), the time-domain product is
          p_a(t) * p_{-b}(t)
      so the mode -b must exist in self.source.modes.
    - Only positive dm = b-a > 0 are used, which avoids mirrored duplicates
      and matches the usual grouped interpretation.
    - The last axis is padded to n_pairs_max because the number of valid pairs
      depends on dm.
    """

    eps = Float(1e-12, desc="small epsilon to avoid division by zero")
    fill_value = Int(-999, desc="fill value for invalid padded pair_modes entries")
    max_abs_dm = Int(0, desc="optional upper limit for automatically used positive dm; 0 means all")

    #: Automatically available positive dm values derived from self.source.modes
    auto_dm_list = Property(depends_on=["source.digest", "max_abs_dm"])

    #: Number of samples in intermediate FFT blocks for the auto-dm setup
    _ns_new_auto = Property(depends_on=["source.sample_freq", "df", "block_size", "auto_dm_list"])

    #: Max index corresponding to fmax on the common rps axis
    _maxfreqind_auto = Property(depends_on=["source.sample_freq", "fmax", "_ns_new_auto"])

    #: Common rotational-frequency axis for the auto-dm setup
    rps_freqs_auto = Property(depends_on=["source.sample_freq", "fmax", "_ns_new_auto"])

    @cached_property
    def _get_auto_dm_list(self):
        dm_list = self._build_auto_dm_list()
        return dm_list

    @cached_property
    def _get__ns_new_auto(self):
        dm_list = self.auto_dm_list
        if dm_list.size == 0:
            raise ValueError("No valid positive dm values available.")
        max_dm = int(dm_list.max())
        return max(int(self.source.sample_freq / self.df) + 1, self.block_size * max_dm)

    @cached_property
    def _get__maxfreqind_auto(self):
        return int(self.fmax * self._ns_new_auto / self.source.sample_freq)

    @cached_property
    def _get_rps_freqs_auto(self):
        fftfreqs = fft.fftfreq(self._ns_new_auto, 1.0 / self.source.sample_freq)
        return np.hstack((fftfreqs[-self._maxfreqind_auto:], fftfreqs[:self._maxfreqind_auto]))

    def _build_auto_dm_list(self):
        """
        Determine all available positive dm values from self.source.modes for which
        at least one valid pair (a, b) exists with b-a=dm and mode -b available.
        """
        modelist = np.asarray(self.source.modes, dtype=int)
        mode_set = set(int(m) for m in modelist)

        dm_values = set()

        for a in modelist:
            a = int(a)
            for b in modelist:
                b = int(b)
                dm = b - a
                if dm <= 0:
                    continue
                if self.max_abs_dm > 0 and dm > self.max_abs_dm:
                    continue
                if -b not in mode_set:
                    continue
                dm_values.add(dm)

        if not dm_values:
            raise ValueError("No valid positive dm values found for the available source.modes.")

        return np.asarray(sorted(dm_values), dtype=int)

    def _build_auto_pair_groups(self):
        """
        Build all valid mode pairs grouped by automatically determined positive dm.

        Returns
        -------
        groups : list of dict
            One dict per dm with keys:
                dm
                a_cols
                negb_cols
                mode_pairs
        pair_modes : ndarray, shape (n_dm, n_pairs_max, 2)
            Padded pair labels (a, b) per dm.
        pair_counts : ndarray, shape (n_dm,)
            Number of valid pairs for each dm.
        dm_arr : ndarray, shape (n_dm,)
            The automatically determined positive dm values.
        """
        modelist = np.asarray(self.source.modes, dtype=int)
        mode_to_col = {int(m): i for i, m in enumerate(modelist)}
        dm_arr = self.auto_dm_list

        groups = []
        pair_counts = []

        for dm in dm_arr:
            entries = []

            for a in modelist:
                a = int(a)
                b = a + int(dm)

                if b not in mode_to_col:
                    continue
                if -b not in mode_to_col:
                    continue

                entries.append((a, b, mode_to_col[a], mode_to_col[-b]))

            entries = sorted(entries, key=lambda x: (x[0], x[1]))

            groups.append({
                "dm": int(dm),
                "a_cols": np.asarray([e[2] for e in entries], dtype=int),
                "negb_cols": np.asarray([e[3] for e in entries], dtype=int),
                "mode_pairs": np.asarray([(e[0], e[1]) for e in entries], dtype=int),
            })
            pair_counts.append(len(entries))

        pair_counts = np.asarray(pair_counts, dtype=int)

        if not np.any(pair_counts > 0):
            raise ValueError("No valid mode pairs found after grouping by automatic dm values.")

        n_dm = len(groups)
        n_pairs_max = int(pair_counts.max())

        pair_modes = np.full((n_dm, n_pairs_max, 2), self.fill_value, dtype=int)
        for k, grp in enumerate(groups):
            n_this = pair_counts[k]
            if n_this > 0:
                pair_modes[k, :n_this, :] = grp["mode_pairs"]

        return groups, pair_modes, pair_counts, dm_arr

    def result(self, num):
        """
        Yield only the aggregated total_spectrum, analogous to RotationalSpeedDetector.result().
        """
        for out in self.pair_result(num=num, store="norm"):
            yield out["total_spectrum"]

    def pair_result(self, num, store="norm"):
        """
        Yield grouped all-pairs spectra with automatically determined dm groups.

        Parameters
        ----------
        num : int
            Number of output time steps per yielded block.
        store : {"norm", "abs", "complex"}
            What to store in pair_spectra:
              - "norm":    abs(pmfpart) / median(abs(pmfpart), axis=0)
              - "abs":     abs(pmfpart)
              - "complex": complex pmfpart itself

        Yields
        ------
        dict with:
            total_spectrum : (n_out, n_freq)
            pair_spectra   : (n_out, n_dm, n_freq, n_pairs_max)
            pair_modes     : (n_dm, n_pairs_max, 2)
            rps_freqs      : (n_freq,)
            dm_list        : (n_dm,)
            pair_counts    : (n_dm,)
        """
        if store not in {"norm", "abs", "complex"}:
            raise ValueError("store must be one of: 'norm', 'abs', 'complex'")

        bs = self.block_size
        step = self.sample_step
        nmodes = len(self.source.modes)

        groups, pair_modes, pair_counts, dm_arr = self._build_auto_pair_groups()

        f_low, f_high = self.band_pass
        if f_low > 0:
            b, a = butter(8, f_low / self.source.sample_freq * 2, btype="highpass")
        if f_high > 0:
            bh, ah = butter(8, f_high / self.source.sample_freq * 2, btype="lowpass")

        mf_ind = self._maxfreqind_auto
        n_freq = 2 * mf_ind
        n_dm = len(groups)
        n_pairs_max = pair_modes.shape[1]
        ns_new_auto = self._ns_new_auto

        result_full = np.zeros((num, n_freq), dtype=float)

        if store == "complex":
            pair_spectra_full = np.zeros((num, n_dm, n_freq, n_pairs_max), dtype=np.complex64)
        else:
            pair_spectra_full = np.zeros((num, n_dm, n_freq, n_pairs_max), dtype=np.float32)

        result_step = np.zeros(n_freq, dtype=float)
        pmt = np.zeros((bs + num, nmodes), dtype=np.complex64)

        ind = 0
        ind_out = 0

        for block in self.source.result(num):
            ns = block.shape[0]
            ind1 = ind + ns

            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]

            ind = ind1

            if ind >= bs:
                # optional band-pass
                if f_low > 0:
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high > 0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for k, grp in enumerate(groups):
                    dm = grp["dm"]
                    n_this = pair_counts[k]

                    if n_this == 0:
                        continue

                    a_cols = grp["a_cols"]
                    negb_cols = grp["negb_cols"]

                    # one product signal per pair: p_a(t) * p_-b(t)
                    pmtprod = pmt_temp[:, a_cols] * pmt_temp[:, negb_cols]

                    # common rotational-frequency mapping for this dm group
                    ns_full = int(ns_new_auto / dm)
                    pmf = fft.fft(
                        np.pad(pmtprod, [(0, ns_full - bs), (0, 0)]),
                        None,
                        axis=0,
                        norm="ortho",
                    )

                    pmfpart = np.zeros((n_freq, n_this), dtype=np.complex64)
                    pmfpart[:mf_ind, :] = pmf[-mf_ind:, :]
                    pmfpart[mf_ind:, :] = pmf[:mf_ind, :]

                    apmf = np.abs(pmfpart)
                    med = np.maximum(np.median(apmf, axis=0, keepdims=True), self.eps)
                    pair_norm = apmf / med

                    # store per-pair spectra in grouped layout
                    if store == "complex":
                        pair_spectra_full[ind_out, k, :, :n_this] = pmfpart
                    elif store == "abs":
                        pair_spectra_full[ind_out, k, :, :n_this] = apmf.astype(np.float32)
                    else:
                        pair_spectra_full[ind_out, k, :, :n_this] = pair_norm.astype(np.float32)

                    # aggregate to total_spectrum in the same spirit as the base detector
                    if n_this > 1:
                        pmfanglediff = np.sum(
                            np.abs(np.angle(pmfpart[:, 1:] * pmfpart[:, :-1].conjugate())),
                            axis=1,
                        )
                        pmfanglediff = np.maximum(pmfanglediff, self.eps)
                    else:
                        pmfanglediff = np.ones(n_freq, dtype=float)

                    result_dm = (1.0 / pmfanglediff) * np.sum(pair_norm, axis=1)
                    result_step += result_dm

                result_full[ind_out, :] = result_step
                result_step[:] = 0.0
                ind_out += 1

                if ind_out >= num:
                    yield {
                        "total_spectrum": result_full.copy(),
                        "pair_spectra": pair_spectra_full.copy(),
                        "pair_modes": pair_modes.copy(),
                        "rps_freqs": self.rps_freqs_auto.copy(),
                        "dm_list": dm_arr.copy(),
                        "pair_counts": pair_counts.copy(),
                    }
                    ind_out = 0
                    result_full[:] = 0.0
                    pair_spectra_full[:] = 0.0

                # shift rolling buffer
                ind -= step
                if ind > 0:
                    pmt[:ind] = pmt[step:step + ind]

        if ind_out > 0:
            yield {
                "total_spectrum": result_full[:ind_out].copy(),
                "pair_spectra": pair_spectra_full[:ind_out].copy(),
                "pair_modes": pair_modes.copy(),
                "rps_freqs": self.rps_freqs_auto.copy(),
                "dm_list": dm_arr.copy(),
                "pair_counts": pair_counts.copy(),
            }

    def save_pair_spectra_h5(
        self,
        filename,
        num=1,
        store="norm",
        compression="gzip",
        compression_opts=4,
    ):
        """
        Save grouped automatic all-pairs spectra to HDF5.

        Datasets
        --------
        total_spectrum : (n_time, n_freq)
        pair_spectra   : (n_time, n_dm, n_freq, n_pairs_max)
        pair_modes     : (n_dm, n_pairs_max, 2)
        rps_freqs      : (n_freq,)
        dm_list        : (n_dm,)
        pair_counts    : (n_dm,)
        """
        first = True
        n_written = 0

        with h5py.File(filename, "w") as h5:
            for out in self.pair_result(num=num, store=store):
                total = out["total_spectrum"]
                pairs = out["pair_spectra"]

                if first:
                    h5.create_dataset("rps_freqs", data=out["rps_freqs"])
                    h5.create_dataset("dm_list", data=out["dm_list"])
                    h5.create_dataset("pair_modes", data=out["pair_modes"])
                    h5.create_dataset("pair_counts", data=out["pair_counts"])

                    h5["pair_modes"].attrs["fill_value"] = int(self.fill_value)

                    h5.create_dataset(
                        "total_spectrum",
                        shape=(0, total.shape[1]),
                        maxshape=(None, total.shape[1]),
                        chunks=(1, total.shape[1]),
                        compression=compression,
                        compression_opts=compression_opts,
                        dtype=total.dtype,
                    )

                    h5.create_dataset(
                        "pair_spectra",
                        shape=(0, pairs.shape[1], pairs.shape[2], pairs.shape[3]),
                        maxshape=(None, pairs.shape[1], pairs.shape[2], pairs.shape[3]),
                        chunks=(1, pairs.shape[1], pairs.shape[2], pairs.shape[3]),
                        compression=compression,
                        compression_opts=compression_opts,
                        dtype=pairs.dtype,
                    )

                    first = False

                n_new = total.shape[0]

                h5["total_spectrum"].resize(n_written + n_new, axis=0)
                h5["total_spectrum"][n_written:n_written + n_new] = total

                h5["pair_spectra"].resize(n_written + n_new, axis=0)
                h5["pair_spectra"][n_written:n_written + n_new] = pairs

                n_written += n_new

        
class RotationalSpeedDetector2( RotationalSpeedDetector ):
    """
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    This version does spline interpolation instead of FFT with zero-padding
    
    --> This is way faster, but not as accurate as RSD(1)
    --> It is not as as accurate as RSD3
    --> RSD2 not to be used for now
    """       
        
    def result( self, num ):
        """
        see "12_frot_detection_weight_and_combine.py" for tests
        assumes that input is in mode order
        """
        
        bs = self.block_size
        step = self.sample_step
        modelist = self.source.modes
        nmodes = len(modelist)
        fmax= self.fmax
        
        f_low, f_high = self.band_pass
                                ### band pass filter ##############################
        if f_low>0:  
            b, a = butter(8, f_low/self.source.sample_freq*2, btype='highpass')
    
        
        nfreqs = 2 * int(fmax / self.df)
        ffreqs = linspace(-fmax,fmax,nfreqs,endpoint=False)

        
        # allocate arrays that hold the needed data
        # largest array that is needed
        maxmaxind = int(fmax / self.source.sample_freq * bs * absolute(self.dm_list).max())
        pmfpart = zeros((2*maxmaxind, self.m_max+1), dtype=complex)
        
        result_full = zeros((num, nfreqs))
        
        #result_step = ones(nfreqs)
        result_step = zeros(nfreqs)
        
        pmt = zeros((self.block_size+num,nmodes), dtype=complex)
        
        ind = 0
        ind_out = 0
        
        for block in self.source.result( num ):
            #print(ind,end=',')
            ns = block.shape[0]
            ind1 = ind+ns
            
            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]
            
            ind = ind1
                        
            if ind >= self.block_size:
                # highpass
                if f_low>0:  
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for dm in self.dm_list:
                    
                    # get index that corresponds to frequency for this mode
                    maxind = int(fmax / self.source.sample_freq * bs * abs(dm))
                    # calculate cross-mode product, only need half of the modes, since everything is mirrored
                    # shift left/right, so that all mode combinations are present (and none doubled) in the resulting array
                    ind_m_b = roll(modelist[::-1], int(0.5-dm/2))[:nmodes//2]
                    ind_m_a = roll(modelist      , int(0.0+dm/2))[:nmodes//2]
                    pmtprod = pmt_temp[:,ind_m_a] * pmt_temp[:,ind_m_b]
                
                    #ns_full = int(self._ns_new/dm)
                    #pmf = fft.fft(pad(pmtprod,[(0, ns_full-bs),(0, 0)]), None, 0, norm="ortho")
                    pmf = fft.fft(pmtprod, None, 0, norm="ortho")
                    
                    
                    # fill array with only part of the data
                    pmfpart[:maxind, :] = pmf[-maxind:, :self.m_max+1]
                    pmfpart[maxind:2*maxind, :] = pmf[ :maxind, :self.m_max+1]
                
                    # weighting array
                    # derivative of angle -> small if rotational velocity is found
                    pmfanglediff=-sum(abs(angle(pmfpart[:2*maxind,1:] * pmfpart[:2*maxind,:-1].conjugate())), axis=1)
                    # most values will not correspond to a rotation, so subtract average
                    pmfanglediff-=pmfanglediff.min()*1.1
                    # normalize max. to 1
                    pmfanglediff/=pmfanglediff.max()
                    
                    #get result for this mode
                    #result_dm = pmfanglediff**4 * absolute(prod(pmfpart[:2*maxind], axis=1))
                    #result_dm = pmfanglediff**4 * absolute(prod(pmfpart[:2*maxind], axis=1))
                    result_dm = pmfanglediff * sum(absolute(pmfpart[:2*maxind]), axis=1)
                    #               result_dm/result_dm.max()  
                    finter = interp1d(linspace(-fmax,fmax,2*maxind,endpoint=False),
                                      result_dm,
                                      kind = 'nearest',
                                      fill_value="extrapolate")
                    result_dm_inter = finter(ffreqs)
                    # average to get overall result
                    #result_step *= result_dm_inter/result_dm_inter.max()
                    result_step += result_dm_inter#/result_dm_inter.max()
                result_full[ind_out,:] = result_step#/result_step.max()
                #result_full[ind_out,:] = (result_step/result_step.max())**40
                result_step[:] = 0
                ind_out += 1
                if ind_out >= num:
                    yield result_full
                    ind_out = 0
            
                # shift rest of data according to step size
                ind -= step#self.block_size
                if ind > 0:
                    pmt[:ind] = pmt[step:step+ind]
                
        if ind_out > 0:
            yield result_full[:ind_out]
        


class RotationalSpeedDetector3( TimeOut ):
    """
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    --> about 4x faster than RSD(1), but not as accurate. 
    --> more accurate than RSD2
    """
    
    #: Maximum delta f of returned spectra in Hz
    df = Float(0.1)
    
    #: number of samples in one block for rotation detection
    block_size = Int(10000)
    
    #: Upper rotational frequency limit that should be resolved
    fmax = Float(100.0)
    
    #: Tuple with (lower, upper) frequency for band pass prior to cross correlation
    band_pass = Tuple((2500.0, 0.0))
    
    #: Result will be yielded for every `sample_step`-th sample. 
    #: If >1, this will result in a decreased `sample_freq`.
    sample_step = Int(10000)

    #: List of delta m (=b-a) to consider
    dm_list = List( [1,3,5,7,9] )#
    
    #: maximum cross mode to consider (max. nmodes/2 )
    m_max = Int(26)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Property( depends_on = ['source.sample_freq', 
                                          'sample_step'])
    
    #: Discrete resolved rotational frequencies
    rps_freqs = Property( depends_on = ['df', 'fmax'])
    
    # number of samples in output
    _ns_new = Property( depends_on = ['source.sample_freq', 'df'])
    

    @cached_property 
    def _get_sample_freq( self ):
        return self.source.sample_freq/self.sample_step
    
    @cached_property 
    def _get__ns_new( self ):
        # number of samples to return
        return int(ceil(2*self.fmax/self.df))

   
    @cached_property 
    def _get_rps_freqs( self ):
        return roll(fft.fftfreq(self._ns_new, 1./(self.df*self._ns_new)),self._ns_new//2)
        
    
    
    def result( self, num ):
        """
        see "12_frot_detection_weight_and_combine.py" for tests
        assumes that input is in mode order
        """
        
        
        
        bs = self.block_size
        fs = self.source.sample_freq
        step = self.sample_step
        modelist = self.source.modes
        nmodes = len(modelist)
        ns_new = self._ns_new
        
        f_low, f_high = self.band_pass
                                ### band pass filter ##############################
        if f_low>0:  
            b, a = butter(8, f_low/fs*2, btype='highpass')
        #TODO: f_high currently not used 
        if f_high>0:
            bh, ah = butter(8, f_high/self.source.sample_freq*2, btype='lowpass')
        #    data = filtfilt(b, a, data, axis=0)
        #if f_high>0:
        #    b, a = butter(8, f_high/fs*2, btype='lowpass')
        #    data = filtfilt(b, a, data, axis=0)
        
        # allocate arrays that hold the needed data
        result_full = zeros((num, ns_new))
        
        result_step = zeros(ns_new)
        
        pmt = zeros((self.block_size+num,nmodes), dtype=complex)
        
        
        # fill array with only part of the data
        #pmfpart[:mf_ind, :] = pmf[-mf_ind:, :self.m_max+1]
        #pmfpart[mf_ind:, :] = pmf[ :mf_ind, :self.m_max+1]
        
        #TODO 1: Diese Variante ist durch alleiniges Downsampling zwar schnell, aber doch etwas ungenau
        #        Ggf. intelligenteres (fractional) Downsampling?
        
        #TODO 2: Durch anti-aliasing-Fitler bei decimate gibt es starken Amplitudenabfall
        #        nach außen, d.h. fmax muss deutlich erhöht werden, um noch auswerten zu können
        
        
        ind = 0
        ind_out = 0
        
        
        for block in self.source.result( num ):
            #print(ind,end=',')
            ns = block.shape[0]
            ind1 = ind+ns
            
            if ind > 0:
                pmt[ind:ind1] = block
            elif ind1 > 0:
                pmt[:ind1] = block[-ind1:]
            
            ind = ind1
                        
            if ind >= bs:
                # highpass
                if f_low>0:  
                    pmt_temp = filtfilt(b, a, pmt[:bs], axis=0)
                    if f_high>0:
                        pmt_temp = filtfilt(bh, ah, pmt_temp, axis=0)
                else:
                    pmt_temp = pmt[:bs]

                for dm in self.dm_list:
            
                    # calculate cross-mode product, only need half of the modes, since everything is mirrored
                    # shift left/right, so that all mode combinations are present (and none doubled) in the resulting array
                    ind_m_b = roll(modelist[::-1], int(0.5-dm/2))[:nmodes//2]
                    ind_m_a = roll(modelist      , int(0.0+dm/2))[:nmodes//2]
                    pmtprod = pmt_temp[:,ind_m_a] * pmt_temp[:,ind_m_b]
                    
                    # subsampling factor
                    D = int(round(fs/(2*self.fmax*dm)))
                    
                    pmtprod = decimate(pmtprod,D, axis=0)
                    ns_down = pmtprod.shape[0]#int(ceil(ns/D))
                    #fs_down = fs / D
                    #print('dm:',dm,', D:',D,', ns_down:',ns_down)                    
                    
                    npad = ns_new-ns_down
                    if npad > 0:
                        pmf = fft.fft(pad(pmtprod,[(0, npad),(0, 0)]), None, axis=0, norm="ortho")
                    else:
                        print("df to high!!! Discarding data!")
                        pmf = fft.fft(pmtprod[:ns_new], None, axis=0, norm="ortho")
                
                
                    # weighting array
                    # derivative of angle -> small if rotational velocity is found
                    pmfanglediff=sum(abs(angle(pmf[:,1:] * pmf[:,:-1].conjugate())), axis=1)
                    
                    #get result for this mode
                    apmf = absolute(pmf)
                    result_dm = 1/pmfanglediff * sum(apmf/median(apmf[:ns_new//4],axis=0), axis=1)
                    
                    # average to get overall result
                    result_step += result_dm
                result_full[ind_out,:] = roll(result_step,ns_new//2)
                result_step *= 0
                ind_out += 1
                if ind_out >= num:
                    yield result_full
                    ind_out = 0
            
                # shift rest of data according to step size
                ind -= step#self.block_size
                if ind > 0:
                    pmt[:ind] = pmt[step:step+ind]
                
        if ind_out > 0:
            yield result_full[:ind_out]



class RotationalSpeed ( TimeOut ):
    '''
    Class to calculate rotational speed for each sample. 
    This implementation is preliminary, interface will change in the future.
    TODO: Maybe include real-time trigger identification.
    '''
    
    #: Number of successive rps values to average over, defaults to 4.
    num_per_average = Int(4, 
        desc = "number of samples to average over")

    #: Trigger data from :class:`acoular.tprocess.Trigger`.
    trigger = Instance(Trigger) 

    # delay of rot signal (due to distance to array)
    delay = Float(0.0)

    #: Trigger signals per revolution,
    #: defaults to 1.
    trigger_per_revo = Int(1,
                           desc ="trigger signals per revolution")
    
    #: Flag to set counter-clockwise (1) or clockwise (-1) rotation,
    #: defaults to -1.
    rot_direction = Int(-1,
                        desc ="mathematical direction of rotation")
    
    # internal identifier
    digest = Property(depends_on=['source.digest', 
                                  'trigger.digest', 
                                  'trigger_per_revo',
                                  'rot_direction',
                                  'start_angle'])

    @cached_property 
    def _get_digest( self ):
        return digest(self)
    
    def result(self, num, rpn = False):
        """
        Python generator that yields the beamformer output block-wise and
        interpolates data at positions of virtually rotating array.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) 
        
        rpn : bool
            If true, yield result in revolution per sample instead of rev per second. 
        Returns
        -------
        Samples in blocks of shape (num, num_channels). 
            The last block may be shorter than num.
        """
        
        if rpn:
            unit = 1
        else:
            unit = self.sample_freq
        
        self.trigger.source = self.source
        ns = self.num_samples
        
        inds_trigger, maxdist, mindist = self.trigger.trigger_data
        # trigger distances in samples
        dtrig = inds_trigger[1:]-inds_trigger[:-1]
        
        #average rpn between triggers
        rpn_avg = 1/dtrig
        
        ###########
        # # calculate moving average of rps
        # rps_rolling_mean = uniform_filter1d(rpn_avg, 
        #                                     self.naverage, 
        #                                     mode='nearest', 
        #                                     origin=-1)  \
        #                    * unit * self.rot_direction
        
        # # setup interpolation function to calculate per-sample
        # linear_fun = interp1d(inds_trigger[1:],
        #                       rps_rolling_mean,
        #                       fill_value="extrapolate")
        
        # #rps_track = linear_fun(arange(ns)) * self.sample_freq
        # #...yield rps_track[ind*num:(ind+1)*num]
        
        # indarray = arange(num)
        # lastblockind = ns%num
        # for ind in range(ns//num):
        #     yield linear_fun(indarray+ind*num)
        # else:
        #     if lastblockind: yield linear_fun(indarray[:lastblockind]+(ind+1)*num)
        ############
        
        rpn_track1 = zeros(ns)

        # set time before first trigger to same rps as after
        rpn_track1[:inds_trigger[0]] += rpn_avg[0]
        ind0 = inds_trigger[0]
        for ind1, rpni in zip(inds_trigger[1:], rpn_avg):
            rpn_track1[ind0:ind1] += rpni
            ind0 = ind1
        # set time after last trigger to same rps as before
        rpn_track1[ind1:] += rpni
        
        # rpn or rps
        rpn_track1 *= unit * self.rot_direction
        nroll =  int(round(abs(self.delay*self.sample_freq)))
        print('roll samples:',nroll)
        rpn_track1 = roll( rpn_track1, nroll)
        
        nblocks = ns//num + (ns%num>0)
        for ind in range(nblocks):
            yield rpn_track1[ind*num:(ind+1)*num]