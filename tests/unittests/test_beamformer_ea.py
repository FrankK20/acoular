# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Correctness tests for :class:`acoular.fbeamform.BeamformerEA`."""

import acoular as ac

import numpy as np
import pytest

BOUNDS = [(-0.5, 0.5), (-0.5, 0.5), (0.3, 1.0)]
SRC_POS = np.array([[0.2], [-0.1], [0.6]])
SRC_STRENGTH = np.array([1e-4])
FREQ = 2000.0


@pytest.fixture(scope='module')
def mics():
    """Planar random array with 16 microphones and an aperture of 1 m."""
    rng = np.random.RandomState(1)
    return ac.MicGeom(pos_total=np.vstack([rng.uniform(-0.5, 0.5, (2, 16)), np.zeros((1, 16))]))


def synthetic_freq_data(mics, pos, strength, freqs):
    """
    Noise-free CSM of uncorrelated monopoles, C = sum_k q_k h_k h_k^H.

    Parameters
    ----------
    mics : acoular.MicGeom
        Microphone geometry.
    pos : numpy.ndarray
        Source positions, shape (3, n).
    strength : numpy.ndarray
        Squared sound pressures of the sources at the reference position, shape (n,).
    freqs : numpy.ndarray
        Frequencies.
    """
    steer = ac.SteeringVector(grid=ac.ImportGrid(pos=pos), mics=mics)
    csm = np.array([(steer.transfer(f).T * strength) @ steer.transfer(f).conj() for f in freqs])
    return ac.PowerSpectraImport(csm=csm, frequencies=freqs)


@pytest.fixture(scope='module')
def single_source_freq_data(mics):
    """Noise-free CSM of a single source."""
    return synthetic_freq_data(mics, SRC_POS, SRC_STRENGTH, np.array([FREQ]))


@pytest.mark.parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
def test_energy_vanishes_at_true_source(mics, single_source_freq_data, r_diag):
    """The energy is zero for the true parameters and positive elsewhere."""
    bf = ac.BeamformerEA(freq_data=single_source_freq_data, steer=ac.SteeringVector(mics=mics), r_diag=r_diag)
    # a single source that explains the whole auto power has normalized strength 1
    assert bf.ecsm(np.r_[SRC_POS[:, 0], 1.0], 0) < 1e-28
    assert bf.ecsm(np.r_[SRC_POS[:, 0] + [0.05, 0.0, 0.0], 1.0], 0) > 1e-3
    assert bf.ecsm(np.r_[SRC_POS[:, 0], 0.5], 0) == pytest.approx(0.25)
    with pytest.raises(ValueError, match='x must have shape'):
        bf.ecsm(np.zeros(3), 0)


@pytest.mark.parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
def test_single_source_recovery(mics, single_source_freq_data, r_diag):
    """Position and strength of a single source are recovered in the noise-free case."""
    bf = ac.BeamformerEA(
        freq_data=single_source_freq_data,
        steer=ac.SteeringVector(mics=mics),
        bounds=BOUNDS,
        r_diag=r_diag,
        maxiter=200,
        cached=False,
    )
    result = bf.synthetic(FREQ, 0)
    np.testing.assert_allclose(bf.pos, SRC_POS, atol=1e-6)
    np.testing.assert_allclose(result, SRC_STRENGTH, rtol=1e-6)
    assert bf.ecsm(np.r_[bf.pos[:, 0], 1.0], 0) < 1e-12


@pytest.mark.parametrize('r_diag', [False, True], ids=['rdiag-False', 'rdiag-True'])
def test_two_source_recovery(mics, r_diag):
    """Positions and strengths of two uncorrelated sources are recovered in the noise-free case."""
    pos = np.array([[-0.3, 0.2], [0.25, -0.1], [0.7, 0.6]])
    strength = np.array([5e-5, 1e-4])
    freq_data = synthetic_freq_data(mics, pos, strength, np.array([FREQ]))
    bf = ac.BeamformerEA(
        freq_data=freq_data,
        steer=ac.SteeringVector(mics=mics),
        bounds=BOUNDS,
        n=2,
        r_diag=r_diag,
        maxiter=300,
        cached=False,
    )
    result = bf.synthetic(FREQ, 0)
    order = np.argsort(bf.pos[0])
    np.testing.assert_allclose(bf.pos[:, order], pos, atol=1e-6)
    np.testing.assert_allclose(result[order], strength, rtol=1e-6)


def test_time_domain_source_recovery(mics):
    """A point source simulated in the time domain is found where BeamformerBase locates it."""
    source = ac.PointSource(
        signal=ac.WNoiseGenerator(sample_freq=51200, num_samples=51200, seed=1),
        mics=mics,
        loc=tuple(SRC_POS[:, 0]),
    )
    freq_data = ac.PowerSpectra(source=source, block_size=256, window='Hanning', cached=False)
    find = 10
    f = freq_data.fftfreq()[find]
    bf = ac.BeamformerEA(
        freq_data=freq_data, steer=ac.SteeringVector(mics=mics), bounds=BOUNDS, maxiter=150, cached=False
    )
    result = bf.synthetic(f, 0)
    pos = bf.pos[:, find]
    np.testing.assert_allclose(pos, SRC_POS[:, 0], atol=1e-2)
    # strength agrees with the level of BeamformerBase at the true source position
    steer = ac.SteeringVector(mics=mics, grid=ac.ImportGrid(pos=SRC_POS), steer_type='true level')
    base = ac.BeamformerBase(freq_data=freq_data, steer=steer, r_diag=False, cached=False).synthetic(f, 0)
    np.testing.assert_allclose(result.sum(), base[0], rtol=1e-2)


def test_comparison_with_base_and_cleansc(mics, single_source_freq_data):
    """BeamformerEA, BeamformerBase and BeamformerCleansc agree for a source on a grid point."""
    grid = ac.RectGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=0.6, increment=0.05)
    steer = ac.SteeringVector(grid=grid, mics=mics, steer_type='true level')
    ea = ac.BeamformerEA(
        freq_data=single_source_freq_data, steer=steer, bounds=BOUNDS, r_diag=False, maxiter=200, cached=False
    )
    ea_result = ea.synthetic(FREQ, 0)
    base = ac.BeamformerBase(freq_data=single_source_freq_data, steer=steer, r_diag=False, cached=False)
    cleansc = ac.BeamformerCleansc(freq_data=single_source_freq_data, steer=steer, cached=False)
    for bf, rtol in [(base, 1e-6), (cleansc, 1e-2)]:
        bf_map = bf.synthetic(FREQ, 0)
        imax = np.argmax(bf_map)
        np.testing.assert_allclose(grid.pos[:, imax], ea.pos[:, 0], atol=grid.increment / 2)
        np.testing.assert_allclose(bf_map.ravel()[imax], ea_result[0], rtol=rtol)


def test_no_side_effects_and_caching(mics, single_source_freq_data, mocker):
    """The steering vector is not modified and the result is calculated only once."""
    grid = ac.RectGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=0.6, increment=0.05)
    steer = ac.SteeringVector(grid=grid, mics=mics)
    steer_digest = steer.digest
    bf = ac.BeamformerEA(freq_data=single_source_freq_data, steer=steer, bounds=BOUNDS, maxiter=10, cached=False)
    bf_digest = bf.digest
    spy = mocker.spy(ac.BeamformerEA, '_calc')
    bf.synthetic(FREQ, 0)
    bf.synthetic(FREQ, 0)
    assert bf.result[0].sum() > 0
    assert spy.call_count == 1
    assert steer.grid is grid
    assert steer.digest == steer_digest
    assert bf.digest == bf_digest


def test_determinism(mics, single_source_freq_data):
    """Results are reproducible for a fixed seed and depend on the seed."""

    def run(seed):
        bf = ac.BeamformerEA(
            freq_data=single_source_freq_data,
            steer=ac.SteeringVector(mics=mics),
            bounds=BOUNDS,
            seed=seed,
            maxiter=5,
            kwargs={'polish': False},
            cached=False,
        )
        return bf.synthetic(FREQ, 0), bf.pos.copy(), bf.digest

    result1, pos1, digest1 = run(3)
    result2, pos2, digest2 = run(3)
    result3, pos3, digest3 = run(4)
    np.testing.assert_array_equal(result1, result2)
    np.testing.assert_array_equal(pos1, pos2)
    assert digest1 == digest2
    assert digest1 != digest3
    assert not np.array_equal(pos1, pos3)
