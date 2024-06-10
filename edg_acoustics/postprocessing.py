"""This module provides postprocessing functionalities for the edg_acoustics package.
"""

import numpy
import scipy


__all__ = ["Monopole_postprocessor", "Sampling_Freq"]

Sampling_Freq = 44100
"""float: Default value of the sampling frequency, set as 44100 Hz."""


class Monopole_postprocessor:
    """Postprocessor for monopole source simulation results.

    :class:`.Monopole_postprocessor` is used to postprocess the simulation results of a monopole source, mainly to correct the source spectrum.
    Furthermore, it collects simulation results to be saved.

    Args:
        sim (edg_acoustics.AcousticsSimulation): The acoustic simulation object instance of :class:`edg_acoustics.AcousticsSimulation`.
        delta_step (float): Factor by which the simulation results are saved.
        sampling_freq (float): The desired sampling frequency. Default is set to Sampling_Freq = 44100 Hz.

    Attributes:
        sim (edg_acoustics.AcousticsSimulation): The acoustic simulation object instance of :class:`edg_acoustics.AcousticsSimulation`.
        dt_old (float): The time step size of the simulation results.
        fs_old (float): The sampling frequency of the simulation results.
        sampling_freq (float): The desired sampling frequency. Default is set to Sampling_Freq = 44100 Hz.
        dt_new (float): The time step size of the desired sampling frequency.
        IRold (numpy.ndarray): The impulse response at the receiver locations.
        IRnew (numpy.ndarray): The resampled impulse response at the receiver locations.
        TR_original (numpy.ndarray): The original transfer function at the receiver locations.
        TR_free (numpy.ndarray): The free transfer function at the receiver locations.
        TR (numpy.ndarray): The corrected transfer function at the receiver locations.
        freqs (numpy.ndarray): The frequency vector.
    """

    def __init__(self, sim, delta_step, sampling_freq=Sampling_Freq):
        self.sim = sim
        self.IRold = sim.prec
        self.dt_old = sim.time_integrator.dt * delta_step
        self.fs_old = round(1 / self.dt_old)
        self.sampling_freq = sampling_freq
        self.dt_new = 1 / sampling_freq

        if self.fs_old != self.sampling_freq:
            self.IRnew = self.apply_resample()
        else:
            self.IRnew = self.IRold

    def apply_resample(self):
        """Resamples the impulse response to the desired sampling frequency.
        Returns:
            IRnew (numpy.ndarray): see :attr:`IRnew`.
        """
        self.IRnew = scipy.signal.resample(
            self.IRold, int(self.IRold.shape[1] * self.sampling_freq / self.fs_old), axis=1
        )
        return self.IRnew

    def apply_correction(self, next_fast_len=False):
        """Corrects the source spectrum of the recorded impulse response.

        Args:
            next_fast_len (bool): If True, the next fast length of input data is used for fft, for zero-padding, etc. Consequently, the singal length is prolonged. Default is False.

        Returns:
            IRnew (numpy.ndarray): see :attr:`IRnew`.
            TR (numpy.ndarray): see :attr:`TR`.
            freqs (numpy.ndarray): see :attr:`freqs`.
        """

        time_vector = numpy.arange(0, self.IRnew.shape[1] * self.dt_new, self.dt_new)
        R = numpy.sqrt(sum((self.sim.IC.source_xyz[:, numpy.newaxis] - self.sim.rec) ** 2))[
            :, numpy.newaxis
        ]
        pout = (
            (R - self.sim.c0 * time_vector)
            / (2 * R)
            * numpy.exp(
                -numpy.log(2) * ((R - self.sim.c0 * time_vector) ** 2) / self.sim.IC.halfwidth**2
            )
        )
        pin = (
            (R + self.sim.c0 * time_vector)
            / (2 * R)
            * numpy.exp(
                -numpy.log(2) * ((R + self.sim.c0 * time_vector) ** 2) / self.sim.IC.halfwidth**2
            )
        )
        p_free = pout + pin

        if next_fast_len:
            n_samples = scipy.fft.next_fast_len(self.IRnew.shape[1])
        else:
            n_samples = self.IRnew.shape[1]

        n_samples = scipy.fft.next_fast_len(self.IRnew.shape[1])
        self.TR_original = scipy.fft.fft(self.IRnew, n=n_samples, axis=1)
        self.TR_free = scipy.fft.fft(p_free, n=n_samples, axis=1)
        self.freqs = scipy.fft.fftfreq(n_samples, self.dt_new)
        wavenumber = 2 * numpy.pi * self.freqs / self.sim.c0
        monopole = numpy.exp(-1j * wavenumber * R) / (4 * numpy.pi * R)
        self.TR = self.TR_original / self.TR_free * monopole
        self.IRnew = numpy.real(scipy.fft.ifft(self.TR))

        return self.IRnew, self.TR, self.freqs
