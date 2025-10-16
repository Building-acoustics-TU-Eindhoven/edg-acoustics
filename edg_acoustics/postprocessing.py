"""This module provides postprocessing functionalities for the edg_acoustics package.
"""

import os
import numpy
import scipy

from scipy.signal import resample_poly
from scipy.fft import fft, ifft
from math import log, pi

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
        self.dt_old = sim.time_integrator.dt
        self.fs_old = round(1 / self.dt_old)
        self.sampling_freq = sampling_freq
        self.dt_new = 1 / sampling_freq

        if self.fs_old != self.sampling_freq:
            self.IRnew = self.apply_resample()
        else:
            self.IRnew = self.IRold
            
    def cconv_python(self, x, h):
        """
        MATLAB-like circular convolution:
        - Output length = len(x) + len(h) - 1
        - Circular (wrap-around) convolution
        """
        x = numpy.asarray(x)
        h = numpy.asarray(h)
        N = len(x) + len(h) - 1

        # Zero-pad both to length N
        x_padded = numpy.pad(x, (0, N - len(x)), mode='constant')
        h_padded = numpy.pad(h, (0, N - len(h)), mode='constant')

        # Perform circular convolution via FFT
        y = numpy.fft.ifft(numpy.fft.fft(x_padded) * numpy.fft.fft(h_padded)).real
        return y


    def fftshift_matlab(self, x):
        """Emulate MATLAB's fftshift in Python (left-biased for even-length arrays)."""
        n = x.shape[-1]
        p2 = int(numpy.floor(n / 2))
        return numpy.roll(x, -p2, axis=-1)


    def apply_resample(self):
        """Resamples the impulse response to the desired sampling frequency.

        Returns:
            IRnew (numpy.ndarray): see :attr:`IRnew`.
        """
        self.IRnew = scipy.signal.resample(
            self.IRold, int(self.IRold.shape[1] * self.sampling_freq / self.fs_old), axis=1
        )
        return self.IRnew

    def generate_blackman_windowed_sinc(self, lower_limit, upper_limit, sample_rate, M):
        x = numpy.arange(-0.5 * M, 0.5 * M + 1)
        x2 = numpy.arange(0, M + 1)
        
        fc = [lower_limit / sample_rate, upper_limit / sample_rate]
        A = []
        del_idx = numpy.where(x == 0)[0][0]  # index where x == 0
        
        for i in range(2):
            sinc_term = numpy.sin(2 * numpy.pi * fc[i] * x) / x
            sinc_term[del_idx] = 2 * numpy.pi * fc[i]  # replace NaN with limit value

            blackman_window = (0.42 - 0.5 * numpy.cos(2 * numpy.pi * x2 / M) +
                            0.08 * numpy.cos(4 * numpy.pi * x2 / M))
            
            A_row = sinc_term * blackman_window
            A_row = A_row / numpy.sum(A_row)
            
            if i == 1:
                A_row *= -1
                A_row[del_idx] += 1

            A.append(A_row)

        B = A[0] + A[1]
        B *= -1
        B[del_idx] += 1

        return B

    def time_windowing(self, signal, Nleft, Nright):
        
        nbt = len(signal)
        
        Nleft = int(numpy.ceil(Nleft))
        Nright = int(min(numpy.floor(Nright), numpy.floor(nbt / 2)))
        
        # Find first and last non-zero sample per row
        if sum(abs(signal)) == 0:
            it0 = 0
        else:
            for i in range(nbt):
                if signal[i] != 0:
                    it0 = i
                    break
                
        if sum(abs(signal)) == 0:
            it3 = nbt - 1
        else:
            for i in reversed(range(nbt)):
                if signal[i] != 0:
                    it3 = i
                    break

        it1 = it0 + Nleft
        it2 = it3 - Nright

        t = numpy.arange(nbt)
        win = numpy.zeros_like(signal)

        if it1 > nbt: it1 = nbt
        if it2 > nbt: it2 = nbt

        if it1 > it0:
            win[it0:it1] = 0.5 * (1 - numpy.cos(
                pi / (it1 - it0) * (t[it0:it1] - t[it0])
            ))
        if it2 > it1:
            win[it1:it2] = 1
        if it3 > it2:
            win[it2:it3] = 0.5 * (1 + numpy.cos(
                pi / (it3 - it2) * (t[it2:it3] - t[it2])
            ))

        signal2 = signal * win
        return signal2


    def apply_correction(self, next_fast_len=False):
        """Corrects the source spectrum of the recorded impulse response.

        Args:
            next_fast_len (bool): If True, the next fast length of input data is used for fft, for zero-padding, etc. Consequently, the singal length is prolonged. Default is False.

        Returns:
            IRnew (numpy.ndarray): see :attr:`IRnew`.
            TR (numpy.ndarray): see :attr:`TR`.
            freqs (numpy.ndarray): see :attr:`freqs`.
        """
        
        # Simulation Parameters
        half_width = self.sim.IC.halfwidth
        c0 = self.sim.c0
        S0 = 1

        saved_every_n_samples = 1
        fs_new = 1 / self.dt_new

        ms_left = 5
        ms_right = 100

        window_length = fs_new / 3 # Heuristic to make sure that f_min_limit stays above 0

        first_band_fmin = 20
        last_band_fmax = self.sim.frequencyLimit
        
        f_min_limit = first_band_fmin - 6.33 * fs_new / window_length 
        f_max_limit = last_band_fmax + 6.33 * fs_new / window_length

        FWHM = 2 * half_width
        alpha = 4 * log(2) / FWHM**2
        dt_old = self.dt_old * saved_every_n_samples
        fs_old = round(1 / dt_old)

        irs_tddg = numpy.array(self.IRold[0], dtype=numpy.float64)

        # Resample if needed
        if fs_old != fs_new:
            irs_tddg_new = resample_poly(irs_tddg, fs_new, fs_old, axis=-1)
        else:
            irs_tddg_new = irs_tddg

        # Frequency Domain Transform
        n_samples = len(irs_tddg_new)
        n_fft = 2**int(numpy.ceil(numpy.log2(n_samples * 2 - 1)))
        n_zeros = n_fft - n_samples + 1
        irs_tddg_new = numpy.pad(irs_tddg_new, (0, n_zeros), mode='constant')

        self.TR_original = fft(irs_tddg_new) / fs_new
        n_samples = self.TR_original.shape[-1]
        df_new = fs_new / (n_samples - 1)
        fv_tddg = numpy.arange(n_samples) * df_new

        # Frequency domain correction
        fv_tddg_shift = self.fftshift_matlab(fv_tddg - fs_new / 2)
        omega = 2 * pi * fv_tddg_shift
        beta_corr = 1 / (pi * S0) * alpha**(1.5) * c0**2 / (1j * numpy.sqrt(pi) * omega) * numpy.exp((omega / c0)**2 / (4 * alpha))
        beta_corr[numpy.isinf(beta_corr)] = 0
        beta_corr[numpy.isnan(beta_corr)] = 0

        self.TR = self.TR_original * beta_corr
        self.TR[0] = 0  # Remove 0 Hz component

        # Frequency masking
        freqWidthPerBand =  fs_new / len(irs_tddg_new)
        index = ((fv_tddg >= f_min_limit) & (fv_tddg <= f_max_limit)) | \
                ((fv_tddg - freqWidthPerBand >= fs_new - f_max_limit) & (fv_tddg - freqWidthPerBand <= fs_new - f_min_limit))
        index = ~index
        self.TR[index] = 0

        # IFFT
        irs_tddg_corrected = ifft(self.TR, axis=-1).real * fs_new

        # Generate filter window (assumed custom implementation)
        blackman_windowed_sinc = self.generate_blackman_windowed_sinc(first_band_fmin, last_band_fmax, fs_new, window_length)

        samples_left = int(ms_left * fs_new * 1e-3)
        samples_right = int(ms_right * fs_new * 1e-3)

        # Final IR construction
        self.IRnew = numpy.zeros_like(irs_tddg_corrected)

        this_ir = irs_tddg_corrected
        this_ir_windowed = self.cconv_python(this_ir, blackman_windowed_sinc)
        shift = -round(window_length / 2)
        this_ir_windowed = numpy.roll(this_ir_windowed, shift)
        this_ir_windowed = self.time_windowing(this_ir_windowed, samples_left, samples_right)
        
        lengthSoundSamples = round(len(irs_tddg)*dt_old*fs_new)
        this_ir_windowed[lengthSoundSamples:-1] = 0
        self.IRnew = this_ir_windowed

        # Final FFT
        tfs_final = fft(self.IRnew, axis=-1) / fs_new
        n_samples = tfs_final.shape[-1]
        df_new = fs_new / n_samples
        fv_final = numpy.arange(n_samples) * df_new
        fv_uncorrected = numpy.arange(self.TR_original.shape[-1]) * df_new
        
        return self.IRnew, self.TR


    def write_results(self, filename, file_format, append=False):
        """Writes the simulation results to a file.

        Args:
            filename (str): The name of the file to save the results.
            file_format (str): The format of the file to save the results. Can be either 'mat' or 'npy'.
        """
        # Load existing data if file exists
        if append and os.path.exists(f"{filename}.{file_format}"):
            object = numpy.load(f"{filename}.{file_format}")
            result_out = {k: object[k] for k in object.files}
        else:
            result_out = {}
            
        result_out = result_out | {
            "IR": self.IRnew,
            "TR": self.TR,
            "freq_limit": self.sim.frequencyLimit,
            #"freqs": self.freqs,
            "dt_old": self.dt_old,
            "dt_simulation": self.sim.time_integrator.dt,
            "fs_old": self.fs_old,
            "sampling_freq": self.sampling_freq,
            "dt_new": self.dt_new,
            "IR_Uncorrected": self.IRold,
            "TR_original": self.TR_original,
            "Ntimesteps": self.sim.Ntimesteps,
            "total_time": self.sim.Ntimesteps * self.sim.time_integrator.dt,
            #"TR_free": self.TR_free,
            #"BC_labels": self.sim.BC_list,
            "BC_para": self.sim.BC.BCpara,
            "rho0": self.sim.rho0,
            "c0": self.sim.c0,
            "mesh_filename": self.sim.mesh.filename,
            "source_xyz": self.sim.IC.source_xyz,
            "source_halfwidth": self.sim.IC.halfwidth,
            "Nx": self.sim.Nx,
            "Nt": self.sim.time_integrator.Nt,
            "CFL": self.sim.time_integrator.CFL,
            "rec": self.sim.rec,
            "total_time_s": self.sim.Ntimesteps * self.sim.time_integrator.dt,
            "N_tets": self.sim.N_tets,
        }
        
        if file_format == "mat":
            scipy.io.savemat(
                f"{filename}.mat",
                **result_out
            )
            print(f"Data saved in MATLAB .mat format to {filename}")
        elif file_format == "npz":
            numpy.savez(
                filename,
                **result_out
            )
            print(f"Data saved in NumPy .npy format to {filename}")
        else:
            raise ValueError("Invalid format. Choose either 'mat' or 'npz'.")
    
    def load_results(self, filename):
        return scipy.io.loadmat(filename)
