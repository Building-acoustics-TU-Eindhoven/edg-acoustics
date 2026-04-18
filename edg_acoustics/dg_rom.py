"""
Non-intrusive ROM for edg-acoustics DG solver.

Same Smolyak + GP approach as PPFFDTD ROM.
Parameterizes by scaling reflection coefficients per surface.
"""

import numpy as np
import time as _time
from pathlib import Path
from edg_acoustics.rom import (
    smolyak_grid, absorption_grid, postprocess_ir,
    compute_metrics, NonIntrusiveROM as _BaseROM
)


class DGROM:
    """Non-intrusive ROM for the edg-acoustics DG solver.

    Parameters
    ----------
    mesh_file : str
        Path to Gmsh .msh file
    BC_labels : dict
        Boundary condition labels {name: physical_tag}
    BC_para_baseline : list
        Baseline BC parameters (RI, RP, CP per surface)
    monopole_xyz : array
        Source position
    rec : array (3, N_rec)
        Receiver positions
    rho0, c0 : float
        Air density and speed of sound
    Nx, Nt_order : int
        Polynomial order (space, time)
    freq_upper_limit : float
        Source frequency limit
    """

    def __init__(self, mesh_file, BC_labels, BC_para_baseline,
                 monopole_xyz, rec, rho0=1.213, c0=343,
                 Nx=4, Nt_order=3, freq_upper_limit=200,
                 impulse_length=0.5):
        self.mesh_file = mesh_file
        self.BC_labels = BC_labels
        self.BC_para_baseline = BC_para_baseline
        self.monopole_xyz = monopole_xyz
        self.rec = rec
        self.rho0 = rho0
        self.c0 = c0
        self.Nx = Nx
        self.Nt_order = Nt_order
        self.freq_upper_limit = freq_upper_limit
        self.impulse_length = impulse_length

        # Count lossy surfaces (non-hard-wall)
        self.lossy_surfaces = [name for name, para in
                               zip(BC_labels.keys(), BC_para_baseline)
                               if para.get('RI', 1) != 1]
        self.n_params = len(self.lossy_surfaces)

    def _run_single(self, BC_para, use_gpu=True):
        """Run one DG simulation, return post-processed IR."""
        import edg_acoustics
        from edg_acoustics.gpu_backend import to_device, sync

        mesh = edg_acoustics.Mesh(self.mesh_file, self.BC_labels)
        sim = edg_acoustics.AcousticsSimulation(
            self.rho0, self.c0, self.Nx, mesh, self.BC_labels)
        flux = edg_acoustics.UpwindFlux(self.rho0, self.c0, sim.n_xyz)
        AbBC = edg_acoustics.AbsorbBC(sim.BCnode, BC_para)
        sim.init_BC(AbBC)
        sim.init_IC(edg_acoustics.Monopole_IC(
            self.monopole_xyz, self.freq_upper_limit))
        sim.init_Flux(flux)
        sim.init_rec(self.rec, 'scipy')
        tsi = edg_acoustics.TSI_TI(
            sim.RHS_operator, sim.dtscale, 0.5, Nt=self.Nt_order)
        sim.init_TimeIntegrator(tsi)

        if use_gpu:
            try:
                from edg_acoustics.gpu_rhs import FusedRHS
                sim.transfer_to_gpu()
                fused = FusedRHS(sim)
                fused.prepare()
                tsi.L_operator = fused
                sim.P = to_device(sim.IC.Pinit(sim.xyz))
                sim.Vx = to_device(sim.IC.VXinit(sim.xyz))
                sim.Vy = to_device(sim.IC.VYinit(sim.xyz))
                sim.Vz = to_device(sim.IC.VZinit(sim.xyz))
            except Exception:
                pass

        sim.time_integration(total_time=self.impulse_length,
                             delta_step=10000)  # minimal printing

        # Post-process
        results = edg_acoustics.Monopole_postprocessor(sim, 10)
        results.apply_correction()
        ir = results.IRnew if hasattr(results, 'IRnew') else results.IRold[0]
        if hasattr(ir, 'shape') and ir.ndim > 1:
            ir = ir[0]
        fs = results.sampling_freq if hasattr(results, 'sampling_freq') else 44100

        return np.asarray(ir), fs

    def _scale_BC_para(self, scales):
        """Create new BC_para with scaled RI values.

        scales: array of length n_params (one per lossy surface)
        """
        import copy
        BC_para_new = copy.deepcopy(self.BC_para_baseline)
        scale_idx = 0
        for i, para in enumerate(BC_para_new):
            if para.get('RI', 1) != 1:  # lossy surface
                if scale_idx < len(scales):
                    # Scale RI: higher scale = more reflective
                    # RI is reflection coefficient, range [0, 1]
                    ri_orig = para['RI']
                    if hasattr(ri_orig, '__len__'):
                        ri_orig = ri_orig[0] if len(ri_orig) > 0 else 0.5
                    ri_new = np.clip(ri_orig * scales[scale_idx], 0.01, 0.99)
                    para['RI'] = ri_new
                    scale_idx += 1
        return BC_para_new

    def train(self, dim=None, level=2, use_gpu=True):
        """Train ROM with Smolyak sparse grid."""
        if dim is None:
            dim = self.n_params
        if dim == 0:
            print("No lossy surfaces to parameterize")
            return

        t0 = _time.perf_counter()

        grid = absorption_grid(dim=dim, level=level, lo=0.5, hi=2.0)
        n_train = len(grid)
        print(f"DG-ROM: {n_train} training cases, {dim}D, "
              f"est. {n_train * 85 / 60:.0f} min on GPU")

        irs = []
        self.fs = None

        for i, scales in enumerate(grid):
            t1 = _time.perf_counter()
            BC_para = self._scale_BC_para(scales)
            ir, fs = self._run_single(BC_para, use_gpu=use_gpu)
            irs.append(ir)
            if self.fs is None:
                self.fs = fs
            dt = _time.perf_counter() - t1

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = _time.perf_counter() - t0
                eta = elapsed / (i + 1) * (n_train - i - 1)
                print(f"  [{i+1}/{n_train}] scales={np.round(scales, 2)} "
                      f"({dt:.0f}s, ~{eta:.0f}s left)")

        # Ensure same length
        min_len = min(len(ir) for ir in irs)
        irs_matrix = np.array([ir[:min_len] for ir in irs])
        self.ir_len = min_len

        # POD
        X = irs_matrix.T
        self.ir_mean = np.mean(X, axis=1)
        X_c = X - self.ir_mean[:, None]
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        r = np.searchsorted(energy, 0.9999) + 1
        r = min(r, len(S))

        self.Phi = U[:, :r]
        self.r = r
        self.training_coeffs = X_c.T @ self.Phi
        self.training_params = grid
        self.training_irs = irs_matrix
        self.n_train = n_train

        print(f"  POD: r={r}, energy={energy[r-1]*100:.4f}%")

        # GP
        self._build_gp()

        elapsed = _time.perf_counter() - t0
        print(f"DG-ROM trained: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    def _build_gp(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        log_params = np.log(self.training_params)
        dim = log_params.shape[1]

        self.gp_models = []
        for j in range(self.r):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                      Matern(nu=2.5, length_scale=np.ones(dim),
                             length_scale_bounds=(0.01, 100.0)) +
                      WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-15, 1e-5)))
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(log_params, self.training_coeffs[:, j])
            self.gp_models.append(gp)

    def evaluate(self, scales):
        """Evaluate ROM at new RI scales. Returns IR array."""
        scales = np.atleast_1d(scales).astype(float)
        log_q = np.log(scales).reshape(1, -1)

        coeffs = np.zeros(self.r)
        for j in range(self.r):
            coeffs[j] = self.gp_models[j].predict(log_q)[0]

        ir = self.ir_mean + self.Phi @ coeffs
        return ir

    def save(self, path):
        np.savez_compressed(str(path),
                            ir_mean=self.ir_mean, Phi=self.Phi,
                            training_params=self.training_params,
                            training_coeffs=self.training_coeffs,
                            training_irs=self.training_irs,
                            fs=self.fs, ir_len=self.ir_len)

    def load(self, path):
        d = np.load(str(path))
        self.ir_mean = d['ir_mean']
        self.Phi = d['Phi']
        self.training_params = d['training_params']
        self.training_coeffs = d['training_coeffs']
        self.training_irs = d['training_irs']
        self.fs = float(d['fs'])
        self.ir_len = int(d['ir_len'])
        self.r = self.Phi.shape[1]
        self.n_train = self.training_params.shape[0]
        self._build_gp()
