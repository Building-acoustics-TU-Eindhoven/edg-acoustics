"""
Non-intrusive ROM for PFFDTD.

Proper design:
  - Parameterize by absorption multiplier (physically meaningful)
  - Smolyak sparse grid training (captures interactions)
  - Post-process BEFORE POD (ROM output = what user hears)
  - Gaussian Process interpolation (with uncertainty estimates)
  - Leave-one-out cross-validation

Offline (~19 min for 33 training runs):
  1. Generate Smolyak grid in absorption-scale space
  2. For each point: scale absorption → fit DEF → run FDTD → post-process IR
  3. POD on post-processed IRs → compact basis
  4. GP regression: parameters → POD coefficients

Online (instant):
  - New absorption scales → GP predict coefficients → reconstruct IR
  - GP posterior variance → uncertainty estimate
"""

import numpy as np
import time as _time
from pathlib import Path
from scipy.signal import butter, sosfiltfilt


# ---- Smolyak sparse grid ----

def _clenshaw_curtis_1d(level):
    """Clenshaw-Curtis nodes on [-1, 1] for given level."""
    if level == 0:
        return np.array([0.0])
    n = 2 ** level
    return -np.cos(np.pi * np.arange(n + 1) / n)


def smolyak_grid(dim, level):
    """Smolyak sparse grid on [-1, 1]^dim.

    Level 2 in 3D gives ~25 points (vs 125 for full tensor grid).
    """
    from itertools import product

    # Generate multi-index set: |alpha|_1 <= level + dim
    def multi_indices(d, max_sum):
        if d == 1:
            return [[i] for i in range(max_sum + 1)]
        result = []
        for i in range(max_sum + 1):
            for rest in multi_indices(d - 1, max_sum - i):
                result.append([i] + rest)
        return result

    indices = [a for a in multi_indices(dim, level)
               if sum(a) <= level]

    # Collect unique points
    points_set = set()
    for alpha in indices:
        grids_1d = [_clenshaw_curtis_1d(a) for a in alpha]
        for pt in product(*grids_1d):
            points_set.add(tuple(np.round(pt, 10)))

    return np.array(sorted(points_set))


def absorption_grid(dim=3, level=2, lo=0.3, hi=3.0):
    """Generate training parameter grid in absorption-scale space.

    Returns array of shape (N, dim) with values in [lo, hi].
    Includes Smolyak grid + corner points.
    """
    # Smolyak in [-1, 1]^dim
    raw = smolyak_grid(dim, level)

    # Map [-1, 1] → [log(lo), log(hi)]
    log_lo, log_hi = np.log(lo), np.log(hi)
    log_mid = (log_lo + log_hi) / 2
    log_half = (log_hi - log_lo) / 2
    log_pts = log_mid + log_half * raw
    pts = np.exp(log_pts)

    # Add corner points
    corners = np.array(list(
        np.array(c) for c in
        np.array(np.meshgrid(*([[lo, hi]] * dim))).T.reshape(-1, dim)
    ))

    # Combine and deduplicate
    all_pts = np.vstack([pts, corners])
    _, idx = np.unique(np.round(all_pts, 6), axis=0, return_index=True)
    return all_pts[sorted(idx)]


# ---- Post-processing ----

def postprocess_ir(ir_raw, fs_native, fmax_grid, fs_out=48000):
    """Apply HP + LP + resample to a raw FDTD IR."""
    # HP at 10 Hz
    sos_hp = butter(4, 10.0, btype='high', fs=fs_native, output='sos')
    ir = sosfiltfilt(sos_hp, ir_raw)

    # LP at 0.9 * grid Nyquist
    fcut = min(0.9 * fmax_grid, 0.45 * fs_native)
    sos_lp = butter(8, fcut, btype='low', fs=fs_native, output='sos')
    ir = sosfiltfilt(sos_lp, ir)

    # Resample
    try:
        import resampy
        ir = resampy.resample(ir, fs_native, fs_out)
    except ImportError:
        from scipy.signal import resample
        n_out = int(len(ir) * fs_out / fs_native)
        ir = resample(ir, n_out)

    return ir


# ---- Metrics ----

def compute_metrics(ir, fs):
    """Compute T30, EDT, C80 from an IR. Returns dict."""
    from scipy.signal import butter, sosfiltfilt

    def _t30_from_edc(edc_db, t):
        i5 = np.argmax(edc_db < -5)
        i35 = np.argmax(edc_db < -35)
        if i35 > i5 + 10:
            return -60.0 / np.polyfit(t[i5:i35], edc_db[i5:i35], 1)[0]
        return np.nan

    def _edt_from_edc(edc_db, t):
        i0 = 0
        i10 = np.argmax(edc_db < -10)
        if i10 > i0 + 10:
            return -60.0 / np.polyfit(t[i0:i10], edc_db[i0:i10], 1)[0]
        return np.nan

    ir2 = ir ** 2
    cum = np.cumsum(ir2[::-1])[::-1]
    cum /= (cum[0] + 1e-30)
    edc_db = 10 * np.log10(cum + 1e-30)
    t = np.arange(len(edc_db)) / fs

    # Broadband
    n80 = int(0.08 * fs)
    early = np.sum(ir2[:n80])
    late = np.sum(ir2[n80:])
    c80 = 10 * np.log10(early / max(late, 1e-30))

    return {
        'T30': _t30_from_edc(edc_db, t),
        'EDT': _edt_from_edc(edc_db, t),
        'C80': c80,
    }


# ---- Main ROM class ----

class NonIntrusiveROM:
    """Non-intrusive ROM for PFFDTD with proper design.

    Parameters
    ----------
    data_dir : str or Path
        PFFDTD simulation data directory (with H5 files)
    pffdtd_python_dir : str or Path
        Path to PFFDTD python/ directory
    """

    def __init__(self, data_dir, pffdtd_python_dir):
        self.data_dir = Path(data_dir)
        self.pffdtd_dir = Path(pffdtd_python_dir)
        self._load_baseline()

    def _load_baseline(self):
        import h5py
        d = self.data_dir

        with h5py.File(d / 'sim_consts.h5', 'r') as f:
            self.c = float(f['c'][()])
            self.Ts = float(f['Ts'][()])
            self.h = float(f['h'][()])
        with h5py.File(d / 'comms_out.h5', 'r') as f:
            self.Nt = int(f['Nt'][()])
            self.Nr = int(f['Nr'][()])
        with h5py.File(d / 'sim_mats.h5', 'r') as f:
            Nmat = int(f['Nmat'][()])
            Mb = f['Mb'][...]
            DEF = np.zeros((Nmat, 12, 3))
            for i in range(Nmat):
                ds = f[f'mat_{i:02d}_DEF'][...]
                DEF[i, :Mb[i]] = ds
            self.DEF_baseline = DEF.copy()
            self.Mb = Mb
            self.Nmat = Nmat

        self.fs_native = 1.0 / self.Ts
        self.fmax_grid = self.c / (2 * self.h)
        self.fs_out = 48000

    def _alpha_to_DEF(self, alpha_scales, baseline_alphas):
        """Convert absorption scale factors to DEF triplets.

        Parameters
        ----------
        alpha_scales : (N_surfaces,) array, range [0.3, 3.0]
        baseline_alphas : dict {surface_name: "a1, a2, a3, a4, a5"}

        Returns
        -------
        DEF : (Nmat, 12, 3) array
        """
        import sys
        pdir = str(self.pffdtd_dir)
        if pdir not in sys.path:
            sys.path.insert(0, pdir)
        from materials.adm_funcs import fit_to_Sabs_oct_11

        import tempfile
        mat_names = sorted(baseline_alphas.keys())
        DEF = np.zeros_like(self.DEF_baseline)

        for i, name in enumerate(mat_names):
            # Parse baseline absorption (5-band)
            alphas_5 = [float(x.strip())
                        for x in baseline_alphas[name].split(",")]

            # Scale
            scale = alpha_scales[i] if i < len(alpha_scales) else 1.0
            alphas_5_scaled = [min(max(a * scale, 0.01), 0.99) for a in alphas_5]

            # Expand to 11 bands
            a = np.array(alphas_5_scaled)
            alphas_11 = np.zeros(11)
            alphas_11[0:3] = a[0]
            alphas_11[3:8] = a
            alphas_11[8:] = a[4]
            alphas_11 = np.clip(alphas_11, 0.01, 0.99)

            # Fit DEF
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
                tmp_path = tmp.name
            fit_to_Sabs_oct_11(alphas_11, tmp_path)

            import h5py, os
            with h5py.File(tmp_path, 'r') as f:
                def_data = f['DEF'][...]
            os.unlink(tmp_path)

            M = def_data.shape[0]
            DEF[i, :M] = def_data

        return DEF

    def _run_single_fdtd(self, DEF, use_gpu=False):
        """Run one FDTD with given DEF triplets. Returns raw IR."""
        import sys
        pdir = str(self.pffdtd_dir)
        if pdir not in sys.path:
            sys.path.insert(0, pdir)

        from fdtd.sim_fdtd import SimEngine

        engine = SimEngine(str(self.data_dir), energy_on=False)
        engine.load_h5_data()
        engine.DEF = DEF.copy()
        engine.setup_mask()
        engine.allocate_mem()
        engine.set_coeffs()
        engine.checks()

        if use_gpu:
            try:
                from gpu_engine import run_gpu, HAS_GPU
                if HAS_GPU:
                    run_gpu(engine)
                else:
                    engine.run_all()
            except Exception:
                engine.run_all()
        else:
            engine.run_all()

        return engine.u_out[engine.out_reorder[0], :]

    def train(self, baseline_alphas, dim=3, level=2, use_gpu=False,
              rec_idx=0):
        """Train ROM with Smolyak sparse grid.

        Parameters
        ----------
        baseline_alphas : dict {surface_name: "a1, a2, a3, a4, a5"}
        dim : int, effective parameter dimensions (3 or 6)
        level : int, Smolyak level (2 recommended)
        use_gpu : bool
        """
        t0 = _time.perf_counter()

        self.baseline_alphas = baseline_alphas
        self.dim = dim

        # Generate training grid
        grid = absorption_grid(dim=dim, level=level)
        n_train = len(grid)
        print(f"ROM train: {n_train} points in {dim}D, "
              f"est. {n_train * 35 / 60:.0f} min on CPU")

        # Map 3D grid to Nmat surfaces
        # dim=3: [floor, ceiling, walls] → expand walls to 4 surfaces
        mat_names = sorted(baseline_alphas.keys())

        # Run all training cases
        raw_irs = []
        for i, scales in enumerate(grid):
            # Expand dim-D scales to Nmat surfaces
            if dim == 3 and self.Nmat == 6:
                full_scales = np.array([
                    scales[0],  # floor
                    scales[1],  # ceiling
                    scales[2], scales[2], scales[2], scales[2]  # 4 walls
                ])
            else:
                full_scales = scales

            t1 = _time.perf_counter()
            DEF = self._alpha_to_DEF(full_scales, baseline_alphas)
            ir_raw = self._run_single_fdtd(DEF, use_gpu=use_gpu)
            dt = _time.perf_counter() - t1
            raw_irs.append(ir_raw)

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = _time.perf_counter() - t0
                eta = elapsed / (i + 1) * (n_train - i - 1)
                print(f"  [{i+1}/{n_train}] scales={np.round(scales, 2)} "
                      f"({dt:.0f}s, ~{eta:.0f}s left)")

        # Post-process ALL IRs before POD
        print("  Post-processing IRs...")
        processed_irs = []
        for ir_raw in raw_irs:
            ir_pp = postprocess_ir(ir_raw, self.fs_native,
                                   self.fmax_grid, self.fs_out)
            processed_irs.append(ir_pp)

        # Ensure same length
        min_len = min(len(ir) for ir in processed_irs)
        irs_matrix = np.array([ir[:min_len] for ir in processed_irs])
        self.ir_len = min_len

        # POD on post-processed IRs
        X = irs_matrix.T  # (N_samples, n_train)
        self.ir_mean = np.mean(X, axis=1)
        X_c = X - self.ir_mean[:, None]

        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        r = np.searchsorted(energy, 0.9999) + 1
        r = min(r, len(S))

        self.Phi = U[:, :r]
        self.r = r
        self.training_coeffs = X_c.T @ self.Phi  # (n_train, r)
        self.training_params = grid
        self.training_irs = irs_matrix
        self.n_train = n_train

        print(f"  POD: r={r}, energy={energy[r-1]*100:.4f}%")

        # GP interpolation
        self._build_gp()

        # Compute metrics for Metric-ROM
        self._build_metric_rom()

        elapsed = _time.perf_counter() - t0
        print(f"ROM trained: {elapsed:.0f}s ({elapsed/60:.1f} min), "
              f"r={r}, {n_train} cases")

    def _build_gp(self):
        """Build GP regressors for each POD coefficient."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        log_params = np.log(self.training_params)

        self.gp_models = []
        for j in range(self.r):
            kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
                      Matern(nu=2.5, length_scale=np.ones(self.dim),
                             length_scale_bounds=(0.01, 100.0)) +
                      WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-15, 1e-5)))
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(log_params, self.training_coeffs[:, j])
            self.gp_models.append(gp)

        print(f"  GP: {self.r} models fitted")

    def _build_metric_rom(self):
        """Build GP regressors for scalar acoustic metrics."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        log_params = np.log(self.training_params)

        self.metric_data = {}
        self.metric_gps = {}

        for i in range(self.n_train):
            m = compute_metrics(self.training_irs[i], self.fs_out)
            for key, val in m.items():
                if key not in self.metric_data:
                    self.metric_data[key] = []
                self.metric_data[key].append(val)

        for key, values in self.metric_data.items():
            values = np.array(values)
            valid = ~np.isnan(values)
            if np.sum(valid) < 5:
                continue
            kernel = (ConstantKernel() *
                      Matern(nu=2.5, length_scale=np.ones(self.dim)) +
                      WhiteKernel(noise_level=1e-8))
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gp.fit(log_params[valid], values[valid])
            self.metric_gps[key] = gp

        print(f"  Metric-ROM: {len(self.metric_gps)} metrics fitted")

    # ---- Evaluation ----

    def evaluate(self, alpha_scales):
        """Evaluate ROM at new absorption scales. Returns post-processed IR.

        Parameters
        ----------
        alpha_scales : (dim,) array, range [0.3, 3.0]

        Returns
        -------
        ir : (N,) array at 48 kHz
        std : float, mean GP uncertainty (0 = confident, >0.1 = extrapolation)
        """
        alpha_scales = np.atleast_1d(alpha_scales).astype(float)
        log_q = np.log(alpha_scales).reshape(1, -1)

        coeffs = np.zeros(self.r)
        stds = np.zeros(self.r)
        for j in range(self.r):
            mean, std = self.gp_models[j].predict(log_q, return_std=True)
            coeffs[j] = mean[0]
            stds[j] = std[0]

        ir = self.ir_mean + self.Phi @ coeffs
        uncertainty = np.mean(stds / (np.std(self.training_coeffs, axis=0) + 1e-10))

        return ir, uncertainty

    def evaluate_metrics(self, alpha_scales):
        """Predict acoustic metrics with uncertainty.

        Returns dict {metric_name: (mean, std)}.
        """
        alpha_scales = np.atleast_1d(alpha_scales).astype(float)
        log_q = np.log(alpha_scales).reshape(1, -1)

        results = {}
        for key, gp in self.metric_gps.items():
            mean, std = gp.predict(log_q, return_std=True)
            results[key] = (float(mean[0]), float(std[0]))
        return results

    # ---- Validation ----

    def validate_loo(self):
        """Leave-one-out cross-validation. Returns error stats."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

        log_params = np.log(self.training_params)
        n = self.n_train
        correlations = []
        t30_errors = []

        print(f"LOO validation ({n} cases)...")

        for i in range(n):
            # Build ROM without case i
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            X = self.training_irs[mask].T
            ir_mean = np.mean(X, axis=1)
            X_c = X - ir_mean[:, None]
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            r = min(self.r, U.shape[1])
            Phi = U[:, :r]
            coeffs = X_c.T @ Phi

            # Predict held-out case
            predicted = np.zeros(r)
            for j in range(r):
                kernel = (ConstantKernel() *
                          Matern(nu=2.5, length_scale=np.ones(self.dim)) +
                          WhiteKernel(noise_level=1e-10))
                gp = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
                gp.fit(log_params[mask], coeffs[:, j])
                predicted[j] = gp.predict(log_params[i:i+1])[0]

            ir_pred = ir_mean + Phi @ predicted
            ir_true = self.training_irs[i, :len(ir_pred)]

            corr = np.corrcoef(ir_true, ir_pred)[0, 1]
            correlations.append(corr)

            m_true = compute_metrics(ir_true, self.fs_out)
            m_pred = compute_metrics(ir_pred, self.fs_out)
            if not np.isnan(m_true['T30']) and not np.isnan(m_pred['T30']):
                err = abs(m_true['T30'] - m_pred['T30']) / m_true['T30'] * 100
                t30_errors.append(err)

        correlations = np.array(correlations)
        t30_errors = np.array(t30_errors)

        print(f"  Correlation: min={correlations.min():.4f}, "
              f"mean={correlations.mean():.4f}, max={correlations.max():.4f}")
        if len(t30_errors) > 0:
            print(f"  T30 error: min={t30_errors.min():.1f}%, "
                  f"mean={t30_errors.mean():.1f}%, max={t30_errors.max():.1f}%")

        return {'correlations': correlations, 't30_errors': t30_errors}

    # ---- Save / Load ----

    def save(self, path):
        np.savez_compressed(str(path),
                            ir_mean=self.ir_mean, Phi=self.Phi,
                            training_params=self.training_params,
                            training_coeffs=self.training_coeffs,
                            training_irs=self.training_irs,
                            DEF_baseline=self.DEF_baseline, Mb=self.Mb,
                            fs_native=self.fs_native, fs_out=self.fs_out,
                            fmax_grid=self.fmax_grid, Nt=self.Nt,
                            Nmat=self.Nmat, dim=self.dim,
                            ir_len=self.ir_len)
        print(f"ROM saved: {path}")

    def load(self, path):
        d = np.load(str(path), allow_pickle=True)
        self.ir_mean = d['ir_mean']
        self.Phi = d['Phi']
        self.training_params = d['training_params']
        self.training_coeffs = d['training_coeffs']
        self.training_irs = d['training_irs']
        self.DEF_baseline = d['DEF_baseline']
        self.Mb = d['Mb']
        self.fs_native = float(d['fs_native'])
        self.fs_out = float(d['fs_out'])
        self.fmax_grid = float(d['fmax_grid'])
        self.Nt = int(d['Nt'])
        self.Nmat = int(d['Nmat'])
        self.dim = int(d['dim'])
        self.ir_len = int(d['ir_len'])
        self.r = self.Phi.shape[1]
        self.n_train = self.training_params.shape[0]

        self._build_gp()
        self._build_metric_rom()
        print(f"ROM loaded: r={self.r}, {self.n_train} training cases")
