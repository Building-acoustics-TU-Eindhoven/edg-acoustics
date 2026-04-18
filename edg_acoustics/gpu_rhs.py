"""
Fused GPU RHS operator for the DG acoustic solver.

Replaces the per-operation Python calls in RHS_operator with
a single function that minimizes kernel launches and memory allocations.
Pre-allocates scratch arrays, avoids .reshape(-1)[:] copies.
"""

from edg_acoustics.gpu_backend import xp, is_gpu
import math


class FusedRHS:
    """Pre-allocated, fused RHS operator for GPU acceleration.

    Call prepare() after transfer_to_gpu() to set up scratch arrays.
    Then use __call__() as a drop-in replacement for RHS_operator.
    """

    def __init__(self, sim):
        """Bind to an AcousticsSimulation instance."""
        self.sim = sim
        self.prepared = False

    def prepare(self):
        """Pre-allocate all scratch arrays on the current device (CPU or GPU)."""
        sim = self.sim
        Nfp4 = sim.Fscale.shape[0]  # 4*Nfp
        N_tets = sim.mesh.N_tets
        Np = sim.Np
        flat_size = Np * N_tets

        # Scratch for jumps (allocated once, reused every call)
        self.dVx = xp.zeros((Nfp4, N_tets), dtype=xp.float64)
        self.dVy = xp.zeros_like(self.dVx)
        self.dVz = xp.zeros_like(self.dVx)
        self.dP = xp.zeros_like(self.dVx)

        # Pre-flatten maps for fast indexing
        self.vmapM = sim.vmapM.ravel()
        self.vmapP = sim.vmapP.ravel()
        self.flat_size = flat_size

        # Pre-compute Taylor coefficients
        self.prepared = True

    def __call__(self, P, Vx, Vy, Vz, BCvar):
        """Compute RHS with minimal kernel launches."""
        sim = self.sim

        # 1. Gather jumps (4 gather operations)
        Pf = P.ravel()
        Vxf = Vx.ravel()
        Vyf = Vy.ravel()
        Vzf = Vz.ravel()

        self.dVx.ravel()[:] = Vxf[self.vmapM] - Vxf[self.vmapP]
        self.dVy.ravel()[:] = Vyf[self.vmapM] - Vyf[self.vmapP]
        self.dVz.ravel()[:] = Vzf[self.vmapM] - Vzf[self.vmapP]
        self.dP.ravel()[:] = Pf[self.vmapM] - Pf[self.vmapP]

        dVx, dVy, dVz, dP = self.dVx, self.dVy, self.dVz, self.dP

        # 2. Fluxes (element-wise, all on GPU — 4 outputs)
        flux = sim.flux
        fluxP = flux.csn1rho * dVx + flux.csn2rho * dVy + flux.csn3rho * dVz - sim.c0 / 2 * dP
        fluxVx = flux.cn1s * dVx + flux.cn1n2 * dVy + flux.cn1n3 * dVz + flux.n1rho * dP
        fluxVy = flux.cn1n2 * dVx + flux.cn2s * dVy + flux.cn2n3 * dVz + flux.n2rho * dP
        fluxVz = flux.cn1n3 * dVx + flux.cn2n3 * dVy + flux.cn3s * dVz + flux.n3rho * dP

        # 3. Boundary conditions (still per-material loop — vectorized within each)
        for index, paras in enumerate(sim.BC.BCpara):
            bmap = sim.BCnode[index]["map"]
            bvmap = sim.BCnode[index]["vmap"]

            n0 = sim.n_xyz[0].ravel()[bmap]
            n1 = sim.n_xyz[1].ravel()[bmap]
            n2 = sim.n_xyz[2].ravel()[bmap]

            vn = n0 * Vxf[bvmap] + n1 * Vyf[bvmap] + n2 * Vzf[bvmap]
            ou = vn + Pf[bvmap] / sim.rho0 / sim.c0
            inc = ou * paras["RI"]

            BCvar[index]["vn"] = vn
            BCvar[index]["ou"] = ou

            for polekey in paras:
                if polekey == "RP":
                    for i in range(paras["RP"].shape[1]):
                        inc = inc + paras["RP"][0, i] * BCvar[index]["phi"][i]
                        BCvar[index]["phi"][i] = ou - paras["RP"][1, i] * BCvar[index]["phi"][i]
                elif polekey == "CP":
                    for i in range(paras["CP"].shape[1]):
                        inc = inc + paras["CP"][0, i] * BCvar[index]["kexi1"][i] + paras["CP"][1, i] * BCvar[index]["kexi2"][i]
                        k1_old = BCvar[index]["kexi1"][i].copy()
                        BCvar[index]["kexi1"][i] = ou - paras["CP"][2, i] * BCvar[index]["kexi1"][i] - paras["CP"][3, i] * BCvar[index]["kexi2"][i]
                        BCvar[index]["kexi2"][i] = -paras["CP"][2, i] * BCvar[index]["kexi2"][i] + paras["CP"][3, i] * k1_old

            BCvar[index]["in"] = inc

            half_ou_in = (ou + inc) / 2
            Prho = Pf[bvmap] / sim.rho0

            fluxVx.ravel()[bmap] = n0 * Prho - n0 * sim.c0 * half_ou_in
            fluxVy.ravel()[bmap] = n1 * Prho - n1 * sim.c0 * half_ou_in
            fluxVz.ravel()[bmap] = n2 * Prho - n2 * sim.c0 * half_ou_in
            fluxP.ravel()[bmap] = sim.c0 ** 2 * sim.rho0 * (vn - 0.5 * (ou - inc))

        # 4. Gradients (3 matmuls + linear combos)
        dUdr = sim.Dr @ P
        dUds = sim.Ds @ P
        dUdt = sim.Dt @ P
        dPdx = sim.rst_xyz[0, 0] * dUdr + sim.rst_xyz[1, 0] * dUds + sim.rst_xyz[2, 0] * dUdt
        dPdy = sim.rst_xyz[0, 1] * dUdr + sim.rst_xyz[1, 1] * dUds + sim.rst_xyz[2, 1] * dUdt
        dPdz = sim.rst_xyz[0, 2] * dUdr + sim.rst_xyz[1, 2] * dUds + sim.rst_xyz[2, 2] * dUdt

        # Velocity divergence (reuse Dr/Ds/Dt for each component)
        dVxdr = sim.Dr @ Vx; dVxds = sim.Ds @ Vx; dVxdt = sim.Dt @ Vx
        divVx = sim.rst_xyz[0, 0] * dVxdr + sim.rst_xyz[1, 0] * dVxds + sim.rst_xyz[2, 0] * dVxdt

        dVydr = sim.Dr @ Vy; dVyds = sim.Ds @ Vy; dVydt = sim.Dt @ Vy
        divVy = sim.rst_xyz[0, 1] * dVydr + sim.rst_xyz[1, 1] * dVyds + sim.rst_xyz[2, 1] * dVydt

        dVzdr = sim.Dr @ Vz; dVzds = sim.Ds @ Vz; dVzdt = sim.Dt @ Vz
        divVz = sim.rst_xyz[0, 2] * dVzdr + sim.rst_xyz[1, 2] * dVzds + sim.rst_xyz[2, 2] * dVzdt

        # 5. Lift + combine
        Fs = sim.Fscale
        RHS_P = -sim.c0 ** 2 * sim.rho0 * (divVx + divVy + divVz) + sim.lift @ (Fs * fluxP)
        RHS_Vx = -dPdx / sim.rho0 + sim.lift @ (Fs * fluxVx)
        RHS_Vy = -dPdy / sim.rho0 + sim.lift @ (Fs * fluxVy)
        RHS_Vz = -dPdz / sim.rho0 + sim.lift @ (Fs * fluxVz)

        return RHS_P, RHS_Vx, RHS_Vy, RHS_Vz, BCvar
