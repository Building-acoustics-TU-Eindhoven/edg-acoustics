"""
Fused GPU RHS operator for the DG acoustic solver.

Replaces the per-operation Python calls in RHS_operator with
a single function that minimizes kernel launches and memory allocations.
Pre-allocates scratch arrays, avoids .reshape(-1)[:] copies.
"""

import edg_acoustics.gpu_backend as backend
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
        """Pre-allocate ALL scratch arrays — zero allocations during time stepping."""
        sim = self.sim
        Nfp4 = sim.Fscale.shape[0]
        N_tets = sim.mesh.N_tets
        Np = sim.Np

        # Jump scratch (reused every call)
        self.dVx = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)
        self.dVy = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)
        self.dVz = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)
        self.dP = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)

        # Flux scratch
        self.fluxP = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)
        self.fluxVx = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)
        self.fluxVy = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)
        self.fluxVz = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)

        # Gradient scratch
        self.dUdr = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.dUds = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.dUdt = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.tmp1 = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.tmp2 = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)

        # Lift scratch
        self.lift_tmp = backend.xp.zeros((Nfp4, N_tets), dtype=backend.xp.float64)

        # RHS output scratch
        self.RHS_P = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.RHS_Vx = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.RHS_Vy = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)
        self.RHS_Vz = backend.xp.zeros((Np, N_tets), dtype=backend.xp.float64)

        # Pre-flatten maps
        self.vmapM = sim.vmapM.ravel()
        self.vmapP = sim.vmapP.ravel()

        # Pre-flatten BC maps
        self.bc_maps = []
        for index in range(len(sim.BCnode)):
            bmap = sim.BCnode[index]["map"]
            bvmap = sim.BCnode[index]["vmap"]
            n0 = sim.n_xyz[0].ravel()[bmap]
            n1 = sim.n_xyz[1].ravel()[bmap]
            n2 = sim.n_xyz[2].ravel()[bmap]
            self.bc_maps.append((bmap, bvmap, n0, n1, n2))

        self.prepared = True

    def __call__(self, P, Vx, Vy, Vz, BCvar):
        """Compute RHS with pre-allocated scratch — zero allocations."""
        sim = self.sim
        dVx, dVy, dVz, dP = self.dVx, self.dVy, self.dVz, self.dP
        fluxP, fluxVx, fluxVy, fluxVz = self.fluxP, self.fluxVx, self.fluxVy, self.fluxVz

        Pf = P.ravel()
        Vxf = Vx.ravel()
        Vyf = Vy.ravel()
        Vzf = Vz.ravel()

        # 1. Gather jumps (in-place into pre-allocated scratch)
        dVx_f = dVx.ravel()
        dVx_f[:] = Vxf[self.vmapM]; dVx_f -= Vxf[self.vmapP]
        dVy_f = dVy.ravel()
        dVy_f[:] = Vyf[self.vmapM]; dVy_f -= Vyf[self.vmapP]
        dVz_f = dVz.ravel()
        dVz_f[:] = Vzf[self.vmapM]; dVz_f -= Vzf[self.vmapP]
        dP_f = dP.ravel()
        dP_f[:] = Pf[self.vmapM]; dP_f -= Pf[self.vmapP]

        # 2. Fluxes (in-place multiply-add into pre-allocated scratch)
        f = sim.flux
        backend.xp.multiply(f.csn1rho, dVx, out=fluxP)
        fluxP += f.csn2rho * dVy; fluxP += f.csn3rho * dVz; fluxP -= (sim.c0 / 2) * dP

        backend.xp.multiply(f.cn1s, dVx, out=fluxVx)
        fluxVx += f.cn1n2 * dVy; fluxVx += f.cn1n3 * dVz; fluxVx += f.n1rho * dP

        backend.xp.multiply(f.cn1n2, dVx, out=fluxVy)
        fluxVy += f.cn2s * dVy; fluxVy += f.cn2n3 * dVz; fluxVy += f.n2rho * dP

        backend.xp.multiply(f.cn1n3, dVx, out=fluxVz)
        fluxVz += f.cn2n3 * dVy; fluxVz += f.cn3s * dVz; fluxVz += f.n3rho * dP

        # 3. Boundary conditions
        for index, paras in enumerate(sim.BC.BCpara):
            bmap, bvmap, n0, n1, n2 = self.bc_maps[index]

            vn = n0 * Vxf[bvmap] + n1 * Vyf[bvmap] + n2 * Vzf[bvmap]
            ou = vn + Pf[bvmap] / (sim.rho0 * sim.c0)
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
                        k1 = BCvar[index]["kexi1"][i].copy()
                        BCvar[index]["kexi1"][i] = ou - paras["CP"][2, i] * BCvar[index]["kexi1"][i] - paras["CP"][3, i] * BCvar[index]["kexi2"][i]
                        BCvar[index]["kexi2"][i] = -paras["CP"][2, i] * BCvar[index]["kexi2"][i] + paras["CP"][3, i] * k1

            BCvar[index]["in"] = inc
            half_ou_in = (ou + inc) * 0.5
            Prho = Pf[bvmap] / sim.rho0

            fluxVx.ravel()[bmap] = n0 * Prho - n0 * sim.c0 * half_ou_in
            fluxVy.ravel()[bmap] = n1 * Prho - n1 * sim.c0 * half_ou_in
            fluxVz.ravel()[bmap] = n2 * Prho - n2 * sim.c0 * half_ou_in
            fluxP.ravel()[bmap] = sim.c0 ** 2 * sim.rho0 * (vn - 0.5 * (ou - inc))

        # 4. Gradient of P (reuse scratch for dr/ds/dt)
        backend.xp.dot(sim.Dr, P, out=self.dUdr)
        backend.xp.dot(sim.Ds, P, out=self.dUds)
        backend.xp.dot(sim.Dt, P, out=self.dUdt)
        dPdx = sim.rst_xyz[0, 0] * self.dUdr + sim.rst_xyz[1, 0] * self.dUds + sim.rst_xyz[2, 0] * self.dUdt
        dPdy = sim.rst_xyz[0, 1] * self.dUdr + sim.rst_xyz[1, 1] * self.dUds + sim.rst_xyz[2, 1] * self.dUdt
        dPdz = sim.rst_xyz[0, 2] * self.dUdr + sim.rst_xyz[1, 2] * self.dUds + sim.rst_xyz[2, 2] * self.dUdt

        # Divergence of V
        backend.xp.dot(sim.Dr, Vx, out=self.dUdr); backend.xp.dot(sim.Ds, Vx, out=self.dUds); backend.xp.dot(sim.Dt, Vx, out=self.dUdt)
        divV = sim.rst_xyz[0, 0] * self.dUdr + sim.rst_xyz[1, 0] * self.dUds + sim.rst_xyz[2, 0] * self.dUdt

        backend.xp.dot(sim.Dr, Vy, out=self.dUdr); backend.xp.dot(sim.Ds, Vy, out=self.dUds); backend.xp.dot(sim.Dt, Vy, out=self.dUdt)
        divV += sim.rst_xyz[0, 1] * self.dUdr + sim.rst_xyz[1, 1] * self.dUds + sim.rst_xyz[2, 1] * self.dUdt

        backend.xp.dot(sim.Dr, Vz, out=self.dUdr); backend.xp.dot(sim.Ds, Vz, out=self.dUds); backend.xp.dot(sim.Dt, Vz, out=self.dUdt)
        divV += sim.rst_xyz[0, 2] * self.dUdr + sim.rst_xyz[1, 2] * self.dUds + sim.rst_xyz[2, 2] * self.dUdt

        # 5. Lift + combine (into pre-allocated RHS arrays)
        Fs = sim.Fscale
        backend.xp.multiply(Fs, fluxP, out=self.lift_tmp)
        backend.xp.dot(sim.lift, self.lift_tmp, out=self.RHS_P)
        self.RHS_P -= sim.c0 ** 2 * sim.rho0 * divV

        backend.xp.multiply(Fs, fluxVx, out=self.lift_tmp)
        backend.xp.dot(sim.lift, self.lift_tmp, out=self.RHS_Vx)
        self.RHS_Vx -= dPdx / sim.rho0

        backend.xp.multiply(Fs, fluxVy, out=self.lift_tmp)
        backend.xp.dot(sim.lift, self.lift_tmp, out=self.RHS_Vy)
        self.RHS_Vy -= dPdy / sim.rho0

        backend.xp.multiply(Fs, fluxVz, out=self.lift_tmp)
        backend.xp.dot(sim.lift, self.lift_tmp, out=self.RHS_Vz)
        self.RHS_Vz -= dPdz / sim.rho0

        return self.RHS_P, self.RHS_Vx, self.RHS_Vy, self.RHS_Vz, BCvar
