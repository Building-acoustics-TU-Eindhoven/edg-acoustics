"""3D visualization of DG acoustic simulation. Dark mode, neon waves."""
import numpy, edg_acoustics, os, glob, scipy.io, time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

plt.style.use('dark_background')
plt.rcParams['font.family'] = 'Segoe UI'

neon_cmap = LinearSegmentedColormap.from_list('neon', [
    (0.00, '#00EEFF'), (0.30, '#002266'), (0.50, '#000000'),
    (0.70, '#660022'), (1.00, '#FF00FF'),
])

OUT = Path('vis3d_dg'); OUT.mkdir(exist_ok=True)

# Setup
rho0, c0 = 1.213, 343
BC_labels = {'hard wall': 11, 'carpet': 13, 'panel': 14}
monopole_xyz = numpy.array([3.04, 2.59, 1.62])
rec = numpy.array([[4.26], [1.76], [1.62]])

BC_para = [{'label': 11, 'RI': 1}]
for mn in ['carpet', 'panel']:
    mf = scipy.io.loadmat(glob.glob(f'examples/scenario1/{mn}*.mat')[0])
    d = {'label': BC_labels[mn], 'RI': mf.get('RI', [[0]])[0]}
    if 'AS' in mf: d['RP'] = numpy.array([mf['AS'][0], mf['lambdaS'][0]])
    if 'BS' in mf: d['CP'] = numpy.array([mf['BS'][0], mf['CS'][0], mf['alphaS'][0], mf['betaS'][0]])
    BC_para.append(d)

mesh = edg_acoustics.Mesh('examples/scenario1/scenario1_coarse.msh', BC_labels)
sim = edg_acoustics.AcousticsSimulation(rho0, c0, 4, mesh, BC_labels)
flux = edg_acoustics.UpwindFlux(rho0, c0, sim.n_xyz)
AbBC = edg_acoustics.AbsorbBC(sim.BCnode, BC_para)
sim.init_BC(AbBC)
sim.init_IC(edg_acoustics.Monopole_IC(monopole_xyz, 200))
sim.init_Flux(flux)
sim.init_rec(rec, 'scipy')
tsi = edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, 0.5, Nt=3)
sim.init_TimeIntegrator(tsi)

# GPU + fused RHS
import edg_acoustics.gpu_backend
from edg_acoustics.gpu_rhs import FusedRHS
from edg_acoustics.gpu_backend import sync, to_device
gpu_ok = sim.transfer_to_gpu()
if gpu_ok:
    fused = FusedRHS(sim)
    fused.prepare()
    tsi.L_operator = fused
    sim.P = to_device(sim.IC.Pinit(sim.xyz))
    sim.Vx = to_device(sim.IC.VXinit(sim.xyz))
    sim.Vy = to_device(sim.IC.VYinit(sim.xyz))
    sim.Vz = to_device(sim.IC.VZinit(sim.xyz))
    print('GPU+fused enabled')
else:
    print('Running on CPU')

# Node coordinates: (3, Np, N_tets)
xyz = sim.xyz
x_all = xyz[0].ravel()
y_all = xyz[1].ravel()
z_all = xyz[2].ravel()

# Room wireframe from mesh vertices
verts = mesh.vertices  # (N_verts, 3)

# Room bounds
xr = [verts[:, 0].min(), verts[:, 0].max()]
yr = [verts[:, 1].min(), verts[:, 1].max()]
zr = [verts[:, 2].min(), verts[:, 2].max()]

print(f'Room: [{xr[0]:.1f},{xr[1]:.1f}] x [{yr[0]:.1f},{yr[1]:.1f}] x [{zr[0]:.1f},{zr[1]:.1f}]')
print(f'Nodes: {len(x_all):,}')

# Run sim and capture snapshots
snap_steps = list(range(1, 100, 2)) + list(range(100, 500, 10)) + list(range(500, 3000, 50))
snap_set = set(snap_steps)
frames_P = []
frames_t = []

print('Running DG simulation...')
t0 = time.perf_counter()
for n in range(3000):
    tsi.step_dt(sim.P, sim.Vx, sim.Vy, sim.Vz, AbBC)
    if n in snap_set:
        P_host = edg_acoustics.gpu_backend.to_host(sim.P) if gpu_ok else sim.P
        frames_P.append(P_host.ravel().copy())
        frames_t.append(n * tsi.dt * 1000)
    if (n+1) % 500 == 0:
        print(f'  step {n+1}/3000 ({time.perf_counter()-t0:.0f}s)')

print(f'Done: {time.perf_counter()-t0:.0f}s, {len(frames_P)} snapshots')

# Global vmax from wavefront
global_vmax = max(numpy.max(numpy.abs(frames_P[i])) for i in range(min(10, len(frames_P))))
global_vmax = max(global_vmax, 1e-10)

# Pick ~30 frames
n_frames = min(35, len(frames_P))
picks = numpy.unique(numpy.linspace(0, len(frames_P)-1, n_frames).astype(int))

print(f'Rendering {len(picks)} frames (vmax={global_vmax:.2e})...')

for fi, si in enumerate(picks):
    P_flat = frames_P[si]
    t_ms = frames_t[si]

    intensity = numpy.abs(P_flat) / global_vmax
    threshold = 0.03
    alpha = numpy.where(intensity < threshold, 0.0,
                        numpy.sqrt((intensity - threshold) / (1 - threshold)))
    alpha = numpy.clip(alpha, 0, 0.9)

    # Only plot points above threshold
    visible = alpha > 0.01
    if numpy.sum(visible) == 0:
        visible = numpy.ones(len(P_flat), dtype=bool)

    colors = neon_cmap((P_flat[visible] / global_vmax + 1) / 2)
    colors[:, 3] = alpha[visible]

    fig = plt.figure(figsize=(11, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Room wireframe
    edges = [
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ]
    # Use first 8 vertices as room corners (assuming box-like)
    if len(verts) >= 8:
        corners = verts[:8]
        for i, j in edges:
            if i < len(corners) and j < len(corners):
                ax.plot3D(*zip(corners[i], corners[j]),
                          color='white', lw=0.8, alpha=0.5)

    # Scatter pressure nodes
    ax.scatter(x_all[visible], y_all[visible], z_all[visible],
               c=colors, s=0.3, depthshade=False, rasterized=True)

    # Source + receiver
    ax.scatter(*monopole_xyz, color='#FF2222', s=150, marker='*',
               edgecolors='white', linewidth=0.5, zorder=10, depthshade=False)
    ax.scatter(rec[0, 0], rec[1, 0], rec[2, 0], color='#22FF22', s=100,
               marker='^', edgecolors='white', linewidth=0.5, zorder=10, depthshade=False)

    ax.set_xlim(xr[0]-0.3, xr[1]+0.3)
    ax.set_ylim(yr[0]-0.3, yr[1]+0.3)
    ax.set_zlim(zr[0]-0.1, zr[1]+0.3)
    ax.set_xlabel('x', color='#444', fontsize=7, labelpad=-4)
    ax.set_ylabel('y', color='#444', fontsize=7, labelpad=-4)
    ax.set_zlabel('z', color='#444', fontsize=7, labelpad=-4)
    ax.tick_params(colors='#222', labelsize=5, pad=-2)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1a1a1a'); ax.yaxis.pane.set_edgecolor('#1a1a1a'); ax.zaxis.pane.set_edgecolor('#1a1a1a')
    ax.grid(False)
    ax.view_init(elev=22, azim=-52 + fi * 0.7)

    fig.text(0.5, 0.96, f'edg-acoustics (DG)    t = {t_ms:.1f} ms',
             color='white', fontsize=15, fontweight='bold', ha='center')

    fig.savefig(OUT / f'frame_{fi:03d}.png', dpi=120, facecolor='black',
                bbox_inches='tight', pad_inches=0.05)
    plt.close()
    if (fi+1) % 10 == 0 or fi == 0:
        print(f'  [{fi+1}/{len(picks)}] t={t_ms:.1f}ms')

# GIF
try:
    from PIL import Image
    frames_img = [Image.open(OUT / f'frame_{fi:03d}.png') for fi in range(len(picks))]
    gif = OUT / 'edg_acoustics_3d.gif'
    frames_img[0].save(gif, save_all=True, append_images=frames_img[1:],
                       duration=200, loop=0, optimize=True)
    print(f'GIF: {gif} ({os.path.getsize(gif)/1e6:.1f} MB)')
except ImportError:
    pass

print('Done!')
