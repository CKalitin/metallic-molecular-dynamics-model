"""
Lennard-Jones Metallic Molecular Dynamics Simulation
=====================================================
Fe-C system using GPU-accelerated Velocity Verlet integration (ModernGL compute shaders)
and real-time visualization (VisPy).

Units: Angstroms, electronvolts (eV), femtoseconds (fs)
kB = 8.617e-5 eV/K
Mass in eV·fs²/Å² (converted from amu)
"""

import os
import numpy as np
import moderngl
from OpenGL.GL import glMemoryBarrier
GL_SSBB = 0x00002000  # GL_SHADER_STORAGE_BARRIER_BIT

# Suppress Qt DPI-awareness warning on Windows (must be set before Qt loads)
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
os.environ["QT_FONT_DPI"] = "96"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false;default.warning=false"

from vispy import app, scene
from vispy.scene import visuals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
KB = 8.617e-5          # Boltzmann constant [eV/K]
# 1 amu = 1.66054e-27 kg; 1 eV·fs²/Å² = 1.60218e-29 kg
# => 1 amu = 1.66054e-27 / 1.60218e-29 = 103.64 eV·fs²/Å²
AMU_TO_EVFS2A2 = 103.64

# ---------------------------------------------------------------------------
# Lennard-Jones parameters (pure species)
# ---------------------------------------------------------------------------
# Fe-Fe  (typical metallic LJ fit)
SIGMA_FE = 2.3193      # Å
EPSILON_FE = 0.4174     # eV

# C-C  (graphite / diamond LJ)
SIGMA_C = 3.4           # Å
EPSILON_C = 0.00239     # eV

# Atomic masses
MASS_FE = 55.845 * AMU_TO_EVFS2A2   # eV·fs²/Å²
MASS_C  = 12.011 * AMU_TO_EVFS2A2   # eV·fs²/Å²

# ---------------------------------------------------------------------------
# Lorentz-Berthelot mixing rules  (Fe-C cross-interaction)
# ---------------------------------------------------------------------------
def lorentz_sigma(sigma_a: float, sigma_b: float) -> float:
    """Arithmetic mean of sigma (Lorentz rule)."""
    return 0.5 * (sigma_a + sigma_b)

def berthelot_epsilon(eps_a: float, eps_b: float) -> float:
    """Geometric mean of epsilon (Berthelot rule)."""
    return np.sqrt(eps_a * eps_b)

SIGMA_FE_C   = lorentz_sigma(SIGMA_FE, SIGMA_C)
EPSILON_FE_C = berthelot_epsilon(EPSILON_FE, EPSILON_C)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N_CELLS_PER_DIM = 6        # FCC unit cells per dimension
# Lattice constant set from the LARGEST sigma (C-C = 3.4 Å) so that every pair
# starts at or beyond its LJ energy minimum.  Nearest-neighbour distance in FCC
# is a/sqrt(2); setting that equal to 2^(1/6)*sigma_C gives:
#   a = 2^(1/6) * sigma_C * sqrt(2) ≈ 5.40 Å
# Fe-Fe pairs will be slightly beyond their minimum (attractive, gentle).
LATTICE_CONST = 2.0 ** (1.0 / 6.0) * SIGMA_C * 2.0 ** 0.5  # Å
DT              = 1.0      # timestep [fs]
TARGET_TEMP     = 300.0    # K  (for initial velocity sampling)

# Cutoff = 2.5 * max sigma  (standard LJ practice)
R_CUT = 2.5 * max(SIGMA_FE, SIGMA_C, SIGMA_FE_C)

# Cell-list grid spacing (must be >= R_CUT)
CELL_SIZE = R_CUT  # Å – adjustable from Python side

# Velocity rescaling buffer (fraction above target used during init)
VEL_BUFFER = 1.05

# ---------------------------------------------------------------------------
# FCC lattice builder
# ---------------------------------------------------------------------------
FCC_BASIS = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
], dtype=np.float32)

def build_fcc_lattice(n: int, a: float, fe_fraction: float = 0.9):
    """
    Build an FCC lattice of n×n×n unit cells with lattice constant *a*.
    Returns positions (N,4), types (N,) where type 0=Fe, 1=C.
    The 4th component of position is unused padding for GPU alignment.
    """
    positions = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                origin = np.array([ix, iy, iz], dtype=np.float32)
                for b in FCC_BASIS:
                    pos = (origin + b) * a
                    positions.append(pos)
    positions = np.array(positions, dtype=np.float32)
    n_atoms = len(positions)

    # Assign types: random Fe/C distribution
    rng = np.random.default_rng(42)
    types = (rng.random(n_atoms) > fe_fraction).astype(np.int32)  # 0=Fe, 1=C

    # Pad positions to vec4  (x, y, z, type_as_float)
    pos4 = np.zeros((n_atoms, 4), dtype=np.float32)
    pos4[:, :3] = positions
    pos4[:, 3] = types.astype(np.float32)

    return pos4, types

# ---------------------------------------------------------------------------
# Initial velocities from Maxwell-Boltzmann
# ---------------------------------------------------------------------------
def maxwell_boltzmann_velocities(n_atoms: int, types: np.ndarray,
                                  temperature: float, buffer: float = 1.0):
    """
    Sample velocities from Maxwell-Boltzmann distribution at *temperature*.
    Returns vel (N,4) — 4th component = 0 padding.
    """
    rng = np.random.default_rng(123)
    masses = np.where(types == 0, MASS_FE, MASS_C)

    vel = np.zeros((n_atoms, 4), dtype=np.float32)
    for dim in range(3):
        sigma_v = np.sqrt(KB * temperature * buffer / masses)
        vel[:, dim] = (rng.normal(size=n_atoms) * sigma_v).astype(np.float32)

    # Remove net momentum
    total_mass = masses.sum()
    for dim in range(3):
        p_total = (masses * vel[:, dim]).sum()
        vel[:, dim] -= (p_total / total_mass).astype(np.float32)

    return vel

# ---------------------------------------------------------------------------
# Temperature from kinetic energy  (Kinetic Theory of Gases)
# ---------------------------------------------------------------------------
def compute_temperature(vel: np.ndarray, types: np.ndarray) -> float:
    """T = (2/3) * KE_total / (N * kB)  where KE = Σ ½ m v²"""
    masses = np.where(types == 0, MASS_FE, MASS_C)
    v2 = np.sum(vel[:, :3].astype(np.float64) ** 2, axis=1)
    ke = 0.5 * np.sum(masses * v2)
    n_atoms = len(types)
    return (2.0 / 3.0) * ke / (n_atoms * KB)

# ---------------------------------------------------------------------------
# Precompute per-pair LJ constants  (avoids pow in shader)
# ---------------------------------------------------------------------------
def precompute_lj_table():
    """
    3 pair types: Fe-Fe (0-0), Fe-C (0-1 / 1-0), C-C (1-1).
    For each: sigma6, sigma12, eps4, eps24, eps48.
    Returns a flat float32 array of shape (3, 8) padded for std430.
    """
    pairs = [
        (SIGMA_FE,   EPSILON_FE),    # Fe-Fe  idx=0
        (SIGMA_FE_C, EPSILON_FE_C),  # Fe-C   idx=1
        (SIGMA_C,    EPSILON_C),     # C-C    idx=2
    ]
    table = np.zeros((3, 8), dtype=np.float32)  # 8 for std430 alignment
    for i, (sig, eps) in enumerate(pairs):
        s6  = sig ** 6
        s12 = sig ** 12
        table[i, 0] = s6
        table[i, 1] = s12
        table[i, 2] = 4.0 * eps          # for potential
        table[i, 3] = 24.0 * eps         # for force (attractive)
        table[i, 4] = 48.0 * eps         # for force (repulsive)
        table[i, 5] = sig               # sigma (for cutoff checks)
        table[i, 6] = R_CUT             # cutoff
        table[i, 7] = R_CUT * R_CUT     # cutoff²
    return table

def pair_index(type_a: int, type_b: int) -> int:
    """Map two atom types to LJ table index: 0=Fe-Fe, 1=Fe-C, 2=C-C."""
    return type_a + type_b  # 0+0=0, 0+1=1, 1+1=2

# ---------------------------------------------------------------------------
# GPU setup  (ModernGL compute shader)
# ---------------------------------------------------------------------------
def load_shader(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def create_gpu_resources(ctx, pos: np.ndarray, vel: np.ndarray,
                         types: np.ndarray, lj_table: np.ndarray,
                         box_size: float):
    """
    Set up compute buffers and shaders on an existing ModernGL context.
    Returns dict with ctx, shaders, buffers.
    """
    n_atoms = len(pos)
    acc = np.zeros_like(pos)  # initial accelerations = 0

    # Masses as vec4-aligned (mass, 0, 0, 0) — kept for per-atom lookup
    masses_arr = np.zeros((n_atoms, 4), dtype=np.float32)
    masses_arr[:, 0] = np.where(types == 0, MASS_FE, MASS_C)

    # Double-buffered state: "current" and "next"
    buf_pos_cur  = ctx.buffer(pos.tobytes())
    buf_vel_cur  = ctx.buffer(vel.tobytes())
    buf_acc_cur  = ctx.buffer(acc.tobytes())

    buf_pos_next = ctx.buffer(pos.tobytes())   # will be overwritten
    buf_vel_next = ctx.buffer(vel.tobytes())
    buf_acc_next = ctx.buffer(acc.tobytes())

    buf_masses   = ctx.buffer(masses_arr.tobytes())
    buf_lj       = ctx.buffer(lj_table.tobytes())

    # Cell-list buffers ---------------------------------------------------
    n_cells_1d = max(1, int(np.floor(box_size / CELL_SIZE)))
    n_cells_total = n_cells_1d ** 3
    # Must hold all atoms in the densest cell — compute worst-case with 2x safety margin
    max_atoms_per_cell = max(64, int(np.ceil(n_atoms / n_cells_total)) * 2)

    # Actual cell width (may differ from CELL_SIZE due to integer floor)
    actual_cell_size = box_size / n_cells_1d

    # cell_counts[c]        = how many atoms in cell c
    # cell_atoms[c * MAX + i] = atom index
    cell_counts = np.zeros(n_cells_total, dtype=np.int32)
    cell_atoms  = np.full(n_cells_total * max_atoms_per_cell, -1, dtype=np.int32)

    buf_cell_counts = ctx.buffer(cell_counts.tobytes())
    buf_cell_atoms  = ctx.buffer(cell_atoms.tobytes())

    # Uniform buffer (simulation constants)
    uniforms = np.array([
        DT,                          # 0: dt
        box_size,                    # 1: box_size
        float(n_atoms),              # 2: n_atoms (as float for shader)
        float(n_cells_1d),           # 3: n_cells_1d
        actual_cell_size,            # 4: cell_size (actual = box/n_cells_1d)
        float(max_atoms_per_cell),   # 5: max_atoms_per_cell
        R_CUT,                       # 6: r_cut
        R_CUT * R_CUT,              # 7: r_cut_sq
    ], dtype=np.float32)
    buf_uniforms = ctx.buffer(uniforms.tobytes())

    # Load & compile compute shaders
    cs_build_cells   = ctx.compute_shader(load_shader("shaders/build_cells.glsl"))
    cs_force         = ctx.compute_shader(load_shader("shaders/force.glsl"))
    cs_integrate_pos = ctx.compute_shader(load_shader("shaders/integrate_pos.glsl"))
    cs_integrate_vel = ctx.compute_shader(load_shader("shaders/integrate_vel.glsl"))

    return {
        "ctx": ctx,
        "n_atoms": n_atoms,
        "box_size": box_size,
        "n_cells_1d": n_cells_1d,
        "n_cells_total": n_cells_total,
        "max_atoms_per_cell": max_atoms_per_cell,
        # Shaders
        "cs_build_cells": cs_build_cells,
        "cs_force": cs_force,
        "cs_integrate_pos": cs_integrate_pos,
        "cs_integrate_vel": cs_integrate_vel,
        # Buffers
        "buf_pos_cur": buf_pos_cur,
        "buf_vel_cur": buf_vel_cur,
        "buf_acc_cur": buf_acc_cur,
        "buf_pos_next": buf_pos_next,
        "buf_vel_next": buf_vel_next,
        "buf_acc_next": buf_acc_next,
        "buf_masses": buf_masses,
        "buf_lj": buf_lj,
        "buf_cell_counts": buf_cell_counts,
        "buf_cell_atoms": buf_cell_atoms,
        "buf_uniforms": buf_uniforms,
    }

# ---------------------------------------------------------------------------
# Step the simulation (one Velocity Verlet cycle)
# ---------------------------------------------------------------------------
def compute_initial_forces(gpu):
    """Compute a(t=0) so the first Velocity Verlet step is correct."""
    n = gpu["n_atoms"]
    wg = max(1, (n + 63) // 64)

    # Build cell lists for initial positions
    gpu["buf_cell_counts"].write(
        np.zeros(gpu["n_cells_total"], dtype=np.int32).tobytes()
    )
    gpu["buf_pos_cur"].bind_to_storage_buffer(0)
    gpu["buf_cell_counts"].bind_to_storage_buffer(1)
    gpu["buf_cell_atoms"].bind_to_storage_buffer(2)
    gpu["buf_uniforms"].bind_to_storage_buffer(3)
    gpu["cs_build_cells"].run(wg)
    glMemoryBarrier(GL_SSBB)

    # Force at initial positions → acc_cur
    gpu["buf_pos_cur"].bind_to_storage_buffer(0)
    gpu["buf_acc_cur"].bind_to_storage_buffer(1)
    gpu["buf_masses"].bind_to_storage_buffer(2)
    gpu["buf_lj"].bind_to_storage_buffer(3)
    gpu["buf_cell_counts"].bind_to_storage_buffer(4)
    gpu["buf_cell_atoms"].bind_to_storage_buffer(5)
    gpu["buf_uniforms"].bind_to_storage_buffer(6)
    gpu["cs_force"].run(wg)
    glMemoryBarrier(GL_SSBB)


def step(gpu):
    """
    One Velocity Verlet cycle:
      1. x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²    [integrate_pos]
      2. Build cell lists for x(t+dt)
      3. a(t+dt) = F(x(t+dt)) / m                    [force]
      4. v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt    [integrate_vel]
      5. Swap buffers
    """
    n = gpu["n_atoms"]
    wg = max(1, (n + 63) // 64)

    # --- 1) Update positions ---
    gpu["buf_pos_cur"].bind_to_storage_buffer(0)
    gpu["buf_vel_cur"].bind_to_storage_buffer(1)
    gpu["buf_acc_cur"].bind_to_storage_buffer(2)
    gpu["buf_pos_next"].bind_to_storage_buffer(3)
    gpu["buf_uniforms"].bind_to_storage_buffer(4)
    gpu["cs_integrate_pos"].run(wg)
    glMemoryBarrier(GL_SSBB)

    # --- 2) Build cell lists for NEW positions ---
    gpu["buf_cell_counts"].write(
        np.zeros(gpu["n_cells_total"], dtype=np.int32).tobytes()
    )
    gpu["buf_pos_next"].bind_to_storage_buffer(0)
    gpu["buf_cell_counts"].bind_to_storage_buffer(1)
    gpu["buf_cell_atoms"].bind_to_storage_buffer(2)
    gpu["buf_uniforms"].bind_to_storage_buffer(3)
    gpu["cs_build_cells"].run(wg)
    glMemoryBarrier(GL_SSBB)

    # --- 3) Compute forces at NEW positions → acc_next ---
    gpu["buf_pos_next"].bind_to_storage_buffer(0)
    gpu["buf_acc_next"].bind_to_storage_buffer(1)
    gpu["buf_masses"].bind_to_storage_buffer(2)
    gpu["buf_lj"].bind_to_storage_buffer(3)
    gpu["buf_cell_counts"].bind_to_storage_buffer(4)
    gpu["buf_cell_atoms"].bind_to_storage_buffer(5)
    gpu["buf_uniforms"].bind_to_storage_buffer(6)
    gpu["cs_force"].run(wg)
    glMemoryBarrier(GL_SSBB)

    # --- 4) Update velocities ---
    gpu["buf_vel_cur"].bind_to_storage_buffer(0)
    gpu["buf_acc_cur"].bind_to_storage_buffer(1)
    gpu["buf_acc_next"].bind_to_storage_buffer(2)
    gpu["buf_vel_next"].bind_to_storage_buffer(3)
    gpu["buf_uniforms"].bind_to_storage_buffer(4)
    gpu["cs_integrate_vel"].run(wg)
    glMemoryBarrier(GL_SSBB)

    # --- 5) Swap buffers ---
    gpu["buf_pos_cur"], gpu["buf_pos_next"] = gpu["buf_pos_next"], gpu["buf_pos_cur"]
    gpu["buf_vel_cur"], gpu["buf_vel_next"] = gpu["buf_vel_next"], gpu["buf_vel_cur"]
    gpu["buf_acc_cur"], gpu["buf_acc_next"] = gpu["buf_acc_next"], gpu["buf_acc_cur"]

def read_positions(gpu) -> np.ndarray:
    raw = gpu["buf_pos_cur"].read()
    return np.frombuffer(raw, dtype=np.float32).reshape(gpu["n_atoms"], 4).copy()

def read_velocities(gpu) -> np.ndarray:
    raw = gpu["buf_vel_cur"].read()
    return np.frombuffer(raw, dtype=np.float32).reshape(gpu["n_atoms"], 4).copy()

# ---------------------------------------------------------------------------
# VisPy real-time renderer
# ---------------------------------------------------------------------------
class MDCanvas:
    def __init__(self, init_pos, init_vel, types, lj_table, box_size):
        self.types = types
        self.step_count = 0

        # 1) Create VisPy canvas first — this establishes the GL context
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(1024, 768), show=True,
            title="Fe-C Molecular Dynamics"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            distance=box_size * 1.8, fov=70, translate_speed=10.0
        )

        # 2) Create ModernGL context from VisPy's active GL context
        ctx = moderngl.create_context(require=430)

        # 3) Set up GPU buffers and compile shaders on this shared context
        self.gpu = create_gpu_resources(ctx, init_pos, init_vel, types,
                                        lj_table, box_size)

        # 4) Compute initial forces
        compute_initial_forces(self.gpu)

        print(f"Cell grid: {self.gpu['n_cells_1d']}³ = {self.gpu['n_cells_total']} cells  "
              f"(cell size = {self.gpu['box_size']/self.gpu['n_cells_1d']:.2f} Å, r_cut = {R_CUT:.2f} Å)")

        # 5) Set up scatter plot from numpy data (no GPU read needed)
        colors = np.where(
            types[:, None] == 0,
            np.array([0.75, 0.75, 0.78, 1.0]),
            np.array([0.25, 0.25, 0.25, 1.0]),
        )

        self.scatter = visuals.Markers(scaling=True)
        self.scatter.set_data(
            init_pos[:, :3], face_color=colors, size=1.0, edge_width=0
        )
        self.view.add(self.scatter)

        # Bounding box
        box = box_size
        corners = np.array([
            [0, 0, 0], [box, 0, 0], [box, box, 0], [0, box, 0], [0, 0, 0],
            [0, 0, box], [box, 0, box], [box, box, box], [0, box, box],
            [0, 0, box],
        ], dtype=np.float32)
        box_lines = visuals.Line(pos=corners, color=(0.4, 0.4, 0.4, 0.5), width=1)
        self.view.add(box_lines)
        for s, e in [([box, 0, 0], [box, 0, box]),
                     ([box, box, 0], [box, box, box]),
                     ([0, box, 0], [0, box, box])]:
            edge = visuals.Line(
                pos=np.array([s, e], dtype=np.float32),
                color=(0.4, 0.4, 0.4, 0.5), width=1
            )
            self.view.add(edge)

        self.colors = colors
        self.timer = app.Timer(interval=0.001, connect=self.on_timer, start=True)

    def on_timer(self, event):
        step(self.gpu)
        self.step_count += 1

        pos = read_positions(self.gpu)[:, :3]
        self.scatter.set_data(pos, face_color=self.colors, size=0.8, edge_width=0)

        # Velocity-rescaling thermostat every step
        vel = read_velocities(self.gpu)
        temp = compute_temperature(vel, self.types)
        if np.isfinite(temp) and temp > 1.0:
            scale = float(np.sqrt(TARGET_TEMP / temp))
            vel[:, :3] = (vel[:, :3] * scale).astype(np.float32)
            self.gpu["buf_vel_cur"].write(vel.tobytes())
        print(f"Step {self.step_count:>6d}  |  T = {temp:8.2f} K")

        self.canvas.update()

    def run(self):
        app.run()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global R_CUT, CELL_SIZE  # May be adjusted for minimum image convention

    print("=" * 60)
    print("  Fe-C Lennard-Jones Molecular Dynamics")
    print("=" * 60)
    print()

    # Derived parameters
    print(f"Fe-Fe:  σ = {SIGMA_FE:.4f} Å   ε = {EPSILON_FE:.4f} eV")
    print(f"C-C:    σ = {SIGMA_C:.4f} Å   ε = {EPSILON_C:.5f} eV")
    print(f"Fe-C:   σ = {SIGMA_FE_C:.4f} Å   ε = {EPSILON_FE_C:.5f} eV  (Lorentz-Berthelot)")
    print()

    # Build lattice
    pos, types = build_fcc_lattice(N_CELLS_PER_DIM, LATTICE_CONST)
    n_atoms = len(pos)
    box_size = N_CELLS_PER_DIM * LATTICE_CONST

    # Enforce minimum image convention: R_CUT must be < box/2
    half_box = box_size / 2.0
    if R_CUT >= half_box:
        R_CUT = half_box - 0.1
        CELL_SIZE = R_CUT
        print(f"WARNING: Cutoff clamped to {R_CUT:.2f} Å (< box/2 = {half_box:.2f} Å)")
        print()

    print(f"Atoms: {n_atoms}  ({np.sum(types == 0)} Fe, {np.sum(types == 1)} C)")
    print(f"Box:   {box_size:.2f} Å")
    print(f"Cutoff: {R_CUT:.2f} Å  (box/2 = {half_box:.2f} Å)")

    # Initial velocities
    vel = maxwell_boltzmann_velocities(n_atoms, types, TARGET_TEMP, VEL_BUFFER)
    t_init = compute_temperature(vel, types)
    print(f"Initial temperature: {t_init:.1f} K  (target {TARGET_TEMP} K, buffer {VEL_BUFFER}x)")
    print()

    # Precompute LJ table
    lj_table = precompute_lj_table()
    print("LJ pair table (precomputed σ⁶, σ¹², 4ε, 24ε, 48ε):")
    for i, name in enumerate(["Fe-Fe", "Fe-C ", "C-C  "]):
        print(f"  {name}: σ⁶={lj_table[i,0]:.4f}  σ¹²={lj_table[i,1]:.4f}  "
              f"4ε={lj_table[i,2]:.6f}  24ε={lj_table[i,3]:.6f}")
    print()

    # Launch viewer (creates VisPy canvas, then ModernGL context, then GPU resources)
    print("Setting up GPU compute shaders...")
    viewer = MDCanvas(pos, vel, types, lj_table, box_size)
    print()
    print("Launching viewer... (close window to stop)")
    viewer.run()

if __name__ == "__main__":
    main()
