#version 430

/*
 * build_cells.glsl — Assign each atom to a spatial cell.
 *
 * Layout: one invocation per atom.
 * Reads positions, writes into cell_counts / cell_atoms arrays.
 */

layout(local_size_x = 64) in;

// Positions: vec4(x, y, z, type)
layout(std430, binding = 0) readonly buffer PosBuffer {
    vec4 positions[];
};

// cell_counts[cell_id] = number of atoms in that cell
layout(std430, binding = 1) buffer CellCounts {
    int cell_counts[];
};

// cell_atoms[cell_id * max_per_cell + local_idx] = atom global index
layout(std430, binding = 2) buffer CellAtoms {
    int cell_atoms[];
};

// Uniforms packed as floats
layout(std430, binding = 3) readonly buffer Uniforms {
    float u_dt;                // 0
    float u_box_size;          // 1
    float u_n_atoms;           // 2
    float u_n_cells_1d;        // 3
    float u_cell_size;         // 4
    float u_max_per_cell;      // 5
    float u_r_cut;             // 6
    float u_r_cut_sq;          // 7
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int n_atoms     = int(u_n_atoms);
    int n_cells_1d  = int(u_n_cells_1d);
    int max_per_cell = int(u_max_per_cell);

    if (gid >= uint(n_atoms)) return;

    vec3 p = positions[gid].xyz;
    float box = u_box_size;
    float cs  = u_cell_size;

    // Wrap into box  [0, box)
    p = mod(p, box);

    // Cell indices
    int cx = clamp(int(p.x / cs), 0, n_cells_1d - 1);
    int cy = clamp(int(p.y / cs), 0, n_cells_1d - 1);
    int cz = clamp(int(p.z / cs), 0, n_cells_1d - 1);

    int cell_id = cx + cy * n_cells_1d + cz * n_cells_1d * n_cells_1d;

    // Atomically add to cell
    int slot = atomicAdd(cell_counts[cell_id], 1);
    if (slot < max_per_cell) {
        cell_atoms[cell_id * max_per_cell + slot] = int(gid);
    }
}
