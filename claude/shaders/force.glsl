#version 430

/*
 * force.glsl — Compute LJ forces using cell lists.
 *
 * One invocation per atom.  Uses precomputed σ⁶, σ¹² and ε multiples
 * to avoid pow() in the inner loop. Writes acceleration = force / mass.
 *
 * LJ potential:   V(r) = 4ε [(σ/r)¹² - (σ/r)⁶]
 * LJ force mag:   F(r) = (1/r) [48ε (σ¹²/r¹²) - 24ε (σ⁶/r⁶)]
 *                       = (1/r²) [48ε σ¹² / r¹² - 24ε σ⁶ / r⁶]
 *
 * With precomputed values we avoid all pow() calls:
 *   r² → r⁶ = r² * r² * r²;   r¹² = r⁶ * r⁶
 */

layout(local_size_x = 64) in;

// Positions: vec4(x, y, z, type_float)
layout(std430, binding = 0) readonly buffer PosBuffer {
    vec4 positions[];
};

// Output accelerations: vec4(ax, ay, az, 0)
layout(std430, binding = 1) buffer AccBuffer {
    vec4 accelerations[];
};

// Masses: vec4(mass, 0, 0, 0)
layout(std430, binding = 2) readonly buffer MassBuffer {
    vec4 masses[];
};

// LJ pair table: 3 rows × 8 floats  (Fe-Fe=0, Fe-C=1, C-C=2)
//   [0]=σ⁶  [1]=σ¹²  [2]=4ε  [3]=24ε  [4]=48ε  [5]=σ  [6]=rcut  [7]=rcut²
layout(std430, binding = 3) readonly buffer LJTable {
    float lj_table[24];   // 3 * 8
};

// Cell lists
layout(std430, binding = 4) readonly buffer CellCounts {
    int cell_counts[];
};

layout(std430, binding = 5) readonly buffer CellAtoms {
    int cell_atoms[];
};

// Uniforms
layout(std430, binding = 6) readonly buffer Uniforms {
    float u_dt;
    float u_box_size;
    float u_n_atoms;
    float u_n_cells_1d;
    float u_cell_size;
    float u_max_per_cell;
    float u_r_cut;
    float u_r_cut_sq;
};

// Fetch LJ table entry
float lj(int pair_idx, int field) {
    return lj_table[pair_idx * 8 + field];
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int n_atoms      = int(u_n_atoms);
    int n_cells_1d   = int(u_n_cells_1d);
    int max_per_cell = int(u_max_per_cell);

    if (gid >= uint(n_atoms)) return;

    vec3 pos_i = positions[gid].xyz;
    int type_i = int(positions[gid].w + 0.5);  // round to int
    float mass_i = masses[gid].x;
    float box = u_box_size;
    float cs  = u_cell_size;
    float rcut_sq = u_r_cut_sq;

    vec3 force = vec3(0.0);

    // Cell of atom i (after wrapping)
    vec3 pw = mod(pos_i, box);
    int cx = clamp(int(pw.x / cs), 0, n_cells_1d - 1);
    int cy = clamp(int(pw.y / cs), 0, n_cells_1d - 1);
    int cz = clamp(int(pw.z / cs), 0, n_cells_1d - 1);

    // Loop over 27 neighboring cells (including self)
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                // Periodic cell indices
                int nx = (cx + dx + n_cells_1d) % n_cells_1d;
                int ny = (cy + dy + n_cells_1d) % n_cells_1d;
                int nz = (cz + dz + n_cells_1d) % n_cells_1d;
                int ncell = nx + ny * n_cells_1d + nz * n_cells_1d * n_cells_1d;

                int count = cell_counts[ncell];
                count = min(count, max_per_cell);

                for (int s = 0; s < count; s++) {
                    int j = cell_atoms[ncell * max_per_cell + s];
                    if (j == int(gid)) continue;

                    vec3 pos_j = positions[j].xyz;
                    int type_j = int(positions[j].w + 0.5);

                    // Minimum image convention
                    vec3 rij = pos_i - pos_j;
                    rij -= box * round(rij / box);

                    float r_sq = dot(rij, rij);
                    if (r_sq >= rcut_sq) continue;

                    // Pair type index: Fe+Fe=0, Fe+C=1, C+C=2
                    int pair_idx = type_i + type_j;

                    // Precomputed LJ parameters
                    float sigma6  = lj(pair_idx, 0);
                    float sigma12 = lj(pair_idx, 1);
                    float eps24   = lj(pair_idx, 3);
                    float eps48   = lj(pair_idx, 4);
                    float sigma   = lj(pair_idx, 5);

                    // Force cap: if r < 0.7*sigma, evaluate forces at r_cap.
                    // Prevents float overflow in the repulsive core and stops
                    // atoms from passing through each other silently.
                    float r_cap_sq = 0.49 * sigma * sigma;  // (0.7*sigma)^2
                    float r2_eff = max(r_sq, r_cap_sq);

                    // r⁶ and r¹² from r² (no pow!)
                    float r2_inv = 1.0 / r2_eff;
                    float r6_inv = r2_inv * r2_inv * r2_inv;
                    float r12_inv = r6_inv * r6_inv;

                    // Force direction always along actual rij, magnitude from capped r
                    // f_over_r_sq = |F| / |rij|²  (F = 48ε σ¹² r⁻¹³ - 24ε σ⁶ r⁻⁷)
                    float f_over_r_sq = eps48 * sigma12 * r12_inv * r2_inv
                                      - eps24 * sigma6  * r6_inv  * r2_inv;

                    force += f_over_r_sq * rij;
                }
            }
        }
    }

    // Acceleration = Force / mass
    accelerations[gid] = vec4(force / mass_i, 0.0);
}
