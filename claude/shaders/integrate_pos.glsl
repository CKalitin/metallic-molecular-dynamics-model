#version 430

/*
 * integrate_pos.glsl — Velocity Verlet POSITION update (phase 1).
 *
 *   x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
 *
 * This runs BEFORE the force computation so that forces are evaluated
 * at the new positions x(t+dt).
 */

layout(local_size_x = 64) in;

// Current state (read)
layout(std430, binding = 0) readonly buffer PosCur {
    vec4 pos_cur[];
};
layout(std430, binding = 1) readonly buffer VelCur {
    vec4 vel_cur[];
};
layout(std430, binding = 2) readonly buffer AccCur {
    vec4 acc_cur[];
};

// New positions (write)
layout(std430, binding = 3) buffer PosNext {
    vec4 pos_next[];
};

// Uniforms
layout(std430, binding = 4) readonly buffer Uniforms {
    float u_dt;
    float u_box_size;
    float u_n_atoms;
    float u_n_cells_1d;
    float u_cell_size;
    float u_max_per_cell;
    float u_r_cut;
    float u_r_cut_sq;
};

void main() {
    uint gid = gl_GlobalInvocationID.x;
    int n_atoms = int(u_n_atoms);

    if (gid >= uint(n_atoms)) return;

    float dt  = u_dt;
    float box = u_box_size;

    vec3 x = pos_cur[gid].xyz;
    vec3 v = vel_cur[gid].xyz;
    vec3 a = acc_cur[gid].xyz;
    float atom_type = pos_cur[gid].w;

    // Velocity Verlet position update
    vec3 x_new = x + v * dt + 0.5 * a * dt * dt;

    // Periodic boundary conditions
    x_new = mod(x_new, box);

    pos_next[gid] = vec4(x_new, atom_type);
}
