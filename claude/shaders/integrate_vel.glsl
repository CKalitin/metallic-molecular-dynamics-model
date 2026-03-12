#version 430

/*
 * integrate_vel.glsl — Velocity Verlet VELOCITY update (phase 2).
 *
 *   v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
 *
 * Runs AFTER force computation at new positions.
 * acc_cur = a(t), acc_next = a(t+dt) computed from the new positions.
 */

layout(local_size_x = 64) in;

// Current velocities (read)
layout(std430, binding = 0) readonly buffer VelCur {
    vec4 vel_cur[];
};

// a(t) — acceleration at old positions
layout(std430, binding = 1) readonly buffer AccCur {
    vec4 acc_cur[];
};

// a(t+dt) — acceleration at new positions (from force shader)
layout(std430, binding = 2) readonly buffer AccNext {
    vec4 acc_next[];
};

// Updated velocities (write)
layout(std430, binding = 3) buffer VelNext {
    vec4 vel_next[];
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

    float dt = u_dt;

    vec3 v     = vel_cur[gid].xyz;
    vec3 a_old = acc_cur[gid].xyz;
    vec3 a_new = acc_next[gid].xyz;

    // Velocity Verlet velocity update
    vec3 v_new = v + 0.5 * (a_old + a_new) * dt;

    vel_next[gid] = vec4(v_new, 0.0);
}
