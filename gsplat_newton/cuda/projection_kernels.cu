#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_kernel_LN(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [N, 3]
    const scalar_t *__restrict__ covars,   // [N, 6] optional
    const scalar_t *__restrict__ quats,    // [N, 4] optional
    const scalar_t *__restrict__ scales,   // [N, 3] optional
    const scalar_t *__restrict__ opacities, // [N] optional
    const scalar_t *__restrict__ viewmats, // [C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [C, N, 2]
    scalar_t *__restrict__ means2d,      // [C, N, 2]
    scalar_t *__restrict__ depths,       // [C, N]
    scalar_t *__restrict__ conics,       // [C, N, 3]
    scalar_t *__restrict__ compensations // [C, N] optional
    //outputs for Local Newton
    mat3x2 *__restrict__ jacobians, // [C, N, 3, 2] dmean2d/dmean3d
    mat3 *__restrict__ H_mean_y, // elements of Hessian d2mux/dmean3d^2 [C, N, 3, 3]
    mat3 *__restirct__ H_mean_x, // elements of hessian d2muy/dmean3d^2 [C, N, 3, 3]
    mat2 *__restrict__ dSigma_dx,        // [C, N, 2, 2] ∂Σ/∂x
    mat2 *__restrict__ dSigma_dy,        // [C, N, 2, 2] ∂Σ/∂y
    mat2 *__restrict__ dSigma_dz         // [C, N, 2, 2] ∂Σ/∂z
    mat2 *__restrict__ H_Sigma, // stores it like that [H_S_xz, H_S_yz, H_S_zz] ∂2Σ/∂p2 [C, N, 3]
    mat3 *__restrict__ dr_dp,
    float *__restrict__ d2r_dp2_compact // hessian of [C, N, 18]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    viewmats += cid * 16;
    Ks += cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );

    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // CALCULATING dr/dpk and d2r/dpk2
    vec3 p_k = glm::make_vec3(means);
    mat3 R_c2w = transpose(R_w2c);
    vec3 camera_origin = -R_c2w * t;
    vec3 d = p_k - camera_origin;
    mat3 dr_dp_local = mat3(0.f);
    float n = length(d);
    float h_compact[18] = {0.f};
    if (n > 1e-7) {
        float n_inv = 1.f / n;
        vec3 r_k = d * n_inv;

        // Jacobian calculation is unchanged
        mat3 I = mat3(1.0f);
        mat3 r_outer_r = outerProduct(r_k, r_k);
        dr_dp_local = n_inv * (I - r_outer_r);

        // ========================================================================
        // 2. EFFICIENT HESSIAN CALCULATION (NO LOOPS)
        // ========================================================================

        // We calculate the 6 unique elements for each of the 3 Hessians directly.
        // H_i is the Hessian of r_k.i, H_ij is the Hessian of r_k.j, etc.
        // J_ij is the (i,j) component of the Jacobian dr_dp_local

        float J[3][3], r[3];
        #pragma unroll
        for(int i=0; i<3; ++i) {
            r[i] = r_k[i];
            for(int j=0; j<3; ++j) J[i][j] = dr_dp_local[i][j];
        }

        float n_inv_neg = -n_inv;

        // Hessian for r_k.x (6 unique terms)
        h_compact[0]  = n_inv_neg * (3 * J[0][0] * r[0]);                            // xx
        h_compact[1]  = n_inv_neg * (J[0][1] * r[1] + J[1][1] * r[0] + J[0][1] * r[1]); // yy
        h_compact[2]  = n_inv_neg * (J[0][2] * r[2] + J[2][2] * r[0] + J[0][2] * r[2]); // zz
        h_compact[3]  = n_inv_neg * (J[0][0] * r[1] + J[0][1] * r[0] + J[0][1] * r[0]); // xy
        h_compact[4]  = n_inv_neg * (J[0][0] * r[2] + J[0][2] * r[0] + J[0][2] * r[0]); // xz
        h_compact[5]  = n_inv_neg * (J[0][1] * r[2] + J[0][2] * r[1] + J[1][2] * r[0]); // yz

        // Hessian for r_k.y (6 unique terms)
        h_compact[6]  = n_inv_neg * (J[1][0] * r[0] + J[0][0] * r[1] + J[1][0] * r[0]); // xx
        h_compact[7]  = n_inv_neg * (3 * J[1][1] * r[1]);                            // yy
        h_compact[8]  = n_inv_neg * (J[1][2] * r[2] + J[2][2] * r[1] + J[1][2] * r[2]); // zz
        h_compact[9]  = n_inv_neg * (J[1][0] * r[1] + J[1][1] * r[0] + J[0][1] * r[1]); // xy
        h_compact[10] = n_inv_neg * (J[1][0] * r[2] + J[1][2] * r[0] + J[0][2] * r[1]); // xz
        h_compact[11] = n_inv_neg * (J[1][1] * r[2] + J[1][2] * r[1] + J[2][2] * r[1]); // yz

        // Hessian for r_k.z (6 unique terms)
        h_compact[12] = n_inv_neg * (J[2][0] * r[0] + J[0][0] * r[2] + J[2][0] * r[0]); // xx
        h_compact[13] = n_inv_neg * (J[2][1] * r[1] + J[1][1] * r[2] + J[2][1] * r[1]); // yy
        h_compact[14] = n_inv_neg * (3 * J[2][2] * r[2]);                            // zz
        h_compact[15] = n_inv_neg * (J[2][0] * r[1] + J[2][1] * r[0] + J[0][1] * r[2]); // xy
        h_compact[16] = n_inv_neg * (J[2][0] * r[2] + J[2][2] * r[0] + J[0][2] * r[2]); // xz
        h_compact[17] = n_inv_neg * (J[2][1] * r[2] + J[2][2] * r[1] + J[1][2] * r[2]); // yz
    }

    // --- 3. Store Results ---
    dr_dp[idx] = dr_dp_local;

    // Store the 18 compact Hessian terms
    // Layout: [H(r_x)_compact, H(r_y)_compact, H(r_z)_compact]
    float* d2r_dp2_ptr = d2r_dp2_compact + idx * 18;
    #pragma unroll
    for(int i = 0; i < 18; ++i) {
        d2r_dp2_ptr[i] = h_compact[i];
    }

    // transform Gaussian center to camera space
    vec3 mean_c;

    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }


    mat3 covar_c;
    covarW2C(R, covar, covar_c);
    // storing all projection derivatives
    if (camera_model == CameraModelType::PINHOLE) {
        // --- Common variables ---
        const float x = mean_c.x, y = mean_c.y, z = mean_c.z;
        const float fx = Ks[0], fy = Ks[4];
        const float rz = 1.f / z, rz2 = rz * rz, rz3 = rz2 * rz;

        // --- 1. Jacobian of the mean (∂π/∂p) ---
        mat3x2 J(
            fx * rz, 0.f,
            0.f, fy * rz,
            -fx * x * rz2, -fy * y * rz2
        );
        jacobians[idx] = J;

        // --- 2. Hessian of the mean (∂²π/∂p²) ---
        mat3 h_mux(0.f);
        h_mux[0][2] = h_mux[2][0] = -fx * rz2;
        h_mux[2][2] = 2.f * fx * x * rz3;
        H_mean_x[idx] = h_mux;

        mat3 h_muy(0.f);
        h_muy[1][2] = h_muy[2][1] = -fy * rz2;
        h_muy[2][2] = 2.f * fy * y * rz3;
        H_mean_y[idx] = h_muy;

        // --- 3. Derivative of the covariance (∂Σ/∂p) ---
        const float s_xx = covar_c[0][0], s_xy = covar_c[0][1], s_xz = covar_c[0][2];
        const float s_yy = covar_c[1][1], s_yz = covar_c[1][2], s_zz = covar_c[2][2];

        // ∂Σ/∂x
        dSigma_dx[idx] = (fx * rz2) * mat2(-2.f * s_xz, -s_yz, -s_yz, 0.f);

        // ∂Σ/∂y
        dSigma_dy[idx] = (fy * rz2) * mat2(0.f, -s_xz, -s_xz, -2.f * s_yz);

        // ∂Σ/∂z
        float A = fx * (x*s_xz + y*s_yz + z*s_zz);
        float B = fy * (x*s_xy + y*s_yy + z*s_yz);
        mat2 dS_dz;
        dS_dz[0][0] = 2.f * (fx * s_xz + A * fx * x * rz2);
        dS_dz[1][1] = 2.f * (fy * s_yz + B * fy * y * rz2);
        dS_dz[0][1] = dS_dz[1][0] = (fx*s_yz + fy*s_xz + A*fx*y*rz2 + B*fy*x*rz2);
        dSigma_dz[idx] = -rz2 * dS_dz;

        // --- Compute H_xz and H_yz (these were already correct) ---
        mat2 H_S_xz = (-2.f * fx * rz3) * mat2(-2.f * s_xz, -s_yz, -s_yz, 0.f);
        mat2 H_S_yz = (-2.f * fy * rz3) * mat2(0.f, -s_xz, -s_xz, -2.f * s_yz);
        // --- Compute H_zz (this was already correct) ---
        mat2 dS_dz_dz;
        float A_dz = fx * s_zz;
        float B_dz = fy * s_yz;
        float d_rz2_dz = -2.f * rz3;
        dS_dz_dz[0][0] = 2.f * (A_dz * fx * x * rz2 + A * fx * x * d_rz2_dz);
        dS_dz_dz[1][1] = 2.f * (B_dz * fy * y * rz2 + B * fy * y * d_rz2_dz);
        dS_dz_dz[0][1] = dS_dz_dz[1][0] = (A_dz * fx * y * rz2 + A * fx * y * d_rz2_dz) +
                                          (B_dz * fy * x * rz2 + B * fy * x * d_rz2_dz);
        mat2 H_S_zz = -d_rz2_dz * dS_dz - rz2 * dS_dz_dz;


        // --- Store the 5 non-zero matrices in the compact output buffer ---
        // The consumer of this buffer must know this specific order.
        mat2* H_Sigma_ptr = H_Sigma + idx * 3;
        H_Sigma_ptr[0] = H_S_xz;
        H_Sigma_ptr[1] = H_S_yz;
        //H_Sigma_ptr[2] = H_S_zx; //  via symmetry
        //H_Sigma_ptr[3] = H_S_zy; //  via symmetry
        H_Sigma_ptr[3] = H_S_zz;
    }
    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[gid];
        if (compensations != nullptr) {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

void launch_projection_ewa_3dgs_fused_fwd_kernel_LN(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [C, N, 2]
    at::Tensor means2d,                    // [C, N, 2]
    at::Tensor depths,                     // [C, N]
    at::Tensor conics,                     // [C, N, 3]
    at::optional<at::Tensor> compensations // [C, N] optional
) {
    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_fwd_kernel",
        [&]() {
            projection_ewa_3dgs_fused_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                      : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                       : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>()
                                         : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr
                );
        }
    );
}



#include <cmath>

__device__ void compute_covariance_derivatives(
    // Input parameters
    const float* quat,        // [w, x, y, z] current rotation
    const float* scale,       // [sx, sy, sz] scale parameters
    const float* view_matrix, // 4x4 view matrix (camera to world)
    const float* proj_matrix, // 3x4 projection matrix (camera to screen)
    const float* position,    // [x, y, z] world position
    // Output arrays
    float* dSigma_dtheta,    // 2x2 matrix (row-major)
    float* d2Sigma_dtheta2   // 2x2 matrix (row-major)
) {
    // 1. Compute 3D covariance in world space
    float R[3][3];
    float qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
    R[0][0] = 1 - 2*qy*qy - 2*qz*qz;
    R[0][1] = 2*qx*qy - 2*qz*qw;
    R[0][2] = 2*qx*qz + 2*qy*qw;
    R[1][0] = 2*qx*qy + 2*qz*qw;
    R[1][1] = 1 - 2*qx*qx - 2*qz*qz;
    R[1][2] = 2*qy*qz - 2*qx*qw;
    R[2][0] = 2*qx*qz - 2*qy*qw;
    R[2][1] = 2*qy*qz + 2*qx*qw;
    R[2][2] = 1 - 2*qx*qx - 2*qy*qy;

    float S[3][3] = {{scale[0]*scale[0], 0, 0},
                     {0, scale[1]*scale[1], 0},
                     {0, 0, scale[2]*scale[2]}};

    // Covariance = R*S*R�
    float RS[3][3], cov_world[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            RS[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                RS[i][j] += R[i][k] * S[k][j];
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov_world[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                cov_world[i][j] += RS[i][k] * R[j][k];
            }
        }
    }

    // 2. Transform position to camera space
    float pos_cam[3];
    float pos_homog[4] = {position[0], position[1], position[2], 1.0f};
    for (int i = 0; i < 3; ++i) {
        pos_cam[i] = 0.0f;
        for (int j = 0; j < 4; ++j) {
            pos_cam[i] += view_matrix[i*4+j] * pos_homog[j];
        }
    }
    float z = pos_cam[2];

    // 3. Compute Jacobian J of projective transform
    float J[2][3] = {
        {1/z, 0, -pos_cam[0]/(z*z)},
        {0, 1/z, -pos_cam[1]/(z*z)}
    };

    // 4. Compute 3D covariance in camera space (M0)
    float cov_cam[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov_cam[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    cov_cam[i][j] += view_matrix[i*4+k] * cov_world[k][l] * view_matrix[j*4+l];
                }
            }
        }
    }

    // 5. Precompute derivative matrices (at θ=0)
    float dR0[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 0}};
    float dR02[3][3] = {{-1, 0, 0}, {0, -1, 0}, {0, 0, 0}};

    // 6. Compute dM/dθ = dR0*M0 + M0*dR0ᵀ
    float dM_dtheta[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            dM_dtheta[i][j] = 0.0f;
            // dR0*M0
            for (int k = 0; k < 3; ++k) {
                dM_dtheta[i][j] += dR0[i][k] * cov_cam[k][j];
            }
            // M0*dR0ᵀ
            for (int k = 0; k < 3; ++k) {
                dM_dtheta[i][j] += cov_cam[i][k] * dR0[j][k];
            }
        }
    }

    // 7. Compute d²M/dθ² = dR02*M0 + M0*dR02ᵀ + 2*(dR0*M0*dR0ᵀ)
    float d2M_dtheta2[3][3];
    // dR02*M0 + M0*dR02ᵀ
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            d2M_dtheta2[i][j] = 0.0f;
            // dR02*M0
            for (int k = 0; k < 3; ++k) {
                d2M_dtheta2[i][j] += dR02[i][k] * cov_cam[k][j];
            }
            // M0*dR02ᵀ
            for (int k = 0; k < 3; ++k) {
                d2M_dtheta2[i][j] += cov_cam[i][k] * dR02[j][k];
            }
        }
    }
    // + 2*(dR0*M0*dR0ᵀ)
    float temp[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                temp[i][j] += dR0[i][k] * cov_cam[k][j];
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float val = 0.0f;
            for (int k = 0; k < 3; ++k) {
                val += temp[i][k] * dR0[j][k];
            }
            d2M_dtheta2[i][j] += 2.0f * val;
        }
    }

    // 8. Project to 2D: Σ = J * M * Jᵀ
    float JdM[2][3], Jd2M[2][3];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            JdM[i][j] = 0.0f;
            Jd2M[i][j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                JdM[i][j] += J[i][k] * dM_dtheta[k][j];
                Jd2M[i][j] += J[i][k] * d2M_dtheta2[k][j];
            }
        }
    }

    // 9. Final 2D covariance derivatives
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            dSigma_dtheta[i*2+j] = 0.0f;
            d2Sigma_dtheta2[i*2+j] = 0.0f;
            for (int k = 0; k < 3; ++k) {
                dSigma_dtheta[i*2+j] += JdM[i][k] * J[j][k];
                d2Sigma_dtheta2[i*2+j] += Jd2M[i][k] * J[j][k];
            }
        }
    }
}