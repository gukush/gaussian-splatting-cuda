#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <glm/glm.hpp>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

using namespace gsplat;

namespace gsplat_newton {

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
    scalar_t *__restrict__ compensations, // [C, N] optional
    //outputs for Local Newton
    mat3x2 *__restrict__ jacobians, // [C, N, 3, 2] dmean2d/dmean3d
    mat3 *__restrict__ H_mean_y, // elements of Hessian d2mux/dmean3d^2 [C, N, 3, 3]
    mat3 *__restrict__ H_mean_x, // elements of hessian d2muy/dmean3d^2 [C, N, 3, 3]
    mat2 *__restrict__ dSigma_dx,        // [C, N, 2, 2] ∂Σ/∂x
    mat2 *__restrict__ dSigma_dy,        // [C, N, 2, 2] ∂Σ/∂y
    mat2 *__restrict__ dSigma_dz,         // [C, N, 2, 2] ∂Σ/∂z
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
    mat3 R_c2w = glm::transpose(R);
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
    at::optional<at::Tensor> compensations, // [C, N] optional
    // outputs for Local Newton
    at::Tensor jacobians,                  // [C, N, 3, 2]
    at::Tensor H_mean_y,                   // [C, N, 3, 3]
    at::Tensor H_mean_x,                   // [C, N, 3, 3]
    at::Tensor dSigma_dx,                  // [C, N, 2, 2]
    at::Tensor dSigma_dy,                  // [C, N, 2, 2]
    at::Tensor dSigma_dz,                  // [C, N, 2, 2]
    at::Tensor H_Sigma,                    // [C, N, 3] (stores [H_S_xz, H_S_yz, H_S_zz])
    at::Tensor dr_dp,                      // [C, N, 3, 3]
    at::Tensor d2r_dp2_compact             // [C, N, 18]
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
        "projection_ewa_3dgs_fused_fwd_kernel_LN",
        [&]() {
            projection_ewa_3dgs_fused_fwd_kernel_LN<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>() : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>() : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>() : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>() : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    // outputs
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value() ? compensations.value().data_ptr<scalar_t>() : nullptr,
                    // outputs for Local Newton
                    reinterpret_cast<mat3x2*>(jacobians.data_ptr<scalar_t>()),
                    reinterpret_cast<mat3*>(H_mean_y.data_ptr<scalar_t>()),
                    reinterpret_cast<mat3*>(H_mean_x.data_ptr<scalar_t>()),
                    reinterpret_cast<mat2*>(dSigma_dx.data_ptr<scalar_t>()),
                    reinterpret_cast<mat2*>(dSigma_dy.data_ptr<scalar_t>()),
                    reinterpret_cast<mat2*>(dSigma_dz.data_ptr<scalar_t>()),
                    reinterpret_cast<mat2*>(H_Sigma.data_ptr<scalar_t>()),
                    reinterpret_cast<mat3*>(dr_dp.data_ptr<scalar_t>()),
                    d2r_dp2_compact.data_ptr<float>()
                );
        }
    );
}



#include <cmath>
////////////////////////////////////////////////////////////////////////////////
// General‑axis version of compute_covariance_derivatives_kernel_impl
////////////////////////////////////////////////////////////////////////////////
__global__ void compute_covariance_derivatives_kernel_impl(
    const uint32_t  N,
    const float* __restrict__ quat,        // [N,4]
    const float* __restrict__ scale,       // [N,3]
    const float* __restrict__ view_matrix, // [4,4]  row‑major
    const float* __restrict__ position,    // [N,3]
    float* __restrict__ dSigma_dtheta,     // [N,2,2]
    float* __restrict__ d2Sigma_dtheta2)   // [N,2,2]
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    //----------------------------------------------------------------------
    // 0.  Pointers to this Gaussian’s data
    //----------------------------------------------------------------------
    const float* q  = quat     + gid*4;
    const float* s  = scale    + gid*3;
    const float* pW = position + gid*3;    // world

    float* dS  = dSigma_dtheta    + gid*4; // 2×2 = 4
    float* dS2 = d2Sigma_dtheta2  + gid*4;

    //----------------------------------------------------------------------
    // 1.  Covariance in world space :  M_W = R  S  Rᵀ
    //----------------------------------------------------------------------
    float R[3][3];
    {
        const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
        R[0][0] = 1.f - 2.f*(qy*qy + qz*qz);
        R[0][1] = 2.f*(qx*qy - qz*qw);
        R[0][2] = 2.f*(qx*qz + qy*qw);
        R[1][0] = 2.f*(qx*qy + qz*qw);
        R[1][1] = 1.f - 2.f*(qx*qx + qz*qz);
        R[1][2] = 2.f*(qy*qz - qx*qw);
        R[2][0] = 2.f*(qx*qz - qy*qw);
        R[2][1] = 2.f*(qy*qz + qx*qw);
        R[2][2] = 1.f - 2.f*(qx*qx + qy*qy);
    }
    const float S[3][3] = { {s[0]*s[0], 0.f, 0.f},
                            {0.f, s[1]*s[1], 0.f},
                            {0.f, 0.f, s[2]*s[2]} };

    float RS[3][3], M_W[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += R[i][k]*S[k][j];
            RS[i][j]=v;
        }
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += RS[i][k]*R[j][k];
            M_W[i][j]=v;
        }

    //----------------------------------------------------------------------
    // 2.  World position -> camera space  &  view direction r_k
    //----------------------------------------------------------------------
    float pC[3];                     // camera‑space position
    {
        const float ph[4] = {pW[0], pW[1], pW[2], 1.f};
        #pragma unroll
        for (int i=0;i<3;++i)
        {
            float v=0.f;
            #pragma unroll
            for (int j=0;j<4;++j) v += view_matrix[i*4+j]*ph[j];
            pC[i]=v;
        }
    }
    const float z      = pC[2];
    const float invL   = rsqrtf(pC[0]*pC[0] + pC[1]*pC[1] + pC[2]*pC[2] + 1e-20f); // |pC|⁻¹
    const float rx     = pC[0]*invL;
    const float ry     = pC[1]*invL;
    const float rz     = pC[2]*invL;

    //----------------------------------------------------------------------
    // 3.  [r]ₓ   and   [r]ₓ²  (skew‑symm matrix and its square)
    //----------------------------------------------------------------------
    float rX[3][3]  = { {   0.f, -rz ,  ry },
                        {  rz  ,  0.f, -rx },
                        { -ry  ,  rx ,  0.f} };

    float rX2[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += rX[i][k]*rX[k][j];
            rX2[i][j]=v;
        }

    //----------------------------------------------------------------------
    // 4.  Covariance in camera space :  M0 = R_cam M_W R_camᵀ
    //----------------------------------------------------------------------
    float M0[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)
            {
                float t=0.f;
                #pragma unroll
                for (int l=0;l<3;++l)
                    t += view_matrix[i*4+l]*M_W[l][k];
                v += t*view_matrix[j*4+k];
            }
            M0[i][j]=v;
        }

    //----------------------------------------------------------------------
    // 5.  First & second derivative of  M(θ)  at θ=0
    //----------------------------------------------------------------------
    float dM[3][3], d2M[3][3];

    // dM = rX*M0 + M0*rXᵀ
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float a=0.f, b=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)  a += rX[i][k]*M0[k][j];
            #pragma unroll
            for (int k=0;k<3;++k)  b += M0[i][k]*rX[j][k];   // rXᵀ
            dM[i][j] = a + b;
        }

    // temp = rX*M0
    float temp[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += rX[i][k]*M0[k][j];
            temp[i][j]=v;
        }

    // d2M = rX2*M0 + M0*rX2ᵀ + 2*temp*rXᵀ
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v1=0.f, v2=0.f, v3=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)  v1 += rX2[i][k]*M0[k][j];
            #pragma unroll
            for (int k=0;k<3;++k)  v2 += M0[i][k]*rX2[j][k];
            #pragma unroll
            for (int k=0;k<3;++k)  v3 += temp[i][k]*rX[j][k];
            d2M[i][j] = v1 + v2 + 2.f*v3;
        }

    //----------------------------------------------------------------------
    // 6.  Jacobian of the perspective projection  J
    //----------------------------------------------------------------------
    const float invZ  = 1.f / z;
    const float invZ2 = invZ * invZ;
    const float J[2][3] = { { invZ, 0.f  , -pC[0]*invZ2 },
                            { 0.f , invZ , -pC[1]*invZ2 } };

    //----------------------------------------------------------------------
    // 7.  Project derivatives :  Σ = J M Jᵀ
    //----------------------------------------------------------------------
    float JdM [2][3], Jd2M[2][3];
    #pragma unroll
    for (int i=0;i<2;++i)
        #pragma unroll
        for (int k=0;k<3;++k)
        {
            float a=0.f, b=0.f;
            #pragma unroll
            for (int j=0;j<3;++j)
            {
                a += J[i][j]*dM [j][k];
                b += J[i][j]*d2M[j][k];
            }
            JdM [i][k]=a;
            Jd2M[i][k]=b;
        }

    //----------------------------------------------------------------------
    // 8.  Final 2×2 blocks (row‑major)
    //----------------------------------------------------------------------
    #pragma unroll
    for (int i=0;i<2;++i)
        #pragma unroll
        for (int j=0;j<2;++j)
        {
            float s1=0.f, s2=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)
            {
                s1 += JdM [i][k]*J[j][k];
                s2 += Jd2M[i][k]*J[j][k];
            }
            dS [i*2+j] = s1;
            dS2[i*2+j] = s2;
        }
}


// Launcher function following the same pattern as your existing code
void launch_compute_covariance_derivatives_kernel(
    const at::Tensor quat,        // [N, 4] quaternions
    const at::Tensor scale,       // [N, 3] scales
    const at::Tensor view_matrix, // [4, 4] view matrix
    const at::Tensor position,    // [N, 3] positions
    at::Tensor dSigma_dtheta,     // [N, 2, 2] output first derivatives
    at::Tensor d2Sigma_dtheta2    // [N, 2, 2] output second derivatives
) {
    const uint32_t N = position.size(0);
    if (N == 0) return;

    const dim3 threads(256);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    compute_covariance_derivatives_kernel_impl<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        quat.data_ptr<float>(),
        scale.data_ptr<float>(),
        view_matrix.data_ptr<float>(),
        position.data_ptr<float>(),
        dSigma_dtheta.data_ptr<float>(),
        d2Sigma_dtheta2.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat_newton