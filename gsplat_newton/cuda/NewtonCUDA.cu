#include "gsplat_newton/Newton.h"
#include "Common.h"
#include "Utils.cuh" // For matrix operations
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

using namespace gsplat;

namespace gsplat_newton {


// -----------------------------------------------------------------------------
// Constants for regularisation ------------------------------------------------
// -----------------------------------------------------------------------------
__constant__ float kMatInvEps = 1e-6f;
__constant__ float kLambdaOpacity = 1e-3f;   // smooth‑L1 λ
__constant__ float kL1Eps        = 1e-4f;    // smooth corner ε

// -----------------------------------------------------------------------------
// Small helpers ---------------------------------------------------------------
// -----------------------------------------------------------------------------

__device__ __forceinline__ vec3 mul_mat3x2_vec2(const mat3x2 &A, const vec2 &v)
{
    // A is column‑major (3 columns, 2 rows)
    return vec3(
        A[0][0] * v.x + A[0][1] * v.y,
        A[1][0] * v.x + A[1][1] * v.y,
        A[2][0] * v.x + A[2][1] * v.y);
}

__device__ __forceinline__ mat3 mul_mat3x2_mat2x3(const mat3x2 &A, const mat2x3 &B)
{
    // Compute C = A * B  (3×2)·(2×3) = (3×3)
    mat3 C(0.0f);
    #pragma unroll
    for (int c = 0; c < 3; ++c) {         // column of C
        #pragma unroll
        for (int r = 0; r < 3; ++r) {     // row   of C
            C[c][r] = A[c][0] * B[0][r]   // col0·row0
                      + A[c][1] * B[1][r];// col1·row1
        }
    }
    return C;
}

__device__ __forceinline__ vec3 mul_mat3_vec3(const mat3 &M, const vec3 &v)
{
    // Column‑major convention: M * v = Σ (col_i * v_i)
    return v.x * vec3(M[0]) + v.y * vec3(M[1]) + v.z * vec3(M[2]);
}

// Quaternion multiply written out to avoid <glm/gtx/quaternion.hpp> ambiguity
__device__ __forceinline__ vec4 quat_mul(const vec4 &q, const vec4 &p)
{
    // components are (x y z w)
    return vec4(
        q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y,
        q.w * p.y - q.x * p.z + q.y * p.w + q.z * p.x,
        q.w * p.z + q.x * p.y - q.y * p.x + q.z * p.w,
        q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z);
}


// This single kernel handles solving and updating for all attributes for one Gaussian.
// Each thread processes one Gaussian.
__global__ void solve_updates_and_backproject_kernel_impl(
    const uint32_t N,
    const int K,
    // Gradients and Hessians
    const float* __restrict__ dL_d_pos,      // [N, 2]
    const float* __restrict__ H_L_pos,       // [N, 2, 2]
    const float* __restrict__ dL_d_scale,    // [N, 2]
    const float* __restrict__ H_L_scale,     // [N, 2, 2]
    const float* __restrict__ dL_d_rot,      // [N]
    const float* __restrict__ H_L_rot,       // [N]
    const float* __restrict__ dL_d_opacity,  // [N]
    const float* __restrict__ H_L_opacity,   // [N]
    const float* __restrict__ dL_d_color,    // [N, 3]
    const float* __restrict__ H_L_color,     // [N, 3, 3]
    // Context tensors
    const float* __restrict__ U_k_bases,     // [N, 2, 3]
    const float* __restrict__ T_k_matrices,  // [N, 2, 3]
    const float* __restrict__ view_dirs,     // [N, 3]
    // Params (in‑place)
    float* __restrict__ means,               // [N, 3]
    float* __restrict__ scales,              // [N, 3]
    float* __restrict__ quats,               // [N, 4]  (x y z w)
    float* __restrict__ opacities,           // [N]
    float* __restrict__ sh_coeffs            // [N, K, 3]
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // ---------------------------------------------------------------------
    // Position update (Δp_k = U^T Δv_k) -----------------------------------
    // ---------------------------------------------------------------------
    mat2  H_pos = glm::make_mat2(H_L_pos  + gid * 4);
    vec2  g_pos = glm::make_vec2(dL_d_pos + gid * 2);
    vec2  delta_vk = -glm::inverse(H_pos + mat2(kMatInvEps)) * g_pos;

    mat2x3 U_k  = glm::make_mat2x3(U_k_bases + gid * 6);
    mat3x2 U_k_T = glm::transpose(U_k);
    vec3  delta_p = mul_mat3x2_vec2(U_k_T, delta_vk);

    means[gid * 3 + 0] += delta_p.x;
    means[gid * 3 + 1] += delta_p.y;
    means[gid * 3 + 2] += delta_p.z;

    // ---------------------------------------------------------------------
    // Scale update  (Δs_k = (T^T T)^{-1} T^T Δλ) ---------------------------
    // ---------------------------------------------------------------------
    mat2  H_scale = glm::make_mat2(H_L_scale  + gid * 4);
    vec2  g_scale = glm::make_vec2(dL_d_scale + gid * 2);
    vec2  delta_lambda = -glm::inverse(H_scale + mat2(kMatInvEps)) * g_scale;

    mat2x3 T_k   = glm::make_mat2x3(T_k_matrices + gid * 6);
    mat3x2 T_k_T = glm::transpose(T_k);

    // (3×2)*(2×3) manual
    mat3 TtT = mul_mat3x2_mat2x3(T_k_T, T_k);
    mat3 TtT_inv = glm::inverse(TtT + mat3(kMatInvEps));

    vec3 temp_vec = mul_mat3x2_vec2(T_k_T, delta_lambda);
    vec3 delta_s  = mul_mat3_vec3(TtT_inv, temp_vec);

    scales[gid * 3 + 0] = fmaxf(1e-4f, scales[gid * 3 + 0] + delta_s.x);
    scales[gid * 3 + 1] = fmaxf(1e-4f, scales[gid * 3 + 1] + delta_s.y);
    scales[gid * 3 + 2] = fmaxf(1e-4f, scales[gid * 3 + 2] + delta_s.z);

    // ---------------------------------------------------------------------
    // Rotation update ------------------------------------------------------
    // ---------------------------------------------------------------------
    float H_rot = H_L_rot[gid];
    if (fabsf(H_rot) > 1e-6f) {
        float g_rot      = dL_d_rot[gid];
        float delta_theta = -g_rot / H_rot;  // small angle

        // axis = view_dir (already world‑space unit)
        vec3 axis = glm::normalize(glm::make_vec3(view_dirs + gid * 3));
        float half = 0.5f * delta_theta;
        float s = sinf(half);
        vec4 dq(axis.x * s, axis.y * s, axis.z * s, cosf(half));

        vec4 q = glm::make_vec4(quats + gid * 4); // (x y z w)
        vec4 q_new = glm::normalize(quat_mul(dq, q));
        quats[gid * 4 + 0] = q_new.x;
        quats[gid * 4 + 1] = q_new.y;
        quats[gid * 4 + 2] = q_new.z;
        quats[gid * 4 + 3] = q_new.w;
    }

    // ---------------------------------------------------------------------
    // Opacity update  (barrier + smooth‑L1) --------------------------------
    // ---------------------------------------------------------------------
    float H_opac = H_L_opacity[gid];
    float sigma  = opacities[gid];

    // Barrier gradient & Hessian
    float g_bar  = -(1.f / sigma) + (1.f / (1.f - sigma));
    float H_bar  =  1.f / (sigma * sigma) + 1.f / ((1.f - sigma) * (1.f - sigma));

    // Smooth‑L1 ≈ |σ| with C2 continuity
    float abs_sig = fabsf(sigma);
    float g_l1    = kLambdaOpacity * (abs_sig > kL1Eps ? copysignf(1.f, sigma)
                                                       : sigma / kL1Eps);
    float H_l1    = kLambdaOpacity * (abs_sig > kL1Eps ? 0.f : 1.f / kL1Eps);

    float g_opac  = dL_d_opacity[gid] + g_bar + g_l1;
    float H_total = H_opac + H_bar + H_l1;

    if (fabsf(H_total) > 1e-6f) {
        float delta_sigma = -g_opac / H_total;
        float new_sig = fminf(1.f - 1e-6f, fmaxf(1e-6f, sigma + delta_sigma));
        opacities[gid] = new_sig;
    }

    // ---------------------------------------------------------------------
    // Colour (SH DC coefficient only, diagonal Hessian) --------------------
    // ---------------------------------------------------------------------
    const float* Hc_ptr = H_L_color + gid * 9;  // 3×3 but we use diag only
    const float* gc_ptr = dL_d_color + gid * 3;
    for (int ch = 0; ch < 3; ++ch) {
        float g = gc_ptr[ch];
        float h = Hc_ptr[ch * 4] + kMatInvEps; // diag indices 0,4,8
        float delta = -g / h;
        sh_coeffs[(gid * K + 0) * 3 + ch] += delta; // DC component (band 0)
    }

    // Higher SH coefficients (band 1..K‑1) --------------------------------
    // Using diagonal Hessian approximation as per your simplification.
    if (K > 1) {
        for (int k = 1; k < K; ++k) {
            for (int ch = 0; ch < 3; ++ch) {
                // grad & Hessian pointers advance by K*3; assume external fill
                int idx = (gid * K + k) * 3 + ch;
                float g = dL_d_color[idx];             // provided grad
                float h = H_L_color[idx * 3 + ch] + kMatInvEps; // diag only
                sh_coeffs[idx] += -g / h;
            }
        }
    }
}

void launch_solve_and_update_all_attributes_kernel(
    const at::Tensor dL_d_pos, const at::Tensor H_L_pos,
    const at::Tensor dL_d_scale, const at::Tensor H_L_scale,
    const at::Tensor dL_d_rot, const at::Tensor H_L_rot,
    const at::Tensor dL_d_opacity, const at::Tensor H_L_opacity,
    const at::Tensor dL_d_color, const at::Tensor H_L_color,
    const at::Tensor U_k_bases, const at::Tensor T_k_matrices,
    const at::Tensor view_dirs,
    at::Tensor means, at::Tensor scales, at::Tensor quats,
    at::Tensor opacities, at::Tensor sh_coeffs
) {
    const uint32_t N = means.size(0);
    if (N == 0) return;
    const int K = sh_coeffs.size(1);
    const dim3 threads(256);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    solve_updates_and_backproject_kernel_impl<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        K,
        dL_d_pos.data_ptr<float>(), H_L_pos.data_ptr<float>(),
        dL_d_scale.data_ptr<float>(), H_L_scale.data_ptr<float>(),
        dL_d_rot.data_ptr<float>(), H_L_rot.data_ptr<float>(),
        dL_d_opacity.data_ptr<float>(), H_L_opacity.data_ptr<float>(),
        dL_d_color.data_ptr<float>(), H_L_color.data_ptr<float>(),
        U_k_bases.data_ptr<float>(), T_k_matrices.data_ptr<float>(),
        view_dirs.data_ptr<float>(),
        means.data_ptr<float>(), scales.data_ptr<float>(), quats.data_ptr<float>(),
        opacities.data_ptr<float>(), sh_coeffs.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat_newton