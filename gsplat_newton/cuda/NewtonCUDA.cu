#include "Newton.h"
#include "Common.h"
#include "Utils.cuh" // For matrix operations
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

namespace gsplat {

// This single kernel handles solving and updating for all attributes for one Gaussian.
// Each thread processes one Gaussian.
__global__ void solve_updates_and_backproject_kernel_impl(
    const uint32_t N,
    // Gradients and Hessians for each attribute
    const float* __restrict__ dL_d_pos,      // [N, 2] (grad for v_k)
    const float* __restrict__ H_L_pos,       // [N, 2, 2] (Hessian for v_k)
    const float* __restrict__ dL_d_scale,    // [N, 2] (grad for lambda)
    const float* __restrict__ H_L_scale,     // [N, 2, 2] (Hessian for lambda)
    const float* __restrict__ dL_d_rot,      // [N] (grad for theta)
    const float* __restrict__ H_L_rot,       // [N] (Hessian for theta)
    const float* __restrict__ dL_d_opacity,  // [N]
    const float* __restrict__ H_L_opacity,   // [N]
    const float* __restrict__ dL_d_color,    // [N, 3]
    const float* __restrict__ H_L_color,     // [N, 3, 3]
    // Tensors from context needed for backprojection
    const float* __restrict__ U_k_bases,     // [N, 2, 3] (Jacobian ∂π/∂p, whose columns form the basis U_k)
    const float* __restrict__ T_k_matrices,  // [N, 2, 3]
    // Gaussian parameters to be updated (in-place)
    float* __restrict__ means,               // [N, 3]
    float* __restrict__ scales,              // [N, 3]
    float* __restrict__ quats,               // [N, 4]
    float* __restrict__ opacities,           // [N, 1]
    float* __restrict__ sh_coeffs            // [N, K, 3]
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    // --- Position Update ---
    mat2 H_pos = glm::make_mat2(H_L_pos + gid * 4);
    vec2 grad_pos = glm::make_vec2(dL_d_pos + gid * 2);
    vec2 delta_vk = -glm::inverse(H_pos + mat2(1e-6)) * grad_pos; // Add regularization
    // Backproject: Δp_k = U_k * Δv_k
    // U_k is formed by the columns of the jacobian ∂π/∂p
    mat2x3 U_k = glm::make_mat2x3(U_k_bases + gid * 6);
    vec3 delta_p = glm::transpose(U_k) * delta_vk;
    atomicAdd(&means[gid * 3 + 0], delta_p.x);
    atomicAdd(&means[gid * 3 + 1], delta_p.y);
    atomicAdd(&means[gid * 3 + 2], delta_p.z);

    // --- Scaling Update ---
    mat2 H_scale = glm::make_mat2(H_L_scale + gid * 4);
    vec2 grad_scale = glm::make_vec2(dL_d_scale + gid * 2);
    vec2 delta_lambda = -glm::inverse(H_scale + mat2(1e-6)) * grad_scale;
    // Backproject: Δs_k = (T_k^T T_k)^-1 T_k^T * Δλ_k
    mat2x3 T_k = glm::make_mat2x3(T_k_matrices + gid * 6);
    mat3 T_k_T = glm::transpose(T_k);
    mat2 T_k_T_T_k = T_k * T_k_T;
    vec3 delta_s = T_k_T * glm::inverse(T_k_T_T_k + mat2(1e-6)) * delta_lambda;
    atomicAdd(&scales[gid * 3 + 0], delta_s.x);
    atomicAdd(&scales[gid * 3 + 1], delta_s.y);
    atomicAdd(&scales[gid * 3 + 2], delta_s.z);

    // --- Rotation Update ---
    float H_rot = H_L_rot[gid];
    if (abs(H_rot) > 1e-6) {
        float grad_rot = dL_d_rot[gid];
        float delta_theta = -grad_rot / H_rot;
        vec4 q_current = glm::make_vec4(quats + gid * 4);
        // Assume rotation axis is view vector (z-axis in camera space), needs to be passed for accuracy
        vec3 axis = vec3(0, 0, 1);
        float angle = delta_theta;
        vec4 delta_q = vec4(cos(angle/2.f), sin(angle/2.f) * axis.x, sin(angle/2.f) * axis.y, sin(angle/2.f) * axis.z);
        vec 4 q_new = delta_q * q_current;
        quats[gid * 4 + 0] = q_new.w;
        quats[gid * 4 + 1] = q_new.x;
        quats[gid * 4 + 2] = q_new.y;
        quats[gid * 4 + 3] = q_new.z;
    }

    // --- Opacity Update ---
    float H_opac = H_L_opacity[gid];
    float current_opac = opacities[gid];
    // Add barrier term derivatives
    float grad_opac = dL_d_opacity[gid] - (1.f / current_opac) + (1.f / (1.f - current_opac));
    H_opac += 1.f / (current_opac * current_opac) + 1.f / ((1.f - current_opac) * (1.f - current_opac));
    if (abs(H_opac) > 1e-6) {
        float delta_opac = -grad_opac / H_opac;
        opacities[gid] = clamp(current_opac + delta_opac, 1e-6f, 1.0f - 1e-6f);
    }

    // --- Color Update ---
    mat3 H_color = glm::make_mat3(H_L_color + gid * 9);
    vec3 grad_color = glm::make_vec3(dL_d_color + gid * 3);
    vec3 delta_color = -glm::inverse(H_color + mat3(1e-6)) * grad_color;
    // Update the DC component of the SH coefficients (assuming K>=1)
    atomicAdd(&sh_coeffs[gid * 16 * 3 + 0], delta_color.x);
    atomicAdd(&sh_coeffs[gid * 16 * 3 + 1], delta_color.y);
    atomicAdd(&sh_coeffs[gid * 16 * 3 + 2], delta_color.z);
}

void launch_solve_and_update_all_attributes_kernel(
    const at::Tensor& dL_d_pos, const at::Tensor& H_L_pos,
    const at::Tensor& dL_d_scale, const at::Tensor& H_L_scale,
    const at::Tensor& dL_d_rot, const at::Tensor& H_L_rot,
    const at::Tensor& dL_d_opacity, const at::Tensor& H_L_opacity,
    const at::Tensor& dL_d_color, const at::Tensor& H_L_color,
    const at::Tensor& U_k_bases, const at::Tensor& T_k_matrices,
    at::Tensor& means, at::Tensor& scales, at::Tensor& quats,
    at::Tensor& opacities, at::Tensor& sh_coeffs
) {
    const uint32_t N = means.size(0);
    if (N == 0) return;

    const dim3 threads(256);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    solve_updates_and_backproject_kernel_impl<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        dL_d_pos.data_ptr<float>(), H_L_pos.data_ptr<float>(),
        dL_d_scale.data_ptr<float>(), H_L_scale.data_ptr<float>(),
        dL_d_rot.data_ptr<float>(), H_L_rot.data_ptr<float>(),
        dL_d_opacity.data_ptr<float>(), H_L_opacity.data_ptr<float>(),
        dL_d_color.data_ptr<float>(), H_L_color.data_ptr<float>(),
        U_k_bases.data_ptr<float>(), T_k_matrices.data_ptr<float>(),
        means.data_ptr<float>(), scales.data_ptr<float>(), quats.data_ptr<float>(),
        opacities.data_ptr<float>(), sh_coeffs.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat