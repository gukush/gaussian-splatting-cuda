#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <glm/gtc/matrix_transform.hpp>
#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace cg = cooperative_groups;
using namespace gsplat;

/**
 * @brief Performs an analytical eigenvalue decomposition for a 2x2 symmetric matrix.
 *
 * @param a The (0,0) component of the matrix.
 * @param b The (0,1) and (1,0) component of the matrix.
 * @param c The (1,1) component of the matrix.
 * @param[out] lambda_min The smaller eigenvalue.
 * @param[out] lambda_max The larger eigenvalue.
 * @param[out] v_min The eigenvector corresponding to lambda_min.
 * @param[out] v_max The eigenvector corresponding to lambda_max.
 */
__device__ __forceinline__ void eigen_decomposition_2d(
    const float a, const float b, const float c,
    float& lambda_min, float& lambda_max,
    vec2& v_min, vec2& v_max
) {
    // Compute eigenvalues using the quadratic formula
    const float tr = a + c;
    const float det = a * c - b * b;
    const float discriminant = sqrtf(max(0.f, tr * tr - 4.f * det));

    lambda_max = (tr + discriminant) / 2.f;
    lambda_min = (tr - discriminant) / 2.f;

    // Compute eigenvectors. For a matrix [[a,b],[b,c]], an eigenvector for
    // an eigenvalue λ is [b, λ - a] or [λ - c, b].
    // We handle the case where b is close to zero to avoid instability.
    if (abs(b) > 1e-6) {
        v_max = {b, lambda_max - a};
        v_min = {b, lambda_min - a};
    } else {
        // Matrix is diagonal, eigenvectors are the axes
        v_max = {1.f, 0.f};
        v_min = {0.f, 1.f};
    }

    // Normalize the eigenvectors
    v_max = normalize(v_max);
    v_min = normalize(v_min);
}


// A 2x3 matrix, useful for the projection Jacobian
/*
struct mat2x3 {
    float data[2][3];
};

// A 3x2 matrix
struct mat3x2 {
    float data[3][2];
};
*/

/**
 * @brief Computes the first derivative ∂c/∂pₖ for a single pixel's contribution.
 * This function implements the first derivative part of Equation (16).
 *
 * @param dc_dc_sh Contribution from ∂c/∂c̃ₖ.
 * @param dc_dG Contribution from ∂c/∂Gₖ.
 * @param dG_dmean2d Local derivative ∂Gₖ/∂πₖ.
 * @param dG_dSigma Local derivative ∂Gₖ/∂Σₖ (as a conic vec3).
 * @param dc_sh_dp Assumed input ∂c̃ₖ/∂pₖ (3x3 matrix).
 * @param J Assumed input projection jacobian ∂πₖ/∂pₖ (2x3 matrix).
 * @param dSigma_dp Assumed input ∂Σₖ/∂pₖ (three 2x2 matrices).
 * @return vec3 The local contribution to ∂c/∂pₖ.
 */
__device__ __forceinline__ vec3
compute_dc_dpk_local(
    const float dc_dc_sh,
    const float dc_dG,
    const vec2& dG_dmean2d,
    const vec3& dG_dSigma, // {dG/dΣxx, dG/dΣxy, dG/dΣyy}
    const mat3& dc_sh_dp,   // ∂c̃ₖ/∂pₖ
    const mat2x3& J,        // ∂πₖ/∂pₖ
    const mat2 dSigma_dpx, const mat2 dSigma_dpy, const mat2 dSigma_dpz)
{
    vec3 grad = {0.f, 0.f, 0.f};

    // Term 1: (∂c/∂c̃ₖ) * (∂c̃ₖ/∂pₖ)
    // The original code sums the columns.
    // dc_sh_dp.data[0][0] + dc_sh_dp.data[1][0] + dc_sh_dp.data[2][0] is the sum of column 0.
    // In GLM, this is dc_sh_dp[0][0] + dc_sh_dp[0][1] + dc_sh_dp[0][2].
    grad.x += dc_dc_sh * (dc_sh_dp[0][0] + dc_sh_dp[0][1] + dc_sh_dp[0][2]); // Sum of column 0
    grad.y += dc_dc_sh * (dc_sh_dp[1][0] + dc_sh_dp[1][1] + dc_sh_dp[1][2]); // Sum of column 1
    grad.z += dc_dc_sh * (dc_sh_dp[2][0] + dc_sh_dp[2][1] + dc_sh_dp[2][2]); // Sum of column 2

    // Term 2: (∂c/∂Gₖ) * [ (∂Gₖ/∂πₖ) * (∂πₖ/∂pₖ) + (∂Gₖ/∂Σₖ) : (∂Σₖ/∂pₖ) ]
    if (abs(dc_dG) > 1e-7) {
        // (∂Gₖ/∂πₖ) * (∂πₖ/∂pₖ) -> vec2 * mat2x3 = vec3
        // J.data[row][col] becomes J[col][row]
        grad.x += dc_dG * (dG_dmean2d.x * J[0][0] + dG_dmean2d.y * J[0][1]);
        grad.y += dc_dG * (dG_dmean2d.x * J[1][0] + dG_dmean2d.y * J[1][1]);
        grad.z += dc_dG * (dG_dmean2d.x * J[2][0] + dG_dmean2d.y * J[2][1]);

        // (∂Gₖ/∂Σₖ) : (∂Σₖ/∂pₖ) -> Frobenius inner product
        // dSigma_dpx.data[row][col] becomes dSigma_dpx[col][row]
        float dG_S_dp_x = dG_dSigma.x * dSigma_dpx[0][0] + 2.f * dG_dSigma.y * dSigma_dpx[1][0] + dG_dSigma.z * dSigma_dpx[1][1];
        float dG_S_dp_y = dG_dSigma.x * dSigma_dpy[0][0] + 2.f * dG_dSigma.y * dSigma_dpy[1][0] + dG_dSigma.z * dSigma_dpy[1][1];
        float dG_S_dp_z = dG_dSigma.x * dSigma_dpz[0][0] + 2.f * dG_dSigma.y * dSigma_dpz[1][0] + dG_dSigma.z * dSigma_dpz[1][1];

        grad.x += dc_dG * dG_S_dp_x;
        grad.y += dc_dG * dG_S_dp_y;
        grad.z += dc_dG * dG_S_dp_z;
    }
    return grad;
}

/**
 * @brief Computes the second derivative ∂²c/∂pₖ² for a single pixel's contribution.
 * This function implements the complex second derivative part of Equation (16).
 *
 * All parameters are local (per-pixel) contributions or assumed (per-Gaussian) inputs.
 * @return mat3 The local contribution to ∂²c/∂pₖ².
 */
__device__ __forceinline__ mat3
compute_d2c_dpk2_local(
    const float dc_dc_sh, const float dc_dG,
    const vec2& dG_dmean2d, const vec3& dG_dSigma,
    const vec3& H_G_mean2d, const float H_G_sigma[6], const float H_G_mixed[6],
    const mat3& H_c_sh_p,
    const mat3 H_pi_px, const mat3 H_pi_py,
    const mat2x3& J,
    const mat3 H_Sigma_pxx, const mat3 H_Sigma_pxy, const mat3 H_Sigma_pyy,
    const mat2 dSigma_dpx, const mat2 dSigma_dpy, const mat2 dSigma_dpz
)
{
    // Initialize a zero matrix using the standard GLM constructor
    mat3 H(0.0f);

    // Term 1: (∂c/∂c̃ₖ) ⋅ (∂²c̃ₖ/∂pₖ²)
    H += dc_dc_sh * H_c_sh_p;

    // Term 2: (∂c/∂Gₖ) ⋅ [ (∂Gₖ/∂πₖ)⋅(∂²πₖ/∂pₖ²) + (∂Gₖ/∂Σₖ):(∂²Σₖ/∂pₖ²) ]
    if (fabsf(dc_dG) > 1e-7f) {
        // Replaced loops with direct matrix operations
        H += dc_dG * (dG_dmean2d.x * H_pi_px + dG_dmean2d.y * H_pi_py);
        H += dc_dG * (dG_dSigma.x * H_Sigma_pxx + 2.f * dG_dSigma.y * H_Sigma_pxy + dG_dSigma.z * H_Sigma_pyy);
    }

    // Term 3: (∂πₖ/∂pₖ)ᵀ (∂²Gₖ/∂πₖ²) (∂πₖ/∂pₖ)
    // Construct H_G_pi with column vectors as per GLM convention
    mat2 H_G_pi(vec2(H_G_mean2d.x, H_G_mean2d.y), vec2(H_G_mean2d.y, H_G_mean2d.z));
    // Replaced direct matrix multiplication with loops due to GLM error
    //H += glm::transpose(J) * H_G_pi * J;
    mat3x2 J_T = glm::transpose(J);
    // Step 1: Compute the intermediate matrix: temp_mat = J_T * H_G_pi (a 3x2 matrix)
    mat3x2 temp_mat;
    // First column of temp_mat
    temp_mat[0][0] = J_T[0][0] * H_G_pi[0][0] + J_T[0][1] * H_G_pi[1][0];
    temp_mat[1][0] = J_T[1][0] * H_G_pi[0][0] + J_T[1][1] * H_G_pi[1][0];
    temp_mat[2][0] = J_T[2][0] * H_G_pi[0][0] + J_T[2][1] * H_G_pi[1][0];
    // Second column of temp_mat
    temp_mat[0][1] = J_T[0][0] * H_G_pi[0][1] + J_T[0][1] * H_G_pi[1][1];
    temp_mat[1][1] = J_T[1][0] * H_G_pi[0][1] + J_T[1][1] * H_G_pi[1][1];
    temp_mat[2][1] = J_T[2][0] * H_G_pi[0][1] + J_T[2][1] * H_G_pi[1][1];
    mat3 Jt_H_J;
    // First column of Jt_H_J
    Jt_H_J[0][0] = temp_mat[0][0] * J[0][0] + temp_mat[0][1] * J[1][0];
    Jt_H_J[1][0] = temp_mat[1][0] * J[0][0] + temp_mat[1][1] * J[1][0];
    Jt_H_J[2][0] = temp_mat[2][0] * J[0][0] + temp_mat[2][1] * J[1][0];
    // Second column of Jt_H_J
    Jt_H_J[0][1] = temp_mat[0][0] * J[0][1] + temp_mat[0][1] * J[1][1];
    Jt_H_J[1][1] = temp_mat[1][0] * J[0][1] + temp_mat[1][1] * J[1][1];
    Jt_H_J[2][1] = temp_mat[2][0] * J[0][1] + temp_mat[2][1] * J[1][1];
    // Third column of Jt_H_J
    Jt_H_J[0][2] = temp_mat[0][0] * J[0][2] + temp_mat[0][1] * J[1][2];
    Jt_H_J[1][2] = temp_mat[1][0] * J[0][2] + temp_mat[1][1] * J[1][2];
    Jt_H_J[2][2] = temp_mat[2][0] * J[0][2] + temp_mat[2][1] * J[1][2];
    // Step 3: Add the result to H
    H[0][0] += Jt_H_J[0][0]; H[0][1] += Jt_H_J[0][1]; H[0][2] += Jt_H_J[0][2];
    H[1][0] += Jt_H_J[1][0]; H[1][1] += Jt_H_J[1][1]; H[1][2] += Jt_H_J[1][2];
    H[2][0] += Jt_H_J[2][0]; H[2][1] += Jt_H_J[2][1]; H[2][2] += Jt_H_J[2][2];
        // Term 4: (∂Σₖ/∂pₖ)ᵀ : (∂²Gₖ/∂Σₖ²) : (∂Σₖ/∂pₖ)
    {
        const mat2 dS_dp[3] = {dSigma_dpx, dSigma_dpy, dSigma_dpz};
        for (int i = 0; i < 3; i++) {
            for (int j = i; j < 3; j++) { // Compute lower triangle + diagonal
                const mat2& dS_i = dS_dp[i];
                const mat2& dS_j = dS_dp[j];

                // Full tensor contraction with corrected [col][row] access
                const float val =
                    dS_i[0][0] * (H_G_sigma[0] * dS_j[0][0] + H_G_sigma[5] * dS_j[1][1] + 2.f*H_G_sigma[3] * dS_j[1][0]) +
                    dS_i[1][1] * (H_G_sigma[5] * dS_j[0][0] + H_G_sigma[1] * dS_j[1][1] + 2.f*H_G_sigma[4] * dS_j[1][0]) +
                    2.f * dS_i[1][0] * (H_G_sigma[3] * dS_j[0][0] + H_G_sigma[4] * dS_j[1][1] + 2.f*H_G_sigma[2] * dS_j[1][0]);

                H[j][i] += val; // H[col][row]
            }
        }
    }

    // Term 5: 2 * (∂Σₖ/∂pₖ)ᵀ : (∂²Gₖ/∂πₖ∂Σₖ) : (∂πₖ/∂pₖ)
    {
        const mat2 dS_dp[3] = {dSigma_dpx, dSigma_dpy, dSigma_dpz};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                const mat2& dS_i = dS_dp[i];
                // H_G_mixed is {H_πxΣxx, H_πxΣxy, H_πxΣyy, H_πyΣxx, H_πyΣxy, H_πyΣyy}
                // Corrected dS_i access to [col][row]
                const float term_pi_x = dS_i[0][0]*H_G_mixed[0] + 2.f*dS_i[1][0]*H_G_mixed[1] + dS_i[1][1]*H_G_mixed[2];
                const float term_pi_y = dS_i[0][0]*H_G_mixed[3] + 2.f*dS_i[1][0]*H_G_mixed[4] + dS_i[1][1]*H_G_mixed[5];

                // Corrected J access to [col][row]
                const float val = term_pi_x * J[j][0] + term_pi_y * J[j][1];

                H[j][i] += 2.0f * val; // H[col][row]
            }
        }
    }

    // Symmetrize the Hessian
    // This correctly copies the lower triangle to the upper triangle
    H[0][1] = H[1][0];
    H[0][2] = H[2][0];
    H[1][2] = H[2][1];

    return H;
}

// Complete scaling Hessian computation
__device__ __forceinline__ void compute_scaling_hessian(
    const float dc_dG,
    const vec3& dG_dSigma,
    const float* H_G_sigma,
    const mat2* dSigma_dsk,
    mat3& hessian_s
) {
    // Precompute the 3-vectors for each dSigma_dsk[i]
    float comp[3][3];
    for (int i=0; i<3; i++) {
        comp[i][0] = dSigma_dsk[i][0][0];  // dΣ_xx / ds_i
        comp[i][1] = dSigma_dsk[i][1][1];  // dΣ_yy / ds_i
        comp[i][2] = dSigma_dsk[i][1][0];  // dΣ_xy / ds_i
    }

    // Compute full contraction
    for (int i=0; i<3; i++) {
        for (int j=0; j<3; j++) {
            float val =
                comp[i][0] * (H_G_sigma[0] * comp[j][0] +
                             H_G_sigma[5] * comp[j][1] +
                             H_G_sigma[3] * comp[j][2]) +
                comp[i][1] * (H_G_sigma[5] * comp[j][0] +
                             H_G_sigma[1] * comp[j][1] +
                             H_G_sigma[4] * comp[j][2]) +
                comp[i][2] * (H_G_sigma[3] * comp[j][0] +
                             H_G_sigma[4] * comp[j][1] +
                             H_G_sigma[2] * comp[j][2]);

            hessian_s[j][i] = dc_dG * val;
        }
    }
}

/**
 * @brief Computes the first and second derivatives w.r.t. rotation angle θₖ.
 *
 * Implements the logic from the "Rotation solve" section (Eq. 24) of the paper.
 * @param dc_dG Summed intermediate ∂c/∂Gₖ.
 * @param dG_dSigma Summed intermediate ∂Gₖ/∂Σₖ.
 * @param H_G_sigma Summed intermediate ∂²Gₖ/∂Σₖ².
 * @param dSigma_dtheta Pre-computed ∂Σₖ/∂θₖ.
 * @param d2Sigma_dtheta2 Pre-computed ∂²Σₖ/∂θₖ².
 * @param out_grad Output for the final gradient ∂c/∂θₖ.
 * @param out_hessian Output for the final Hessian ∂²c/∂θₖ².
 */
__device__ __forceinline__ void compute_rotation_derivatives(
    const float dc_dG,
    const vec3& dG_dSigma,
    const float* H_G_sigma,
    const mat2& dSigma_dtheta,
    const mat2& d2Sigma_dtheta2,
    float& out_grad,
    float& out_hessian
) {
    // --- First Derivative: ∂c/∂θₖ ---
    // Corrected matrix access to [col][row]
    float grad_term = dG_dSigma.x * dSigma_dtheta[0][0] +
                      2.f * dG_dSigma.y * dSigma_dtheta[1][0] +
                      dG_dSigma.z * dSigma_dtheta[1][1];
    out_grad = dc_dG * grad_term;

    // --- Second Derivative: ∂²c/∂θₖ² ---
    out_hessian = 0.f;

    // Term 1: (∂c/∂Gₖ) * [ (∂Σₖ/∂θₖ)ᵀ : (∂²Gₖ/∂Σₖ²) : (∂Σₖ/∂θₖ) ]
    // Corrected matrix access to [col][row]
    const float dS_xx = dSigma_dtheta[0][0];
    const float dS_yy = dSigma_dtheta[1][1];
    const float dS_xy = dSigma_dtheta[1][0];

    // H_G_sigma layout: {H_xxxx, H_yyyy, H_xyxy, H_xxxy, H_yyxy, H_xxyy}
    const float hess_term1 =
        dS_xx * (H_G_sigma[0] * dS_xx + H_G_sigma[5] * dS_yy + 2.f*H_G_sigma[3] * dS_xy) +
        dS_yy * (H_G_sigma[5] * dS_xx + H_G_sigma[1] * dS_yy + 2.f*H_G_sigma[4] * dS_xy) +
        2.f * dS_xy * (H_G_sigma[3] * dS_xx + H_G_sigma[4] * dS_yy + 2.f*H_G_sigma[2] * dS_xy);

    // Term 2: (∂c/∂Gₖ) * [ (∂Gₖ/∂Σₖ) : (∂²Σₖ/∂θₖ²) ]
    // Corrected matrix access to [col][row]
    const float hess_term2 = dG_dSigma.x * d2Sigma_dtheta2[0][0] +
                             2.f * dG_dSigma.y * d2Sigma_dtheta2[1][0] +
                             dG_dSigma.z * d2Sigma_dtheta2[1][1];

    out_hessian = dc_dG * (hess_term1 + hess_term2);
}

// Include the file with the helper functions (mat2x3, mat3, compute_..._totals, etc.)
//#include "NewtonHelpers.cu"

/**
 * @brief Assembles the final first and second order derivatives w.r.t. pₖ.
 *
 * This kernel is launched with one thread per Gaussian. Each thread reads the
 * summed intermediate products from the backward rasterization pass and the
 * pre-computed projection derivatives to compute the final derivatives without
 * the need for atomic operations.
 *
 * @param num_gaussians Total number of Gaussians to process.
 * @param dc_dcSH_totals Intermediate tensor from Kernel 1.
 * @param dc_dG_totals Intermediate tensor from Kernel 1.
 * @param dG_dmean2d_totals Intermediate tensor from Kernel 1.
 * @param dG_dSigma_totals Intermediate tensor from Kernel 1.
 * @param H_G_mean2d_totals Intermediate tensor from Kernel 1.
 * @param H_G_sigma_totals Intermediate tensor from Kernel 1.
 * @param H_G_mixed_totals Intermediate tensor from Kernel 1.
 * @param jacobians Assumed input ∂πₖ/∂pₖ.
 * @param dSigma_dpx Assumed input ∂Σₖ/∂pₖ_x.
 * ... (and all other assumed projection derivative inputs) ...
 * @param dc_dpk Final output tensor for ∂c/∂pₖ.
 * @param d2c_dpk2 Final output tensor for ∂²c/∂pₖ².
 */
__global__ void assemble_newton_derivatives_kernel(
    // INPUTS:
    const int num_gaussians,
    const float *__restrict__ dc_dcSH_totals,
    const float *__restrict__ dc_dG_totals,
    const vec2 *__restrict__ dG_dmean2d_totals,
    const vec3 *__restrict__ dG_dSigma_totals,
    const vec3 *__restrict__ H_G_mean2d_totals,
    const float *__restrict__ H_G_sigma_totals,
    const float *__restrict__ H_G_mixed_totals,
    const float *__restrict__    dc_dG_opacity_totals,
    const vec3 *__restrict__     dG_dSigma_opacity_totals,
    const float *__restrict__    H_G_sigma_opacity_totals,  // length = num_gaussians * 6
    float *__restrict__          dc_dopacity,                // output ∂c_op/∂pₖ
    float *__restrict__          d2c_dopacity2,              // output ∂²c_op/∂pₖ²
    const mat2x3 *__restrict__ jacobians,
    const mat2 *__restrict__ dSigma_dpx,
    const mat2 *__restrict__ dSigma_dpy,
    const mat2 *__restrict__ dSigma_dpz,
    const mat3 *__restrict__ dc_sh_dp,
    const mat3 *__restrict__ H_pi_px,
    const mat3 *__restrict__ H_pi_py,
    const mat3 *__restrict__ H_c_sh_p,
    const mat3 *__restrict__ H_Sigma_pxx,
    const mat3 *__restrict__ H_Sigma_pxy,
    const mat3 *__restrict__ H_Sigma_pyy,
    const mat2 *__restrict__   dSigma_dtheta_inputs,
    const mat2 *__restrict__   d2Sigma_dtheta2_inputs,
    const mat2x3 *__restrict__ T_matrices,
    const vec3* __restrict__ conics_2d,
    const vec3 *__restrict__ p_k,
    const vec3 camera_pos,
    // OUTPUTS:
    vec2 *__restrict__ d_c_vk,
    mat2 *__restrict__ H_c_vk,
    vec2 *__restrict__ dc_dlambda,
    mat3 *__restrict__ d2c_dlambda2,
    float *__restrict__   dc_dtheta,
    float *__restrict__   d2c_dtheta2
//    float *__restrict__  dc_dcolor,
//    float *__restrict__ dc_dsigma
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;

    // Load data
    const float dc_dc_sh = dc_dcSH_totals[g_idx];
    const float dc_dG = dc_dG_totals[g_idx];
    const vec2 dG_dmean2d = dG_dmean2d_totals[g_idx];
    const vec3 dG_dSigma = dG_dSigma_totals[g_idx];
    const vec3 H_G_mean2d = H_G_mean2d_totals[g_idx];
    const float* H_G_sigma = H_G_sigma_totals + g_idx * 6;
    const float* H_G_mixed = H_G_mixed_totals + g_idx * 6;

    vec3 r_k = normalize(camera_pos - p_k[g_idx]);
    vec3 u_y = normalize(cross(r_k, cross(r_k, vec3(0, 1, 0))));  // Eq. 14
    vec3 u_x = normalize(cross(r_k, u_y));
    mat2x3 U_k = {u_x, u_y};
    // Position solve
    vec3 final_grad = compute_dc_dpk_local(
        dc_dc_sh, dc_dG, dG_dmean2d, dG_dSigma,
        dc_sh_dp[g_idx], jacobians[g_idx],
        dSigma_dpx[g_idx], dSigma_dpy[g_idx], dSigma_dpz[g_idx]
    );

    mat3 final_hessian = compute_d2c_dpk2_local(
        dc_dc_sh, dc_dG, dG_dmean2d, dG_dSigma,
        H_G_mean2d, H_G_sigma, H_G_mixed,
        H_c_sh_p[g_idx], H_pi_px[g_idx], H_pi_py[g_idx], jacobians[g_idx],
        H_Sigma_pxx[g_idx], H_Sigma_pxy[g_idx], H_Sigma_pyy[g_idx],
        dSigma_dpx[g_idx], dSigma_dpy[g_idx], dSigma_dpz[g_idx]
    );

    vec2 dc_dvk = {dot(U_k[0], final_grad), dot(U_k[1], final_grad)};
    mat2 d2c_dvk2 = mat2(
        dot(U_k[0], final_hessian * U_k[0]), dot(U_k[0], final_hessian * U_k[1]),
        dot(U_k[1], final_hessian * U_k[0]), dot(U_k[1], final_hessian * U_k[1])
    );
    d_c_vk[g_idx] = dc_dvk;
    H_c_vk[g_idx] = d2c_dvk2;

    // Scaling solve
    const vec3 cov_2d = conics_2d[g_idx];
    float lambda_min, lambda_max;
    vec2 v_min, v_max;

    eigen_decomposition_2d(
        cov_2d.x, cov_2d.y, cov_2d.z, // Inputs: a, b, c
        lambda_min, lambda_max,       // Outputs: eigenvalues
        v_min, v_max                  // Outputs: eigenvectors
    );

    // The eigenvector matrix V_k is formed from the resulting eigenvectors.
    //[cite_start]// The decomposition gives Σₖ = Vₖᵀ Λₖ Vₖ[cite: 275].
    mat2 V = { {v_min.x, v_max.x}, {v_min.y, v_max.y} };
    //const mat2 V = eigenvectors_2d[g_idx];
    const mat2x3 T_k = T_matrices[g_idx];

       // Compute ∂Σₖ/∂λₖ using the outer product, which is the idiomatic GLM approach.
    // The derivative ∂Σ/∂λᵢ is the outer product vᵢvᵢᵀ.
    mat2 dSigma_dlambda[2];
    dSigma_dlambda[0] = glm::outerProduct(V[0], V[0]); // Using v_min (1st col of V)
    dSigma_dlambda[1] = glm::outerProduct(V[1], V[1]); // Using v_max (2nd col of V)

    // Compute ∂Σₖ/∂sₖ
    mat2 dSigma_dsk[3];
    for (int i=0; i<3; ++i) {
        // Replaced element-wise assignment with direct matrix arithmetic.
        // T_k is mat2x3, so T_k[i] is a vec2 (column i).
        dSigma_dsk[i] = dSigma_dlambda[0] * T_k[i][0] + dSigma_dlambda[1] * T_k[i][1];
    }

    // Compute gradient ∂c/∂sₖ
    vec3 grad_s(0.0f);
    for(int i=0; i<3; ++i) {
        // Corrected access for grad_s vector and dSigma_dsk matrix [col][row]
        grad_s[i] = dG_dSigma.x * dSigma_dsk[i][0][0] +
                    2.f * dG_dSigma.y * dSigma_dsk[i][1][0] +
                    dG_dSigma.z * dSigma_dsk[i][1][1];
    }

    //dc_dsk[g_idx] = grad_s;

    // Compute Hessian ∂²c/∂sₖ²
    mat3 hessian_s(0.0f);
    compute_scaling_hessian(
        dc_dG,
        dG_dSigma,
        H_G_sigma,
        dSigma_dsk,
        hessian_s
    );
    //d2c_dsk2[g_idx] = hessian_s;

        // 3) form the 2×2 Gram matrix M = T Tᵀ and invert it:
    mat2 M;                               // M = T * Tᵀ
    M[0][0] = glm::dot(T_k[0],T_k[0]);
    M[0][1] = glm::dot(T_k[0],T_k[1]);
    M[1][0] = M[0][1];
    M[1][1] = glm::dot(T_k[1],T_k[1]);
    mat2 M_inv = inverse(M);             // your favorite small‐matrix inverse

    // 4) compute ∂c/∂λₖ = M⁻¹ * (T * grad_s)
    vec2 grad_lambda;
    {
      vec2 Tgs;
      Tgs.x = glm::dot(T_k[0], grad_s);
      Tgs.y = glm::dot(T_k[1], grad_s);
      grad_lambda = M_inv * Tgs;
    }

    // 5) compute H_λ = M⁻¹ * (T * H_s * Tᵀ) * M⁻¹
    mat2 H_temp;
    for(int i=0;i<2;++i)
      for(int j=0;j<2;++j){
        // (T * H_s * Tᵀ)[i][j]
        H_temp[i][j] =
          T_k[i].x*(hessian_s[0][0]*T_k[j].x + hessian_s[0][1]*T_k[j].y + hessian_s[0][2]*T_k[j].z)
        + T_k[i].y*(hessian_s[1][0]*T_k[j].x + hessian_s[1][1]*T_k[j].y + hessian_s[1][2]*T_k[j].z)
        + T_k[i].z*(hessian_s[2][0]*T_k[j].x + hessian_s[2][1]*T_k[j].y + hessian_s[2][2]*T_k[j].z);
      }
    mat2 H_lambda = M_inv * (H_temp * M_inv);
    dc_dlambda   [g_idx] = grad_lambda;
    d2c_dlambda2[g_idx] = H_lambda;
      // ========================================================================
    // Rotation Solve (θₖ) - New logic
    // ========================================================================
    {
        // 1. Load data for the rotation solve
        const float dc_dG = dc_dG_totals[g_idx];
        const vec3 dG_dSigma = dG_dSigma_totals[g_idx];
        const float* H_G_sigma = H_G_sigma_totals + g_idx * 6;
        const mat2 dSigma_dtheta = dSigma_dtheta_inputs[g_idx];
        const mat2 d2Sigma_dtheta2 = d2Sigma_dtheta2_inputs[g_idx];

        float grad_theta = 0.f;
        float hessian_theta = 0.f;

        // 2. Call the helper function to compute derivatives
        compute_rotation_derivatives(
            dc_dG,
            dG_dSigma,
            H_G_sigma,
            dSigma_dtheta,
            d2Sigma_dtheta2,
            grad_theta,
            hessian_theta
        );

        // 3. Write results directly to global memory
        dc_dtheta[g_idx] = grad_theta;
        d2c_dtheta2[g_idx] = hessian_theta;
    }
     // ========================================================================
    // Opacity solve (αₖ) not called here
    // in compute_intermediate_derivatives_kernel we compute the dc_dopac quantity
    // which is of our interest here
    // d2c_dopacity2 is always zero.
    // ========================================================================
    /*
    {
        const float  dc_dG_op    = dc_dG_opacity_totals    [g_idx];
        const vec3   dG_dSigma_op    = dG_dSigma_opacity_totals[g_idx];
        const float* H_Gsigma_op     = H_G_sigma_opacity_totals + g_idx*6;

        float grad_op = 0.f, hess_op = 0.f;
        compute_opacity_derivatives(
            dc_dG_op,
            dG_dSigma_op,
            H_Gsigma_op,
            jacobians[g_idx],
            dSigma_dpx[g_idx],
            dSigma_dpy[g_idx],
            dSigma_dpz[g_idx],
            grad_op,
            hess_op
        );

        dc_dopacity[g_idx]    = grad_op;
        d2c_dopacity2[g_idx]  = hess_op;
    }
    */

    // ========================================================================
    // Color solve (R, G, B) - NOT CALLED HERE
    // DC_DCOLOR IS CALCULATED IN BACKWARD OF SPHERICAL HARMONICS
    // D2C_DCOLOR2 IS ALWAYS 0
    // ========================================================================
    /*
    for (int chan = 0; chan < 3; ++chan) {
        dc_dcolor   [idx3] = grad_c;
        d2c_dcolor2 [idx3] = hess_c;
    }
    */
}



void launch_assemble_newton_derivatives_kernel(
    const int num_gaussians,
    const at::Tensor dc_dcSH_totals,
    const at::Tensor dc_dG_totals,
    const at::Tensor dG_dmean2d_totals,
    const at::Tensor dG_dSigma_totals,
    const at::Tensor H_G_mean2d_totals,
    const at::Tensor H_G_sigma_totals,
    const at::Tensor H_G_mixed_totals,
    const at::Tensor dc_dG_opacity_totals,
    const at::Tensor dG_dSigma_opacity_totals,
    const at::Tensor H_G_sigma_opacity_totals,
    at::Tensor dc_dopacity,
    at::Tensor d2c_dopacity2,
    const at::Tensor jacobians,
    const at::Tensor dSigma_dpx,
    const at::Tensor dSigma_dpy,
    const at::Tensor dSigma_dpz,
    const at::Tensor dc_sh_dp,
    const at::Tensor H_pi_px,
    const at::Tensor H_pi_py,
    const at::Tensor H_c_sh_p,
    const at::Tensor H_Sigma_pxx,
    const at::Tensor H_Sigma_pxy,
    const at::Tensor H_Sigma_pyy,
    const at::Tensor dSigma_dtheta_inputs,
    const at::Tensor d2Sigma_dtheta2_inputs,
    const at::Tensor T_matrices,
    const at::Tensor conics_2d,
    const at::Tensor p_k,
    const at::Tensor camera_pos,
    at::Tensor d_c_vk,
    at::Tensor H_c_vk,
    at::Tensor dc_dlambda,
    at::Tensor d2c_dlambda2,
    at::Tensor dc_dtheta,
    at::Tensor d2c_dtheta2
//    at::Tensor dc_dcolor,
//    at::Tensor dc_dsigma
) {
    // Configure kernel launch
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;

    // Get pointers to tensor data
    auto camera_pos_ptr = reinterpret_cast<const vec3*>(camera_pos.data_ptr<float>());

    assemble_newton_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        dc_dcSH_totals.data_ptr<float>(),
        dc_dG_totals.data_ptr<float>(),
        reinterpret_cast<const vec2*>(dG_dmean2d_totals.data_ptr<float>()),
        reinterpret_cast<const vec3*>(dG_dSigma_totals.data_ptr<float>()),
        reinterpret_cast<const vec3*>(H_G_mean2d_totals.data_ptr<float>()),
        H_G_sigma_totals.data_ptr<float>(),
        H_G_mixed_totals.data_ptr<float>(),
        dc_dG_opacity_totals.data_ptr<float>(),
        reinterpret_cast<const vec3*>(dG_dSigma_opacity_totals.data_ptr<float>()),
        H_G_sigma_opacity_totals.data_ptr<float>(),
        dc_dopacity.data_ptr<float>(),
        d2c_dopacity2.data_ptr<float>(),
        reinterpret_cast<const mat2x3*>(jacobians.data_ptr<float>()),
        reinterpret_cast<const mat2*>(dSigma_dpx.data_ptr<float>()),
        reinterpret_cast<const mat2*>(dSigma_dpy.data_ptr<float>()),
        reinterpret_cast<const mat2*>(dSigma_dpz.data_ptr<float>()),
        reinterpret_cast<const mat3*>(dc_sh_dp.data_ptr<float>()),
        reinterpret_cast<const mat3*>(H_pi_px.data_ptr<float>()),
        reinterpret_cast<const mat3*>(H_pi_py.data_ptr<float>()),
        reinterpret_cast<const mat3*>(H_c_sh_p.data_ptr<float>()),
        reinterpret_cast<const mat3*>(H_Sigma_pxx.data_ptr<float>()),
        reinterpret_cast<const mat3*>(H_Sigma_pxy.data_ptr<float>()),
        reinterpret_cast<const mat3*>(H_Sigma_pyy.data_ptr<float>()),
        reinterpret_cast<const mat2*>(dSigma_dtheta_inputs.data_ptr<float>()),
        reinterpret_cast<const mat2*>(d2Sigma_dtheta2_inputs.data_ptr<float>()),
        reinterpret_cast<const mat2x3*>(T_matrices.data_ptr<float>()),
        reinterpret_cast<const vec3*>(conics_2d.data_ptr<float>()),
        reinterpret_cast<const vec3*>(p_k.data_ptr<float>()),
        *camera_pos_ptr,
        reinterpret_cast<vec2*>(d_c_vk.data_ptr<float>()),
        reinterpret_cast<mat2*>(H_c_vk.data_ptr<float>()),
        reinterpret_cast<vec2*>(dc_dlambda.data_ptr<float>()),
        reinterpret_cast<mat3*>(d2c_dlambda2.data_ptr<float>()),
        dc_dtheta.data_ptr<float>(),
        d2c_dtheta2.data_ptr<float>()
//       dc_dcolor.data_ptr<float>(),
//        dc_dsigma.data_ptr<float>()
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
