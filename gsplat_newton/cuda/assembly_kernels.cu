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
namespace gsplat_newton {
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

__device__ __forceinline__ void chain_attribute(
        const vec3& dL_dc,
        const vec3& d2L_dc2,
        const vec2& dC_dyk,
        const mat2& d2C_dyk2,
        vec2&       dL_dyk,
        mat2&       H_L_yk)
{
    float S1, S2;
    S1 = dL_dc.x  + dL_dc.y  + dL_dc.z;
    S2 = d2L_dc2.x+ d2L_dc2.y+ d2L_dc2.z;

    dL_dyk = S1 * dC_dyk;

    H_L_yk[0][0] = S2 * dC_dyk.x * dC_dyk.x + S1 * d2C_dyk2[0][0];
    H_L_yk[0][1] = S2 * dC_dyk.x * dC_dyk.y + S1 * d2C_dyk2[0][1];
    H_L_yk[1][0] = H_L_yk[0][1];
    H_L_yk[1][1] = S2 * dC_dyk.y * dC_dyk.y + S1 * d2C_dyk2[1][1];
}

// Repetition from Projection kernel in gsplat
///////////////////////////////
// Quaternion
///////////////////////////////

inline __device__ mat3 quat_to_rotmat(const vec4 quat) {
    float w = quat[0], x = quat[1], y = quat[2], z = quat[3];
    // normalize
    float inv_norm = rsqrt(x * x + y * y + z * z + w * w);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
    w *= inv_norm;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    return mat3(
        (1.f - 2.f * (y2 + z2)),
        (2.f * (xy + wz)),
        (2.f * (xz - wy)), // 1st col
        (2.f * (xy - wz)),
        (1.f - 2.f * (x2 + z2)),
        (2.f * (yz + wx)), // 2nd col
        (2.f * (xz + wy)),
        (2.f * (yz - wx)),
        (1.f - 2.f * (x2 + y2)) // 3rd col
    );
}


/**
 * @brief Corrected position gradient computation with RGB channel handling
 */
__device__ __forceinline__ vec3 compute_dc_dpk_local_rgb(
    const float dc_dc_sh,
    const float dc_dG,
    const vec2& dG_dmean2d,
    const vec3& dG_dSigma,
    const mat3& dc_sh_dp,     // 3x3 matrix: channels x position components
    const mat2x3& J,
    const mat2 dSigma_dpx, const mat2 dSigma_dpy, const mat2 dSigma_dpz
) {
    vec3 grad = {0.f, 0.f, 0.f};

    // Term 1: Spherical harmonics contribution - corrected for RGB channels
    // dc_sh_dp[i][j] = ∂(color_channel_i)/∂(position_component_j)
    // We need to sum over color channels for each position component
    grad.x += dc_dc_sh * (dc_sh_dp[0][0] + dc_sh_dp[1][0] + dc_sh_dp[2][0]); // Sum RGB for px
    grad.y += dc_dc_sh * (dc_sh_dp[0][1] + dc_sh_dp[1][1] + dc_sh_dp[2][1]); // Sum RGB for py
    grad.z += dc_dc_sh * (dc_sh_dp[0][2] + dc_sh_dp[1][2] + dc_sh_dp[2][2]); // Sum RGB for pz

    // Term 2: Gaussian weight contribution (unchanged)
    if (fabsf(dc_dG) > 1e-7f) {
        grad.x += dc_dG * (dG_dmean2d.x * J[0][0] + dG_dmean2d.y * J[0][1]);
        grad.y += dc_dG * (dG_dmean2d.x * J[1][0] + dG_dmean2d.y * J[1][1]);
        grad.z += dc_dG * (dG_dmean2d.x * J[2][0] + dG_dmean2d.y * J[2][1]);

        float dG_S_dp_x = dG_dSigma.x * dSigma_dpx[0][0] + 2.0f * dG_dSigma.z * dSigma_dpx[1][0] + dG_dSigma.y * dSigma_dpx[1][1];
        float dG_S_dp_y = dG_dSigma.x * dSigma_dpy[0][0] + 2.0f * dG_dSigma.z * dSigma_dpy[1][0] + dG_dSigma.y * dSigma_dpy[1][1];
        float dG_S_dp_z = dG_dSigma.x * dSigma_dpz[0][0] + 2.0f * dG_dSigma.z * dSigma_dpz[1][0] + dG_dSigma.y * dSigma_dpz[1][1];

        grad.x += dc_dG * dG_S_dp_x;
        grad.y += dc_dG * dG_S_dp_y;
        grad.z += dc_dG * dG_S_dp_z;
    }
    return grad;
}

// HELPERS Sigma_inv to Sigma for derivatives conversion

/**
 * @brief Complete conversion functions for derivatives from Σ^{-1} to Σ
 *
 * For Gaussian splatting, we need to convert derivatives computed w.r.t. the
 * inverse covariance matrix Σ^{-1} to derivatives w.r.t. the covariance matrix Σ.
 *
 * Mathematical background:
 * - Covariance matrix Σ = R * S * S^T * R^T (always symmetric)
 * - Gaussian weight: G = exp(-0.5 * (π - x)^T * Σ^{-1} * (π - x))
 * - We compute ∂G/∂Σ^{-1} and ∂²G/∂(Σ^{-1})² directly
 * - Need to convert to ∂G/∂Σ and ∂²G/∂Σ² using chain rule
 */

/**
 * @brief Converts first derivative from inverse covariance to covariance
 *
 * Chain rule: ∂G/∂Σ_{ij} = Σ_{kl} (∂G/∂(Σ^{-1})_{kl}) * (∂(Σ^{-1})_{kl}/∂Σ_{ij})
 * Where: ∂(Σ^{-1})_{kl}/∂Σ_{ij} = -(Σ^{-1})_{ki} * (Σ^{-1})_{jl}
 *
 * @param dG_dSigma_inv Input: ∂G/∂Σ^{-1} as vec3 {∂G/∂Σ^{-1}_{00}, ∂G/∂Σ^{-1}_{11}, ∂G/∂Σ^{-1}_{01}}
 * @param Sigma_inv The inverse covariance matrix Σ^{-1}
 * @return vec3 Output: ∂G/∂Σ as {∂G/∂Σ_{00}, ∂G/∂Σ_{11}, ∂G/∂Σ_{01}}
 */
__device__ __forceinline__ vec3 convert_first_derivative_inverse_to_covariance(
    const vec3& dG_dSinv,
    const mat2& Sigma_inv
) {
    // For 2x2 symmetric matrix stored as vec3(Σ_{00}, Σ_{11}, Σ_{01})
    // We apply: ∂G/∂Σ_{ij} = -Σ_{kl} (∂G/∂(Σ^{-1})_{kl}) * (Σ^{-1})_{ki} * (Σ^{-1})_{jl}

    const float Sinv_00 = Sigma_inv[0][0];
    const float Sinv_11 = Sigma_inv[1][1];
    const float Sinv_01 = Sigma_inv[1][0]; // = Sigma_inv[0][1] due to symmetry

    vec3 dG_dSigma;

    // ∂G/∂Σ_{00} = -[(∂G/∂Σ^{-1}_{00}) * Σ^{-1}_{00} * Σ^{-1}_{00}
    //               + (∂G/∂Σ^{-1}_{11}) * Σ^{-1}_{01} * Σ^{-1}_{01}
    //               + (∂G/∂Σ^{-1}_{01}) * (Σ^{-1}_{00} * Σ^{-1}_{01} + Σ^{-1}_{01} * Σ^{-1}_{00})]
    dG_dSigma.x = -(dG_dSinv.x * Sinv_00 * Sinv_00 +
                    dG_dSinv.y * Sinv_01 * Sinv_01 +
                    dG_dSinv.z * 2.0f * Sinv_00 * Sinv_01);

    // ∂G/∂Σ_{11} = -[(∂G/∂Σ^{-1}_{00}) * Σ^{-1}_{01} * Σ^{-1}_{01}
    //               + (∂G/∂Σ^{-1}_{11}) * Σ^{-1}_{11} * Σ^{-1}_{11}
    //               + (∂G/∂Σ^{-1}_{01}) * (Σ^{-1}_{01} * Σ^{-1}_{11} + Σ^{-1}_{11} * Σ^{-1}_{01})]
    dG_dSigma.y = -(dG_dSinv.x * Sinv_01 * Sinv_01 +
                    dG_dSinv.y * Sinv_11 * Sinv_11 +
                    dG_dSinv.z * 2.0f * Sinv_01 * Sinv_11);

    // ∂G/∂Σ_{01} = -[(∂G/∂Σ^{-1}_{00}) * Σ^{-1}_{00} * Σ^{-1}_{01}
    //               + (∂G/∂Σ^{-1}_{11}) * Σ^{-1}_{01} * Σ^{-1}_{11}
    //               + (∂G/∂Σ^{-1}_{01}) * (Σ^{-1}_{00} * Σ^{-1}_{11} + Σ^{-1}_{01} * Σ^{-1}_{01})]
    dG_dSigma.z = -(dG_dSinv.x * Sinv_00 * Sinv_01 +
                    dG_dSinv.y * Sinv_01 * Sinv_11 +
                    dG_dSinv.z * (Sinv_00 * Sinv_11 + Sinv_01 * Sinv_01));

    return dG_dSigma;
}

/**
 * @brief Converts second derivative (Hessian) from inverse covariance to covariance
 *
 * This implements the full chain rule for second derivatives:
 * ∂²G/∂Σ_{ij}∂Σ_{pq} = Σ_{kl,mn} [
 *   (∂²G/∂(Σ^{-1})_{kl}∂(Σ^{-1})_{mn}) * (∂(Σ^{-1})_{kl}/∂Σ_{ij}) * (∂(Σ^{-1})_{mn}/∂Σ_{pq})
 *   + (∂G/∂(Σ^{-1})_{kl}) * (∂²(Σ^{-1})_{kl}/∂Σ_{ij}∂Σ_{pq})
 * ]
 *
 * Where:
 * ∂(Σ^{-1})_{kl}/∂Σ_{ij} = -(Σ^{-1})_{ki} * (Σ^{-1})_{jl}
 * ∂²(Σ^{-1})_{kl}/∂Σ_{ij}∂Σ_{pq} = (Σ^{-1})_{kp} * (Σ^{-1})_{qi} * (Σ^{-1})_{jl} + (Σ^{-1})_{ki} * (Σ^{-1})_{jp} * (Σ^{-1})_{ql}
 *
 * @param dG_dSigma_inv First derivative ∂G/∂Σ^{-1}
 * @param H_G_Sigma_inv Second derivative ∂²G/∂(Σ^{-1})² (6 components for 2x2 symmetric)
 * @param Sigma_inv The inverse covariance matrix Σ^{-1}
 * @param H_G_Sigma Output array for ∂²G/∂Σ² (6 components)
 */
__device__ __forceinline__ void convert_second_derivative_inverse_to_covariance(
    const vec3& dG_dSinv,
    const float* H_G_Sinv,  // [H_{00,00}, H_{11,11}, H_{01,01}, H_{00,01}, H_{01,11}, H_{00,11}]
    const mat2& Sigma_inv,
    float* H_G_Sigma        // Output: same layout as H_G_Sinv
) {
    const float Sinv_00 = Sigma_inv[0][0];
    const float Sinv_11 = Sigma_inv[1][1];
    const float Sinv_01 = Sigma_inv[1][0];

    // Pre-compute commonly used products
    const float Sinv_00_sq = Sinv_00 * Sinv_00;
    const float Sinv_11_sq = Sinv_11 * Sinv_11;
    const float Sinv_01_sq = Sinv_01 * Sinv_01;

    // Mapping for symmetric 2x2 Hessian storage:
    // H[0] = H_{00,00}, H[1] = H_{11,11}, H[2] = H_{01,01}
    // H[3] = H_{00,01}, H[4] = H_{01,11}, H[5] = H_{00,11}

    // ∂²G/∂Σ_{00}²
    {
        // Term 1: Chain rule with second derivatives of G
        float term1 = H_G_Sinv[0] * (Sinv_00_sq * Sinv_00_sq) +          // H_{00,00} term
                      H_G_Sinv[1] * (Sinv_01_sq * Sinv_01_sq) +          // H_{11,11} term
                      H_G_Sinv[2] * (4.0f * Sinv_00_sq * Sinv_01_sq) +   // H_{01,01} term
                      2.0f * H_G_Sinv[3] * (Sinv_00_sq * 2.0f * Sinv_00 * Sinv_01) + // H_{00,01} term
                      2.0f * H_G_Sinv[4] * (Sinv_01_sq * 2.0f * Sinv_00 * Sinv_01) + // H_{01,11} term
                      2.0f * H_G_Sinv[5] * (2.0f * Sinv_00 * Sinv_01 * Sinv_01_sq);  // H_{00,11} term

        // Term 2: First derivatives with second derivatives of inverse
        float term2 = dG_dSinv.x * 2.0f * (Sinv_00 * Sinv_00_sq) +      // 2 * Σ^{-1}_{00}³
                      dG_dSinv.y * 2.0f * (Sinv_01 * Sinv_01_sq) +      // 2 * Σ^{-1}_{01}³
                      dG_dSinv.z * 4.0f * (Sinv_00 * Sinv_00 * Sinv_01 + Sinv_01 * Sinv_00_sq);

        H_G_Sigma[0] = term1 + term2;
    }

    // ∂²G/∂Σ_{11}²
    {
        float term1 = H_G_Sinv[0] * (Sinv_01_sq * Sinv_01_sq) +
                      H_G_Sinv[1] * (Sinv_11_sq * Sinv_11_sq) +
                      H_G_Sinv[2] * (4.0f * Sinv_01_sq * Sinv_11_sq) +
                      2.0f * H_G_Sinv[3] * (Sinv_01_sq * 2.0f * Sinv_01 * Sinv_11) +
                      2.0f * H_G_Sinv[4] * (2.0f * Sinv_01 * Sinv_11 * Sinv_11_sq) +
                      2.0f * H_G_Sinv[5] * (Sinv_01_sq * Sinv_11_sq);

        float term2 = dG_dSinv.x * 2.0f * (Sinv_01 * Sinv_01_sq) +
                      dG_dSinv.y * 2.0f * (Sinv_11 * Sinv_11_sq) +
                      dG_dSinv.z * 4.0f * (Sinv_01 * Sinv_01 * Sinv_11 + Sinv_11 * Sinv_01_sq);

        H_G_Sigma[1] = term1 + term2;
    }

    // ∂²G/∂Σ_{01}²
    {
        float term1 = H_G_Sinv[0] * (Sinv_00_sq * Sinv_01_sq) +
                      H_G_Sinv[1] * (Sinv_01_sq * Sinv_11_sq) +
                      H_G_Sinv[2] * ((Sinv_00 * Sinv_11 + Sinv_01_sq) * (Sinv_00 * Sinv_11 + Sinv_01_sq)) +
                      2.0f * H_G_Sinv[3] * (Sinv_00 * Sinv_01 * (Sinv_00 * Sinv_11 + Sinv_01_sq)) +
                      2.0f * H_G_Sinv[4] * (Sinv_01 * Sinv_11 * (Sinv_00 * Sinv_11 + Sinv_01_sq)) +
                      2.0f * H_G_Sinv[5] * (Sinv_00 * Sinv_01 * Sinv_01 * Sinv_11);

        float term2 = dG_dSinv.x * (Sinv_01_sq + Sinv_00 * Sinv_11) +
                      dG_dSinv.y * (Sinv_01_sq + Sinv_00 * Sinv_11) +
                      dG_dSinv.z * 2.0f * (Sinv_00 * Sinv_01 + Sinv_01 * Sinv_11);

        H_G_Sigma[2] = term1 + term2;
    }

    // ∂²G/∂Σ_{00}∂Σ_{01} = H_{00,01}
    {
        float term1 = H_G_Sinv[0] * (Sinv_00_sq * Sinv_00 * Sinv_01) +
                      H_G_Sinv[1] * (Sinv_01_sq * Sinv_01 * Sinv_11) +
                      H_G_Sinv[2] * (4.0f * Sinv_00 * Sinv_01 * (Sinv_00 * Sinv_11 + Sinv_01_sq)) +
                      H_G_Sinv[3] * (Sinv_00_sq * (Sinv_00 * Sinv_11 + Sinv_01_sq) + 2.0f * Sinv_00 * Sinv_01 * Sinv_00 * Sinv_01) +
                      H_G_Sinv[4] * (2.0f * Sinv_00 * Sinv_01 * Sinv_01 * Sinv_11 + Sinv_01_sq * (Sinv_00 * Sinv_11 + Sinv_01_sq)) +
                      H_G_Sinv[5] * (2.0f * Sinv_00 * Sinv_01 * Sinv_01_sq + Sinv_00_sq * Sinv_01 * Sinv_11);

        float term2 = dG_dSinv.x * (Sinv_00_sq * Sinv_01 + 2.0f * Sinv_00 * Sinv_00 * Sinv_01) +
                      dG_dSinv.y * (Sinv_01_sq * Sinv_11 + 2.0f * Sinv_01 * Sinv_01 * Sinv_11) +
                      dG_dSinv.z * (2.0f * Sinv_00 * (Sinv_00 * Sinv_11 + Sinv_01_sq) + 2.0f * Sinv_01 * 2.0f * Sinv_00 * Sinv_01);

        H_G_Sigma[3] = term1 + term2;
    }

    // ∂²G/∂Σ_{01}∂Σ_{11} = H_{01,11}
    {
        float term1 = H_G_Sinv[0] * (Sinv_01_sq * Sinv_00 * Sinv_01) +
                      H_G_Sinv[1] * (Sinv_11_sq * Sinv_01 * Sinv_11) +
                      H_G_Sinv[2] * ((Sinv_00 * Sinv_11 + Sinv_01_sq) * 4.0f * Sinv_01 * Sinv_11) +
                      H_G_Sinv[3] * (Sinv_00 * Sinv_01 * Sinv_01_sq + (Sinv_00 * Sinv_11 + Sinv_01_sq) * Sinv_01_sq) +
                      H_G_Sinv[4] * ((Sinv_00 * Sinv_11 + Sinv_01_sq) * Sinv_11_sq + 2.0f * Sinv_01 * Sinv_11 * Sinv_01 * Sinv_11) +
                      H_G_Sinv[5] * (Sinv_00 * Sinv_01 * 2.0f * Sinv_01 * Sinv_11 + Sinv_01_sq * Sinv_11_sq);

        float term2 = dG_dSinv.x * (Sinv_00 * Sinv_01_sq + Sinv_01 * Sinv_00 * Sinv_01) +
                      dG_dSinv.y * (Sinv_01 * Sinv_11_sq + Sinv_11 * Sinv_01 * Sinv_11) +
                      dG_dSinv.z * ((Sinv_00 * Sinv_11 + Sinv_01_sq) + 2.0f * Sinv_01 * (Sinv_01 + Sinv_11));

        H_G_Sigma[4] = term1 + term2;
    }

    // ∂²G/∂Σ_{00}∂Σ_{11} = H_{00,11}
    {
        float term1 = H_G_Sinv[0] * (Sinv_00_sq * Sinv_01_sq) +
                      H_G_Sinv[1] * (Sinv_01_sq * Sinv_11_sq) +
                      H_G_Sinv[2] * (4.0f * Sinv_00 * Sinv_01 * Sinv_01 * Sinv_11) +
                      H_G_Sinv[3] * (2.0f * Sinv_00 * Sinv_01 * Sinv_01_sq) +
                      H_G_Sinv[4] * (2.0f * Sinv_01 * Sinv_11 * Sinv_01_sq) +
                      H_G_Sinv[5] * (Sinv_00_sq * Sinv_11_sq + 2.0f * Sinv_00 * Sinv_01 * Sinv_01 * Sinv_11);

        float term2 = dG_dSinv.x * (Sinv_01_sq) +
                      dG_dSinv.y * (Sinv_01_sq) +
                      dG_dSinv.z * (2.0f * Sinv_00 * Sinv_01 + 2.0f * Sinv_01 * Sinv_11);

        H_G_Sigma[5] = term1 + term2;
    }
}

/**
 * @brief Helper function to compute the inverse covariance matrix with numerical stability
 *
 * @param conic_2d The 2D covariance matrix as vec3 {Σ_{00}, Σ_{11}, Σ_{01}}
 * @return mat2 The inverse covariance matrix Σ^{-1}
 */
__device__ __forceinline__ mat2 compute_inverse_covariance_2d(const vec3& conic_2d) {
    const float det = conic_2d.x * conic_2d.y - conic_2d.z * conic_2d.z;
    const float inv_det = 1.0f / fmaxf(det, 1e-10f); // Numerical stability

    mat2 Sigma_inv;
    Sigma_inv[0][0] = conic_2d.y * inv_det;     // Σ^{-1}_{00} = Σ_{11} / det
    Sigma_inv[1][1] = conic_2d.x * inv_det;     // Σ^{-1}_{11} = Σ_{00} / det
    Sigma_inv[1][0] = -conic_2d.z * inv_det;    // Σ^{-1}_{01} = -Σ_{01} / det
    Sigma_inv[0][1] = Sigma_inv[1][0];          // Symmetry

    return Sigma_inv;
}

/**
 * @brief Converts mixed partial derivatives from inverse covariance to covariance
 *
 * This handles the conversion of ∂²G/∂π∂(Σ^{-1}) to ∂²G/∂π∂Σ using chain rule:
 * ∂²G/∂π_i∂Σ_{jk} = -∑_{pq} (∂²G/∂π_i∂(Σ^{-1})_{pq}) * (Σ^{-1})_{pj} * (Σ^{-1})_{kq}
 *
 * @param H_G_mixed_inv Input mixed derivatives w.r.t. Σ^{-1}
 *                      Layout: {H_πxΣ^{-1}_{00}, H_πxΣ^{-1}_{01}, H_πxΣ^{-1}_{11},
 *                               H_πyΣ^{-1}_{00}, H_πyΣ^{-1}_{01}, H_πyΣ^{-1}_{11}}
 * @param Sigma_inv The inverse covariance matrix Σ^{-1}
 * @param H_G_mixed Output mixed derivatives w.r.t. Σ (same layout but for Σ)
 */
__device__ __forceinline__ void convert_mixed_derivatives_inverse_to_covariance(
    const float* H_G_mixed_inv,  // 6 components
    const mat2& Sigma_inv,
    float* H_G_mixed             // 6 components output
) {
    const float Sinv_00 = Sigma_inv[0][0];
    const float Sinv_11 = Sigma_inv[1][1];
    const float Sinv_01 = Sigma_inv[1][0];

    // Input layout: {H_πxΣ^{-1}_{00}, H_πxΣ^{-1}_{01}, H_πxΣ^{-1}_{11},
    //                H_πyΣ^{-1}_{00}, H_πyΣ^{-1}_{01}, H_πyΣ^{-1}_{11}}
    const float H_px_Sinv00 = H_G_mixed_inv[0];
    const float H_px_Sinv01 = H_G_mixed_inv[1];
    const float H_px_Sinv11 = H_G_mixed_inv[2];
    const float H_py_Sinv00 = H_G_mixed_inv[3];
    const float H_py_Sinv01 = H_G_mixed_inv[4];
    const float H_py_Sinv11 = H_G_mixed_inv[5];

    // Convert: ∂²G/∂π_x∂Σ_{00} = -[(∂²G/∂π_x∂Σ^{-1}_{00}) * Σ^{-1}_{00} * Σ^{-1}_{00}
    //                              + (∂²G/∂π_x∂Σ^{-1}_{01}) * Σ^{-1}_{01} * Σ^{-1}_{00} * 2
    //                              + (∂²G/∂π_x∂Σ^{-1}_{11}) * Σ^{-1}_{01} * Σ^{-1}_{01}]
    H_G_mixed[0] = -(H_px_Sinv00 * Sinv_00 * Sinv_00 +
                     H_px_Sinv01 * Sinv_01 * Sinv_00 * 2.0f +
                     H_px_Sinv11 * Sinv_01 * Sinv_01);

    // ∂²G/∂π_x∂Σ_{01}
    H_G_mixed[1] = -(H_px_Sinv00 * Sinv_00 * Sinv_01 +
                     H_px_Sinv01 * (Sinv_01 * Sinv_01 + Sinv_00 * Sinv_11) +
                     H_px_Sinv11 * Sinv_01 * Sinv_11);

    // ∂²G/∂π_x∂Σ_{11}
    H_G_mixed[2] = -(H_px_Sinv00 * Sinv_01 * Sinv_01 +
                     H_px_Sinv01 * Sinv_01 * Sinv_11 * 2.0f +
                     H_px_Sinv11 * Sinv_11 * Sinv_11);

    // ∂²G/∂π_y∂Σ_{00}
    H_G_mixed[3] = -(H_py_Sinv00 * Sinv_00 * Sinv_00 +
                     H_py_Sinv01 * Sinv_01 * Sinv_00 * 2.0f +
                     H_py_Sinv11 * Sinv_01 * Sinv_01);

    // ∂²G/∂π_y∂Σ_{01}
    H_G_mixed[4] = -(H_py_Sinv00 * Sinv_00 * Sinv_01 +
                     H_py_Sinv01 * (Sinv_01 * Sinv_01 + Sinv_00 * Sinv_11) +
                     H_py_Sinv11 * Sinv_01 * Sinv_11);

    // ∂²G/∂π_y∂Σ_{11}
    H_G_mixed[5] = -(H_py_Sinv00 * Sinv_01 * Sinv_01 +
                     H_py_Sinv01 * Sinv_01 * Sinv_11 * 2.0f +
                     H_py_Sinv11 * Sinv_11 * Sinv_11);
}

/**
 * @brief Complete conversion function that handles all derivative types
 *
 * @param dG_dSinv Input first derivative w.r.t. Σ^{-1}
 * @param H_G_Sinv Input second derivative w.r.t. Σ^{-1}
 * @param H_G_mixed_inv Input mixed derivatives w.r.t. Σ^{-1}
 * @param conic_2d The covariance matrix as vec3
 * @param dG_dSigma Output first derivative w.r.t. Σ
 * @param H_G_Sigma Output second derivative w.r.t. Σ
 * @param H_G_mixed Output mixed derivatives w.r.t. Σ
 */
__device__ __forceinline__ void convert_all_inverse_derivatives(
    const vec3& dG_dSinv,
    const float* H_G_Sinv,
    const float* H_G_mixed_inv,
    const vec3& conic_2d,
    vec3& dG_dSigma,
    float* H_G_Sigma,
    float* H_G_mixed
) {
    // Compute inverse covariance matrix
    mat2 Sigma_inv = compute_inverse_covariance_2d(conic_2d);

    // Convert first derivative
    dG_dSigma = convert_first_derivative_inverse_to_covariance(dG_dSinv, Sigma_inv);

    // Convert second derivative
    convert_second_derivative_inverse_to_covariance(dG_dSinv, H_G_Sinv, Sigma_inv, H_G_Sigma);

    // Convert mixed derivatives
    convert_mixed_derivatives_inverse_to_covariance(H_G_mixed_inv, Sigma_inv, H_G_mixed);
}

// END Sigma_inv

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
 * @brief Kernel 1: Position derivatives with proper basis transformation
 */
__global__ void compute_position_derivatives_kernel(
    const int num_gaussians,
    const float *__restrict__ dc_dcSH_totals,
    const float *__restrict__ dc_dG_totals,
    const vec2 *__restrict__ dG_dmean2d_totals,
    const vec3 *__restrict__ dG_dSigma_inv_totals,
    const vec3 *__restrict__ H_G_mean2d_totals,
    const float *__restrict__ H_G_sigma_inv_totals,
    const float *__restrict__ H_G_mixed_inv_totals,
    const mat3x2 *__restrict__ jacobians,
    const float *__restrict__ dSigma_dp,        // dSigma_dpx, dSigma_dpy, dSigma_dpz,
    const mat3 *__restrict__ dc_sh_dp,        // 3x3 for RGB channels
    const float *__restrict__ H_mean2d_dp,
    const mat3 *__restrict__ H_c_sh_p,        // 3x3 Hessian for RGB
    const float *__restrict__ H_Sigma_dp,
    const vec3* __restrict__ conics_2d,
    const vec3 *__restrict__ p_k,
    const vec3 *__restrict__ dL_dc,
    const vec3 *__restrict__ d2L_dc2,
    const vec3 *__restrict__ campos,
    // OUTPUTS:
    vec2 *__restrict__ d_L_vk,
    mat2 *__restrict__ H_L_vk
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;
    const vec3 camera_pos = campos[0];
    // Load and convert inverse derivatives
    const vec3 dG_dSigma_inv = dG_dSigma_inv_totals[g_idx];
    const float* H_G_sigma_inv = H_G_sigma_inv_totals + g_idx * 6;
    const float* H_G_mixed_inv = H_G_mixed_inv_totals + g_idx * 6;
    int base_S    = g_idx * 9;
    int base_Hm   = g_idx * 18;
    // ── unpack ∂Σ/∂p into three 2×2 mats ────────────────────────────
    //    row‑major order: mat2(ptr[0],ptr[1], ptr[3],ptr[4])
    mat2 dSigma_dpx = glm::make_mat2(&dSigma_dp[base_S + 0]);  // col 0
    mat2 dSigma_dpy = glm::make_mat2(&dSigma_dp[base_S + 3]);  // col 1
    mat2 dSigma_dpz = glm::make_mat2(&dSigma_dp[base_S + 6]);  // col 2

    // ── unpack ∂²Σ/∂p² the same way ─────────────────────────────────
    mat2 H_Sigma_pxx = glm::make_mat2(&H_Sigma_dp[base_S + 0]);
    mat2 H_Sigma_pxy = glm::make_mat2(&H_Sigma_dp[base_S + 3]);
    mat2 H_Sigma_pyy = glm::make_mat2(&H_Sigma_dp[base_S + 6]);

    // ── unpack ∂²π/∂p² into two vec3 rows ────────────────────────────
    //    first 3 floats = ∂²π_x/∂p², next 3 = ∂²π_y/∂p²
    mat3 H_pi_px = glm::make_mat3(&H_mean2d_dp[base_Hm + 0]);  // row 0
    mat3 H_pi_py = glm::make_mat3(&H_mean2d_dp[base_Hm + 9]);  // r
    vec3 dG_dSigma;
    float H_G_sigma[6];
    float H_G_mixed[6];
    convert_all_inverse_derivatives(
        dG_dSigma_inv, H_G_sigma_inv, H_G_mixed_inv,
        conics_2d[g_idx], dG_dSigma, H_G_sigma, H_G_mixed
    );

    // Construct orthonormal basis (corrected implementation)
    vec3 r_k = normalize(camera_pos - p_k[g_idx]);
    vec3 world_up(0, 1, 0);
    vec3 r_cross_up = cross(r_k, world_up);

    // Handle case where r_k is parallel to world_up
    if (length(r_cross_up) < 1e-6f) {
        world_up = vec3(1, 0, 0);
        r_cross_up = cross(r_k, world_up);
    }

    vec3 u_y = normalize(world_up - dot(world_up, r_k) * r_k);
    vec3 u_x = normalize(cross(r_k, u_y));
    mat2x3 U_k = {u_x, u_y};

    // Compute derivatives w.r.t. position - sum over RGB channels properly
    const float dc_dc_sh = dc_dcSH_totals[g_idx];
    const float dc_dG = dc_dG_totals[g_idx];
    const vec2 dG_dmean2d = dG_dmean2d_totals[g_idx];
    const vec3 H_G_mean2d = H_G_mean2d_totals[g_idx];

    // Position gradient computation with RGB channel handling
    vec3 final_grad = compute_dc_dpk_local_rgb(
        dc_dc_sh, dc_dG, dG_dmean2d, dG_dSigma,
        dc_sh_dp[g_idx], jacobians[g_idx],
        dSigma_dpx, dSigma_dpy, dSigma_dpz
    );

    // Position Hessian computation
    mat3 final_hessian = compute_d2c_dpk2_local(
        dc_dc_sh, dc_dG, dG_dmean2d, dG_dSigma,
        H_G_mean2d, H_G_sigma, H_G_mixed,
        H_c_sh_p[g_idx], H_pi_px, H_pi_py, jacobians[g_idx],
        H_Sigma_pxx, H_Sigma_pxy, H_Sigma_pyy,
        dSigma_dpx, dSigma_dpy, dSigma_dpz
    );

    // Transform to reduced coordinates
    vec2 dc_dvk = {dot(U_k[0], final_grad), dot(U_k[1], final_grad)};
    mat2 d2c_dvk2 = mat2(
        dot(U_k[0], final_hessian * U_k[0]), dot(U_k[0], final_hessian * U_k[1]),
        dot(U_k[1], final_hessian * U_k[0]), dot(U_k[1], final_hessian * U_k[1])
    );

    vec2 dL_dv;
    mat2 H_v;
    chain_attribute(dL_dc[g_idx],d2L_dc2[g_idx],dc_dvk,d2c_dvk2,dL_dv,H_v);
    d_L_vk[g_idx] = dL_dv;
    H_L_vk[g_idx] = H_v;
    //d_c_vk[g_idx] = dc_dvk;
    //H_c_vk[g_idx] = d2c_dvk2;
}

/**
 * @brief Kernel 2: Scale derivatives with eigenvalue decomposition
 */
__global__ void compute_scale_derivatives_kernel(
    const int num_gaussians,
    const float *__restrict__ dc_dG_totals,
    const vec3 *__restrict__ dG_dSigma_inv_totals,
    const float *__restrict__ H_G_sigma_inv_totals,
    const mat3x2 *__restrict__ jacobians,
    const float *__restrict__ viewmats, // [C, 4, 4]
    const float *__restrict__ quats,    // [N, 4]
    const vec3* __restrict__ conics_2d,
    const vec3* __restrict__ dL_dc,
    const vec3* __restrict__ d2L_dc,
    // OUTPUTS:
    vec2 *__restrict__ dL_lambda,
    mat2 *__restrict__ H_L_dlambda
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;

    // Convert inverse derivatives
    const vec3 dG_dSigma_inv = dG_dSigma_inv_totals[g_idx];
    const float* H_G_sigma_inv = H_G_sigma_inv_totals + g_idx * 6;

    vec3 dG_dSigma;
    float H_G_sigma[6];
    float dummy_mixed[6]; // Not used in scale computation
    convert_all_inverse_derivatives(
        dG_dSigma_inv, H_G_sigma_inv, nullptr,
        conics_2d[g_idx], dG_dSigma, H_G_sigma, dummy_mixed
    );

    const float dc_dG = dc_dG_totals[g_idx];
    const vec3 cov_2d = conics_2d[g_idx];
    // Eigenvalue decomposition
    float lambda_min, lambda_max;
    vec2 v_min, v_max;
    eigen_decomposition_2d(
        cov_2d.x, cov_2d.y, cov_2d.z,
        lambda_min, lambda_max, v_min, v_max
    );
    mat2 V = {{v_min.x, v_max.x}, {v_min.y, v_max.y}};
    mat3 W_k = mat3(
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

    quats += g_idx * 4;
    mat3 R_k = quat_to_rotmat(glm::make_vec4(quats));
    //const mat2x3 T_k = T_matrices[g_idx];
    const mat2x3 T_k = V * jacobians[g_idx]* W_k * R_k;


    // Compute ∂Σₖ/∂λₖ
    mat2 dSigma_dlambda[2];
    dSigma_dlambda[0] = glm::outerProduct(V[0], V[0]);
    dSigma_dlambda[1] = glm::outerProduct(V[1], V[1]);

    // Compute ∂Σₖ/∂sₖ
    mat2 dSigma_dsk[3];
    for (int i = 0; i < 3; ++i) {
        dSigma_dsk[i] = dSigma_dlambda[0] * T_k[i][0] + dSigma_dlambda[1] * T_k[i][1];
    }

    // Compute gradient ∂c/∂sₖ
    vec3 grad_s(0.0f);
    for (int i = 0; i < 3; ++i) {
        grad_s[i] = dG_dSigma.x * dSigma_dsk[i][0][0] +
                    2.0f * dG_dSigma.z * dSigma_dsk[i][1][0] +
                    dG_dSigma.y * dSigma_dsk[i][1][1];
    }

    // Compute Hessian ∂²c/∂sₖ²
    mat3 hessian_s(0.0f);
    compute_scaling_hessian(dc_dG, dG_dSigma, H_G_sigma, dSigma_dsk, hessian_s);

    // Transform to eigenvalue coordinates
    mat2 M;
    M[0][0] = glm::dot(T_k[0], T_k[0]);
    M[0][1] = glm::dot(T_k[0], T_k[1]);
    M[1][0] = M[0][1];
    M[1][1] = glm::dot(T_k[1], T_k[1]);
    mat2 M_inv = inverse(M);

    // Gradient in eigenvalue space
    vec2 grad_lambda;
    vec2 Tgs;
    Tgs.x = glm::dot(T_k[0], grad_s);
    Tgs.y = glm::dot(T_k[1], grad_s);
    grad_lambda = M_inv * Tgs;

    // Hessian in eigenvalue space
    mat2 H_temp;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            H_temp[i][j] =
                T_k[i].x * (hessian_s[0][0] * T_k[j].x + hessian_s[0][1] * T_k[j].y + hessian_s[0][2] * T_k[j].z) +
                T_k[i].y * (hessian_s[1][0] * T_k[j].x + hessian_s[1][1] * T_k[j].y + hessian_s[1][2] * T_k[j].z) +
                T_k[i].z * (hessian_s[2][0] * T_k[j].x + hessian_s[2][1] * T_k[j].y + hessian_s[2][2] * T_k[j].z);
        }
    }
    mat2 H_c_lambda = M_inv * (H_temp * M_inv);

    vec2 dL_dlambda;
    mat2 H_lambda;
    chain_attribute(dL_dc[g_idx],d2L_dc[g_idx],grad_lambda,H_c_lambda,dL_dlambda,H_lambda);
    dL_lambda[g_idx] = dL_dlambda;
    H_L_dlambda[g_idx] = H_lambda;
}

/**
 * @brief Kernel 3: Rotation derivatives
 */
__global__ void compute_rotation_derivatives_kernel(
    const int num_gaussians,
    const float *__restrict__ dc_dG_totals,
    const vec3 *__restrict__ dG_dSigma_inv_totals,
    const float *__restrict__ H_G_sigma_inv_totals,
    const mat2 *__restrict__ dSigma_dtheta_inputs,
    const mat2 *__restrict__ d2Sigma_dtheta2_inputs,
    const vec3* __restrict__ conics_2d,
    const vec3* __restrict__ dL_dc,
    const vec3* __restrict__ H_L_dc,
    // OUTPUTS:
    float *__restrict__ dL_dtheta,
    float *__restrict__ d2L_dtheta2
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;

    // Convert inverse derivatives
    const vec3 dG_dSigma_inv = dG_dSigma_inv_totals[g_idx];
    const float* H_G_sigma_inv = H_G_sigma_inv_totals + g_idx * 6;

    vec3 dG_dSigma;
    float H_G_sigma[6];
    float dummy_mixed[6];
    convert_all_inverse_derivatives(
        dG_dSigma_inv, H_G_sigma_inv, nullptr,
        conics_2d[g_idx], dG_dSigma, H_G_sigma, dummy_mixed
    );

    const float dc_dG = dc_dG_totals[g_idx];
    const mat2 dSigma_dtheta = dSigma_dtheta_inputs[g_idx];
    const mat2 d2Sigma_dtheta2 = d2Sigma_dtheta2_inputs[g_idx];

    float grad_theta, hess_theta;
    compute_rotation_derivatives(
        dc_dG, dG_dSigma, H_G_sigma,
        dSigma_dtheta, d2Sigma_dtheta2,
        grad_theta, hess_theta
    );
    float S1 = dL_dc[g_idx].x + dL_dc[g_idx].y + dL_dc[g_idx].z;
    float S2 = H_L_dc[g_idx].x + H_L_dc[g_idx].y + H_L_dc[g_idx].z;
    //float H_L_theta;
    //float dL_theta;
    dL_dtheta[g_idx] = S1 * grad_theta;
    d2L_dtheta2[g_idx] = S2 * grad_theta * grad_theta + S1 * hess_theta;
}


/**
 * @brief Kernel: Computes first and second-order derivatives for opacity.
 * Chains the loss derivatives with the opacity derivatives.
 */
__global__ void compute_opacity_derivatives_kernel(
    const int num_gaussians,
    const vec3 *__restrict__ dc_dopac,  // Input: ∂c_RASTERIZED / ∂σ_k
    const vec3 *__restrict__ dL_dc,     // Input: ∂L / ∂c_RASTERIZED
    const vec3 *__restrict__ H_L_dc,    // Input: Diagonal of ∂²L / ∂c_RASTERIZED²
    // --- OUTPUTS ---
    float *__restrict__ dL_dopac,  // Output: ∂L / ∂σ_k
    float *__restrict__ H_L_dopac  // Output: ∂²L / ∂σ_k²
)
{
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;

    // Fetch the derivatives for the current Gaussian
    const vec3 grad_L_c = dL_dc[g_idx];
    const vec3 hess_L_c_diag = H_L_dc[g_idx];
    const vec3 grad_c_opac = dc_dopac[g_idx];

    // 1. --- Calculate the final Gradient: ∂L/∂σ_k ---
    // The chain rule for a scalar output is the dot product of the input gradients.
    // ∂L/∂σ = Σ (∂L/∂c_χ) * (∂c_χ/∂σ)
    dL_dopac[g_idx] = dot(grad_L_c, grad_c_opac);

    // 2. --- Calculate the final Hessian: ∂²L/∂σ_k² ---
    // The second term of the Hessian chain rule is zero because ∂²c/∂σ² = 0.
    // We only need to compute: Σ (∂c_χ/∂σ)² * (∂²L/∂c_χ²)
    float final_hessian = 0.0f;
    final_hessian += grad_c_opac.x * grad_c_opac.x * hess_L_c_diag.x;
    final_hessian += grad_c_opac.y * grad_c_opac.y * hess_L_c_diag.y;
    final_hessian += grad_c_opac.z * grad_c_opac.z * hess_L_c_diag.z;
    H_L_dopac[g_idx] = final_hessian;
}


/**
 * @brief Kernel 5: Color derivatives with proper RGB channel handling.
 */
__global__ void compute_color_derivatives_kernel(
    const int num_gaussians,
    const int num_sh_coeffs,   // e.g., 16 for SH degree 3
    const float *__restrict__ dcRAST_dck, // Input: ∂c_RASTERIZED / ∂c_k (size: num_gaussians * num_sh_coeffs)
    const vec3 *__restrict__ dL_dc,         // Input: ∂L / ∂c_RASTERIZED
    const vec3 *__restrict__ H_L_dc,        // Input: Diagonal of ∂²L / ∂c_RASTERIZED²
    // --- OUTPUTS ---
    vec3 *__restrict__ dL_dcoeffs,  // Output: ∂L / ∂c_k as a vec3 (size: num_gaussians * num_sh_coeffs)
    vec3 *__restrict__ H_L_dcoeffs // Output: Diagonal of ∂²L / ∂c_k² (size: num_gaussians * num_sh_coeffs)
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;

    // Fetch the derivatives of the loss w.r.t the final rasterized color.
    const vec3 loss_grad = dL_dc[g_idx];
    const vec3 loss_hess_diag = H_L_dc[g_idx];

    // Pre-calculate the sum of the diagonal components of the loss Hessian.
    //const float S_H = loss_hess_diag.x + loss_hess_diag.y + loss_hess_diag.z;

    // Process each of the 16 SH coefficients for the current Gaussian
    for (int m = 0; m < num_sh_coeffs; ++m) {
        const int coeff_idx = g_idx * num_sh_coeffs + m;

        // Fetch the pre-computed derivative: ∂c_RASTERIZED / ∂c_{k,m}
        const float dc_dck_m = dcRAST_dck[coeff_idx];

        // 1. --- Calculate the final Gradient for all channels ---
        vec3 final_grad;
        vec3 final_hess;
        // Small loop over RGB channels (can be unrolled by the compiler)
        #pragma unroll
        for (int c = 0; c < 3; ++c) { // 3 is just CDIM but such approach for now
            final_grad[c] = loss_grad[c] * dc_dck_m;
            final_hess[c] = dc_dck_m * dc_dck_m * loss_hess_diag[c];
        }
        dL_dcoeffs[coeff_idx] = final_grad;

        // 2. --- Calculate the final Hessian diagonal ---
        // This is calculated once per coefficient, not per channel.
        H_L_dcoeffs[coeff_idx] = final_hess;
    }
}


void launch_compute_position_derivatives_kernel(
    const int num_gaussians,
    const at::Tensor dc_dcSH_totals,
    const at::Tensor dc_dG_totals,
    const at::Tensor dG_dmean2d_totals,
    const at::Tensor dG_dSigma_inv_totals,
    const at::Tensor H_G_mean2d_totals,
    const at::Tensor H_G_sigma_inv_totals,
    const at::Tensor H_G_mixed_inv_totals,
    const at::Tensor jacobians,
    const at::Tensor dSigma_dp,
    const at::Tensor dc_sh_dp,
    const at::Tensor H_mean2d_dp,
    const at::Tensor H_c_sh_p,
    const at::Tensor H_Sigma_dp,
    const at::Tensor conics_2d,
    const at::Tensor p_k,
    const at::Tensor dL_dc,
    const at::Tensor H_L_dc,
    const at::Tensor camera_pos,
    at::Tensor d_L_vk,
    at::Tensor H_L_vk
) {
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;
    compute_position_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        dc_dcSH_totals.data_ptr<float>(),
        dc_dG_totals.data_ptr<float>(),
        reinterpret_cast<const vec2*>(dG_dmean2d_totals.data_ptr<float>()),
        reinterpret_cast<const vec3*>(dG_dSigma_inv_totals.data_ptr<float>()),
        reinterpret_cast<const vec3*>(H_G_mean2d_totals.data_ptr<float>()),
        H_G_sigma_inv_totals.data_ptr<float>(),
        H_G_mixed_inv_totals.data_ptr<float>(),
        reinterpret_cast<const mat3x2*>(jacobians.data_ptr<float>()),
        dSigma_dp.data_ptr<float>(),
        reinterpret_cast<const mat3*>(dc_sh_dp.data_ptr<float>()),
        H_mean2d_dp.data_ptr<float>(),
        reinterpret_cast<const mat3*>(H_c_sh_p.data_ptr<float>()),
        H_Sigma_dp.data_ptr<float>(),
        reinterpret_cast<const vec3*>(conics_2d.data_ptr<float>()),
        reinterpret_cast<const vec3*>(p_k.data_ptr<float>()),
        reinterpret_cast<vec3*>(dL_dc.data_ptr<float>()),
        reinterpret_cast<vec3*>(H_L_dc.data_ptr<float>()),
        reinterpret_cast<const vec3*>(camera_pos.data_ptr<float>()),
        reinterpret_cast<vec2*>(d_L_vk.data_ptr<float>()),
        reinterpret_cast<mat2*>(H_L_vk.data_ptr<float>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_compute_scale_derivatives_kernel(
    const int num_gaussians,
    const at::Tensor dc_dG_totals,
    const at::Tensor dG_dSigma_inv_totals,
    const at::Tensor H_G_sigma_inv_totals,
    const at::Tensor jacobians,
    const at::Tensor viewmats,
    const at::Tensor quats,
    const at::Tensor conics_2d,
    const at::Tensor dL_dc,
    const at::Tensor H_L_dc,
    at::Tensor dc_dlambda,
    at::Tensor d2c_dlambda2
) {
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;
    compute_scale_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        dc_dG_totals.data_ptr<float>(),
        reinterpret_cast<const vec3*>(dG_dSigma_inv_totals.data_ptr<float>()),
        H_G_sigma_inv_totals.data_ptr<float>(),
        reinterpret_cast<const mat3x2*>(jacobians.data_ptr<float>()),
        viewmats.data_ptr<float>(),
        quats.data_ptr<float>(),
        reinterpret_cast<const vec3*>(conics_2d.data_ptr<float>()),
        reinterpret_cast<vec3*>(dL_dc.data_ptr<float>()),
        reinterpret_cast<vec3*>(H_L_dc.data_ptr<float>()),
        reinterpret_cast<vec2*>(dc_dlambda.data_ptr<float>()),
        reinterpret_cast<mat2*>(d2c_dlambda2.data_ptr<float>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_compute_rotation_derivatives_kernel(
    const int num_gaussians,
    const at::Tensor dc_dG_totals,
    const at::Tensor dG_dSigma_inv_totals,
    const at::Tensor H_G_sigma_inv_totals,
    const at::Tensor dSigma_dtheta_inputs,
    const at::Tensor d2Sigma_dtheta2_inputs,
    const at::Tensor conics_2d,
    const at::Tensor dL_dc,
    const at::Tensor H_L_dc,
    at::Tensor dL_dtheta,
    at::Tensor d2L_dtheta2
) {
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;
    compute_rotation_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        dc_dG_totals.data_ptr<float>(),
        reinterpret_cast<const vec3*>(dG_dSigma_inv_totals.data_ptr<float>()),
        H_G_sigma_inv_totals.data_ptr<float>(),
        reinterpret_cast<const mat2*>(dSigma_dtheta_inputs.data_ptr<float>()),
        reinterpret_cast<const mat2*>(d2Sigma_dtheta2_inputs.data_ptr<float>()),
        reinterpret_cast<const vec3*>(conics_2d.data_ptr<float>()),
        reinterpret_cast<const vec3*>(dL_dc.data_ptr<float>()),
        reinterpret_cast<const vec3*>(H_L_dc.data_ptr<float>()),
        dL_dtheta.data_ptr<float>(),
        d2L_dtheta2.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void launch_compute_opacity_derivatives_kernel(
    const int num_gaussians,
    const at::Tensor dc_dopac,
    const at::Tensor dL_dc,
    const at::Tensor H_L_dc,
    at::Tensor dL_dopac,
    at::Tensor H_L_dopac
)
{
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;
    compute_opacity_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        reinterpret_cast<const vec3*>(dc_dopac.data_ptr<float>()),
        reinterpret_cast<vec3*>(dL_dc.data_ptr<float>()),
        reinterpret_cast<vec3*>(H_L_dc.data_ptr<float>()),
        dL_dopac.data_ptr<float>(),
        H_L_dopac.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch_compute_color_derivatives_kernel(
    const int num_gaussians,
    const int num_coeffs,
    const at::Tensor dcRAST_dck,
    const at::Tensor dL_dc,
    const at::Tensor H_L_dc,
    at::Tensor dL_dcolor,
    at::Tensor H_L_dcolor
) {
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;
    compute_color_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        num_coeffs,
        dcRAST_dck.data_ptr<float>(),
        reinterpret_cast<const vec3*>(dL_dc.data_ptr<float>()),
        reinterpret_cast<const vec3*>(H_L_dc.data_ptr<float>()),
        reinterpret_cast<vec3*>(dL_dcolor.data_ptr<float>()),
        reinterpret_cast<vec3*>(H_L_dcolor.data_ptr<float>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


} // namespace gsplat_newton

