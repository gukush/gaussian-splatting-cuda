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
 * @brief Helper function to compute the inverse covariance matrix with numerical stability
 *
 * @param conic_2d The 2D covariance matrix as vec3 {Σ_{00}, Σ_{11}, Σ_{01}}
 * @return mat2 The inverse covariance matrix Σ^{-1}
 */
__device__ __forceinline__ mat2 compute_inverse_covariance_2d(const vec3& conic_2d) {
    const float det = conic_2d.x * conic_2d.y - conic_2d.z * conic_2d.z;

    // Check for near-singular matrix
    if (fabsf(det) < 1e-6f) {
        // Fallback to regularized inverse
        const float reg = 1e-4f;
        const float reg_det = (conic_2d.x + reg) * (conic_2d.y + reg) - conic_2d.z * conic_2d.z;
        const float inv_det = 1.0f / reg_det;

        mat2 Sigma_inv;
        Sigma_inv[0][0] = (conic_2d.y + reg) * inv_det;
        Sigma_inv[1][1] = (conic_2d.x + reg) * inv_det;
        Sigma_inv[1][0] = Sigma_inv[0][1] = -conic_2d.z * inv_det;
        return Sigma_inv;
    }

    const float inv_det = 1.0f / det;
    mat2 Sigma_inv;
    Sigma_inv[0][0] = conic_2d.y * inv_det;
    Sigma_inv[1][1] = conic_2d.x * inv_det;
    Sigma_inv[1][0] = Sigma_inv[0][1] = -conic_2d.z * inv_det;
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
    if (H_G_mixed_inv != nullptr) {
        convert_mixed_derivatives_inverse_to_covariance(H_G_mixed_inv, Sigma_inv, H_G_mixed);
    }
}


/**
 * @brief New kernel to perform all derivative conversions once per Gaussian
 */
__global__ void convert_inverse_derivatives_kernel(
    const int num_gaussians,
    const vec3 *__restrict__ dG_dSigma_inv_totals,
    const float *__restrict__ H_G_sigma_inv_totals,
    const float *__restrict__ H_G_mixed_inv_totals,
    const vec3* __restrict__ conics_2d,
    // OUTPUTS:
    vec3 *__restrict__ dG_dSigma_totals,      // Converted first derivatives
    float *__restrict__ H_G_sigma_totals,     // Converted second derivatives (6 components each)
    float *__restrict__ H_G_mixed_totals      // Converted mixed derivatives (6 components each)
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;

    // Load inverse derivatives
    const vec3 dG_dSigma_inv = dG_dSigma_inv_totals[g_idx];
    const float* H_G_sigma_inv = H_G_sigma_inv_totals + g_idx * 6;
    const float* H_G_mixed_inv = H_G_mixed_inv_totals + g_idx * 6;

    // Output arrays for this Gaussian
    vec3& dG_dSigma = dG_dSigma_totals[g_idx];
    float* H_G_sigma = H_G_sigma_totals + g_idx * 6;
    float* H_G_mixed = H_G_mixed_totals + g_idx * 6;

    // Perform the conversion once
    convert_all_inverse_derivatives(
        dG_dSigma_inv, H_G_sigma_inv, H_G_mixed_inv,
        conics_2d[g_idx], dG_dSigma, H_G_sigma, H_G_mixed
    );
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


/*
__device__ __forceinline__ void compute_rotation_derivatives(
    const float dc_dG,
    const vec3& dL_dSigma,
    const float* H_L_sigma,
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
*/
// Include the file with the helper functions (mat2x3, mat3, compute_..._totals, etc.)
//#include "NewtonHelpers.cu"


// END Sigma_inv
// chains ∂L/∂π_k and ∂L/∂Σ_k down to ∂L/∂p_k (3‑vector)
__device__ __forceinline__ vec3 compute_dL_dpk_local(
    const vec2& dL_dpi,         // { ∂L/∂π_x, ∂L/∂π_y }
    const vec3& dL_dSigma,      // { ∂L/∂Σ₀₀, ∂L/∂Σ₀₁, ∂L/∂Σ₁₁ }
    const mat3x2& J_pi,         // ∂π_k/∂p_k (2×3 Jacobian)
    const mat2& dSigma_dpx,     // ∂Σ_k/∂p_x  (2×2)
    const mat2& dSigma_dpy,     // ∂Σ_k/∂p_y
    const mat2& dSigma_dpz      // ∂Σ_k/∂p_z
) {
    vec3 grad = vec3(0.0f);
    // — chain through π_k —
    //   grad[p] += ∂L/∂π_i * ∂π_i/∂p
    grad.x += dL_dpi.x * J_pi[0][0] + dL_dpi.y * J_pi[0][1];
    grad.y += dL_dpi.x * J_pi[1][0] + dL_dpi.y * J_pi[1][1];
    grad.z += dL_dpi.x * J_pi[2][0] + dL_dpi.y * J_pi[2][1];

    // — chain through Σ_k —
    //   contract ∂L/∂Σ : ∂Σ/∂p  taking into account symmetry Σ₀₁==Σ₁₀
    auto contractSigma = [&](const mat2& dS_dp){
      return dL_dSigma.x * dS_dp[0][0]
           + 2.0f         * dL_dSigma.y * dS_dp[1][0]
           + dL_dSigma.z * dS_dp[1][1];
    };
    grad += vec3(
      contractSigma(dSigma_dpx),
      contractSigma(dSigma_dpy),
      contractSigma(dSigma_dpz)
    );
    return grad;
}

// ——— replaces compute_d2c_dpk2_local ———
// chains the (π,Σ)-level Hessian blocks plus intrinsic curvature into
// H_L_p   = ∂²L/∂p_k²   (3×3)
__device__ __forceinline__
mat3 compute_H_L_p_local(
    // — first‐order weights —
    const vec2&   dL_dpi,            // ∂L/∂π_u, ∂L/∂π_v
    const float   dL_dSigma_xx,      // ∂L/∂Σ₁₁
    const float   dL_dSigma_xy,      // ∂L/∂Σ₁₂
    const float   dL_dSigma_yy,      // ∂L/∂Σ₂₂
    const float   dL_dSigma_xz,      // ∂L/∂Σ₁₃
    const float   dL_dSigma_yz,      // ∂L/∂Σ₂₃
    const float   dL_dSigma_zz,      // ∂L/∂Σ₃₃

    // — second‐order π‐blocks —
    const mat2&   H_pi_pi,           // H_{ππ} (2×2)
    const mat2x3& H_pi_Sigma,        // H_{πΣ} (2→Σ)
    const mat3&   H_Sigma_Sigma,     // H_{ΣΣ} (Σ→Σ)
    const mat3x2& J_pi,              // ∂π/∂p  (2×3)
    const mat3&   H_pi2_px,          // ∂²π/∂p_x²  (3×3)
    const mat3&   H_pi2_py,          // ∂²π/∂p_x∂p_y (3×3)

    // — first‐order Σ‐blocks (build J_Sigma yourself before) —
    const mat2&   dSigma_dpx,        // ∂Σ/∂p_x  (2→plane)
    const mat2&   dSigma_dpy,        // ∂Σ/∂p_y
    const mat2&   dSigma_dpz,        // ∂Σ/∂p_z

    // — second‐order Σ‐blocks
    const mat3&   H_Sigma2_pxx,      // ∂²Σ/∂p_x²  (3×3)
    const mat3&   H_Sigma2_pxy,      // ∂²Σ/∂p_x∂p_y
    const mat3&   H_Sigma2_pyy,      // ∂²Σ/∂p_y²
    const mat3&   H_Sigma2_pxz,      // ∂²Σ/∂p_x∂p_z
    const mat3&   H_Sigma2_pyz,      // ∂²Σ/∂p_y∂p_z
    const mat3&   H_Sigma2_pzz       // ∂²Σ/∂p_z²
) {
    // — build the little J_Sigma (3 Σ‐entries ← 3 p‐dims) —
    //    row 0 = ∂Σ₁₁/∂p,  row 1 = ∂Σ₂₂/∂p,  row 2 = ∂Σ₁₂/∂p
    mat3 J_Sigma;
    J_Sigma[0] = vec3(dSigma_dpx[0][0],
                      dSigma_dpy[0][0],
                      dSigma_dpz[0][0]);
    J_Sigma[1] = vec3(dSigma_dpx[1][1],
                      dSigma_dpy[1][1],
                      dSigma_dpz[1][1]);
    J_Sigma[2] = vec3(dSigma_dpx[0][1],
                      dSigma_dpy[0][1],
                      dSigma_dpz[0][1]);

    mat3 H(0.0f);

    // 1) (∂π/∂p)ᵀ H_{ππ} (∂π/∂p)
    for(int a=0;a<2;++a) for(int b=0;b<2;++b) {
      float hij = H_pi_pi[a][b];
      for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        H[i][j] += J_pi[i][a] * hij * J_pi[j][b];
    }

    // 2) 2*(∂π/∂p)ᵀ H_{πΣ} (∂Σ/∂p)
    for(int a=0;a<2;++a) for(int b=0;b<3;++b) {
      float hij = H_pi_Sigma[a][b];
      for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        H[i][j] += 2.f * J_pi[j][a] * hij * J_Sigma[b][i];
    }

    // 3) (∂Σ/∂p)ᵀ H_{ΣΣ} (∂Σ/∂p)
    for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
      float hij = H_Sigma_Sigma[a][b];
      for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        H[i][j] += J_Sigma[a][j] * hij * J_Sigma[b][i];
    }

    // 4) intrinsic from ∂²π/∂p² weighted by ∂L/∂π
    for(int i=0;i<3;++i) for(int j=0;j<3;++j){
      H[i][j] += dL_dpi.x * H_pi2_px[i][j]
               + dL_dpi.y * H_pi2_py[i][j];
    }

    // 5) intrinsic from ∂²Σ/∂p² weighted by ∂L/∂Σ
    for(int i=0;i<3;++i) for(int j=0;j<3;++j){
      H[i][j] += dL_dSigma_xx * H_Sigma2_pxx[i][j]
               + dL_dSigma_xy * H_Sigma2_pxy[i][j]
               + dL_dSigma_yy * H_Sigma2_pyy[i][j]
               + dL_dSigma_xz * H_Sigma2_pxz[i][j]
               + dL_dSigma_yz * H_Sigma2_pyz[i][j]
               + dL_dSigma_zz * H_Sigma2_pzz[i][j];
    }

    return H;
}

/**
 * @brief Kernel 1: Position derivatives with proper basis transformation
 * Now also outputs dSigma_dp for use by scale kernel
 */
__global__ void compute_position_derivatives_kernel(
    const int num_gaussians,
    const vec3 *__restrict__ dL_dG_totals,
    const vec2 *__restrict__ dL_dmean2d,
    const vec3 *__restrict__ dL_dSigma_totals,        // Changed: now pre-converted
    const vec3 *__restrict__ H_L_mean2d,
    const float *__restrict__ H_L_sigma_totals,       // Changed: now pre-converted
    const float *__restrict__ H_L_mixed_totals,       // Changed: now pre-converted
    const vec3 *__restrict__ p_k,
    const mat3 *__restrict__ K,                         // [1, 3, 3]
    const mat3 *__restrict__ covars,
    const int32_t* __restrict__ radii,
    const vec3 *__restrict__ campos,
    const vec3 *__restrict__ dLcolor_dp,               // [N,3] gradients from the color path
    const float *__restrict__ H_Lcolor_dp,              // [N, 6]w.r.t position
    // OUTPUTS:
    vec2 *__restrict__ d_L_vk,
    mat2 *__restrict__ H_L_vk,
    // NEW OUTPUT for scale kernel:
    mat2x3 *__restrict__ U_k_bases_out,     // [N,2,3] per‐Gaussian basis
    float *__restrict__ dSigma_dp_out        // [N * 12] - packed dSigma/dp matrices
) {
    __shared__ mat3 K_shared;
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int tid = threadIdx.x;

    // Load K matrix into shared memory (only first 9 threads do this)
    if (tid == 0) {
        K_shared = K[0];
    }
    __syncthreads();
    if (g_idx >= num_gaussians) return;

    if(radii[g_idx*2] <= 0 || radii[g_idx*2+1] <= 0) return;

    const vec3 camera_pos = campos[0];

    // Load pre-converted derivatives (no conversion needed!)
    const vec3 dL_dSigma = dL_dSigma_totals[g_idx];
    //const float* H_L_sigma = H_L_sigma_totals + g_idx * 6;
    //const float* H_L_mixed = H_L_mixed_totals + g_idx * 6;

    vec3 splat_pos = p_k[g_idx];

    mat3 covar = glm::transpose(covars[g_idx]); // because QuatScaleToCovar transposes in row major

    // Construct orthonormal basis
    vec3 r_k = normalize(camera_pos - splat_pos);
    vec3 world_up(0, 1, 0);
    vec3 r_cross_up = cross(r_k, world_up);

    if (length(r_cross_up) < 1e-6f) {
        world_up = vec3(1, 0, 0);
        r_cross_up = cross(r_k, world_up);
    }

    vec3 u_y = normalize(world_up - dot(world_up, r_k) * r_k);
    vec3 u_x = normalize(cross(r_k, u_y));
    mat2x3 U_k = {u_x, u_y};

    // Compute derivatives w.r.t. position
    const vec2 dL_dpi = dL_dmean2d[g_idx];

    // Load Hessian blocks
    mat2 H_pi_pi = mat2(H_L_mean2d[g_idx].x, H_L_mean2d[g_idx].y,
                        H_L_mean2d[g_idx].y, H_L_mean2d[g_idx].z);

    const float* Hmix = H_L_mixed_totals + 6*g_idx;
    const float* Hc = H_L_sigma_totals + 6*g_idx;

    // Unpack H_{πΣ} (2×3) into mat3x2 (3 cols, 2 rows)
    mat3x2 H_pi_Sigma;
    H_pi_Sigma[0][0] = Hmix[0];  // ∂²L/∂πₓ∂Σ₁₁
    H_pi_Sigma[0][1] = Hmix[1];  // ∂²L/∂π_y∂Σ₁₁
    H_pi_Sigma[1][0] = Hmix[3];  // ∂²L/∂πₓ∂Σ₁₂
    H_pi_Sigma[1][1] = Hmix[4];  // ∂²L/∂π_y∂Σ₁₂
    H_pi_Sigma[2][0] = Hmix[2];  // ∂²L/∂πₓ∂Σ₂₂
    H_pi_Sigma[2][1] = Hmix[5];  // ∂²L/∂π_y∂Σ₂₂

    // Unpack H_{ΣΣ} (3×3) into mat3
    mat3 H_Sigma_Sigma;
    H_Sigma_Sigma[0][0] = Hc[0]; // (Σ₁₁,Σ₁₁)
    H_Sigma_Sigma[0][1] = Hc[3]; // (Σ₂₂,Σ₁₁)
    H_Sigma_Sigma[0][2] = Hc[2]; // (Σ₁₂,Σ₁₁)
    H_Sigma_Sigma[1][0] = Hc[3];
    H_Sigma_Sigma[1][1] = Hc[5];
    H_Sigma_Sigma[1][2] = Hc[4];
    H_Sigma_Sigma[2][0] = Hc[2];
    H_Sigma_Sigma[2][1] = Hc[4];
    H_Sigma_Sigma[2][2] = Hc[1];

    // Pinhole projection derivatives and hessians
    const float rz = 1.f / splat_pos.z, rz2 = rz*rz, rz3 = rz2 * rz, rz4 = rz3 * rz;
    const float fx = K_shared[0][0], fy = K_shared[1][1];
    mat3x2 J(
        fx * rz, 0.f,
        0.f,     fy * rz,
        -fx * splat_pos.x * rz2, -fy * splat_pos.y * rz2
    );

    mat3 H_pi_px(0.f);
    H_pi_px[0][2] = H_pi_px[2][0] = -fx * rz2;
    H_pi_px[2][2] = 2.f * fx * splat_pos.x * rz3;

    mat3 H_pi_py(0.f);
    H_pi_py[1][2] = H_pi_py[2][1] = -fy * rz2;
    H_pi_py[2][2] = 2.f * fy * splat_pos.y * rz3;

    const float s_xx = covar[0][0], s_xy = covar[0][1], s_xz = covar[0][2];
    const float s_yy = covar[1][1], s_yz = covar[1][2], s_zz = covar[2][2];

    mat2 dSigma_dpx = (fx * rz2) * mat2(-2.f * s_xz, -s_yz, -s_yz, 0.f);
    mat2 dSigma_dpy = (fy * rz2) * mat2(0.f, -s_xz, -s_xz, -2.f * s_yz);

    float A = fx * (splat_pos.x*s_xz + splat_pos.y*s_yz + splat_pos.z*s_zz);
    float B = fy * (splat_pos.x*s_xy + splat_pos.y*s_yy + splat_pos.z*s_yz);
    mat2 dS_dz;
    dS_dz[0][0] = 2.f * (fx * s_xz + A * fx * splat_pos.x * rz2);
    dS_dz[1][1] = 2.f * (fy * s_yz + B * fy * splat_pos.y * rz2);
    dS_dz[0][1] = dS_dz[1][0] = (fx*s_yz + fy*s_xz + A*fx*splat_pos.y*rz2 + B*fy*splat_pos.x*rz2);
    mat2 dSigma_dpz = -rz2 * dS_dz;

    // Compute Hessian matrices for position derivatives
    glm::mat3x2 dJdx(
        0.f,        0.f,   -fx * rz2,
        0.f,        0.f,         0.f
    );
    glm::mat3x2 dJdy(
        0.f,        0.f,        0.f,
        0.f,        0.f,  -fy * rz2
    );
    glm::mat3x2 dJdz(
        -fx * rz2,     0.f,   2.f * fx * splat_pos.x * rz3,
        0.f,  -fy * rz2,   2.f * fy * splat_pos.y * rz3
    );

    glm::mat3x2 d2Jdxz(
        0.f,        0.f,    2.f * fx * rz3,
        0.f,        0.f,          0.f
    );
    glm::mat3x2 d2Jdyz(
        0.f,        0.f,          0.f,
        0.f,        0.f,   2.f * fy * rz3
    );
    glm::mat3x2 d2Jdzz(
        2.f * fx * rz3,    0.f,   -6.f * fx * splat_pos.x * rz4,
        0.f,   2.f * fy * rz3,  -6.f * fy * splat_pos.y * rz4
    );

    glm::mat2 H_Sigma_pxx = 2.f * (dJdx * covar * glm::transpose(dJdx));
    glm::mat2 H_Sigma_pyy = 2.f * (dJdy * covar * glm::transpose(dJdy));
    glm::mat2 H_Sigma_pxy = dJdx * covar * glm::transpose(dJdy) + dJdy * covar * glm::transpose(dJdx);
    glm::mat2 H_Sigma_pxz = d2Jdxz * covar * glm::transpose(J) + dJdx * covar * glm::transpose(dJdz) + dJdz * covar * glm::transpose(dJdx) + J * covar * glm::transpose(d2Jdxz);
    glm::mat2 H_Sigma_pyz = d2Jdyz * covar * glm::transpose(J) + dJdy * covar * glm::transpose(dJdz) + dJdz * covar * glm::transpose(dJdy) + J * covar * glm::transpose(d2Jdyz);
    glm::mat2 H_Sigma_pzz = d2Jdzz * covar * glm::transpose(J) + 2.f * (dJdz * covar * glm::transpose(dJdz)) + J * covar * glm::transpose(d2Jdzz);

    // Position gradient computation
    vec3 final_grad = compute_dL_dpk_local(
        dL_dpi, dL_dSigma,
        J,
        dSigma_dpx, dSigma_dpy, dSigma_dpz
    );

    // Position Hessian computation (note: fixed the function call to match the signature)
    mat3 final_hessian = compute_H_L_p_local(
        dL_dpi, dL_dSigma.x, dL_dSigma.y, dL_dSigma.z,
        0.f, 0.f, 0.f,  // Assuming these are zero for 2D case
        H_pi_pi, H_pi_Sigma, H_Sigma_Sigma,
        J, H_pi_px, H_pi_py,
        dSigma_dpx, dSigma_dpy, dSigma_dpz,
        H_Sigma_pxx, H_Sigma_pxy, H_Sigma_pyy,
        H_Sigma_pxz, H_Sigma_pyz, H_Sigma_pzz
    );
     // Load and add color gradients and hessians
    // Load 3-element gradient from color path
    const vec3* color_grad_ptr = dLcolor_dp + g_idx * 3;
    vec3 color_grad = *color_grad_ptr;

    // Load 6-element symmetric hessian from color path
    const float* color_hess_ptr = H_Lcolor_dp + g_idx * 6;
    mat3 color_hessian;

    // Reconstruct symmetric 3x3 hessian from 6 unique elements
    // Based on your storage format: [xx, yy, zz, xy, xz, yz]
    color_hessian[0][0] = color_hess_ptr[0];  // xx
    color_hessian[1][1] = color_hess_ptr[1];  // yy
    color_hessian[2][2] = color_hess_ptr[2];  // zz
    color_hessian[0][1] = color_hessian[1][0] = color_hess_ptr[3];  // xy = yx
    color_hessian[0][2] = color_hessian[2][0] = color_hess_ptr[4];  // xz = zx
    color_hessian[1][2] = color_hessian[2][1] = color_hess_ptr[5];  // yz = zy

    final_grad += color_grad;
    final_hessian += color_hessian;
    // Transform to reduced coordinates
    vec2 dL_dv = {dot(U_k[0], final_grad), dot(U_k[1], final_grad)};
    mat2 H_v = mat2(
        dot(U_k[0], final_hessian * U_k[0]), dot(U_k[0], final_hessian * U_k[1]),
        dot(U_k[1], final_hessian * U_k[0]), dot(U_k[1], final_hessian * U_k[1])
    );

    // Store main outputs
    d_L_vk[g_idx] = dL_dv;
    H_L_vk[g_idx] = H_v;
    U_k_bases_out[g_idx] = U_k;
    // NEW: Store dSigma_dp for scale kernel
    // Pack dSigma_dp: 12 floats per Gaussian (3 mat2 matrices = 3 * 4 floats)
    const int base_idx = g_idx * 12;
    float* dSigma_dp_ptr = &dSigma_dp_out[base_idx];

    // Store dSigma_dpx (4 floats)
    dSigma_dp_ptr[0] = dSigma_dpx[0][0];
    dSigma_dp_ptr[1] = dSigma_dpx[0][1];
    dSigma_dp_ptr[2] = dSigma_dpx[1][0];
    dSigma_dp_ptr[3] = dSigma_dpx[1][1];

    // Store dSigma_dpy (4 floats)
    dSigma_dp_ptr[4] = dSigma_dpy[0][0];
    dSigma_dp_ptr[5] = dSigma_dpy[0][1];
    dSigma_dp_ptr[6] = dSigma_dpy[1][0];
    dSigma_dp_ptr[7] = dSigma_dpy[1][1];

    // Store dSigma_dpz (4 floats)
    dSigma_dp_ptr[8] = dSigma_dpz[0][0];
    dSigma_dp_ptr[9] = dSigma_dpz[0][1];
    dSigma_dp_ptr[10] = dSigma_dpz[1][0];
    dSigma_dp_ptr[11] = dSigma_dpz[1][1];
}


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

__device__ __forceinline__
mat3 unpack_H_SigmaSigma(const float* Hc) {
    // Hc = [H11, H22, H12_12, H11_12, H12_22, H12_12] in your layout:
    //   [ Σ₁₁Σ₁₁, Σ₂₂Σ₂₂, Σ₁₂Σ₁₂, Σ₁₁Σ₁₂, Σ₁₂Σ₂₂, Σ₁₂Σ₁₂ ]
    mat3 H;
    // column = wrt Σ₁₁
    H[0][0] = Hc[0];  // (Σ₁₁,Σ₁₁)
    H[0][1] = Hc[3];  // (Σ₂₂,Σ₁₁)? careful: we want symmetry
    H[0][2] = Hc[2];  // (Σ₁₂,Σ₁₁)
    // column = wrt Σ₁₂
    H[1][0] = Hc[3];
    H[1][1] = Hc[5];
    H[1][2] = Hc[4];
    // column = wrt Σ₂₂
    H[2][0] = Hc[2];
    H[2][1] = Hc[4];
    H[2][2] = Hc[1];
    return H;
}


/**
 * @brief Kernel 2: Scale derivatives (now receives dSigma_dp from position kernel)
 */
__global__ void compute_scale_derivatives_kernel(
    const int   num_gaussians,
    const int32_t* __restrict__ radii,
    // — loss‑space derivatives ready to go —
    const vec3* __restrict__ dL_dSigma_totals,   // [N] vec3{∂L/∂Σ₁₁,∂L/∂Σ₁₂,∂L/∂Σ₂₂}
    const float* __restrict__ H_L_conic_totals,   // [N×6] packed HΣΣ as above
    // — geometry inputs (unchanged) —
    const float* __restrict__ viewmats,          // [C×16]
    const float* __restrict__ quats,             // [N×4]
    const vec3*   __restrict__ conics_2d,        // [N] vec3{Σ₁₁,Σ₁₂,Σ₂₂}
    const float* __restrict__ dSigma_dp,         // [N*12] - now passed from position kernel
    // — outputs —
    vec2*  __restrict__ dL_dlambda,  // [N]
    mat2*  __restrict__ H_L_dlambda,   // [N]
    mat2x3* __restrict__ T_matrices     // [N, 2, 3]
) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;
    if (radii[g_idx*2] <= 0 || radii[g_idx*2 + 1] <= 0) return;

    // 1) load the loss-space gradient & Hessian‐block
    vec3 L_Sigma = dL_dSigma_totals[g_idx];
    const float* Hc = H_L_conic_totals + g_idx*6;
    mat3  H_SigmaSigma = unpack_H_SigmaSigma(Hc);

    // Load dSigma_dp (now passed from position kernel)
    constexpr int STRIDE_DS = 12;  // 3 * 4 floats per Gaussian
    int base_S = g_idx * STRIDE_DS;

    mat2 dSigma_dpx = glm::make_mat2(&dSigma_dp[base_S + 0]);
    mat2 dSigma_dpy = glm::make_mat2(&dSigma_dp[base_S + 4]);
    mat2 dSigma_dpz = glm::make_mat2(&dSigma_dp[base_S + 8]);

    // 2) eigen‑decompose Σ
    vec3 cov = conics_2d[g_idx];     // {Σ₁₁, Σ₁₂, Σ₂₂}
    float λmin, λmax;
    vec2  vmin, vmax;
    eigen_decomposition_2d(
        cov.x, cov.y, cov.z,
        λmin, λmax,
        vmin, vmax
    );

    glm::vec3 J_proj[3] = {
      // row0 = ∂Σ₂d₁₁/∂[Σ₃d₁₁,Σ₃d₁₂,Σ₃d₂₂]
      glm::vec3(
        dSigma_dpx[0][0],
        dSigma_dpx[0][1],
        dSigma_dpx[1][1]
      ),
      // row1 = ∂Σ₂d₁₂/∂[…]
      glm::vec3(
        dSigma_dpy[0][0],
        dSigma_dpy[0][1],
        dSigma_dpy[1][1]
      ),
      // row2 = ∂Σ₂d₂₂/∂[…]
      glm::vec3(
        dSigma_dpz[0][0],
        dSigma_dpz[0][1],
        dSigma_dpz[1][1]
      )
    };

    // E2 coefficients: how λₘᵢₙ/ₘₐₓ depend on the 3 diag‑entries
    float a1 = vmin.x*vmin.x,
          b1 = 2.f*vmin.x*vmin.y,
          c1 = vmin.y*vmin.y;
    float a2 = vmax.x*vmax.x,
          b2 = 2.f*vmax.x*vmax.y,
          c2 = vmax.y*vmax.y;
    glm::vec3 col0 = a1 * J_proj[0]
                   + b1 * J_proj[1]
                   + c1 * J_proj[2];
    glm::vec3 col1 = a2 * J_proj[0]
                   + b2 * J_proj[1]
                   + c2 * J_proj[2];
    T_matrices[g_idx] = glm::mat2x3(col0, col1);

    // 3) build the two "direction" vectors gₘᵢₙ, gₘₐₓ in the flattened Σ→ℝ³
    vec3 g_min = { vmin.x*vmin.x,
                   vmin.y*vmin.y,
                   2.f*vmin.x*vmin.y };
    vec3 g_max = { vmax.x*vmax.x,
                   vmax.y*vmax.y,
                   2.f*vmax.x*vmax.y };

    // 4) chain once via the sandwich to get ∂L/∂λ  and  ∂²L/∂λ²
    vec2 grad_lambda;
    grad_lambda.x = dot(L_Sigma, g_min);
    grad_lambda.y = dot(L_Sigma, g_max);

    mat2 H_lambda;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      const vec3& gi = (i==0 ? g_min : g_max);
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        const vec3& gj = (j==0 ? g_min : g_max);
        float sum = 0.f;
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
          #pragma unroll
          for (int q = 0; q < 3; ++q) {
            sum += gi[p] * H_SigmaSigma[p][q] * gj[q];
          }
        }
        H_lambda[i][j] = sum;
      }
    }

    // 5) write out
    dL_dlambda [g_idx] = grad_lambda;
    H_L_dlambda[g_idx] = H_lambda;
}

/**
 * @brief Kernel 3: Rotation derivatives
 */
__global__ void compute_rotation_derivatives_kernel(
    const int num_gaussians,
    const int32_t* __restrict__ radii,
    const float *__restrict__ dL_dG_totals,
    const vec3 *__restrict__ dL_dSigma_totals,        // Changed: now pre-converted
    const float *__restrict__ H_L_sigma_totals,       // Changed: now pre-converted
    const mat2 *__restrict__ dSigma_dtheta_inputs,
    const mat2 *__restrict__ d2Sigma_dtheta2_inputs,
    // OUTPUTS:
    float *__restrict__ dL_dtheta,
    float *__restrict__ d2L_dtheta2
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;
    if (radii[g_idx*2] <= 0 || radii[g_idx*2 + 1] <= 0) return;
    // Load pre-converted derivatives (no conversion needed!)
    // 1) load the loss-space gradient & Hessian‐block
    vec3 L_Sigma = dL_dSigma_totals[g_idx];
    const float* Hc = H_L_sigma_totals + g_idx*6;
    mat3  H_SigmaSigma = unpack_H_SigmaSigma(Hc);


    // Rest of the kernel logic remains exactly the same...
    const float dL_dG = dL_dG_totals[g_idx];
    const mat2 dSigma_dtheta = dSigma_dtheta_inputs[g_idx];
    const mat2 H_Sigma_dtheta = d2Sigma_dtheta2_inputs[g_idx];


    vec3 g_theta = {
       dSigma_dtheta[0][0],          // ∂Σ₁₁/∂θ
       dSigma_dtheta[1][0],          // ∂Σ₁₂/∂θ
       dSigma_dtheta[1][1]           // ∂Σ₂₂/∂θ
    };
    vec3 Q_theta = {
       H_Sigma_dtheta[0][0],        // ∂²Σ₁₁/∂θ²
       H_Sigma_dtheta[1][0],        // ∂²Σ₁₂/∂θ²
       H_Sigma_dtheta[1][1]         // ∂²Σ₂₂/∂θ²
    };

    // 3) chain once: gradient
    float grad_theta = dot(L_Sigma, g_theta);

    // 4) sandwich for Hessian + intrinsic curvature
    //    H_sand = gθᵀ HΣΣ gθ
    float sand = 0.f;
    #pragma unroll
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        sand += g_theta[i] * H_SigmaSigma[i][j] * g_theta[j];
    //    H_intr = LΣᵀ Qθ
    float intr = dot(L_Sigma, Q_theta);
    float hess_theta = sand + intr;

    // 5) write out
    dL_dtheta[g_idx] = grad_theta;
    d2L_dtheta2[g_idx] = hess_theta;
}


/**
 * @brief Kernel 5: Color derivatives with proper RGB channel handling.
 */
__global__ void compute_color_derivatives_kernel(
    const int num_gaussians,
    const int num_sh_coeffs,   // e.g., 16 for SH degree 3
    const int32_t* __restrict__ radii,
    const float *__restrict__ dcRAST_dck, // Input: ∂c_RASTERIZED / ∂c_k (size: num_gaussians * num_sh_coeffs)
    const vec3 *__restrict__ dL_dc,         // Input: ∂L / ∂c_RASTERIZED
    const vec3 *__restrict__ H_L_dc,        // Input: Diagonal of ∂²L / ∂c_RASTERIZED²
    // --- OUTPUTS ---
    vec3 *__restrict__ dL_dcoeffs,  // Output: ∂L / ∂c_k as a vec3 (size: num_gaussians * num_sh_coeffs)
    vec3 *__restrict__ H_L_dcoeffs // Output: Diagonal of ∂²L / ∂c_k² (size: num_gaussians * num_sh_coeffs)
) {
    const int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= num_gaussians) return;
    if (radii[g_idx*2] <= 0 || radii[g_idx*2 + 1] <= 0) return;
    // Fetch the derivatives of the loss w.r.t the final rasterized color.
    const vec3 loss_grad = dL_dc[g_idx];
    const vec3 loss_hess_diag = H_L_dc[g_idx];

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

void launch_assemble_derivatives_kernels(
    const int num_gaussians,
    // Input tensors for conversion
    const at::Tensor conics_2d,
    const at::Tensor campos,
    const at::Tensor viewmats,
    const at::Tensor radii,
    const at::Tensor Ks,
    const at::Tensor covars,
    const at::Tensor quats,
    // ... other input tensors for the main kernels
    const at::Tensor dL_dcSH_totals,
    const at::Tensor dL_dG_totals,
    const at::Tensor dL_dmean2d_totals,
    const at::Tensor dL_dconic_totals,
    const at::Tensor H_L_mean2d_totals,
    const at::Tensor H_L_conic_totals,
    const at::Tensor H_L_mixedinv_totals,
    // REMOVED: jacobians, dSigma_dp, H_Sigma_dp (no longer needed as inputs)
    const at::Tensor p_k,
    const at::Tensor dSigma_dtheta,
    const at::Tensor H_Sigma_dtheta,
    const at::Tensor dLcolor_dp,
    const at::Tensor H_Lcolor_dp,
    // OUTPUTS:
    at::Tensor d_L_vk, // [N, 2]
    at::Tensor H_L_vk, // [N, 3]
    at::Tensor dL_dlambda,  // [N]
    at::Tensor H_L_dlambda,   // [N]
    at::Tensor dL_dtheta, // [N]
    at::Tensor d2L_dtheta2, // [N]
    at::Tensor dL_dcoeffs,  // Output: ∂L / ∂c_k as a vec3 [N, num_sh_coeffs]
    at::Tensor H_L_dcoeffs, // Output: Diagonal of ∂²L / ∂c_k² [N, num_sh_coeffs]
    // temporary outputs
    at::Tensor dL_dSigma,
    at::Tensor H_L_sigma,
    at::Tensor H_L_mixed,
    at::Tensor U_k_bases,
    at::Tensor T_matrices,
    at::Tensor dSigma_dp_temp
) {
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;

    // Create temporary tensor for dSigma_dp
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

    // STEP 1: Convert all inverse derivatives once
    convert_inverse_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        reinterpret_cast<const vec3*>(dL_dconic_totals.data_ptr<float>()),
        H_L_conic_totals.data_ptr<float>(),
        H_L_mixedinv_totals.data_ptr<float>(),
        reinterpret_cast<const vec3*>(conics_2d.data_ptr<float>()),
        reinterpret_cast<vec3*>(dL_dSigma.data_ptr<float>()),
        H_L_sigma.data_ptr<float>(),
        H_L_mixed.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // STEP 2: Position derivatives kernel (now outputs dSigma_dp)
    compute_position_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        reinterpret_cast<const vec3*>(dL_dG_totals.data_ptr<float>()),
        reinterpret_cast<const vec2*>(dL_dmean2d_totals.data_ptr<float>()),
        reinterpret_cast<const vec3*>(dL_dSigma.data_ptr<float>()),
        reinterpret_cast<const vec3*>(H_L_mean2d_totals.data_ptr<float>()),
        H_L_sigma.data_ptr<float>(),
        H_L_mixed.data_ptr<float>(),
        reinterpret_cast<const vec3*>(p_k.data_ptr<float>()),
        reinterpret_cast<const mat3*>(Ks.data_ptr<float>()),
        reinterpret_cast<const mat3*>(covars.data_ptr<float>()),
        radii.data_ptr<int32_t>(),
        reinterpret_cast<const vec3*>(campos.data_ptr<float>()),
        reinterpret_cast<const vec3*>(dLcolor_dp.data_ptr<float>()),
        H_Lcolor_dp.data_ptr<float>(),
        reinterpret_cast<vec2*>(d_L_vk.data_ptr<float>()),
        reinterpret_cast<mat2*>(H_L_vk.data_ptr<float>()),
        // NEW: output for scale kernel
        reinterpret_cast<mat2x3*>(U_k_bases.data_ptr<float>()),
        dSigma_dp_temp.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // STEP 3: Scale derivatives kernel (uses dSigma_dp from position kernel)
    compute_scale_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        radii.data_ptr<int32_t>(),
        reinterpret_cast<const vec3*>(dL_dSigma.data_ptr<float>()),
        H_L_sigma.data_ptr<float>(),
        viewmats.data_ptr<float>(),
        quats.data_ptr<float>(),
        reinterpret_cast<const vec3*>(conics_2d.data_ptr<float>()),
        dSigma_dp_temp.data_ptr<float>(),
        reinterpret_cast<vec2*>(dL_dlambda.data_ptr<float>()),
        reinterpret_cast<mat2*>(H_L_dlambda.data_ptr<float>()),
        reinterpret_cast<mat2x3*>(T_matrices.data_ptr<float>())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // STEP 4: Rotation derivatives kernel (unchanged)
    compute_rotation_derivatives_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_gaussians,
        radii.data_ptr<int32_t>(),
        dL_dG_totals.data_ptr<float>(),
        reinterpret_cast<const vec3*>(dL_dSigma.data_ptr<float>()),
        H_L_sigma.data_ptr<float>(),
        reinterpret_cast<const mat2*>(dSigma_dtheta.data_ptr<float>()),
        reinterpret_cast<const mat2*>(H_Sigma_dtheta.data_ptr<float>()),
        dL_dtheta.data_ptr<float>(),
        d2L_dtheta2.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // The opacity and color kernels don't need the conversion, so they remain unchanged
}
} // namespace gsplat_newton

