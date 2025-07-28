#pragma once
#include "gsplat_newton/local_newton_context.hpp" // Your main context struct
#include "core/splat_data.hpp"
#include <torch/torch.h>
#include <tuple>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include "Common.h"

// Forward declarations for core data structures if needed
namespace gs {
    struct RenderOutput;
//    class SplatData;
}

namespace at {
class Tensor;
}


using gsplat::vec2;

namespace gsplat_newton {

/**
 * @brief Namespace for all CUDA kernel launcher functions related to the
 * local Newton optimization method for 3D Gaussian Splatting.
 */

// ========================================================================
// 1. PROJECTION KERNELS
// ========================================================================

/**
 * @brief Performs the fused forward projection of 3D Gaussians to 2D and simultaneously
 * computes all first and second-order derivatives of the projection outputs
 * (2D mean, 2D covariance, view direction) with respect to the 3D position.
 * @param means3D The 3D centers of the Gaussians.
 * @param rotations The quaternion rotations of the Gaussians.
 * @param scales The scales of the Gaussians.
 * @param viewmat The world-to-camera transformation matrix.
 * @param K The camera intrinsics matrix.
 * @param image_width The width of the output image.
 * @param image_height The height of the output image.
 * @param context A reference to the LocalNewtonContext struct where all output
 * tensors (forward results and derivatives) will be stored.
 */
/*
void projection_fwd_with_derivatives(
    const torch::Tensor means3D,
    const torch::Tensor rotations,
    const torch::Tensor scales,
    const torch::Tensor viewmat,
    const torch::Tensor K,
    uint32_t image_width,
    uint32_t image_height,
    LocalNewtonContext& context // Output parameter
);*/
/*
std::tuple<
    at::Tensor,  // radii
    at::Tensor,  // means2d
    at::Tensor,  // depths
    at::Tensor,  // conics
    at::Tensor,  // compensations
    at::Tensor,  // jacobians
    at::Tensor,  // H_mean_y
    at::Tensor,  // H_mean_x
    at::Tensor,  // dSigma_dx
    at::Tensor,  // dSigma_dy
    at::Tensor,  // dSigma_dz
    at::Tensor,  // H_Sigma
    at::Tensor,  // dr_dp
    at::Tensor>  // d2r_dp2_compact
projection_ewa_3dgs_fused_fwd_LN(
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
    const bool calc_compensations,
    const gsplat::CameraModelType camera_model
);
*/
std::tuple<at::Tensor, at::Tensor> compute_covariance_derivatives(
    const at::Tensor means,       // [N, 3] world positions
    const at::Tensor quats,       // [N, 4] quaternions [w, x, y, z]
    const at::Tensor scales,      // [N, 3] scale parameters
    const at::Tensor view_matrix // [4, 4] view matrix (camera to world)
);


// ========================================================================
// 2. SPHERICAL HARMONICS (SH) KERNELS
// ========================================================================

/**
 * @brief Evaluates spherical harmonics to compute colors and also calculates the
 * first and second derivatives of the SH color with respect to the
 * normalized view direction (r_k).
 * @param degree The maximum degree of SH to evaluate.
 * @param view_dirs The normalized view directions.
 * @param sh_coeffs The spherical harmonic coefficients.
 * @param context A reference to the context struct to store the output derivatives
 * (∂c̃/∂r and ∂²c̃/∂r²).
 * @return A tensor containing the computed RGB colors.
 */
torch::Tensor sh_fwd_with_derivatives(
    int degree,
    const torch::Tensor view_dirs,
    const torch::Tensor sh_coeffs,
    LocalNewtonContext& context // Output parameter for derivatives
);

/**
 * @brief Applies the chain rule to compute the derivatives of the SH color with
 * respect to the 3D position (p_k). It combines ∂c̃/∂r (from SH kernels)
 * and ∂r/∂p (from projection kernels).
 * @param context A reference to the context struct, which contains the input
 * derivatives and will be updated with the final ∂c̃/∂p and ∂²c̃/∂p².
 */
void chain_rule_sh_position(LocalNewtonContext& context);

std::tuple<at::Tensor, at::Tensor, at::Tensor,  at::Tensor> spherical_harmonics_LN(
    const uint32_t      degrees_to_use,
    const at::Tensor   dirs,       // [..., 3]
    const at::Tensor   coeffs,     // [..., K, 3]
    const at::Tensor v_colors
);

std::pair<at::Tensor,at::Tensor> chain_rule_color_position(
    const at::Tensor p_k,             // [...,3]
    const at::Tensor radii,
    const at::Tensor camera_center,   // [3]
    const at::Tensor color_dir_grad,  // [...,3]
    const at::Tensor color_dir_hess   // [...,6]
);

// ========================================================================
// 3. LOSS & RASTERIZATION BACKWARD KERNELS
// ========================================================================

/**
 * @brief Computes the SSIM and L2 loss between the rendered and ground truth images,
 * and crucially, calculates their first (∂L/∂c) and second (∂²L/∂c²) derivatives
 * with respect to the rendered pixel colors.
 * @param rendered_image The image produced by the forward rasterizer.
 * @param gt_image The ground truth image.
 * @param lambda_ssim The weight for the SSIM component of the loss.
 * @return A tuple containing: the total loss (scalar Tensor), the first derivative
 * tensor (∂L/∂c), and the second derivative tensor (∂²L/∂c²).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_loss_and_derivatives(
    const torch::Tensor rendered_image,
    const torch::Tensor gt_image,
    float lambda_ssim
);

// Returns tuple(ssim_map, mu1, mu2, sigma1_sq, sigma2_sq, sigma12)
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> fusedssim_LN(
    float C1,
    float C2,
    torch::Tensor img1,
    torch::Tensor img2,
    bool train
);

// Backward SSIM
// Returns tuple(dL_dimg1, d2L_dimg1)
std::tuple<
    torch::Tensor, torch::Tensor
> fusedssim_backward_LN(
    float C1,
    float C2,
    torch::Tensor img1,
    torch::Tensor img2,
    torch::Tensor dL_dmap,
    torch::Tensor mu1_map,
    torch::Tensor mu2_map,
    torch::Tensor s1_map,
    torch::Tensor s2_map,
    torch::Tensor s12_map
);

/**
 * @brief The "backward" rasterization pass. It aggregates the contributions of all pixels
 * to the intermediate derivatives for each Gaussian. This replaces the standard
 * autograd backward pass for the rasterizer.
 * @param context The context struct to be populated with aggregated derivatives.
 * @param render_output The output from the forward rasterization pass.
 * @param dL_dcolor The first derivative of the loss w.r.t. rendered colors.
 * @param d2L_dcolor2 The second derivative of the loss w.r.t. rendered colors.
 */
void aggregate_intermediate_derivatives(
    LocalNewtonContext& context,
    const gs::RenderOutput& render_output,
    const torch::Tensor dL_dcolor,
    const torch::Tensor d2L_dcolor2
);


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
compute_intermediate_derivatives_bwd(
    // Gaussian parameters
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // forward outputs
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    // gradients of outputs
    const at::Tensor dL_dcIMG,
    const at::Tensor H_L_dcIMG,
    // output shapes
    const int64_t N
);


// ========================================================================
// 4. ASSEMBLY & UPDATE KERNELS
// ========================================================================
std::tuple<at::Tensor, at::Tensor, at::Tensor,  at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor > // this row is temporary!!!
assemble_derivatives_split(
    // Intermediate derivatives
    const at::Tensor dL_dcSH_totals,
    const at::Tensor dL_dG_totals,
    const at::Tensor dL_dmean2d_totals,
    const at::Tensor dL_dconic_totals,
    const at::Tensor H_L_mean2d_totals,
    const at::Tensor H_L_conic_totals,
    const at::Tensor H_L_mixedinv_totals,
    const at::Tensor radii,
    const at::Tensor Ks,
    const at::Tensor covars,
    const at::Tensor viewmat,
    const at::Tensor dLcolor_dp,
    const at::Tensor H_Lcolor_dp,
    const at::Tensor dSigma_dtheta_inputs,
    const at::Tensor H_Sigma_dtheta_inputs,
    const at::Tensor quats,
    const at::Tensor conics_2d,
    const at::Tensor p_k,
    const at::Tensor scales,
    const at::Tensor camera_pos,
    const at::Tensor dL_dck
//    const at::Tensor dL_dc,
//    const at::Tensor H_L_dc
);


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
    const gsplat::CameraModelType camera_model,
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
);


void launch_compute_covariance_derivatives_kernel(
    const at::Tensor quat,        // [N, 4] quaternions
    const at::Tensor scale,       // [N, 3] scales
    const at::Tensor view_matrix, // [4, 4] view matrix
    const at::Tensor position,    // [N, 3] positions
    at::Tensor dSigma_dtheta,     // [N, 2, 2] output first derivatives
    at::Tensor d2Sigma_dtheta2    // [N, 2, 2] output second derivatives
);

void launch_spherical_harmonics_LN_kernel(
    const uint32_t      degrees_to_use,
    const at::Tensor   dirs,       // [..., 3]
    const at::Tensor   coeffs,     // [..., K, 3]
    const at::Tensor   v_colors,   // dL / dc_SH
    //outputs
    at::Tensor         v_coeffs,   // dc_RAST / dc_attribute
    at::Tensor         H_coeffs,   // d2c_RAST / dc_attribute^2 (diagonal terms only)
    at::Tensor         v_dir,      // [..., 3]     (dL / dr)
    at::Tensor         H_dir       // [..., 6]     (d2L / dr2)
);

void launch_chain_rule_color_position_kernel(
    const at::Tensor p_k,             // [...,3]
    const at::Tensor radii,
    const at::Tensor camera_center,   // [3]
    const at::Tensor color_dir_grad,  // [...,3]
    const at::Tensor color_dir_hess,  // [...,6]
    at::Tensor       color_pos_grad,  // [...,3]
    at::Tensor       color_pos_hess   // [...,6]
);

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
    const at::Tensor scales,
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
);

template <uint32_t CDIM>
void launch_compute_intermediate_derivatives_kernel(
    const bool packed,
    // --- Forward Pass Inputs ---
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // --- Forward Pass Outputs ---
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    // --- Loss Gradients (ADDED) ---
    const at::Tensor dL_dcIMG,
    const at::Tensor H_L_dcIMG,
    // --- OUTPUTS (per-Gaussian) ---
    at::Tensor dL_dcSH,
    at::Tensor H_L_dcSH,  // Fixed name from H_L_cSH
    at::Tensor dL_dG,
    at::Tensor dL_dmean2d,
    at::Tensor dL_dconic,
    at::Tensor H_L_mean2d,
    at::Tensor H_L_conic,   // Fixed name from H_L_sigma
    at::Tensor H_L_mixed,
    at::Tensor dL_dopac,
    at::Tensor H_L_dopac    // ADDED
);



void launch_fusedssim_backward_LN_kernel(
    int64_t B, int64_t CH, int64_t H, int64_t W,
    float C1, float C2,
    const float* img1,
    const float* img2,
    const float* dL_dmap,
    const float* mu1_map,
    const float* mu2_map,
    const float* s1_map,
    const float* s2_map,
    const float* s12_map,
    float* dL_dimg1,
    float* d2L_dimg1,
    cudaStream_t stream
);

void launch_fusedssim_LN_kernel(
    int64_t B, int64_t CH, int64_t H, int64_t W,
    float C1, float C2,
    const float* img1,
    const float* img2,
    float* ssim_map,
    float* mu1_map,
    float* mu2_map,
    float* s1_map,
    float* s2_map,
    float* s12_map,
    bool train,
    cudaStream_t stream
);


// Newton

void local_newton_backward(
    const LocalNewtonContext& context,
    at::Tensor means,
    at::Tensor scales,
    at::Tensor quats,
    at::Tensor opacities,
    at::Tensor sh_coeffs,
    const at::Tensor dL_d_color_img,
    const at::Tensor H_L_color_img, // Assuming this is passed in
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    uint32_t image_width,
    uint32_t image_height
);

void local_newton_backward(
    LocalNewtonContext& context,
    SplatData& gaussian_model,
    uint32_t image_width,
    uint32_t image_height,
    bool print
);

void solve_and_update(
    const LocalNewtonContext& context,
    SplatData& gaussian_model,
    uint32_t image_width,
    uint32_t image_height
);

void launch_solve_and_update_all_attributes_kernel(
    const at::Tensor dL_d_pos,
    const at::Tensor H_L_pos,
    const at::Tensor dL_d_scale,
    const at::Tensor H_L_scale,
    const at::Tensor dL_d_rot,
    const at::Tensor H_L_rot,
    const at::Tensor dL_d_opacity,
    const at::Tensor H_L_opacity,
    const at::Tensor dL_d_color,
    const at::Tensor H_L_color,
    const at::Tensor U_k_bases,
    const at::Tensor T_k_matrices,
    const at::Tensor view_dirs,
    const at::Tensor radii,
    at::Tensor means, at::Tensor scales, at::Tensor quats,
    at::Tensor opacities, at::Tensor sh_coeffs
);

} // namespace gsplat_newton