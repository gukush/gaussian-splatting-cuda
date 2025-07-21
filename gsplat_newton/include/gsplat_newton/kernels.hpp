#pragma once

#include "core/local_newton_context.hpp" // Your main context struct
#include <torch/torch.h>
#include <tuple>

// Forward declarations for core data structures if needed
namespace gs {
    struct RenderOutput;
    class SplatData;
}

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
void projection_fwd_with_derivatives(
    const torch::Tensor& means3D,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& viewmat,
    const torch::Tensor& K,
    uint32_t image_width,
    uint32_t image_height,
    gs::LocalNewtonContext& context // Output parameter
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
    const torch::Tensor& view_dirs,
    const torch::Tensor& sh_coeffs,
    gs::LocalNewtonContext& context // Output parameter for derivatives
);

/**
 * @brief Applies the chain rule to compute the derivatives of the SH color with
 * respect to the 3D position (p_k). It combines ∂c̃/∂r (from SH kernels)
 * and ∂r/∂p (from projection kernels).
 * @param context A reference to the context struct, which contains the input
 * derivatives and will be updated with the final ∂c̃/∂p and ∂²c̃/∂p².
 */
void chain_rule_sh_position(gs::LocalNewtonContext& context);


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
    const torch::Tensor& rendered_image,
    const torch::Tensor& gt_image,
    float lambda_ssim
);

// Returns tuple(ssim_map, mu1, mu2, sigma1_sq, sigma2_sq, sigma12)
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor
> fusedssim_LN(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    bool train
);

// Backward SSIM
// Returns tuple(dL_dimg1, d2L_dimg1)
std::tuple<
    torch::Tensor, torch::Tensor
> fusedssim_backward_LN(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    torch::Tensor& dL_dmap,
    torch::Tensor& mu1_map,
    torch::Tensor& mu2_map,
    torch::Tensor& s1_map,
    torch::Tensor& s2_map,
    torch::Tensor& s12_map
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
    gs::LocalNewtonContext& context,
    const gs::RenderOutput& render_output,
    const torch::Tensor& dL_dcolor,
    const torch::Tensor& d2L_dcolor2
);


// ========================================================================
// 4. ASSEMBLY & UPDATE KERNELS
// ========================================================================

/**
 * @brief Assembles the final gradients (g) and Hessians (H) for each parameter
 * group by applying the chain rule to the aggregated intermediate derivatives.
 * This kernel runs one thread per Gaussian.
 * @param context The context struct containing all necessary input derivatives.
 * The final g and H for each parameter group will be stored back into the context.
 */
void assemble_newton_derivatives(gs::LocalNewtonContext& context);

/**
 * @brief Solves the local Newton system and updates the 3D positions of the Gaussians.
 * @param context The context struct containing the final position gradients and Hessians.
 * The underlying SplatData within the context will be updated.
 */
void update_position(gs::LocalNewtonContext& context, gs::SplatData& model);

/**
 * @brief Solves the local Newton system and updates the scaling parameters.
 * @param context The context struct containing the final scaling gradients and Hessians.
 * The underlying SplatData will be updated.
 */
void update_scaling(gs::LocalNewtonContext& context, gs::SplatData& model);

/**
 * @brief Solves the local Newton system and updates the rotation quaternions.
 * @param context The context struct containing the final rotation gradients and Hessians.
 * The underlying SplatData will be updated.
 */
void update_rotation(gs::LocalNewtonContext& context, gs::SplatData& model);

/**
 * @brief Solves the local Newton system (with log barrier) and updates the opacities.
 * @param context The context struct containing the final opacity gradients and Hessians.
 * The underlying SplatData will be updated.
 */
void update_opacity(gs::LocalNewtonContext& context, gs::SplatData& model);

/**
 * @brief Solves the local Newton system and updates the SH color coefficients.
 * @param context The context struct containing the final color gradients and Hessians.
 * The underlying SplatData will be updated.
 */
void update_color(gs::LocalNewtonContext& context, gs::SplatData& model);

} // namespace gsplat_newton