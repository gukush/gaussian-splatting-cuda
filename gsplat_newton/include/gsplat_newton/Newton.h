// =================================================================================
// Newton.h
// This file defines the interface for the local Newton optimization's backward and update steps.
// =================================================================================
#pragma once

#include <ATen/core/Tensor.h>
#include <tuple>
#include "core/local_newton_context.hpp" // Include the user's context struct definition

namespace gsplat {

/**
 * @brief Performs the backward, assembly, solve, and update stages of a local Newton step.
 *
 * This function assumes the forward pass has already been run and the LocalNewtonContext
 * is populated with the necessary projection derivatives. It will modify the provided
 * Gaussian parameter tensors in-place.
 *
 * @param context A populated LocalNewtonContext struct from the forward pass.
 * @param means The 3D means of the Gaussians to be updated [N, 3].
 * @param scales The scales of the Gaussians to be updated [N, 3].
 * @param quats The rotations (quaternions) of the Gaussians to be updated [N, 4].
 * @param opacities The opacities of the Gaussians to be updated [N, 1].
 * @param sh_coeffs The Spherical Harmonics coefficients to be updated [N, K, 3].
 * @param dL_d_color_img The gradient of the loss w.r.t. the rendered pixel colors [C, H, W, 3].
 * @param H_L_color_img The Hessian of the loss w.r.t. the rendered pixel colors [C, H, W, 3, 3] or diagonal.
 * @param render_alphas The alpha values from the forward rasterization pass [C, H, W, 1].
 * @param last_ids The index of the last contributing Gaussian per pixel [C, H, W].
 * @param tile_offsets The tile offsets for the intersection data structure [C, tile_h, tile_w].
 * @param flatten_ids The flattened intersection indices [n_isects].
 * @param image_width The width of the rendered image.
 * @param image_height The height of the rendered image.
 */
void local_newton_backward_and_update(
    const gsplat_newton::LocalNewtonContext& context,
    at::Tensor& means,
    at::Tensor& scales,
    at::Tensor& quats,
    at::Tensor& opacities,
    at::Tensor& sh_coeffs,
    const at::Tensor& dL_d_color_img,
    const at::Tensor& H_L_color_img, // Assuming this is passed in
    const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    uint32_t image_width,
    uint32_t image_height
);

// Forward declaration for the unified CUDA kernel launcher
void launch_solve_and_update_all_attributes_kernel(
    const at::Tensor& dL_d_pos,
    const at::Tensor& H_L_pos,
    const at::Tensor& dL_d_scale,
    const at::Tensor& H_L_scale,
    const at::Tensor& dL_d_rot,
    const at::Tensor& H_L_rot,
    const at::Tensor& dL_d_opacity,
    const at::Tensor& H_L_opacity,
    const at::Tensor& dL_d_color,
    const at::Tensor& H_L_color, // Placeholder for color Hessian
    // Tensors from context needed for backprojection
    const at::Tensor& U_k_bases, // Basis for position update
    const at::Tensor& T_k_matrices, // Basis for scale update
    // Gaussian parameters to be updated (in-out)
    at::Tensor& means,
    at::Tensor& scales,
    at::Tensor& quats,
    at::Tensor& opacities,
    at::Tensor& sh_coeffs
);

} // namespace gsplat