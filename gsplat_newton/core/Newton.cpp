#include "gsplat_newton/Newton.h"
#include "Common.h"
#include "gsplat_newton/kernels.hpp" // User's wrappers
//#include "utils/cuda_utils.cuh"
#include <ATen/TensorUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include "core/splat_data.hpp"

namespace gsplat {

void local_newton_backward(
    const LocalNewtonContext& context,
    SplatData& gaussian_model,
    uint32_t image_width,
    uint32_t image_height
) {

    at::Tensor& means = gaussian_model.get_means();
    at::Tensor& scales = gaussian_model.get_scaling();
    at::Tensor& quats = gaussian_model.get_rotation();
    at::Tensor& opacities = gaussian_model.get_opacity();
    at::Tensor& sh_coeffs = gaussian_model.get_shs();
    const at::Tensor& render_alphas = context.render_alphas;
    const at::Tensor& last_ids = context.last_ids;
    const at::Tensor& tile_offsets = context.tile_offsets;
    const at::Tensor& flatten_ids = context.flatten_ids;
    const at::Tensor& dL_d_color_img = context.dL_d_color_img;
    const at::Tensor& H_L_color_img = context.H_L_color_img;
    DEVICE_GUARD(means);
    // Input checks
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(opacities);
    CHECK_INPUT(sh_coeffs);
    CHECK_INPUT(dL_d_color_img);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    // --- Stage 1: Backward Pass through Rasterizer ---
    // Computes per-Gaussian aggregated derivatives from image-space loss derivatives.
    auto intermediate_derivs = gsplat_newton::compute_intermediate_derivatives_bwd(
        context.means2d,
        context.conics,
        sh_coeffs.slice(1, 0, 1).squeeze(1), // Use DC term as representative color
        opacities,
        at::nullopt, // backgrounds
        at::nullopt, // masks
        image_width,
        image_height,
        16, // tile_size
        tile_offsets,
        flatten_ids,
        render_alphas,
        last_ids,
        dL_d_color_img,
        at::nullopt, // v_render_alphas (assuming not needed or combined in dL_d_color_img)
        means.size(0)
    );

    auto& dc_dcSH_totals = std::get<0>(intermediate_derivs);
    auto& dc_dG_totals = std::get<1>(intermediate_derivs);
    auto& dG_dmean2d_totals = std::get<2>(intermediate_derivs);
    auto& dG_dSigma_totals = std::get<3>(intermediate_derivs);
    auto& H_G_mean2d_totals = std::get<4>(intermediate_derivs);
    auto& H_G_sigma_totals = std::get<5>(intermediate_derivs);
    auto& H_G_mixed_totals = std::get<6>(intermediate_derivs);
    auto& v_opac = std::get<7>(intermediate_derivs);

    // --- Stage 2: "Backward pass" for Spherical Harmonics
    //
    auto sh_outputs = launch_spherical_harmonics_LN_kernel(
        context.sh_degree,
        context.view_dirs,
        context.coeffs,
        dc_dcSH_totals
    );
    auto& dcRAST_dck = std::get<0>(sh_outputs);
    auto& dcRAST_dr = std::get<1>(sh_outputs);
    auto& H_cRAST_r = std::get<2>(sh_outputs);
    auto chained_outputs = launch_chain_rule_color_position_kernel(
        means,
        viewmats.slice(0,0,1).inverse().slice(1,3,4).squeeze(), //camera_pos ??????
        dcRAST_dr,
        H_cRAST_r
    );
    auto& d_cSH_dp = std::get<0>(chained_outputs);
    auto& H_cSH_dp = std::get<1>(chained_outputs);
    // --- Stage 2: Assemble Local Newton Systems ---
    // Combines projection derivatives (from context) and rasterization derivatives (from above).
    auto newton_systems = gsplat_newton::assemble_newton_derivatives(
        // Inputs from intermediate derivatives
        dc_dcSH_totals,
        dc_dG_totals,
        dG_dmean2d_totals,
        dG_dSigma_totals,
        H_G_mean2d_totals,
        H_G_sigma_totals,
        H_G_mixed_totals,
        dc_dG_totals, // Using dc_dG for opacity part as placeholder
        dG_dSigma_totals, // Using dG_dSigma for opacity part as placeholder
        H_G_sigma_totals, // Using H_G_sigma for opacity part as placeholder
        // Projection derivatives from context
        context.d_mean2d_dp, // jacobians
        context.d_Sigma_dp.select(2, 0), // dSigma_dpx
        context.d_Sigma_dp.select(2, 1), // dSigma_dpy
        context.d_Sigma_dp.select(2, 2), // dSigma_dpz
        d_cSH_dp, // dc_sh_dp
        context.H_mean2d_dp.select(2, 0), // H_pi_px
        context.H_mean2d_dp.select(2, 1), // H_pi_py
        H_cSH_dp, // H_c_sh_p
        context.H_Sigma_dp.select(2, 0), // H_Sigma_pxx
        context.H_Sigma_dp.select(2, 1), // H_Sigma_pxy
        context.H_Sigma_dp.select(2, 2), // H_Sigma_pyy
        context.d_Sigma_dtheta,
        context.H_Sigma_dtheta,
        context.T_matrices,
        context.conics,
        means, // p_k
        viewmats.slice(0,0,1).inverse().slice(1,3,4).squeeze() // camera_pos
    );

    context.dL_d_pos      = std::get<0>(newton_systems);
    context.H_L_pos       = std::get<1>(newton_systems);
    context.dL_d_scale    = std::get<2>(newton_systems);
    context.H_L_scale     = std::get<3>(newton_systems);
    context.dL_d_rot      = std::get<4>(newton_systems);
    context.H_L_rot       = std::get<5>(newton_systems);
    context.dL_d_opacity  = std::get<6>(newton_systems);
    context.H_L_opacity   = std::get<7>(newton_systems);
    // Note: assemble_newton_derivatives returns 8 tensors, color is not separate.
    // We will need to compute color derivatives separately or assume they are part of another tensor.
    // For now, creating placeholder tensors for color update.
    context.dL_d_color = torch::zeros({means.size(0), 3}, means.options());
    context.H_L_color = torch::zeros({means.size(0), 3, 3}, means.options());

}

} // namespace gsplat

solve_and_update(
    const gsplat_newton::LocalNewtonContext& context,
    SplatData& gaussian_model,
    uint32_t image_width,
    uint32_t image_height
) {

    auto& means = gaussian_model.get_means();
    auto& scales = gaussian_model.get_scaling();
    auto& quats = gaussian_model.get_rotation();
    auto& opacities = gaussian_model.get_opacity();
    auto& sh_coeffs = gaussian_model.get_shs();
    DEVICE_GUARD(means);
    // Input checks
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(opacities);
    CHECK_INPUT(sh_coeffs);
    auto& dL_d_pos = context.dL_d_pos;
    auto& H_L_pos = context.H_L_pos;
    auto& dL_d_scale = context.dL_d_scale;
    auto& H_L_scale = context.H_L_scale;
    auto& dL_d_rot = context.dL_d_rot;
    auto& H_L_rot = context.H_L_rot;
    auto& dL_d_opacity = context.dL_d_opacity;
    auto& H_L_opacity = context.H_L_opacity;
    auto& dL_d_color = context.dL_d_color;
    auto& H_L_color = context.H_L_color;
    launch_solve_and_update_all_attributes_kernel(
        dL_d_pos, H_L_pos,
        dL_d_scale, H_L_scale,
        dL_d_rot, H_L_rot,
        dL_d_opacity, H_L_opacity,
        dL_d_color, H_L_color,
        context.d_mean2d_dp, // Basis U_k is implicitly defined by this jacobian
        context.T_matrices,
        means, scales, quats, opacities, sh_coeffs // Pass by reference to update in-place
    );
}