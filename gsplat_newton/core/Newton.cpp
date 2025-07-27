#include "core/splat_data.hpp"
#include "gsplat_newton/Newton.h"
#include "Common.h"
#include "gsplat_newton/kernels.hpp" // User's wrappers
//#include "utils/cuda_utils.cuh"
#include <ATen/TensorUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include "core/splat_data.hpp"

namespace gsplat_newton {

void local_newton_backward(
    LocalNewtonContext& context,
    SplatData& gaussian_model,
    uint32_t image_width,
    uint32_t image_height
) {

    auto means = gaussian_model.get_means();
    auto scales = gaussian_model.get_scaling();
    auto quats = gaussian_model.get_rotation();
    auto opacities = gaussian_model.get_opacity();
    auto sh_coeffs = gaussian_model.get_shs();
    const at::Tensor& viewmat = context.viewmat;
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
    auto intermediate_derivs = compute_intermediate_derivatives_bwd(
        context.means2d,
        context.conics,
        context.colors, // Use DC term as representative color
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
        context.dL_d_color_img,
        context.H_L_color_img,
        means.size(0)
    );


    auto dtheta_out = compute_covariance_derivatives(
        means,
        quats,
        scales,
        viewmat
    );

    auto& dSigma_dtheta = std::get<0>(dtheta_out);
    auto& H_Sigma_dtheta = std::get<1>(dtheta_out);

    auto& dL_dcSH_totals = std::get<0>(intermediate_derivs);
    auto& H_L_dcSH_totals = std::get<1>(intermediate_derivs);
    auto& dL_dG_totals = std::get<2>(intermediate_derivs);
    auto& dL_dmean2d_totals = std::get<3>(intermediate_derivs);
    auto& dL_dconic_totals = std::get<4>(intermediate_derivs);
    auto& H_L_mean2d_totals = std::get<5>(intermediate_derivs);
    auto& H_L_conic_totals = std::get<6>(intermediate_derivs);
    auto& H_L_mixedinv_totals = std::get<7>(intermediate_derivs);

    context.dL_d_opacity = std::get<8>(intermediate_derivs);
    context.H_L_opacity = std::get<9>(intermediate_derivs);

    std::cout << "dL_dcSH_totals NaN: " << torch::isnan(dL_dcSH_totals).any().item<bool>() << std::endl;
    std::cout << "dL_dmean2d_totals NaN: " << torch::isnan(dL_dmean2d_totals).any().item<bool>() << std::endl;
    std::cout << "dL_dconic_totals NaN: " << torch::isnan(dL_dconic_totals).any().item<bool>() << std::endl;

    if (torch::isnan(dL_dcSH_totals).any().item<bool>()) {
        std::cout << "ERROR: NaN detected in intermediate derivatives!" << std::endl;
        return; // Early exit to prevent further propagation
    }
    // --- Stage 2: "Backward pass" for Spherical Harmonics
    //
    auto sh_outputs = spherical_harmonics_LN(
        context.sh_degree,
        context.view_dirs,
        context.coeffs,
        dL_dcSH_totals
    );
    context.dL_d_color = std::get<0>(sh_outputs);
    context.H_L_color = std::get<1>(sh_outputs);
    auto& dLcolor_dr = std::get<2>(sh_outputs);
    auto& H_Lcolor_r = std::get<3>(sh_outputs);

    auto chained_outputs = chain_rule_color_position(
        means,
        context.campos, //camera_pos ??????
        dLcolor_dr,
        H_Lcolor_r
    );
    auto& dLcolor_dp = std::get<0>(chained_outputs);
    auto& H_Lcolor_dp = std::get<1>(chained_outputs);
    // --- Stage 2: Assemble Local Newton Systems ---
    // this one does formulas for position, scale and rotation
    // opacity is calculated in bwd of rasterization
    // color is calculated in spherical harmonics LN (dcRAST_dck)
    //
    // Combines projection derivatives (from context) and rasterization derivatives (from above).
        // DEBUG: Check inputs to assembly
    std::cout << "=== Before Assembly ===" << std::endl;
    std::cout << "jacobians NaN: " << torch::isnan(context.d_mean2d_dp).any().item<bool>() << std::endl;
    std::cout << "d_Sigma_dp NaN: " << torch::isnan(context.d_Sigma_dp).any().item<bool>() << std::endl;
    std::cout << "H_mean2d_dp NaN: " << torch::isnan(context.H_mean2d_dp).any().item<bool>() << std::endl;
    std::cout << "H_Sigma_dp NaN: " << torch::isnan(context.H_Sigma_dp).any().item<bool>() << std::endl;

    // Check magnitudes
    std::cout << "jacobians norm: " << context.d_mean2d_dp.norm().item<float>() << std::endl;
    std::cout << "d_Sigma_dp norm: " << context.d_Sigma_dp.norm().item<float>() << std::endl;
    std::cout << "H_mean2d_dp norm: " << context.H_mean2d_dp.norm().item<float>() << std::endl;
    std::cout << "H_Sigma_dp norm: " << context.H_Sigma_dp.norm().item<float>() << std::endl;

    // Check intermediate inputs
    std::cout << "dL_dmean2d_totals norm: " << dL_dmean2d_totals.norm().item<float>() << std::endl;
    std::cout << "dL_dconic_totals norm: " << dL_dconic_totals.norm().item<float>() << std::endl;
    std::cout << "H_L_mean2d_totals norm: " << H_L_mean2d_totals.norm().item<float>() << std::endl;
    std::cout << "H_L_conic_totals norm: " << H_L_conic_totals.norm().item<float>() << std::endl;


    auto newton_systems = assemble_derivatives_split(
        // Inputs from intermediate derivatives
        dL_dcSH_totals,
        dL_dG_totals,
        dL_dmean2d_totals,
        dL_dconic_totals,
        H_L_mean2d_totals,
        H_L_conic_totals,
        H_L_mixedinv_totals,
        // Projection derivatives from context
        context.d_mean2d_dp, // jacobians
        context.viewmat,
        context.d_Sigma_dp, // ∂Σ/∂p  [N,3,3]
        dLcolor_dp,
        context.H_mean2d_dp, // ∂²π/∂p² [N,2,3,3]
        H_Lcolor_dp,
        context.H_Sigma_dp, // ∂²Σ/∂p² [N,3,3]
        dSigma_dtheta,
        H_Sigma_dtheta,
        quats,
        context.conics,
        means, // p_k
        context.campos,// camera_pos
        context.dL_d_color
    );
        // DEBUG: Check outputs from assembly
    std::cout << "=== After Assembly ===" << std::endl;
    auto& dL_d_pos_result = std::get<0>(newton_systems);
    auto& H_L_pos_result = std::get<1>(newton_systems);

    std::cout << "dL_d_pos_result NaN: " << torch::isnan(dL_d_pos_result).any().item<bool>() << std::endl;
    std::cout << "H_L_pos_result NaN: " << torch::isnan(H_L_pos_result).any().item<bool>() << std::endl;
    std::cout << "dL_d_pos_result norm: " << dL_d_pos_result.norm().item<float>() << std::endl;
    std::cout << "H_L_pos_result norm: " << H_L_pos_result.norm().item<float>() << std::endl;

    context.dL_d_pos      = std::get<0>(newton_systems);
    context.H_L_pos       = std::get<1>(newton_systems);
    context.dL_d_scale    = std::get<2>(newton_systems);
    context.H_L_scale     = std::get<3>(newton_systems);
    context.dL_d_rot      = std::get<4>(newton_systems);
    context.H_L_rot       = std::get<5>(newton_systems);
    //context.dL_d_color    = std::get<6>(newton_systems);
    //context.H_L_color     = std::get<7>(newton_systems);
    context.T_matrices    = std::get<8>(newton_systems);

    // We will need to compute color derivatives separately or assume they are part of another tensor.
    // For now, creating placeholder tensors for color update.
    //context.dL_d_color = torch::zeros({means.size(0), 3}, means.options());
    //context.H_L_color = torch::zeros({means.size(0), 3, 3}, means.options());

}





void solve_and_update(
    const LocalNewtonContext& context,
    SplatData& gaussian_model,
    uint32_t image_width,
    uint32_t image_height
) {

    auto means = gaussian_model.get_means();
    auto scales = gaussian_model.get_scaling();
    auto quats = gaussian_model.get_rotation();
    auto opacities = gaussian_model.get_opacity();
    auto sh_coeffs = gaussian_model.get_shs();
    DEVICE_GUARD(means);
    // Input checks
    CHECK_INPUT(means);
    CHECK_INPUT(scales);
    CHECK_INPUT(quats);
    CHECK_INPUT(opacities);
    CHECK_INPUT(sh_coeffs);
    const auto dL_d_pos = context.dL_d_pos;
    const auto H_L_pos = context.H_L_pos;
    const auto dL_d_scale = context.dL_d_scale;
    const auto H_L_scale = context.H_L_scale;
    const auto dL_d_rot = context.dL_d_rot;
    const auto H_L_rot = context.H_L_rot;
    const auto dL_d_opacity = context.dL_d_opacity;
    const auto H_L_opacity = context.H_L_opacity;
    const auto dL_d_color = context.dL_d_color;
    const auto H_L_color = context.H_L_color;
    const auto d_mean2d_dp = context.d_mean2d_dp;
    //const auto T_matrices = context.T_matrices;
    const auto view_dirs = context.view_dirs;
    const auto T_matrices = context.T_matrices;
    CHECK_INPUT(dL_d_pos);
    CHECK_INPUT(H_L_pos);
    CHECK_INPUT(dL_d_scale);
    CHECK_INPUT(H_L_scale);
    CHECK_INPUT(dL_d_rot);
    CHECK_INPUT(H_L_rot);
    CHECK_INPUT(dL_d_opacity);
    CHECK_INPUT(H_L_opacity);
    CHECK_INPUT(dL_d_color);
    CHECK_INPUT(H_L_color);
    CHECK_INPUT(d_mean2d_dp);
    CHECK_INPUT(T_matrices);
    CHECK_INPUT(view_dirs);

    // Check for singular matrices
    auto H_pos_det = H_L_pos.det();
    auto min_det = H_pos_det.min().item<float>();
    auto max_det = H_pos_det.max().item<float>();
    std::cout << "H_L_pos det range: [" << min_det << ", " << max_det << "]" << std::endl;

    if (min_det < 1e-10f) {
        std::cout << "WARNING: Nearly singular Hessian detected!" << std::endl;
    }
    launch_solve_and_update_all_attributes_kernel(
        dL_d_pos, H_L_pos,
        dL_d_scale, H_L_scale,
        dL_d_rot, H_L_rot,
        dL_d_opacity, H_L_opacity,
        dL_d_color, H_L_color,
        d_mean2d_dp, // Basis U_k is implicitly defined by this jacobian
        T_matrices,
        view_dirs,
        means, scales, quats, opacities, sh_coeffs // Pass by reference to update in-place
    );
}

} // namespace gsplat_newton

