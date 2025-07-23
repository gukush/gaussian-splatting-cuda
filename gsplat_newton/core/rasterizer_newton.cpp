#include "core/splat_data.hpp"
#include "gsplat_newton/rasterizer_newton.hpp"
#include "gsplat_newton/local_newton_context.hpp"
#include "gsplat_newton/kernels.hpp" // Your custom Newton kernels
#include "Ops.h"       // For gsplat::rasterize_to_pixels_3dgs_fwd, etc.
#include <torch/torch.h>
#include "gsplat_newton/newton_functions.hpp"
#include "core/camera.hpp"
#include "core/rasterizer.hpp"

namespace gs {
using torch::indexing::None;
using torch::indexing::Slice;
// This is a placeholder for the forward rasterization part of the original pipeline.
// In a real scenario, you would have a non-autograd version of this.
// For now, we'll use the existing forward kernel.
namespace gsplat = ::gsplat;

    inline std::tuple<torch::Tensor> spherical_harmonics(
        int sh_degree,
        const torch::Tensor& dirs,
        const torch::Tensor& coeffs,
        const torch::Tensor& masks,
        LocalNewtonContext& context) {

        // Validate inputs
        TORCH_CHECK((sh_degree + 1) * (sh_degree + 1) <= coeffs.size(-2),
                    "coeffs K dimension must be at least ", (sh_degree + 1) * (sh_degree + 1),
                    ", got ", coeffs.size(-2));
        TORCH_CHECK(dirs.sizes().slice(0, dirs.dim() - 1) == coeffs.sizes().slice(0, coeffs.dim() - 2),
                    "dirs and coeffs batch dimensions must match");
        TORCH_CHECK(dirs.size(-1) == 3, "dirs last dimension must be 3, got ", dirs.size(-1));
        TORCH_CHECK(coeffs.size(-1) == 3, "coeffs last dimension must be 3, got ", coeffs.size(-1));

        if (masks.defined()) {
            TORCH_CHECK(masks.sizes() == dirs.sizes().slice(0, dirs.dim() - 1),
                        "masks shape must match dirs shape without last dimension");
        }

        // Create sh_degree tensor
        auto sh_degree_tensor = torch::tensor({sh_degree},
                                              torch::TensorOptions().dtype(torch::kInt32).device(dirs.device()));

        // Call the autograd function
        auto out = SphericalHarmonicsForward(
            context,
            sh_degree_tensor,
            dirs.contiguous(),
            coeffs.contiguous(),
            masks.defined() ? masks.contiguous() : masks);
        return std::get<0>(out);
    }



RenderOutput rasterize_newton_step(
    Camera& viewpoint_camera,
    const ::SplatData& gaussian_model,
    torch::Tensor& bg_color,
    const torch::Tensor& gt_image,
    float scaling_modifier,
    bool antialiased,
    RenderMode render_mode,
    LocalNewtonContext* context) {

    // ========================================================================
    // 1. INPUT VALIDATION AND SETUP
    // (Preserving checks from rasterizer.cpp and rasterizer_autograd.cpp)
    // ========================================================================

    const int image_height = static_cast<int>(viewpoint_camera.image_height());
    const int image_width = static_cast<int>(viewpoint_camera.image_width());

    // Prepare camera parameters
    auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
    context->viewmat = viewmat;
    TORCH_CHECK(viewmat.dim() == 3 && viewmat.size(0) == 1 && viewmat.size(1) == 4 && viewmat.size(2) == 4,
                "viewmat must be [1, 4, 4], got ", viewmat.sizes());
    TORCH_CHECK(viewmat.is_cuda(), "viewmat must be on CUDA");

    const auto K = viewpoint_camera.K().to(torch::kCUDA);
    TORCH_CHECK(K.dim() == 3 && K.size(0) == 1 && K.size(1) == 3 && K.size(2) == 3,
                "K must be [1, 3, 3], got ", K.sizes());
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");

    // Get and validate Gaussian parameters
    auto means3D = gaussian_model.get_means().contiguous();
    auto opacities = gaussian_model.get_opacity().contiguous();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }
    auto scales = gaussian_model.get_scaling().contiguous();
    auto rotations = gaussian_model.get_rotation().contiguous();
    auto sh_coeffs = gaussian_model.get_shs().contiguous();
    const int sh_degree = gaussian_model.get_active_sh_degree();

    const int N = static_cast<int>(means3D.size(0));
    TORCH_CHECK(means3D.dim() == 2 && means3D.size(1) == 3, "means3D must be [N, 3], got ", means3D.sizes());
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N, "opacities must be [N], got ", opacities.sizes());
    TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == 3, "scales must be [N, 3], got ", scales.sizes());
    TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == N && rotations.size(1) == 4, "rotations must be [N, 4], got ", rotations.sizes());
    TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(0) == N && sh_coeffs.size(2) == 3, "sh_coeffs must be [N, K, 3], got ", sh_coeffs.sizes());

    const int required_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
    TORCH_CHECK(sh_coeffs.size(1) >= required_sh_coeffs, "Not enough SH coefficients. Expected at least ", required_sh_coeffs, " but got ", sh_coeffs.size(1));

    TORCH_CHECK(means3D.is_cuda(), "means3D must be on CUDA");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be on CUDA");
    TORCH_CHECK(rotations.is_cuda(), "rotations must be on CUDA");
    TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA");
    TORCH_CHECK(gt_image.is_cuda(), "gt_image must be on CUDA");

    // Handle background color
    torch::Tensor prepared_bg_color;
    if (bg_color.defined() && bg_color.numel() > 0) {
        prepared_bg_color = bg_color.view({1, -1}).to(torch::kCUDA);
        TORCH_CHECK(prepared_bg_color.size(1) == 3, "bg_color must have 3 channels, got ", prepared_bg_color.size(1));
    }

    // Apply scaling modifier
    auto scaled_scales = scales * scaling_modifier;

    // ========================================================================
    // 2. PROJECTION WITH DERIVATIVES
    // ========================================================================
    // This kernel computes 2D projections and all first and second order
    // derivatives with respect to 3D position, storing them in the context.

    // Prepare output tensors for the projection
    auto means2d = torch::empty({1, N, 2}, means3D.options());
    auto depths = torch::empty({1, N}, means3D.options());
    auto conics = torch::empty({1, N, 3}, means3D.options());
    auto compensations = torch::empty({1, N}, means3D.options());

    // Prepare tensors for Local Newton derivatives
    auto jacobians = torch::empty({1, N, 3, 2}, means3D.options());
    auto H_mean_y = torch::empty({1, N, 3, 3}, means3D.options());
    auto H_mean_x = torch::empty({1, N, 3, 3}, means3D.options());
    auto dSigma_dx = torch::empty({1, N, 2, 2}, means3D.options());
    auto dSigma_dy = torch::empty({1, N, 2, 2}, means3D.options());
    auto dSigma_dz = torch::empty({1, N, 2, 2}, means3D.options());
    auto H_Sigma = torch::empty({1, N, 3, 2, 2}, means3D.options()); // Stores 3 2x2 matrices
    auto dr_dp = torch::empty({1, N, 3, 3}, means3D.options());
    auto d2r_dp2_compact = torch::empty({1, N, 18}, means3D.options().dtype(torch::kFloat));

    const float eps2d = 0.3f;
    const float near_plane = 0.01f;
    const float far_plane = 10000.0f;
    const float radius_clip = 0.0f;
    const bool calc_compensations = antialiased;

    // Call the projection function with all outputs
    auto projection_outputs = gsplat_newton::projection_ewa_3dgs_fused_fwd_LN(
        means3D,
        c10::nullopt, // covars (not used since we're using quats+scales)
        rotations,
        scaled_scales,
        opacities,
        viewmat,
        K,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        gsplat::CameraModelType::PINHOLE // Assuming pinhole camera model
    );

    // Unpack the tuple outputs
    auto radii = std::get<0>(projection_outputs);
    means2d = std::get<1>(projection_outputs);
    depths = std::get<2>(projection_outputs);
    conics = std::get<3>(projection_outputs);
    compensations = std::get<4>(projection_outputs);
    jacobians = std::get<5>(projection_outputs);
    H_mean_y = std::get<6>(projection_outputs);
    H_mean_x = std::get<7>(projection_outputs);
    dSigma_dx = std::get<8>(projection_outputs);
    dSigma_dy = std::get<9>(projection_outputs);
    dSigma_dz = std::get<10>(projection_outputs);
    H_Sigma = std::get<11>(projection_outputs);
    dr_dp = std::get<12>(projection_outputs);
    d2r_dp2_compact = std::get<13>(projection_outputs);

    // Store the outputs in the context if needed
    if (context) {
        context->radii = radii;
        context->means2d = means2d;
        context->depths = depths;
        context->conics = conics;
        //context->compensations = compensations;
        context->d_mean2d_dp = jacobians;
        context->H_mean2d_dp = torch::stack({
        H_mean_x,  // [C, N, 3, 3] for x component
        H_mean_y   // [C, N, 3, 3] for y component
    }, /*dim=*/2);
            context->d_Sigma_dp = torch::stack({
        dSigma_dx,  // [C, N, 2, 2] derivative w.r.t. x
        dSigma_dy,   // [C, N, 2, 2] derivative w.r.t. y
        dSigma_dz    // [C, N, 2, 2] derivative w.r.t. z
    }, /*dim=*/2);
        context->H_Sigma_dp = H_Sigma;
        context->d_r_dp = dr_dp;
        context->H_r_dp = d2r_dp2_compact;
    }

    // ========================================================================
    // 3. SPHERICAL HARMONICS WITH DERIVATIVES
    // ========================================================================
    // These kernels compute color and its derivatives w.r.t. view direction,
    // then use the chain rule to find derivatives w.r.t. 3D position.
    // Step 2: Compute colors from SH
    // First, compute camera position from inverse viewmat
    auto viewmat_inv = torch::inverse(viewmat);
    auto campos = viewmat_inv.index({Slice(), Slice(None, 3), 3}); // [C, 3]

    // Compute directions from camera to each Gaussian
    auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1); // [C, N, 3]

    // Create masks based on radii
    auto masks = (radii > 0).all(-1); // [C, N]
    auto coeffs_flat = sh_coeffs.reshape({-1, sh_coeffs.size(-2), 3});
    // The Python code broadcasts colors from [N, K, 3] to [C, N, K, 3] if needed
    //auto shs = sh_coeffs.unsqueeze(0); // [1, N, K, 3]

    auto colors_tuple = spherical_harmonics(
        sh_degree,
        context->view_dirs, // Input from projection context
        coeffs_flat,
        masks,
        *context            // Populates SH derivatives in context
    );
    auto colors = std::get<0>(colors_tuple);
    // This kernel computes ∂c̃/∂p, ∂²c̃/∂p² using the chain rule - chain rule already computed in the spherical harmonics function
    // gsplat_newton::chain_rule_sh_position(context);

    // Apply standard color transformation for rendering
    colors = torch::clamp_min(colors + 0.5f, 0.0f);


    // Handle different render modes
    torch::Tensor render_colors;
    torch::Tensor final_bg;
    switch (render_mode) {
    case RenderMode::RGB:
        render_colors = colors;
        final_bg = prepared_bg_color.defined() ? prepared_bg_color : torch::Tensor();
        break;
    // Other render modes (D, ED, etc.) can be added here if needed
    default:
        TORCH_CHECK(false, "Unsupported render mode for Newton step.");
    }

    if (!final_bg.defined()) {
        final_bg = at::empty({0}, colors.options());
    }

    // Step 4: Apply opacity with compensations
    torch::Tensor final_opacities;
    if (calc_compensations && compensations.defined() && compensations.numel() > 0) {
        final_opacities = opacities.unsqueeze(0) * compensations;
    } else {
        final_opacities = opacities.unsqueeze(0);
    }
    TORCH_CHECK(final_opacities.is_cuda(), "final_opacities must be on CUDA");

    // ========================================================================
    // 4. FORWARD RASTERIZATION
    // ========================================================================
    // We still need to perform a standard forward pass to get the rendered image.

    // Tiling and Intersection (standard, no derivatives)
    const int tile_size = 16;
    const int tile_width = (image_width + tile_size - 1) / tile_size;
    const int tile_height = (image_height + tile_size - 1) / tile_size;

    const auto isect_results = gsplat::intersect_tile(
        context->means2d, radii, context->depths, {}, {},
        1, tile_size, tile_width, tile_height,
        true);
    const auto flatten_ids = std::get<2>(isect_results);
    auto isect_offsets = gsplat::intersect_offset(std::get<1>(isect_results), 1, tile_width, tile_height);
    const auto tiles_per_gauss = std::get<0>(isect_results);
    const auto isect_ids = std::get<1>(isect_results);

    TORCH_CHECK(tiles_per_gauss.is_cuda(), "tiles_per_gauss must be on CUDA");
    TORCH_CHECK(isect_ids.is_cuda(), "isect_ids must be on CUDA");
    TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
    TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");

    // Forward Rasterization Kernel
    auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
        context->means2d, context->conics, render_colors, opacities.unsqueeze(0),
        final_bg, {}, // No masks
        image_width, image_height, tile_size,
        isect_offsets, flatten_ids);

    auto rendered_image = std::get<0>(raster_results);
    auto rendered_alpha = std::get<1>(raster_results);
    auto last_ids = std::get<2>(raster_results);
    if(context) {
        context->flatten_ids = flatten_ids;
        context->last_ids = last_ids;
        context->tile_offsets = isect_offsets;
        context->render_alphas = rendered_alpha;
    }
    // ========================================================================
    // 5. LOSS & DERIVATIVE AGGREGATION
    // ========================================================================
    // Compute loss and its derivatives, then run the aggregation kernel which
    // acts as the "backward" pass for the rasterizer.
    /*
    // Compute loss and its derivatives w.r.t. pixel colors
    auto [loss, dL_dcolor, d2L_dcolor2] = gsplat_newton::compute_loss_and_derivatives(
        rendered_image,
        gt_image,
        0.2f // lambda_ssim, can be a parameter
    );

    // This kernel aggregates all per-pixel derivatives into per-Gaussian sums
    gsplat_newton::aggregate_intermediate_derivatives(
        context,
        rendered_image,
        rendered_alpha,
        last_ids,
        dL_dcolor,
        d2L_dcolor2
    );
    */
    // ========================================================================
    // 6. PREPARE OUTPUT
    // ========================================================================
    // The context is now fully populated. The next steps would be assembly and
    // update, but we stop here as requested.

    RenderOutput result;
    result.image = torch::clamp(rendered_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);
    result.alpha = rendered_alpha.squeeze(0).permute({2, 0, 1});
    result.depth = torch::Tensor(); // Or compute if needed
    result.means2d = context->means2d.squeeze(0);
    result.depths = context->depths.squeeze(0);
    result.radii = std::get<0>(radii.squeeze(0).max(-1));
    result.visibility = (result.radii > 0);
    result.width = image_width;
    result.height = image_height;

    // The 'context' object is now the main output, containing all the derivatives
    // needed for the assembly and update steps.
    return result;
}

} //namespace gs