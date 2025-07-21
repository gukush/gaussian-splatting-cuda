#include "core/rasterizer.hpp"
#include "core/local_newton_context.hpp" // Your new header
#include "Ops.h"
#include "core/rasterizer_autograd.hpp"
#include <torch/torch.h>
#include <tuple>

namespace gs
{
    // Forward declarations for new CUDA kernels/wrappers you will need to create.
    namespace gsplat
    {
        inline torch::Tensor spherical_harmonics(
        int sh_degree,
        const torch::Tensor& dirs,
        const torch::Tensor& coeffs,
        const torch::Tensor& masks = {}) {

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
        return SphericalHarmonicsFunction::apply(
            sh_degree_tensor,
            dirs.contiguous(),
            coeffs.contiguous(),
            masks.defined() ? masks.contiguous() : masks)[0];
    }


        // This is the C++ wrapper for your new, expanded projection kernel.
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        projection_fwd_with_newton(
            const at::Tensor& means, const at::optional<at::Tensor>& covars,
            const at::optional<at::Tensor>& quats, const at::optional<at::Tensor>& scales,
            const at::optional<at::Tensor>& opacities, const at::Tensor& viewmats,
            const at::Tensor& Ks, uint32_t image_width, uint32_t image_height,
            float eps2d, float near_plane, float far_plane, float radius_clip,
            CameraModelType camera_model);

        // This is the C++ wrapper for your custom rasterization backward kernel.
        void compute_intermediate_derivatives(
            const torch::Tensor& means2d, const torch::Tensor& conics, const torch::Tensor& colors,
            const torch::Tensor& opacities, const at::optional<at::Tensor>& backgrounds,
            uint32_t width, uint32_t height, uint32_t tile_size,
            const torch::Tensor& tile_offsets, const torch::Tensor& flatten_ids,
            const torch::Tensor& render_alphas, const torch::Tensor& last_ids,
            const torch::Tensor& v_render_colors, const torch::Tensor& v_render_alphas,
            // Outputs to populate the context
            LocalNewtonContext& context);

        // *** THIS IS THE MISSING KERNEL YOU NEED TO IMPLEMENT ***
        // It applies the chain rule to get from derivatives w.r.t. view direction (r)
        // to derivatives w.r.t. 3D position (p).
        std::tuple<torch::Tensor, torch::Tensor> sh_chain_rule_derivatives(
            const torch::Tensor& dirs,              // [C, N, 3]
            const torch::Tensor& sh_coeffs,         // [C, N, K, 3]
            const torch::Tensor& v_colors,          // [C, N, 3] (from dL/dc)
            const torch::Tensor& dr_dp,             // [C, N, 3, 3] from projection
            const torch::Tensor& d2r_dp2_compact);  // [C, N, 18] from projection

    } // namespace gsplat

    /**
     * @brief Performs rasterization and computes all intermediate derivatives required for
     * the local Newton optimization method.
     * @return A tuple containing the standard RenderOutput and the populated LocalNewtonContext.
     */
    std::tuple<RenderOutput, LocalNewtonContext> rasterize_with_newton_context(
        Camera& viewpoint_camera,
        const SplatData& gaussian_model,
        torch::Tensor& bg_color,
        const torch::Tensor& gt_image, // Ground truth image is needed to get loss gradients
        float scaling_modifier = 1.0f)
    {
        // --- 1. Initial Setup ---
        const int image_height = static_cast<int>(viewpoint_camera.image_height());
        const int image_width = static_cast<int>(viewpoint_camera.image_width());
        auto viewmat = viewpoint_camera.world_view_transform().to(torch::kCUDA);
        const auto K = viewpoint_camera.K().to(torch::kCUDA);
        auto means3D = gaussian_model.get_means();
        auto opacities = gaussian_model.get_opacity();
        const auto scales = gaussian_model.get_scaling();
        const auto rotations = gaussian_model.get_rotation();
        const auto sh_coeffs = gaussian_model.get_shs();
        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int N = static_cast<int>(means3D.size(0));
        const int C = static_cast<int>(viewmat.size(0));

        LocalNewtonContext context;

        // --- 2. Projection and Derivative Computation ---
        // Call your new, comprehensive projection kernel.
        auto proj_outputs = gsplat::projection_fwd_with_newton(
            means3D, {}, rotations, scales * scaling_modifier, opacities, viewmat, K,
            image_width, image_height, 0.3f, 0.01f, 10000.0f, 0.0f, gsplat::CameraModelType::PINHOLE);

        // Unpack all results into the context struct
        auto radii = std::get<0>(proj_outputs);
        context.means2d = std::get<1>(proj_outputs);
        auto depths = std::get<2>(proj_outputs);
        context.conics = std::get<3>(proj_outputs);
        context.d_mean2d_p = std::get<4>(proj_outputs);
        context.H_mean2d_p = std::get<5>(proj_outputs);
        context.d_Sigma_p = std::get<6>(proj_outputs);
        context.H_Sigma_p = std::get<7>(proj_outputs);
        context.d_r_p = std::get<8>(proj_outputs);
        context.H_r_p_compact = std::get<9>(proj_outputs);
        context.eigenvectors_2d = std::get<10>(proj_outputs);
        context.d_Sigma_theta = std::get<11>(proj_outputs);
        context.H_Sigma_theta = std::get<12>(proj_outputs);
        context.T_matrices = std::get<13>(proj_outputs);
        // Note: You will need to add more outputs from your kernel as needed.

        // --- 3. Spherical Harmonics Forward Pass ---
        auto viewmat_inv = torch::inverse(viewmat);
        auto campos = viewmat_inv.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3), 3});
        auto dirs = means3D.unsqueeze(0) - campos.unsqueeze(1);
        auto masks = (radii > 0).all(-1);
        auto shs = sh_coeffs.unsqueeze(0);
        auto colors = spherical_harmonics(sh_degree, dirs, shs, masks);
        colors = torch::clamp_min(colors + 0.5f, 0.0f);

        // --- 4. Standard Forward Rasterization ---
        const int tile_size = 16;
        const int tile_width = (image_width + tile_size - 1) / tile_size;
        const int tile_height = (image_height + tile_size - 1) / tile_size;
        auto isect_results = gsplat::intersect_tile(context.means2d, radii, depths, {}, {}, C, tile_size, tile_width, tile_height, true);
        auto isect_ids = std::get<1>(isect_results);
        auto flatten_ids = std::get<2>(isect_results);
        auto isect_offsets = gsplat::intersect_offset(isect_ids, C, tile_width, tile_height).reshape({C, tile_height, tile_width});

        auto raster_outputs = gsplat::rasterize_to_pixels_3dgs_fwd(
            context.means2d, context.conics, colors, opacities.unsqueeze(0),
            bg_color, {}, image_width, image_height, tile_size,
            isect_offsets, flatten_ids);

        auto rendered_image = std::get<0>(raster_outputs);
        auto rendered_alpha = std::get<1>(raster_outputs);
        auto last_ids = std::get<2>(raster_outputs);

        // --- 5. Compute Loss and Initial Gradients (dL/dc) ---
        // This would typically be part of your main training loop.
        auto l1_loss = torch::l1_loss(rendered_image.permute({0, 3, 1, 2}), gt_image);
        auto ssim_loss = 1.0f - fused_ssim(rendered_image.permute({0, 3, 1, 2}), gt_image, "valid", true);
        auto loss = (1.0f - 0.2f) * l1_loss + 0.2f * ssim_loss;

        // Get the initial gradients w.r.t. the rendered image and alpha
        auto grads = torch::autograd::grad({loss}, {rendered_image, rendered_alpha}, {}, false, false, true);
        auto v_render_colors = grads[0];
        auto v_render_alphas = grads[1];

        // --- 6. Custom Backward Pass to Aggregate Derivatives ---
        // This kernel populates the rasterization-dependent fields in the context.
        gsplat::compute_intermediate_derivatives(
            context, rendered_alpha, last_ids, v_render_colors, v_render_alphas);

        // --- 7. *** MISSING PIECE: SH Chain Rule Kernel *** ---
        // This kernel calculates the final SH derivatives w.r.t. 3D position (p_k)
        // by combining the view direction derivatives from projection with the SH derivatives
        // w.r.t. view direction.
        auto sh_deriv_outputs = gsplat::sh_chain_rule_derivatives(
            dirs, shs, context.d_c_cSH, context.d_r_p, context.H_r_p_compact);
        context.d_cSH_p = std::get<0>(sh_deriv_outputs);
        context.H_cSH_p = std::get<1>(sh_deriv_outputs);

        // At this point, the `context` object is fully populated and ready for the final assembly kernel.

        // --- 8. Prepare Standard RenderOutput ---
        RenderOutput result;
        result.image = torch::clamp(rendered_image.squeeze(0).permute({2, 0, 1}), 0.0f, 1.0f);
        result.alpha = rendered_alpha.squeeze(0).permute({2, 0, 1});
        result.means2d = context.means2d.squeeze(0);
        result.depths = depths.squeeze(0);
        result.radii = std::get<0>(radii.squeeze(0).max(-1));
        result.visibility = (result.radii > 0);
        result.width = image_width;
        result.height = image_height;

        return {result, context};
    }

} // namespace gs
