#include "gsplat_newton/local_newton_context.hpp"
#include "Ops.h"
#include "gsplat_newton/kernels.hpp"
#include <torch/torch.h>
#include <tuple>


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizationFunctionForward(
    LocalNewtonContext* ctx,
    torch::Tensor means2d,       // [C, N, 2]
    torch::Tensor conics,        // [C, N, 3]
    torch::Tensor colors,        // [C, N, channels] - may include depth
    torch::Tensor opacities,     // [C, N]
    torch::Tensor bg_color,      // [C, channels] - may include depth, can be empty
    torch::Tensor isect_offsets, // [C, tile_height, tile_width]
    torch::Tensor flatten_ids,   // [nnz]
    torch::Tensor settings) {    // [3] containing width, height, tile_size

    // Extract settings
    const auto width = settings[0].item<int>();
    const auto height = settings[1].item<int>();
    const auto tile_size = settings[2].item<int>();

    const int C = static_cast<int>(means2d.size(0));
    const int N = static_cast<int>(means2d.size(1));
    const int channels = static_cast<int>(colors.size(2)); // Get actual channel count

    // Input validation - DO NOT hardcode channels to 3!
    TORCH_CHECK(means2d.dim() == 3 && means2d.size(2) == 2,
                "means2d must be [C, N, 2], got ", means2d.sizes());
    TORCH_CHECK(conics.dim() == 3 && conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3,
                "conics must be [C, N, 3], got ", conics.sizes());
    TORCH_CHECK(colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N,
                "colors must be [C, N, channels], got ", colors.sizes());
    TORCH_CHECK(opacities.dim() == 2 && opacities.size(0) == C && opacities.size(1) == N,
                "opacities must be [C, N], got ", opacities.sizes());

    // Only validate bg_color if it's not empty
    if (bg_color.defined() && bg_color.numel() > 0) {
        TORCH_CHECK(bg_color.dim() == 2 && bg_color.size(0) == C && bg_color.size(1) == channels,
                    "bg_color must be [C, ", channels, "], got ", bg_color.sizes());
        TORCH_CHECK(bg_color.is_cuda(), "bg_color must be on CUDA");
        bg_color = bg_color.contiguous();
    }

    // Device checks
    TORCH_CHECK(means2d.is_cuda(), "means2d must be on CUDA");
    TORCH_CHECK(conics.is_cuda(), "conics must be on CUDA");
    TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be on CUDA");
    TORCH_CHECK(isect_offsets.is_cuda(), "isect_offsets must be on CUDA");
    TORCH_CHECK(flatten_ids.is_cuda(), "flatten_ids must be on CUDA");
    TORCH_CHECK(settings.is_cuda(), "settings must be on CUDA");

    // Ensure tensors are contiguous
    means2d = means2d.contiguous();
    conics = conics.contiguous();
    colors = colors.contiguous();
    opacities = opacities.contiguous();
    isect_offsets = isect_offsets.contiguous();
    flatten_ids = flatten_ids.contiguous();

    // Convert empty tensor to optional for CUDA function
    at::optional<at::Tensor> bg_color_opt;
    if (bg_color.defined() && bg_color.numel() > 0) {
        bg_color_opt = bg_color;
    }

    // Call rasterization with optional background
    auto raster_results = gsplat::rasterize_to_pixels_3dgs_fwd(
        means2d, conics, colors, opacities,
        bg_color_opt, {}, // bg_color_opt might not have value, masks is empty optional
        width, height, tile_size,
        isect_offsets, flatten_ids);

    auto rendered_image = std::get<0>(raster_results).contiguous();
    auto rendered_alpha = std::get<1>(raster_results).to(torch::kFloat32).contiguous();
    auto last_ids = std::get<2>(raster_results).contiguous();

    // Validate outputs - use actual channel count
    TORCH_CHECK(rendered_image.dim() == 4 && rendered_image.size(0) == C &&
                    rendered_image.size(1) == height && rendered_image.size(2) == width &&
                    rendered_image.size(3) == channels,
                "rendered_image must be [C, H, W, ", channels, "], got ", rendered_image.sizes());
    TORCH_CHECK(rendered_alpha.dim() == 4 && rendered_alpha.size(0) == C &&
                    rendered_alpha.size(1) == height && rendered_alpha.size(2) == width &&
                    rendered_alpha.size(3) == 1,
                "rendered_alpha must be [C, H, W, 1], got ", rendered_alpha.sizes());

    // Device checks for outputs
    TORCH_CHECK(rendered_image.is_cuda(), "rendered_image must be on CUDA");
    TORCH_CHECK(rendered_alpha.is_cuda(), "rendered_alpha must be on CUDA");
    TORCH_CHECK(last_ids.is_cuda(), "last_ids must be on CUDA");

    // Store relevant tensors in LocalNewtonContext
    ctx->means2d = means2d;
    ctx->conics = conics;
    ctx->tile_offsets = isect_offsets;
    ctx->flatten_ids = flatten_ids;

    return {rendered_image, rendered_alpha, last_ids};
}




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
    at::Tensor   // d2r_dp2_compact
>
ProjectionFunctionForward(
    LocalNewtonContext& ctx,
    at::Tensor means3D,              // [N,3]
    at::Tensor quats,                // [N,4]
    at::Tensor scales,               // [N,3]
    at::Tensor opacities,            // [N] or undefined
    at::Tensor viewmat,              // [C,4,4]
    at::Tensor K,                    // [C,3,3]
    at::Tensor settings              // [7]
) {
    // --- same validation as before ---
    const int N = means3D.size(0);
    const int C = viewmat.size(0);
    TORCH_CHECK(means3D.dim()==2 && means3D.size(1)==3,
                "means3D must be [N,3]");
    TORCH_CHECK(quats.dim()==2 && quats.size(0)==N && quats.size(1)==4,
                "quats must be [N,4]");
    TORCH_CHECK(scales.dim()==2 && scales.size(0)==N && scales.size(1)==3,
                "scales must be [N,3]");
    if (opacities.defined())
        TORCH_CHECK(opacities.dim()==1 && opacities.size(0)==N,
                    "opacities must be [N]");
    TORCH_CHECK(viewmat.dim()==3 && viewmat.size(1)==4 && viewmat.size(2)==4,
                "viewmat must be [C,4,4]");
    TORCH_CHECK(K.dim()==3 && K.size(0)==C && K.size(1)==3 && K.size(2)==3,
                "K must be [C,3,3]");
    TORCH_CHECK(settings.dim()==1 && settings.size(0)==7,
                "settings must be [7]");

    TORCH_CHECK(means3D.is_cuda(),    "means3D must be CUDA");
    TORCH_CHECK(quats.is_cuda(),      "quats must be CUDA");
    TORCH_CHECK(scales.is_cuda(),     "scales must be CUDA");
    if (opacities.defined())
        TORCH_CHECK(opacities.is_cuda(),"opacities must be CUDA");
    TORCH_CHECK(viewmat.is_cuda(),    "viewmat must be CUDA");
    TORCH_CHECK(K.is_cuda(),          "K must be CUDA");
    TORCH_CHECK(settings.is_cuda(),   "settings must be CUDA");

    // --- extract settings ---
    auto width            = settings[0].item<uint32_t>();
    auto height           = settings[1].item<uint32_t>();
    auto eps2d            = settings[2].item<float>();
    auto near_plane       = settings[3].item<float>();
    auto far_plane        = settings[4].item<float>();
    auto radius_clip      = settings[5].item<float>();
    auto scaling_modifier = settings[6].item<float>();

    // --- make contiguous and prep scales ---
    means3D = means3D.contiguous();
    quats   = quats.contiguous();
    scales  = scales.contiguous();
    if (opacities.defined()) opacities = opacities.contiguous();
    viewmat = viewmat.contiguous();
    K       = K.contiguous();

    auto scaled_scales = scales * scaling_modifier;

    // --- call your LN‐aware kernel ---
    auto proj = gsplat_newton::projection_ewa_3dgs_fused_fwd_LN(
        means3D,
        /*covars=*/{}, /*quats=*/quats,
        /*scales=*/scaled_scales,
        /*opacities=*/opacities,
        /*viewmats=*/viewmat,
        /*Ks=*/K,
        width, height,
        eps2d, near_plane, far_plane, radius_clip,
        /*calc_compensations=*/false,
        gsplat::CameraModelType::PINHOLE
    );

    // --- unpack and make contiguous ---
    auto radii        = std::get<0>(proj).contiguous();  // [C,N,2]
    auto means2d      = std::get<1>(proj).contiguous();  // [C,N,2]
    auto depths       = std::get<2>(proj).contiguous();  // [C,N]
    auto conics       = std::get<3>(proj).contiguous();  // [C,N,3]
    auto compensations= std::get<4>(proj);
    if (!compensations.defined())
        compensations = at::empty({0}, means3D.options());
    auto jacobians    = std::get<5>(proj).contiguous();  // [C,N,3,2]
    auto H_mean_y     = std::get<6>(proj).contiguous();  // [C,N,3,3]
    auto H_mean_x     = std::get<7>(proj).contiguous();  // [C,N,3,3]
    auto dSigma_dx    = std::get<8>(proj).contiguous();  // [C,N,2,2]
    auto dSigma_dy    = std::get<9>(proj).contiguous();  // [C,N,2,2]
    auto dSigma_dz    = std::get<10>(proj).contiguous(); // [C,N,2,2]
    auto H_Sigma      = std::get<11>(proj).contiguous();// [C,N,3,2,2]
    auto dr_dp        = std::get<12>(proj).contiguous();// [C,N,3,3]
    auto d2r_dp2      = std::get<13>(proj).contiguous();// [C,N,18]

    // --- fill your LocalNewtonContext ---
    ctx.means2d     = means2d;
    ctx.conics      = conics;
    ctx.depths      = depths;
    // you may compute view_dirs separately if needed
    // ----  ∂π/∂p  ----
    // kernel gave [C,N,3,2] (∂πᵧ/∂p, ∂πₓ/∂p]) so:
    ctx.d_mean2d_dp = jacobians.permute({0,1,3,2});       // [C,N,2,3]
    // ----  Hessian of π  ----
    // stack X & Y Hessians into [C,N,2,3,3]
    ctx.H_mean2d_dp = at::stack({ H_mean_x, H_mean_y }, /*dim=*/2);
    // ----  ∂Σ/∂p  ----
    ctx.d_Sigma_dp  = at::stack({ dSigma_dx, dSigma_dy, dSigma_dz }, /*dim=*/2);
    // ----  ∂²Σ/∂p² (compact)  ----
    // if you need 6×2×2, reshape/expand H_Sigma appropriately here:
    ctx.H_Sigma_dp = H_Sigma.view({C, N, /*3→6?*/ 6, 2, 2});
    // ----  ∂r/∂p  ----
    ctx.d_r_dp = dr_dp;
    // ----  ∂²r/∂p²  ----
    ctx.H_r_dp = d2r_dp2.view({C, N, 3, 3, 3});

    // other LocalNewtonContext fields (d_cSH_dp, etc.) can be set later

    // --- return everything in a tuple ---
    return std::make_tuple(
        radii, means2d, depths, conics, compensations,
        jacobians, H_mean_y, H_mean_x,
        dSigma_dx, dSigma_dy, dSigma_dz,
        H_Sigma, dr_dp, d2r_dp2
    );
}


/*
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_local_newton_backward(
    LocalNewtonContext& ctx,
    const torch::Tensor& grad_image,
    const torch::Tensor& grad_alpha) {

    // Get saved variables from context
    const auto& means2d = ctx.means2d;
    const auto& conics = ctx.conics;
    const auto& colors = ctx.colors;
    const auto& opacities = ctx.opacities;
    const auto& bg_color = ctx.bg_color;
    const auto& isect_offsets = ctx.isect_offsets;
    const auto& flatten_ids = ctx.flatten_ids;
    const auto& rendered_alpha = ctx.rendered_alpha;
    const auto& last_ids = ctx.last_ids;
    const auto& settings = ctx.settings;

    // Ensure gradients are contiguous
    auto grad_image_contig = grad_image.contiguous();
    auto grad_alpha_contig = grad_alpha.contiguous();

    // Extract settings
    const auto width = settings[0].item<int>();
    const auto height = settings[1].item<int>();
    const auto tile_size = settings[2].item<int>();

    // Convert empty tensor to optional for CUDA function
    at::optional<at::Tensor> bg_color_opt;
    if (bg_color.defined() && bg_color.numel() > 0) {
        bg_color_opt = bg_color;
    }

    // Get number of Gaussians from means2d
    const int64_t N = means2d.size(-2);

    // Call backward to compute intermediate derivatives
    auto intermediate_derivs = compute_intermediate_derivatives_bwd(
        means2d, conics, colors, opacities,
        bg_color_opt, {}, // bg_color_opt might not have value, masks is empty optional
        width, height, tile_size,
        isect_offsets, flatten_ids,
        rendered_alpha, last_ids,
        grad_image_contig, grad_alpha_contig,
        N);

    // Unpack results
    auto dc_dcSH = std::get<0>(intermediate_derivs).contiguous();
    auto dc_dG = std::get<1>(intermediate_derivs).contiguous();
    auto dG_dmean2d = std::get<2>(intermediate_derivs).contiguous();
    auto dG_dSigma = std::get<3>(intermediate_derivs).contiguous();
    auto H_G_mean2d = std::get<4>(intermediate_derivs).contiguous();
    auto H_G_sigma = std::get<5>(intermediate_derivs).contiguous();
    auto H_G_mixed = std::get<6>(intermediate_derivs).contiguous();
    auto v_opac = std::get<7>(intermediate_derivs).contiguous();

    // Store intermediate derivatives in context
    ctx.dc_dcSH = dc_dcSH;
    ctx.dc_dG = dc_dG;
    ctx.dG_dmean2d = dG_dmean2d;
    ctx.dG_dSigma = dG_dSigma;
    ctx.H_G_mean2d = H_G_mean2d;
    ctx.H_G_sigma = H_G_sigma;
    ctx.H_G_mixed = H_G_mixed;
    ctx.v_opac = v_opac;

    // Compute background gradient if needed
    torch::Tensor v_bg_color;
    if (bg_color.defined() && bg_color.numel() > 0) {
        auto one_minus_alpha = 1.0f - rendered_alpha;
        v_bg_color = (grad_image_contig * one_minus_alpha).sum({1, 2});
    } else {
        v_bg_color = torch::Tensor();
    }

    // Return gradients for input tensors
    return {
        dG_dmean2d,    // gradient for means2d
        dG_dSigma,     // gradient for conics
        dc_dcSH,       // gradient for colors
        v_opac,        // gradient for opacities
        v_bg_color     // gradient for background color
    };
}
*/


    // SphericalHarmonicsFunction implementation
   std::tuple<torch::Tensor> SphericalHarmonicsForward(
        LocalNewtonContext& ctx,
        torch::Tensor sh_degree_tensor, // [1] containing sh_degree
        torch::Tensor dirs,             // [..., 3]
        torch::Tensor coeffs,           // [..., K, 3]
        torch::Tensor masks,
        const torch::Tensor& means3D,
        const torch::Tensor& viewmat) {          // [...] optional

        const int sh_degree = sh_degree_tensor.item<int>();
        const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);

        // Input validation
        TORCH_CHECK(dirs.size(-1) == 3,
                    "dirs last dimension must be 3, got ", dirs.size(-1));
        TORCH_CHECK(coeffs.size(-1) == 3,
                    "coeffs last dimension must be 3, got ", coeffs.size(-1));
        TORCH_CHECK(coeffs.size(-2) >= num_sh_coeffs,
                    "coeffs K dimension must be at least ", num_sh_coeffs, ", got ", coeffs.size(-2));

        // Get batch dimensions
        auto batch_dims = dirs.sizes().slice(0, dirs.dim() - 1);

        TORCH_CHECK(dirs.sizes().slice(0, dirs.dim() - 1) == coeffs.sizes().slice(0, coeffs.dim() - 2),
                    "dirs and coeffs batch dimensions must match");

        if (masks.defined()) {
            TORCH_CHECK(masks.sizes() == batch_dims,
                        "masks must match dirs batch dims, got ", masks.sizes());
        }

        // Device checks
        TORCH_CHECK(dirs.is_cuda(), "dirs must be on CUDA");
        TORCH_CHECK(coeffs.is_cuda(), "coeffs must be on CUDA");
        TORCH_CHECK(sh_degree_tensor.is_cuda(), "sh_degree_tensor must be on CUDA");
        if (masks.defined()) {
            TORCH_CHECK(masks.is_cuda(), "masks must be on CUDA");
        }

        // Ensure tensors are contiguous
        dirs = dirs.contiguous();
        coeffs = coeffs.contiguous();
        if (masks.defined()) {
            masks = masks.contiguous();
        } else {
            // Create default masks (all true) with proper shape
            masks = torch::ones(batch_dims, torch::TensorOptions().dtype(torch::kBool).device(dirs.device()));
        }
        auto batch_dims_vec = dirs.sizes().vec();
        auto original_shape = batch_dims_vec;
        original_shape[original_shape.size() - 1] = 3;
        const int64_t num_gaussians = dirs.numel() / 3;
        // Flatten batch dimensions for CUDA kernel
        //auto dirs_flat = dirs.reshape({-1, 3});
        //auto coeffs_flat = coeffs.reshape({-1, coeffs.size(-2), 3});
        //auto masks_flat = masks.reshape({-1});
        // Reshape inputs for the kernel: [..., D] -> [1, N, D]
        auto dirs_reshaped = dirs.reshape({1, num_gaussians, 3});
        auto coeffs_reshaped = coeffs.reshape({1, num_gaussians, coeffs.size(-2), 3});

        // Flatten batch dimensions for CUDA kernel
        auto dirs_flat = dirs.reshape({-1, 3});
        auto coeffs_flat = coeffs.reshape({-1, coeffs.size(-2), 3});
        auto masks_flat = masks.reshape({-1});
        // Call spherical harmonics forward - pass FULL coeffs!
        auto colors =  gsplat::spherical_harmonics_fwd(sh_degree, dirs_flat, coeffs_flat, masks_flat);


        auto means3D_no_cam_dim = means3D;
        if (means3D.dim() == 3 && means3D.size(0) == 1) {
            means3D_no_cam_dim = means3D.squeeze(0);
        }

        // Reshape output back to original batch dimensions
        auto output_shape = dirs.sizes().vec();
        output_shape[output_shape.size() - 1] = 3; // Ensure last dimension is 3
        colors = colors.reshape(output_shape).contiguous();

        TORCH_CHECK(colors.is_cuda(), "colors must be on CUDA after SH computation");

        // Save for backward - save everything as-is
        //ctx->save_for_backward({dirs, coeffs, masks});
        ctx.dirs = dirs;
        ctx.coeffs = coeffs;
        ctx.masks = masks;
        ctx.sh_degree = sh_degree;
        ctx.num_bases = coeffs.size(-2);
        //ctx->saved_data["sh_degree"] = sh_degree;
        //ctx->saved_data["num_bases"] = coeffs.size(-2); // Save the full K dimension

        return {colors};
    }