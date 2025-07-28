#include "gsplat_newton/local_newton_context.hpp"
#include "gsplat_newton/kernels.hpp"
//#include "Utils.cuh" // For AT_DISPATCH_FLOATING_TYPES, etc.
#include "Projection.h"   // For CameraModelType
#include "Common.h"
//#include "kernels/ssim.h"               // For SSIM functions
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

using namespace gsplat;

namespace gsplat_newton {

// ========================================================================
// 1. PROJECTION KERNELS
// ========================================================================
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
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    } else {
        assert(quats.has_value() && scales.has_value());
        CHECK_INPUT(quats.value());
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(0);    // number of gaussians
    uint32_t C = viewmats.size(0); // number of cameras

    // Standard outputs
    at::Tensor radii = at::empty({C, N, 2}, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor depths = at::empty({C, N}, means.options());
    at::Tensor conics = at::empty({C, N, 3}, means.options());
    at::Tensor compensations = {};
    if (calc_compensations) {
        compensations = at::zeros({C, N}, means.options());
    }

    // Local Newton outputs
    at::Tensor jacobians = at::empty({C, N, 3, 2}, means.options());
    at::Tensor H_mean_y = at::empty({C, N, 3, 3}, means.options());
    at::Tensor H_mean_x = at::empty({C, N, 3, 3}, means.options());
    at::Tensor dSigma_dx = at::empty({C, N, 2, 2}, means.options());
    at::Tensor dSigma_dy = at::empty({C, N, 2, 2}, means.options());
    at::Tensor dSigma_dz = at::empty({C, N, 2, 2}, means.options());
    at::Tensor H_Sigma = at::empty({C, N, 3, 2, 2}, means.options()); // Stores 3 2x2 matrices
    at::Tensor dr_dp = at::empty({C, N, 3, 3}, means.options());
    at::Tensor d2r_dp2_compact = at::empty({C, N, 18}, means.options().dtype(at::kFloat));

    launch_projection_ewa_3dgs_fused_fwd_kernel_LN(
        // inputs
        means,
        covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        camera_model,
        // standard outputs
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations) : c10::nullopt,
        // Local Newton outputs
        jacobians,
        H_mean_y,
        H_mean_x,
        dSigma_dx,
        dSigma_dy,
        dSigma_dz,
        H_Sigma,
        dr_dp,
        d2r_dp2_compact
    );

    return std::make_tuple(
        radii,
        means2d,
        depths,
        conics,
        compensations,
        jacobians,
        H_mean_y,
        H_mean_x,
        dSigma_dx,
        dSigma_dy,
        dSigma_dz,
        H_Sigma,
        dr_dp,
        d2r_dp2_compact
    );
}
*/

std::tuple<at::Tensor, at::Tensor> compute_covariance_derivatives(
    const at::Tensor means,       // [N, 3] world positions
    const at::Tensor quats,       // [N, 4] quaternions [w, x, y, z]
    const at::Tensor scales,      // [N, 3] scale parameters
    const at::Tensor view_matrix // [4, 4] view matrix (camera to world)
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(view_matrix);

    uint32_t N = means.size(0);  // number of gaussians

    // Create output tensors
    at::Tensor dSigma_dtheta = at::empty({N, 2, 2}, means.options());    // [N, 2, 2] first derivatives
    at::Tensor d2Sigma_dtheta2 = at::empty({N, 2, 2}, means.options());  // [N, 2, 2] second derivatives

    // Launch the kernel
    launch_compute_covariance_derivatives_kernel(
        quats,
        scales,
        view_matrix,
        means,
        dSigma_dtheta,
        d2Sigma_dtheta2
    );

    return std::make_tuple(dSigma_dtheta, d2Sigma_dtheta2);
}


// ========================================================================
// 2. SPHERICAL HARMONICS (SH) KERNELS
// ========================================================================


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> spherical_harmonics_LN(
    const uint32_t      degrees_to_use,
    const at::Tensor   dirs,       // [..., 3]
    const at::Tensor   coeffs,     // [..., K, 3]
    const at::Tensor v_colors
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    CHECK_INPUT(v_colors);

    TORCH_CHECK(coeffs.size(-1) == 3,     "coeffs last dim must be 3");
    TORCH_CHECK(dirs.size(-1) == 3,       "dirs last dim must be 3");
    TORCH_CHECK(v_colors.size(-1) == 3,   "v_colors last dim must be 3 (CDIM)");// the reason is it is the same scalar for all channels in this case

    // outputs
    auto batch_dims = dirs.sizes().slice(0, dirs.dim() - 1);
    auto grad_shape = batch_dims.vec();
    grad_shape.push_back(3); // For the 3 output color channels
    grad_shape.push_back(3); // For the 3 input direction components (x, y, z)
    at::Tensor d_color_d_dir = at::empty(grad_shape, dirs.options());

    // 3. Initialize the `H_color_d_dir` (Hessian) output tensor
    //    Shape: [..., 3, 6]
    auto hess_shape = batch_dims.vec();
    hess_shape.push_back(3); // For the 3 output color channels
    hess_shape.push_back(6); // For the 6 packed Hessian elements (xx,yy,zz,xy,xz,yz)
    at::Tensor H_color_d_dir = at::empty(hess_shape, dirs.options());
    // replace last dim with 6 for Hessians
    //std::vector<int64_t> H_sizes = dirs.sizes().vec();
    //H_sizes.back() = 6;
    //at::Tensor H_dir = at::empty(H_sizes, dirs.options());
    //    d_color_d_dir,
    //    H_color_d_dir,
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;
    auto v_coeffs = at::empty({(int64_t)N, (int64_t)K, 3}, dirs.options());
    auto H_coeffs = at::empty({(int64_t)N, (int64_t)K, 3}, dirs.options());
    //auto v_dir    = at::empty({(int64_t)N,          3}, dirs.options());
    //auto H_dir    = at::empty({(int64_t)N,          6}, dirs.options());

    launch_spherical_harmonics_LN_kernel(
        degrees_to_use,
        dirs,
        coeffs,
        v_colors,
        v_coeffs,
        H_coeffs,
        d_color_d_dir,
        H_color_d_dir
    );

    return std::make_tuple(v_coeffs,H_coeffs,d_color_d_dir,H_color_d_dir);
}



std::pair<at::Tensor,at::Tensor> chain_rule_color_position(
    const at::Tensor p_k,             // [...,3]
    const at::Tensor radii,
    const at::Tensor camera_center,   // [3]
    const at::Tensor color_dir_grad,  // [...,3]
    const at::Tensor color_dir_hess   // [...,6]
) {
    DEVICE_GUARD(p_k);
    CHECK_INPUT(p_k);
    CHECK_INPUT(camera_center);
    CHECK_INPUT(color_dir_grad);
    CHECK_INPUT(color_dir_hess);

    TORCH_CHECK(p_k.size(-1)            == 3, "p_k must have shape [...,3]");
    TORCH_CHECK(camera_center.numel()   == 3, "camera_center must have 3 elements");
    TORCH_CHECK(color_dir_grad.size(-1) == 3, "color_dir_grad must have shape [...,3]");
    TORCH_CHECK(color_dir_hess.size(-1) == 6, "color_dir_hess must have shape [...,6]");
    const uint32_t N = p_k.numel() / 3;
    auto color_pos_grad = at::empty({(int64_t)N, 3}, p_k.options());
    auto color_pos_hess = at::empty({(int64_t)N, 6}, p_k.options());
    launch_chain_rule_color_position_kernel(
        p_k,
        radii,
        camera_center,
        color_dir_grad,
        color_dir_hess,
        color_pos_grad,
        color_pos_hess
    );
    return {color_pos_grad, color_pos_hess};
}

/*
void chain_rule_sh_position(LocalNewtonContext &context,
                            const torch::Tensor &means3D,
                            const torch::Tensor &viewmat,
                            const torch::Tensor &d_color_d_dir,
                            const torch::Tensor &H_color_d_dir) {
    const uint32_t N = means3D.size(0);
    const uint32_t C = viewmat.size(0);
    TORCH_CHECK(C == 1, "Only single camera supported for now in SH kernels.");

    // Compute camera center from view matrix
    auto viewmat_inv = torch::inverse(viewmat.squeeze(0));
    auto camera_center = viewmat_inv.slice(0, 0, 3).slice(1, 3, 4).contiguous();

    if (N == 0)
        return;

    const dim3 threads(256, 1, 1);
    const dim3 blocks(GET_BLOCKS(N, threads.x), 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        means3D.scalar_type(), "chain_rule_color_position_kernel_batched",
        ([&] {
            chain_rule_color_position_kernel_batched<scalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    N, means3D.data_ptr<scalar_t>(),
                    camera_center.data_ptr<scalar_t>(),
                    d_color_d_dir.data_ptr<scalar_t>(),
                    H_color_d_dir.data_ptr<scalar_t>(),
                    context.d_r_dp.data_ptr<scalar_t>(),
                    context.H_r_dp.data_ptr<scalar_t>(),
                    // outputs
                    context.d_cSH_dp.data_ptr<scalar_t>(),
                    context.H_cSH_dp.data_ptr<scalar_t>());
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
*/

// ========================================================================
// 3. BACKWARD PASS FOR RASTERIZATION KERNEL (GRADIENTS COMPUTATION)
// ========================================================================

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
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(dL_dcIMG);
    CHECK_INPUT(H_L_dcIMG);
    //CHECK_INPUT(v_render_colors);
    //CHECK_INPUT(v_render_alphas);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }

    uint32_t channels = colors.size(-1);
    bool packed = means2d.dim() == 2;

    // Create output tensors
    at::Tensor dL_dcSH = at::zeros({N,3}, means2d.options());
    at::Tensor H_L_dcSH = at::zeros({N,3}, means2d.options());
    at::Tensor dL_dG = at::zeros({N,3}, means2d.options());
    at::Tensor dL_dmean2d = at::zeros({N, 2}, means2d.options());
    at::Tensor dL_dconic = at::zeros({N, 3}, means2d.options());
    at::Tensor H_L_mean2d = at::zeros({N, 3}, means2d.options());
    at::Tensor H_L_conic = at::zeros({N, 6}, means2d.options());
    at::Tensor H_L_mixed = at::zeros({N, 6}, means2d.options());
    at::Tensor dL_dopac = at::zeros({N}, means2d.options());
    at::Tensor H_L_dopac = at::zeros({N}, means2d.options());

#define __LAUNCH_KERNEL__(CHANNELS) \
    case CHANNELS: \
        launch_compute_intermediate_derivatives_kernel<CHANNELS>( \
            packed, \
            means2d, \
            conics, \
            colors, \
            opacities, \
            backgrounds, \
            masks, \
            image_width, \
            image_height, \
            tile_size, \
            tile_offsets, \
            flatten_ids, \
            render_alphas, \
            last_ids, \
            dL_dcIMG, \
            H_L_dcIMG, \
            dL_dcSH, \
            H_L_dcSH, \
            dL_dG, \
            dL_dmean2d, \
            dL_dconic, \
            H_L_mean2d, \
            H_L_conic, \
            H_L_mixed, \
            dL_dopac, \
            H_L_dopac \
        ); \
        break;

    switch (channels) {
        __LAUNCH_KERNEL__(1)
        __LAUNCH_KERNEL__(2)
        __LAUNCH_KERNEL__(3)
        __LAUNCH_KERNEL__(4)
        __LAUNCH_KERNEL__(5)
        __LAUNCH_KERNEL__(8)
        __LAUNCH_KERNEL__(9)
        __LAUNCH_KERNEL__(16)
        __LAUNCH_KERNEL__(17)
        __LAUNCH_KERNEL__(32)
        __LAUNCH_KERNEL__(33)
        __LAUNCH_KERNEL__(64)
        __LAUNCH_KERNEL__(65)
        __LAUNCH_KERNEL__(128)
        __LAUNCH_KERNEL__(129)
        __LAUNCH_KERNEL__(256)
        __LAUNCH_KERNEL__(257)
        __LAUNCH_KERNEL__(512)
        __LAUNCH_KERNEL__(513)
    default:
        AT_ERROR("Unsupported number of channels: ", channels);
    }
#undef __LAUNCH_KERNEL__

    return std::make_tuple(
        dL_dcSH, H_L_dcSH, dL_dG, dL_dmean2d, dL_dconic,
        H_L_mean2d, H_L_conic, H_L_mixed, dL_dopac, H_L_dopac
    );
}

// ===================================================================================================
// 3. ASSEMBLE NEWTON DERIVATIVES (it returns gradients of loss and hessians of loss w.r.t attributes)
// ===================================================================================================

//TODO: REMOVE COLOR TERMS
// BECAUSE WE DO NOT ASSEMBLE THOSE DERIVATIVES HERE
// Wrapper function to call all split kernels
std::tuple<at::Tensor, at::Tensor, at::Tensor,  at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
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
    const at::Tensor dc_sh_dp,
    const at::Tensor H_c_sh_p,
    const at::Tensor dSigma_dtheta_inputs,
    const at::Tensor H_Sigma_dtheta_inputs,
    const at::Tensor quats,
    const at::Tensor conics_2d,
    const at::Tensor p_k,
    const at::Tensor camera_pos,
    const at::Tensor dL_dck
//    const at::Tensor dL_dc,
//    const at::Tensor H_L_dc
) {
    const int num_gaussians = p_k.size(0);
    const int num_coeffs = dL_dck.size(-2);
    auto options = p_k.options();

    // Allocate output tensors
    auto dL_dvk = torch::empty({num_gaussians, 2}, options);
    auto H_L_dvk = torch::empty({num_gaussians, 2, 2}, options);
    auto dL_dlambda = torch::empty({num_gaussians, 2}, options);
    auto H_L_dlambda = torch::empty({num_gaussians, 2, 2}, options);
    auto dL_dtheta = torch::empty({num_gaussians}, options);
    auto H_L_dtheta = torch::empty({num_gaussians}, options);
    auto dL_dcolor = torch::empty({num_gaussians, num_coeffs, 3}, options);
    auto H_L_dcolor = torch::empty({num_gaussians, num_coeffs, 3}, options);
    auto dL_dSigma = torch::empty({num_gaussians,  3}, options);
    auto H_L_sigma = torch::empty({num_gaussians, 6}, options);
    auto H_L_mixed = torch::empty({num_gaussians, 6}, options);
    // T_k matrix is of size 2x3
    auto T_matrices = torch::empty({num_gaussians, 2, 3}, options);
    auto U_k_bases = at::empty({num_gaussians, 2, 3}, options);
    auto dSigma_dp_temp = at::empty({num_gaussians * 12}, options); // 12 floats per Gaussian
    /*
    TORCH_CHECK(dSigma_dp.sizes()[1] == num_gaussians &&
                dSigma_dp.sizes()[2] == 3 &&
                dSigma_dp.sizes()[3] == 2 &&
                dSigma_dp.sizes()[4] == 2,
            "Expected dimensions of dSigm_dp to be [N, 3, 2, 2]");
    */
            // Launch split kernels
        launch_assemble_derivatives_kernels(
            num_gaussians,
            conics_2d,
            camera_pos,
            viewmat,
            radii,
            Ks,
            covars,
            quats,
            dL_dcSH_totals,
            dL_dG_totals,
            dL_dmean2d_totals,
            dL_dconic_totals,
            H_L_mean2d_totals,
            H_L_conic_totals,
            H_L_mixedinv_totals,
            p_k,
            dSigma_dtheta_inputs,
            H_Sigma_dtheta_inputs,
            dL_dvk,
            H_L_dvk,
            dL_dlambda,
            H_L_dlambda,
            dL_dtheta,
            H_L_dtheta,
            dL_dcolor,
            H_L_dcolor,
            dL_dSigma,
            H_L_sigma,
            H_L_mixed,
            U_k_bases,
            T_matrices,
            dSigma_dp_temp
        );

    return std::make_tuple(dL_dvk, H_L_dvk, dL_dlambda, H_L_dlambda,
                          dL_dtheta, H_L_dtheta,
                          dL_dcolor, H_L_dcolor, U_k_bases,T_matrices);
}
// =====================
// Losses
// ====================
std::tuple<at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor>
fusedssim_LN(
    float C1,
    float C2,
    at::Tensor img1,
    at::Tensor img2,
    bool train
) {
    img1 = img1.contiguous();
    img2 = img2.contiguous();
    DEVICE_GUARD(img1);
    CHECK_INPUT(img1);
    CHECK_INPUT(img2);
    TORCH_CHECK(img1.sizes() == img2.sizes(), "img1/img2 size mismatch");

    auto sizes = img1.sizes();
    int64_t B  = sizes[0],
            CH = sizes[1],
            H  = sizes[2],
            W  = sizes[3];

    // allocate outputs
    auto ssim_map = at::empty_like(img1);
    auto mu1_map  = at::empty({B,CH,H,W}, img1.options());
    auto mu2_map  = at::empty({B,CH,H,W}, img1.options());
    auto s1_map   = at::empty({B,CH,H,W}, img1.options());
    auto s2_map   = at::empty({B,CH,H,W}, img1.options());
    auto s12_map  = at::empty({B,CH,H,W}, img1.options());

    // call the CUDA launcher
    launch_fusedssim_LN_kernel(
        B, CH, H, W,
        C1, C2,
        img1.data_ptr<float>(),
        img2.data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? mu1_map.data_ptr<float>()  : nullptr,
        train ? mu2_map.data_ptr<float>()  : nullptr,
        train ? s1_map.data_ptr<float>()   : nullptr,
        train ? s2_map.data_ptr<float>()   : nullptr,
        train ? s12_map.data_ptr<float>()  : nullptr,
        train,
        at::cuda::getCurrentCUDAStream()
    );

    return {ssim_map, mu1_map, mu2_map, s1_map, s2_map, s12_map};
}

std::tuple<at::Tensor, at::Tensor>
fusedssim_backward_LN(
    float C1,
    float C2,
    at::Tensor img1,
    at::Tensor img2,
    at::Tensor dL_dmap,
    at::Tensor mu1_map,
    at::Tensor mu2_map,
    at::Tensor s1_map,
    at::Tensor s2_map,
    at::Tensor s12_map
) {
    DEVICE_GUARD(img1);
    img1 = img1.contiguous();
    img2 = img2.contiguous();
    CHECK_INPUT(img1);   CHECK_INPUT(img2);
    CHECK_INPUT(dL_dmap);
    CHECK_INPUT(mu1_map); CHECK_INPUT(mu2_map);
    CHECK_INPUT(s1_map);  CHECK_INPUT(s2_map);
    CHECK_INPUT(s12_map);
    TORCH_CHECK(img1.sizes() == img2.sizes(), "img1/img2 size mismatch");

    auto sizes = img1.sizes();
    int64_t B  = sizes[0],
            CH = sizes[1],
            H  = sizes[2],
            W  = sizes[3];

    auto dL_dimg1  = at::empty_like(img1);
    auto d2L_dimg1 = at::empty_like(img1);

    launch_fusedssim_backward_LN_kernel(
        B, CH, H, W,
        C1, C2,
        img1.data_ptr<float>(),
        img2.data_ptr<float>(),
        dL_dmap.data_ptr<float>(),
        mu1_map.data_ptr<float>(),
        mu2_map.data_ptr<float>(),
        s1_map.data_ptr<float>(),
        s2_map.data_ptr<float>(),
        s12_map.data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        d2L_dimg1.data_ptr<float>(),
        at::cuda::getCurrentCUDAStream()
    );

    return {dL_dimg1, d2L_dimg1};
}

} // namespace gsplat_newton