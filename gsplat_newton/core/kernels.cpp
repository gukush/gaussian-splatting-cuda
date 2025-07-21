#include "core/kernels.hpp"
#include "core/local_newton_context.hpp"
#include "utils/cuda_utils.cuh" // For AT_DISPATCH_FLOATING_TYPES, etc.
#include "utils/projection.h"   // For CameraModelType
#include "ssim.h"               // For SSIM functions
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace gsplat_newton {

// ========================================================================
// 1. PROJECTION KERNELS
// ========================================================================
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd(
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
    const CameraModelType camera_model
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

    at::Tensor radii = at::empty({C, N, 2}, means.options().dtype(at::kInt));
    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor depths = at::empty({C, N}, means.options());
    at::Tensor conics = at::empty({C, N, 3}, means.options());
    at::Tensor compensations = {};
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = at::zeros({C, N}, means.options());
    }

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
        // outputs
        radii,
        means2d,
        depths,
        conics,
        calc_compensations ? at::optional<at::Tensor>(compensations)
                           : c10::nullopt
    );
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

// ========================================================================
// 2. SPHERICAL HARMONICS (SH) KERNELS
// ========================================================================

torch::Tensor sh_fwd_with_derivatives(int degree,
                                      const torch::Tensor &view_dirs,
                                      const torch::Tensor &sh_coeffs,
                                      torch::Tensor &d_color_d_dir,
                                      torch::Tensor &H_color_d_dir) {
    const uint32_t N = view_dirs.size(1);
    const uint32_t C = view_dirs.size(0);
    TORCH_CHECK(C == 1, "Only single camera supported for now in SH kernels.");

    auto colors = torch::zeros({C, N, 3}, view_dirs.options());
    if (N == 0)
        return colors;

    const dim3 threads(256, 1, 1);
    const dim3 blocks(GET_BLOCKS(N, threads.x), 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        view_dirs.scalar_type(), "sh_fwd_with_derivatives_kernel", ([&] {
            sh_fwd_with_derivatives_kernel<scalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    N, degree, view_dirs.data_ptr<scalar_t>(),
                    sh_coeffs.data_ptr<scalar_t>(),
                    colors.data_ptr<scalar_t>(),
                    d_color_d_dir.data_ptr<scalar_t>(),
                    H_color_d_dir.data_ptr<scalar_t>());
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return colors;
}

void chain_rule_sh_position(gs::LocalNewtonContext &context,
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
