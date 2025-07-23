   std::tuple<torch::Tensor> SphericalHarmonicsForward(
        LocalNewtonContext* ctx,
        torch::Tensor sh_degree_tensor, // [1] containing sh_degree
        torch::Tensor dirs,             // [..., 3]
        torch::Tensor coeffs,           // [..., K, 3]
        torch::Tensor masks,
        const torch::Tensor& means3D,
        const torch::Tensor& viewmat);



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
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RasterizationFunctionForward(
    LocalNewtonContext* ctx,
    torch::Tensor means2d,       // [C, N, 2]
    torch::Tensor conics,        // [C, N, 3]
    torch::Tensor colors,        // [C, N, channels] - may include depth
    torch::Tensor opacities,     // [C, N]
    torch::Tensor bg_color,      // [C, channels] - may include depth, can be empty
    torch::Tensor isect_offsets, // [C, tile_height, tile_width]
    torch::Tensor flatten_ids,   // [nnz]
    torch::Tensor settings);