   std::tuple<torch::Tensor> SphericalHarmonicsForward_with_gradients(
        LocalNewtonContext ctx,
        torch::Tensor sh_degree_tensor, // [1] containing sh_degree
        torch::Tensor dirs,             // [..., 3]
        torch::Tensor coeffs,           // [..., K, 3]
        torch::Tensor masks);