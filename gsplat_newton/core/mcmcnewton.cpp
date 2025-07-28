#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include "core/rasterizer.hpp"
#include "gsplat_newton/mcmcnewton.hpp"
#include <random>
#include <iostream>
#include "core/splat_data.hpp"

MCMCNewton::MCMCNewton(SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}

torch::Tensor MCMCNewton::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
    const int64_t num_elements = weights.size(0);

    // PyTorch's multinomial has a limit of 2^24 elements
    if (num_elements <= (1 << 24)) {
        return torch::multinomial(weights, n, replacement);
    } else {
        // For larger arrays, implement sampling manually
        auto weights_normalized = weights / (weights.sum() + 1e-8);
        auto weights_cpu = weights_normalized.cpu();

        std::vector<int64_t> sampled_indices;
        sampled_indices.reserve(n);

        // Create cumulative distribution
        auto cumsum = weights_cpu.cumsum(0);
        auto cumsum_data = cumsum.accessor<float, 1>();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        for (int i = 0; i < n; ++i) {
            float u = dis(gen);
            // Binary search for the index
            int64_t idx = 0;
            int64_t left = 0, right = num_elements - 1;
            while (left <= right) {
                int64_t mid = (left + right) / 2;
                if (cumsum_data[mid] < u) {
                    left = mid + 1;
                } else {
                    idx = mid;
                    right = mid - 1;
                }
            }
            sampled_indices.push_back(idx);
        }

        auto result = torch::tensor(sampled_indices, torch::kLong);
        return result.to(weights.device());
    }
}

void MCMCNewton::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    // Move tensors to GPU and set requires_grad
    const auto dev = torch::kCUDA;
    _splat_data.means() = _splat_data.means().to(dev).set_requires_grad(true);
    _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true);
    _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true);
    _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);
    _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);
    _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);

    // Initialize binomial coefficients
    const int n_max = 51;
    _binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = _binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    _binoms = _binoms.to(dev);

    // No optimizer or scheduler needed!
}

void MCMCNewton::inject_noise(int iter) {
    torch::NoGradGuard no_grad;

    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto scales = _splat_data.get_scaling();
    auto quats = _splat_data.get_rotation();

    // Get covariance matrices
    auto covar_result = gsplat::quat_scale_to_covar_preci_fwd(
        quats, scales, true, false, false
    );
    auto covars = std::get<0>(covar_result);

    // Opacity sigmoid
    const float k = 100.0f;
    const float x0 = 0.995f;
    auto op_sigmoid = 1.0f / (1.0f + torch::exp(-k * ((1.0f - opacities) - x0)));

    // Calculate learning rate decay
    float decay_factor = std::pow(_lr_decay_rate, static_cast<float>(iter) / _params->iterations);
    float current_noise_scale = _base_noise_lr * decay_factor;

    // Use gradient magnitude from Newton context to scale noise
    if (_newton_context && _newton_context->dL_d_pos.defined()) {
        float grad_norm = _newton_context->dL_d_pos.norm().item<float>();
        float adaptive_scale = 1.0f / (1.0f + grad_norm);
        current_noise_scale *= adaptive_scale;
    }

    // Generate and apply noise
    auto noise = torch::randn_like(_splat_data.means()) * op_sigmoid.unsqueeze(-1) * current_noise_scale;
    noise = torch::bmm(covars, noise.unsqueeze(-1)).squeeze(-1);
    _splat_data.means().add_(noise);
}

void MCMCNewton::step(int iter) {
    // No optimizer step needed - Newton handles the updates
    // Just clear any residual gradients
    torch::NoGradGuard no_grad;
    if (_splat_data.means().grad().defined()) {
        _splat_data.means().mutable_grad().reset();
    }
    if (_splat_data.rotation_raw().grad().defined()) {
        _splat_data.rotation_raw().mutable_grad().reset();
    }
    if (_splat_data.scaling_raw().grad().defined()) {
        _splat_data.scaling_raw().mutable_grad().reset();
    }
    if (_splat_data.sh0().grad().defined()) {
        _splat_data.sh0().mutable_grad().reset();
    }
    if (_splat_data.shN().grad().defined()) {
        _splat_data.shN().mutable_grad().reset();
    }
    if (_splat_data.opacity_raw().grad().defined()) {
        _splat_data.opacity_raw().mutable_grad().reset();
    }
}

bool MCMCNewton::is_refining(int iter) const {
    return (iter < _params->stop_refine &&
            iter > _params->start_refine &&
            iter % _params->refine_every == 0);
}

void MCMCNewton::post_backward(int iter, gs::RenderOutput& render_output) {
    // Store visibility mask
    if (_params->selective_adam) {
        _last_visibility_mask = render_output.visibility;
    }

    // Increment SH degree
    torch::NoGradGuard no_grad;
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    // Refine Gaussians
    if (is_refining(iter)) {
        relocate_gs();
        add_new_gs();
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // Inject noise
    inject_noise(iter);
}


int MCMCNewton::relocate_gs() {
    // Get opacities and handle both [N] and [N, 1] shapes
    torch::NoGradGuard no_grad;
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto dead_mask = opacities <= _params->min_opacity;
    auto dead_indices = dead_mask.nonzero().squeeze(-1);
    int n_dead = dead_indices.numel();

    if (n_dead == 0)
        return 0;

    auto alive_mask = ~dead_mask;
    auto alive_indices = alive_mask.nonzero().squeeze(-1);

    if (alive_indices.numel() == 0)
        return 0;

    // Sample from alive Gaussians based on opacity
    auto probs = opacities.index_select(0, alive_indices);
    if (probs.sum().item<float>()==0) return 0;
    auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
    auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences of each sampled index
    auto ratios = torch::zeros({opacities.size(0)}, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kCUDA); // ??? Use opacities.options().dtype(torch::kFloat32) and drop the explicit .to.
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::TensorOptions().dtype(torch::kFloat32)));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    // IMPORTANT: Clamp and convert to int as in Python implementation
    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous(); // Convert to int!

    // Call the CUDA relocation function from gsplat
    auto relocation_result = gsplat::relocation(
        sampled_opacities,
        sampled_scales,
        ratios,
        _binoms,
        n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    // Clamp new opacities
    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    // Update parameters for sampled indices
    // Handle opacity shape properly
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Copy from sampled to dead indices
    _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
    _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
    _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
    _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
    _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
    _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));


    return n_dead;
}

int MCMCNewton::add_new_gs() {
    torch::NoGradGuard no_grad;

    const int current_n = _splat_data.size();
    const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
    const int n_new = std::max(0, n_target - current_n);

    if (n_new == 0)
        return 0;

    // Get opacities and handle shapes
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto probs = opacities.flatten();
    auto sampled_idxs = multinomial_sample(probs, n_new, true);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences
    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    // Clamp and convert to int
    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous();

    // Call CUDA relocation function
    auto relocation_result = gsplat::relocation(
        sampled_opacities,
        sampled_scales,
        ratios,
        _binoms,
        n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    // Clamp new opacities
    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    // Update existing Gaussians
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Prepare new Gaussians
    auto new_means = _splat_data.means().index_select(0, sampled_idxs);
    auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
    auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
    auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
    auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
    auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

    // Concatenate all parameters - NO OPTIMIZER STATE MANAGEMENT NEEDED!
    _splat_data.means() = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
    _splat_data.sh0() = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
    _splat_data.shN() = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
    _splat_data.scaling_raw() = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
    _splat_data.rotation_raw() = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
    _splat_data.opacity_raw() = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

    return n_new;
}