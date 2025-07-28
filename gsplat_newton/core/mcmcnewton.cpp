#include "gsplat_newton/mcmcnewton.hpp"
#include "gsplat_newton/local_newton_context.hpp"


void MCMCNewton::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    // Move tensors to GPU and set requires_grad (even though we won't use autograd)
    const auto dev = torch::kCUDA;
    _splat_data.means() = _splat_data.means().to(dev).set_requires_grad(true);
    _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true);
    _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true);
    _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);
    _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);
    _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);

    // Initialize binomial coefficients (same as before)
    const int n_max = 51;
    _binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = _binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            // Compute binomial coefficient C(n,k)
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    _binoms = _binoms.to(dev);

    // should I have that here?
    const double gamma = std::pow(0.01, 1.0 / _params->iterations);
    _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, 0);
}

void MCMCNewton::inject_noise(int iter) {
    torch::NoGradGuard no_grad;

    // Get opacities and scales
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

    // Calculate learning rate decay (instead of reading from optimizer)
    float decay_factor = std::pow(_lr_decay_rate, static_cast<float>(iter) / _params->iterations);
    float current_noise_scale = _base_noise_lr * decay_factor;

    // Optionally, use gradient magnitude from Newton context to scale noise
    if (_newton_context && _newton_context->dL_d_pos.defined()) {
        // Scale noise based on gradient magnitude
        float grad_norm = _newton_context->dL_d_pos.norm().item<float>();
        float adaptive_scale = 1.0f / (1.0f + grad_norm); // Example scaling
        current_noise_scale *= adaptive_scale;
    }

    // Generate and apply noise
    auto noise = torch::randn_like(_splat_data.means()) * op_sigmoid.unsqueeze(-1) * current_noise_scale;
    noise = torch::bmm(covars, noise.unsqueeze(-1)).squeeze(-1);
    _splat_data.means().add_(noise);
}

void MCMCNewton::step(int iter) {
    // No optimizer step needed - Newton handles the updates
    // Just clear any residual gradients if they exist
    torch::NoGradGuard no_grad;
    if (_splat_data.means().grad().defined()) {
        _splat_data.means().mutable_grad() = {};
    }
    if (_splat_data.rotation_raw().grad().defined()) {
        _splat_data.rotation_raw().mutable_grad() = {};
    }
        if (_splat_data.scaling_raw().grad().defined()) {
        _splat_data.scaling_raw().mutable_grad() = {};
    }
    if (_splat_data.sh0().grad().defined()) {
        _splat_data.sh0().mutable_grad() = {};
    }
    if (_splat_data.shN().grad().defined()) {
        _splat_data.shN().mutable_grad() = {};
    }
    if (_splat_data.opacity_raw().grad().defined()) {
        _splat_data.opacity_raw().mutable_grad() = {};
    }
    // ... clear other gradients ...
}

void MCMCNewton::post_backward(int iter, gs::RenderOutput& render_output) {
    // Store visibility mask for selective adam
    if (_params->selective_adam) {
        _last_visibility_mask = render_output.visibility;
    }

    // Increment SH degree every 1000 iterations
    torch::NoGradGuard no_grad;
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    // Refine Gaussians
    if (is_refining(iter)) {
        // Relocate dead Gaussians
        relocate_gs();

        // Add new Gaussians
        add_new_gs();

        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    // Inject noise to positions
    inject_noise();
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
    auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
    auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences of each sampled index
    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
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



int MCMC::add_new_gs() {
    // Add this check at the beginning
    torch::NoGradGuard no_grad;
    if (!_optimizer) {
        std::cerr << "Warning: add_new_gs called but optimizer not initialized" << std::endl;
        return 0;
    }

    const int current_n = _splat_data.size();
    const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
    const int n_new = std::max(0, n_target - current_n);

    if (n_new == 0)
        return 0;

    // Get opacities and handle both [N] and [N, 1] shapes
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

    // Update existing Gaussians FIRST (before concatenation)
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Prepare new Gaussians to concatenate
    auto new_means = _splat_data.means().index_select(0, sampled_idxs);
    auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
    auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
    auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
    auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
    auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

    // Step 1: Concatenate all parameters
    auto concat_means = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
    auto concat_sh0 = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
    auto concat_shN = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
    auto concat_scaling = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
    auto concat_rotation = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
    auto concat_opacity = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

    // Step 2: SAFER optimizer state update
    // Store the new parameters in a temporary array first
    std::array<torch::Tensor*, 6> new_params = {
        &concat_means, &concat_sh0, &concat_shN,
        &concat_scaling, &concat_rotation, &concat_opacity};

    // Collect old parameter keys and states
    std::vector<void*> old_param_keys;
    std::vector<std::unique_ptr<torch::optim::OptimizerParamState>> saved_states;

    for (int i = 0; i < 6; ++i) {
        auto& old_param = _optimizer->param_groups()[i].params()[0];
        void* old_param_key = old_param.unsafeGetTensorImpl();
        old_param_keys.push_back(old_param_key);

        // Check if state exists
        auto state_it = _optimizer->state().find(old_param_key);
        if (state_it != _optimizer->state().end()) {
            // Clone the state before modifying - handle both optimizer types
            if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(state_it->second.get())) {
                // Standard Adam state
                torch::IntArrayRef new_shape;
                if (i == 0)
                    new_shape = new_means.sizes();
                else if (i == 1)
                    new_shape = new_sh0.sizes();
                else if (i == 2)
                    new_shape = new_shN.sizes();
                else if (i == 3)
                    new_shape = new_scaling.sizes();
                else if (i == 4)
                    new_shape = new_rotation.sizes();
                else
                    new_shape = new_opacity.sizes();

                auto zeros_to_add = torch::zeros(new_shape, adam_state->exp_avg().options());
                auto new_exp_avg = torch::cat({adam_state->exp_avg(), zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({adam_state->exp_avg_sq(), zeros_to_add}, 0);

                // Create new state
                auto new_state = std::make_unique<torch::optim::AdamParamState>();
                new_state->step(adam_state->step());
                new_state->exp_avg(new_exp_avg);
                new_state->exp_avg_sq(new_exp_avg_sq);
                if (adam_state->max_exp_avg_sq().defined()) {
                    auto new_max_exp_avg_sq = torch::cat({adam_state->max_exp_avg_sq(), zeros_to_add}, 0);
                    new_state->max_exp_avg_sq(new_max_exp_avg_sq);
                }

                saved_states.push_back(std::move(new_state));
            } else if (auto* selective_adam_state = dynamic_cast<gs::SelectiveAdam::AdamParamState*>(state_it->second.get())) {
                // SelectiveAdam state
                torch::IntArrayRef new_shape;
                if (i == 0)
                    new_shape = new_means.sizes();
                else if (i == 1)
                    new_shape = new_sh0.sizes();
                else if (i == 2)
                    new_shape = new_shN.sizes();
                else if (i == 3)
                    new_shape = new_scaling.sizes();
                else if (i == 4)
                    new_shape = new_rotation.sizes();
                else
                    new_shape = new_opacity.sizes();

                auto zeros_to_add = torch::zeros(new_shape, selective_adam_state->exp_avg.options());
                auto new_exp_avg = torch::cat({selective_adam_state->exp_avg, zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({selective_adam_state->exp_avg_sq, zeros_to_add}, 0);

                // Create new state
                auto new_state = std::make_unique<gs::SelectiveAdam::AdamParamState>();
                new_state->step_count = selective_adam_state->step_count;
                new_state->exp_avg = new_exp_avg;
                new_state->exp_avg_sq = new_exp_avg_sq;
                if (selective_adam_state->max_exp_avg_sq.defined()) {
                    auto new_max_exp_avg_sq = torch::cat({selective_adam_state->max_exp_avg_sq, zeros_to_add}, 0);
                    new_state->max_exp_avg_sq = new_max_exp_avg_sq;
                }

                saved_states.push_back(std::move(new_state));
            } else {
                saved_states.push_back(nullptr);
            }
        } else {
            saved_states.push_back(nullptr);
        }
    }

    // Now remove all old states
    for (auto key : old_param_keys) {
        _optimizer->state().erase(key);
    }

    // Update parameters and add new states
    for (int i = 0; i < 6; ++i) {
        _optimizer->param_groups()[i].params()[0] = *new_params[i];

        if (saved_states[i]) {
            void* new_param_key = new_params[i]->unsafeGetTensorImpl();
            _optimizer->state()[new_param_key] = std::move(saved_states[i]);
        }
    }

    // Step 3: Finally update the model's parameters
    _splat_data.means() = concat_means;
    _splat_data.sh0() = concat_sh0;
    _splat_data.shN() = concat_shN;
    _splat_data.scaling_raw() = concat_scaling;
    _splat_data.rotation_raw() = concat_rotation;
    _splat_data.opacity_raw() = concat_opacity;

    return n_new;
}
