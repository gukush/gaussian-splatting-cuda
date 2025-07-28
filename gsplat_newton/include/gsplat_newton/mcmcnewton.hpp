#pragma once
#include "core/istrategy.hpp"
#include "gsplat_newton/local_newton_context.hpp"
#include "Ops.h"
#include <c10/cuda/CUDACachingAllocator.h>

class MCMCNewton : public IStrategy {
public:
    MCMCNewton(SplatData&& splat_data);

    // IStrategy interface
    void initialize(const gs::param::OptimizationParameters& optimParams) override;
    void post_backward(int iter, gs::RenderOutput& render_output) override;
    bool is_refining(int iter) const override;
    void step(int iter) override;
    SplatData& get_model() override { return _splat_data; }
    const SplatData& get_model() const override { return _splat_data; }

    // Newton-specific method to receive context
    void set_newton_context(const LocalNewtonContext* ctx) { _newton_context = ctx; }

private:
    // Core MCMC operations
    torch::Tensor multinomial_sample(const torch::Tensor& weights, int n, bool replacement = true);
    int relocate_gs();
    int add_new_gs();
    void inject_noise(int iter);

    SplatData _splat_data;
    std::unique_ptr<const gs::param::OptimizationParameters> _params;
    torch::Tensor _binoms;
    torch::Tensor _last_visibility_mask;
    const LocalNewtonContext* _newton_context = nullptr;

    // Newton-specific parameters
    float _base_noise_lr = 5e-4;
    float _lr_decay_rate = 0.01;   // Decay to 1% over iterations, if not use this 0.9995f
};