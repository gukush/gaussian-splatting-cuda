#pragma once

#include "core/itrainer.hpp"

namespace gs {

    class StandardTrainer : public ITrainer {
    public:
        // Constructor that takes ownership of strategy and shares datasets
        StandardTrainer(std::shared_ptr<CameraDataset> dataset,
                       std::unique_ptr<IStrategy> strategy,
                       const param::TrainingParameters& params);

        // Destructor
        ~StandardTrainer() override = default;

        // Main training method implementation
        void train() override;

    protected:
        // Protected method for processing a single training step
        bool train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode) override;

    private:
        // Private method for computing loss (standard implementation)
        torch::Tensor compute_loss(const RenderOutput& render_output,
                                   const torch::Tensor& gt_image,
                                   const SplatData& splatData,
                                   const param::OptimizationParameters& opt_params);
    };

} // namespace gs