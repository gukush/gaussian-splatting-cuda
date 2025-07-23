#pragma once
#include "core/splat_data.hpp"
#include "core/itrainer.hpp"
#include "gsplat_newton/camera_knn.hpp"


namespace gs {

    class LocalNewtonTrainer : public ITrainer {
    public:
        // Constructor that takes ownership of strategy and shares datasets
        LocalNewtonTrainer(std::shared_ptr<CameraDataset> dataset,
                          std::unique_ptr<IStrategy> strategy,
                          const param::TrainingParameters& params);

        // Destructor
        ~LocalNewtonTrainer() override = default;

        // Main training method implementation
        void train() override;

    protected:
        // Protected method for processing a single training step
        bool train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode) override;

    private:
        // Private method for computing loss with gradients and Hessians (Newton implementation)
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_loss_grads(
            const RenderOutput& render_output,
            const torch::Tensor& gt_image,
            const SplatData& splatData,
            const param::OptimizationParameters& opt_params);

        // Newton-specific initialization
        void initialize_newton_components(std::shared_ptr<CameraDataset> dataset);

        // Camera KNN for finding nearby views
        std::unique_ptr<gs::CameraKNN> camera_knn_;
    };

} // namespace gs