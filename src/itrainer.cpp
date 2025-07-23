#include "core/itrainer.hpp"
#include "visualizer/detail.hpp"
#include <chrono>
#include <iostream>
#include <torch/torch.h>

namespace gs {

    ITrainer::ITrainer(std::shared_ptr<CameraDataset> dataset,
                       std::unique_ptr<IStrategy> strategy,
                       const param::TrainingParameters& params)
        : strategy_(std::move(strategy)),
          params_(params) {

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        setup_datasets(dataset);
        common_initialization();
    }

    ITrainer::~ITrainer() {
        // Ensure training is stopped
        stop_requested_ = true;
    }

    void ITrainer::setup_datasets(std::shared_ptr<CameraDataset> dataset) {
        // Handle dataset split based on evaluation flag
        if (params_.optimization.enable_eval) {
            // Create train/val split
            train_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params_.dataset, CameraDataset::Split::TRAIN);
            val_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params_.dataset, CameraDataset::Split::VAL);

            std::cout << "Created train/val split: "
                      << train_dataset_->size().value() << " train, "
                      << val_dataset_->size().value() << " val images" << std::endl;
        } else {
            // Use all images for training
            train_dataset_ = dataset;
            val_dataset_ = nullptr;

            std::cout << "Using all " << train_dataset_->size().value()
                      << " images for training (no evaluation)" << std::endl;
        }

        train_dataset_size_ = train_dataset_->size().value();
    }

    void ITrainer::common_initialization() {
        strategy_->initialize(params_.optimization);

        // Initialize bilateral grid if enabled
        initialize_bilateral_grid();

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(torch::kCUDA);

        progress_ = std::make_unique<TrainingProgress>(
            params_.optimization.iterations,
            /*bar_width=*/100);

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params_);

        // Print render mode configuration
        std::cout << "Render mode: " << params_.optimization.render_mode << std::endl;
        std::cout << "Visualization: " << (params_.optimization.enable_viz ? "enabled" : "disabled") << std::endl;
    }

    void ITrainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return;
        }

        bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
            train_dataset_size_,
            params_.optimization.bilateral_grid_X,
            params_.optimization.bilateral_grid_Y,
            params_.optimization.bilateral_grid_W);

        bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
            std::vector<torch::Tensor>{bilateral_grid_->parameters()},
            torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                .eps(1e-15));
    }

    GSViewer* ITrainer::create_and_get_viewer() {
        if (!params_.optimization.enable_viz) {
            return nullptr;
        }

        if (!viewer_) {
            viewer_ = std::make_unique<GSViewer>("GS-CUDA", 1280, 720);
            viewer_->setTrainer(this);
        }

        return viewer_.get();
    }

    void ITrainer::handle_control_requests(int iter) {
        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            progress_->pause();
            std::cout << "\nTraining paused at iteration " << iter << std::endl;
            std::cout << "Click 'Resume Training' to continue." << std::endl;
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            progress_->resume(iter, current_loss_, static_cast<int>(strategy_->get_model().size()));
            std::cout << "\nTraining resumed at iteration " << iter << std::endl;
        }

        // Handle save request
        if (save_requested_.load()) {
            save_requested_ = false;
            std::cout << "\nSaving checkpoint at iteration " << iter << "..." << std::endl;
            strategy_->get_model().save_ply(params_.dataset.output_path / "checkpoints", iter, /*join=*/true);
            std::cout << "Checkpoint saved to " << (params_.dataset.output_path / "checkpoints").string() << std::endl;
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::cout << "\nStopping training permanently at iteration " << iter << "..." << std::endl;
            std::cout << "Saving final model..." << std::endl;
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

} // namespace gs