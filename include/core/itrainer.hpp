#pragma once

#include "core/bilateral_grid.hpp"
#include "core/dataset.hpp"
#include "core/istrategy.hpp"
#include "core/metrics.hpp"
#include "core/parameters.hpp"
#include "core/training_progress.hpp"
#include <atomic>
#include <memory>
#include <torch/torch.h>

namespace gs {

    class GSViewer;

    class ITrainer {
    public:
        // Constructor that takes ownership of strategy and shares datasets
        ITrainer(std::shared_ptr<CameraDataset> dataset,
                 std::unique_ptr<IStrategy> strategy,
                 const param::TrainingParameters& params);

        // Delete copy operations
        ITrainer(const ITrainer&) = delete;
        ITrainer& operator=(const ITrainer&) = delete;

        // Allow move operations
        ITrainer(ITrainer&&) = default;
        ITrainer& operator=(ITrainer&&) = default;

        virtual ~ITrainer();

        // Main training method - pure virtual
        virtual void train() = 0;

        // Create viewer and return it for main thread execution
        GSViewer* create_and_get_viewer();

        // Control methods for GUI interaction
        void request_pause() { pause_requested_ = true; }
        void request_resume() { pause_requested_ = false; }
        void request_save() { save_requested_ = true; }
        void request_stop() { stop_requested_ = true; }

        bool is_paused() const { return is_paused_; }
        bool is_running() const { return is_running_; }
        bool is_training_complete() const { return training_complete_; }
        bool has_stopped() const { return stop_requested_; }

        // Get current training state
        int get_current_iteration() const { return current_iteration_; }
        float get_current_loss() const { return current_loss_; }

        // For viewer to get model
        const IStrategy& get_strategy() const { return *strategy_; }

    protected:
        // Protected method for processing a single training step - pure virtual
        virtual bool train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode) = 0;

        // Common initialization methods
        void initialize_bilateral_grid();
        void handle_control_requests(int iter);
        void setup_datasets(std::shared_ptr<CameraDataset> dataset);
        void common_initialization();

        // Common member variables
        std::shared_ptr<CameraDataset> train_dataset_;
        std::shared_ptr<CameraDataset> val_dataset_;
        std::unique_ptr<IStrategy> strategy_;
        param::TrainingParameters params_;

        std::unique_ptr<GSViewer> viewer_;

        torch::Tensor background_{};
        std::unique_ptr<TrainingProgress> progress_;
        size_t train_dataset_size_;

        // Bilateral grid components
        std::unique_ptr<gs::BilateralGrid> bilateral_grid_;
        std::unique_ptr<torch::optim::Adam> bilateral_grid_optimizer_;

        // Metrics evaluator - handles all evaluation logic
        std::unique_ptr<metrics::MetricsEvaluator> evaluator_;

        // Control flags for thread communication
        std::atomic<bool> pause_requested_{false};
        std::atomic<bool> save_requested_{false};
        std::atomic<bool> stop_requested_{false};
        std::atomic<bool> is_paused_{false};
        std::atomic<bool> is_running_{false};
        std::atomic<bool> training_complete_{false};

        // Current training state
        std::atomic<int> current_iteration_{0};
        std::atomic<float> current_loss_{0.0f};
    };

} // namespace gs