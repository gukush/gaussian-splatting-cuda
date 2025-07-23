#include "core/splat_data.hpp"
#include "gsplat_newton/local_newton_trainer.hpp"
#include "gsplat_newton/rasterizer_newton.hpp"
#include "gsplat_newton/kernels.hpp"
#include "kernels/fused_ssim.cuh"
#include "visualizer/detail.hpp"
#include "gsplat_newton/Newton.h"
#include <chrono>
#include <iostream>
#include <numeric>
#include <torch/torch.h>
#include "gsplat_newton/local_newton_context.hpp"

namespace gs {

    static inline torch::Tensor ensure_4d(const torch::Tensor& image) {
        return image.dim() == 3 ? image.unsqueeze(0) : image;
    }

    static torch::Tensor spherical_distance(const torch::Tensor& v1, const torch::Tensor& v2) {
        // Normalize vectors to ensure they are on the unit sphere
        auto v1_norm = v1 / v1.norm();
        auto v2_norm = v2 / v2.norm();
        // The angle is acos of the dot product
        return torch::acos(torch::dot(v1_norm, v2_norm));
    }

    static torch::Tensor get_projected_camera_positions(
        const std::vector<std::shared_ptr<Camera>>& cameras,
        const SplatData& model) {

        // 1. Estimate scene center and bounding sphere
        torch::Tensor scene_center = model.get_means().mean(/*dim=*/0);

        std::vector<torch::Tensor> projected_positions;
        projected_positions.reserve(cameras.size());

        for (const auto& cam : cameras) {
            // 2. Project camera poses to the surface of the bounding sphere
            const auto& c2w_matrix = cam->world_view_transform();
            torch::Tensor cam_pos = c2w_matrix.index({0, torch::indexing::Slice(0, 3), 3});
            torch::Tensor centered_pos = cam_pos - scene_center;
            // Project by normalizing the vector from the scene center to the camera
            projected_positions.push_back(centered_pos / centered_pos.norm());
        }

        return torch::stack(projected_positions, 0);
    }

    LocalNewtonTrainer::LocalNewtonTrainer(std::shared_ptr<CameraDataset> dataset,
                                         std::unique_ptr<IStrategy> strategy,
                                         const param::TrainingParameters& params)
        : ITrainer(dataset, std::move(strategy), params) {

        // Initialize Newton-specific components
        initialize_newton_components(dataset);
    }
    /*
    static torch::Tensor get_camera_positions(const std::vector<std::shared_ptr<Camera>>& cameras) {
        std::vector<torch::Tensor> positions;
        positions.reserve(cameras.size());
        for (const auto* cam : cameras) {
            positions.push_back(cam.);
        }
        return torch::stack(positions, 0);
    }
    */

    void LocalNewtonTrainer::initialize_newton_components(std::shared_ptr<CameraDataset> dataset) {
        // Initialize camera KNN using spherical distance as the metric, inspired by the paper.
        auto projected_camera_positions = get_projected_camera_positions(
            dataset->get_cameras(),
            strategy_->get_model()
        );

        // The CameraKNN class should be initialized with these projected positions
        // and use a spherical distance metric for finding neighbors.
        camera_knn_ = std::make_unique<gs::CameraKNN>(projected_camera_positions, true); // Assuming a bool flag enables spherical distance
        std::cout << "Camera KNN initialized for Newton trainer using spherical distance." << std::endl;
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> LocalNewtonTrainer::compute_loss_grads(
        const RenderOutput& render_output,
        const torch::Tensor& gt_image,
        const SplatData& splatData,
        const param::OptimizationParameters& opt_params) {

        // Ensure images have same dimensions
        torch::Tensor rendered = render_output.image;
        torch::Tensor gt = gt_image;

        // Ensure both tensors are 4D (batch, height, width, channels)
        rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
        gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

        TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch – rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());

        // Constants for SSIM (these should be defined elsewhere or passed as parameters)
        const float C1 = 0.01f * 0.01f;
        const float C2 = 0.03f * 0.03f;
        const bool train = true;

        torch::Tensor ssim_map, mu1, mu2, s1, s2, s12;
        // CALL APPROPRIATE SSIM LOSS KERNEL
        std::tie(ssim_map, mu1, mu2, s1, s2, s12) = gsplat_newton::fusedssim_LN(C1, C2, rendered, gt, train);

        // MSE part
        auto diff = rendered - gt;
        auto mse = diff.pow(2).mean();

        auto loss = opt_params.lambda_dssim * (1.0f - ssim_map.mean()) + (1 - opt_params.lambda_dssim) * mse;
        int64_t numel = ssim_map.numel();
        auto dL_dmap = torch::full_like(ssim_map, -1.0f / float(numel));

        auto dL_mse = (1 - opt_params.lambda_dssim) * 2.0f * diff / float(numel);

        // CALL BACKWARD KERNEL
        torch::Tensor dL_ssim, H_ssim;
        std::tie(dL_ssim, H_ssim) = gsplat_newton::fusedssim_backward_LN(
            C1, C2,
            rendered, gt,      // inputs
            dL_dmap,          // upstream grad on the map
            mu1, mu2, s1, s2, s12
        );

        float mse_weight = 1 - opt_params.lambda_dssim;
        float H_mse_const = mse_weight * 2.0f / float(numel);
        auto H_mse = torch::full_like(H_ssim, H_mse_const);

        auto dL_c = dL_ssim + dL_mse;
        auto H_L_c = H_ssim + H_mse;

        return {loss, dL_c, H_L_c};
    }

    bool LocalNewtonTrainer::train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode) {
        current_iteration_ = iter;
        LocalNewtonContext ctx;

        // Check control requests at the beginning
        handle_control_requests(iter);

        // If stop requested, return false to end training
        if (stop_requested_) {
            return false;
        }

        // If paused, wait
        while (is_paused_ && !stop_requested_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            handle_control_requests(iter);
        }

        // Check stop again after potential pause
        if (stop_requested_) {
            return false;
        }

        // --- Example of how to get nearest neighbors ---
        if (iter % 1000 == 0) { // Example: print every 1000 iterations
            if (camera_knn_) {
                auto neighbors = camera_knn_->find_neighbors(cam->uid(), 3);
                std::cout << "Iter " << iter << ": Camera " << cam->uid() << " neighbors: ";
                for (int neighbor_idx : neighbors) {
                    std::cout << neighbor_idx << " ";
                }
                std::cout << std::endl;
            }
        }
        // --- End of example ---
        auto render_fn = [this, &cam, render_mode, gt_image, &ctx]() {
                return gs::rasterize_newton_step(
                    *cam,
                    strategy_->get_model(),
                    background_,
                    gt_image,
                    1.0f,
                    false,
                    render_mode,
                    &ctx
                );
            };

        RenderOutput r_output;

        if (viewer_) {
            std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
            r_output = render_fn();
        } else {
            r_output = render_fn();
        }

        // Apply bilateral grid if enabled
        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
            r_output.image = bilateral_grid_->apply(r_output.image, cam->uid());
        }

        // Compute loss using the Newton-specific function
        torch::Tensor loss, dL_c, H_L_c;
        std::tie(loss, dL_c, H_L_c) = compute_loss_grads(r_output,
                                                         gt_image,
                                                         strategy_->get_model(),
                                                         params_.optimization);

        current_loss_ = loss.item<float>();

        // Use local Newton instead of loss.backward()
        gsplat_newton::local_newton_backward(
            ctx,
            strategy_->get_model(),
            static_cast<int>(cam->image_width()),
            static_cast<int>(cam->image_height())
        );

        // Here we would need to do the overshoot prevention.
        // but it requires 2 smaller images to be completely iterated in the same manner.
        auto neighbors = camera_knn_->find_neighbors(cam->uid(), 3);
        for(int n_idx: neighbors) {
            auto neighbor_cam_from_set = train_dataset_->get(n_idx);
            auto& neighbor_cam = neighbor_cam_from_set.data.camera;
            const int downsample_factor = 3;
            int low_res_h = neighbor_cam->image_height() / downsample_factor;
            int low_res_w = neighbor_cam->image_width() / downsample_factor;
            Camera temp_neighbor_cam(*neighbor_cam, low_res_w, low_res_h);
            LocalNewtonContext tmp_ctx;
            RenderOutput tmp_r_output;
            auto tmp_image = neighbor_cam_from_set.target;

            auto tmp_render_fn = [this, &temp_neighbor_cam, render_mode, tmp_image, &tmp_ctx]() {
                return gs::rasterize_newton_step(
                    temp_neighbor_cam,
                    strategy_->get_model(),
                    background_,
                    tmp_image,
                    false,
                    1.0f,
                    render_mode,
                    &tmp_ctx
                );
            };

            if (viewer_) {
                std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
                tmp_r_output = tmp_render_fn();
            } else {
                tmp_r_output = tmp_render_fn();
            }

            gsplat_newton::local_newton_backward(
                tmp_ctx,
                strategy_->get_model(),
                static_cast<int>(temp_neighbor_cam.image_width()),
                static_cast<int>(temp_neighbor_cam.image_height())
            );

            // aggregate hessians and gradients
            ctx.dL_d_pos += tmp_ctx.dL_d_pos;
            ctx.H_L_pos  += tmp_ctx.H_L_pos;
            ctx.dL_d_scale += tmp_ctx.dL_d_scale;
            ctx.H_L_scale += tmp_ctx.H_L_scale;
            ctx.dL_d_rot += tmp_ctx.dL_d_rot;
            ctx.H_L_rot += tmp_ctx.H_L_rot;
            ctx.dL_d_opacity += tmp_ctx.dL_d_opacity;
            ctx.H_L_opacity += tmp_ctx.H_L_opacity;
            ctx.dL_d_color += tmp_ctx.dL_d_color;
            ctx.H_L_color += tmp_ctx.H_L_color;
        }

        // --- Stage 3: Solve Systems, Backproject, and Apply Updates ---
        gsplat_newton::solve_and_update(
            ctx,
            strategy_->get_model(),
            static_cast<int>(cam->image_width()),
            static_cast<int>(cam->image_height())
        );

        {
            torch::NoGradGuard no_grad;

            // Clean evaluation - let the evaluator handle everything
            if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                evaluator_->print_evaluation_header(iter);
                auto metrics = evaluator_->evaluate(iter,
                                                    strategy_->get_model(),
                                                    val_dataset_,
                                                    background_);
                std::cout << metrics.to_string() << std::endl;
            }

            // Save model at specified steps
            for (size_t save_step : params_.optimization.save_steps) {
                if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                    const bool join_threads = (iter == params_.optimization.save_steps.back());
                    strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/join_threads);
                }
            }

            auto do_strategy = [&]() {
                strategy_->post_backward(iter, r_output);
                strategy_->step(iter);
            };

            if (viewer_) {
                std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
                do_strategy();
            } else {
                do_strategy();
            }

            if (params_.optimization.use_bilateral_grid) {
                bilateral_grid_optimizer_->step();
                bilateral_grid_optimizer_->zero_grad(true);
            }
        }

        progress_->update(iter, loss.item<float>(),
                          static_cast<int>(strategy_->get_model().size()),
                          strategy_->is_refining(iter));

        if (viewer_) {
            if (viewer_->info_) {
                auto& info = viewer_->info_;
                std::lock_guard<std::mutex> lock(viewer_->info_->mtx);
                info->updateProgress(iter, params_.optimization.iterations);
                info->updateNumSplats(static_cast<size_t>(strategy_->get_model().size()));
                info->updateLoss(loss.item<float>());
            }

            if (viewer_->notifier_) {
                auto& notifier = viewer_->notifier_;
                std::unique_lock<std::mutex> lock(notifier->mtx);
                notifier->cv.wait(lock, [&notifier] { return notifier->ready; });
            }
        }

        // Return true if we should continue training
        return iter < params_.optimization.iterations && !stop_requested_;
    }

    void LocalNewtonTrainer::train() {
        is_running_ = false; // Don't start running until notified
        training_complete_ = false;

        // Wait for the start signal from GUI if visualization is enabled
        if (viewer_ && viewer_->notifier_) {
            auto& notifier = viewer_->notifier_;
            std::unique_lock<std::mutex> lock(notifier->mtx);
            notifier->cv.wait(lock, [&notifier] { return notifier->ready; });
        }

        is_running_ = true; // Now we can start

        int iter = 1;
        const int epochs_needed = (params_.optimization.iterations + train_dataset_size_ - 1) / train_dataset_size_;

        const int num_workers = 4;

        const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

        bool should_continue = true;

        for (int epoch = 0; epoch < epochs_needed && should_continue; ++epoch) {
            auto train_dataloader = create_dataloader_from_dataset(train_dataset_, num_workers);

            for (auto& batch : *train_dataloader) {
                auto camera_with_image = batch[0].data;
                Camera* cam = camera_with_image.camera;
                torch::Tensor gt_image = std::move(camera_with_image.image);

                should_continue = train_step(iter, cam, gt_image, render_mode);

                if (!should_continue) {
                    break;
                }

                ++iter;
            }
        }

        // Final save if not already saved by stop request
        if (!stop_requested_) {
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        }

        progress_->complete();
        evaluator_->save_report();
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));

        is_running_ = false;
        training_complete_ = true;
    }

} // namespace gs