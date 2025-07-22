#include "core/camera_knn.hpp"

namespace gs {

    CameraKNN::CameraKNN(const torch::Tensor& camera_positions) {
        // Ensure positions are on CPU and contiguous
        positions_ = camera_positions.to(torch::kCPU).contiguous();

        adaptor_ = std::make_unique<CameraPositionAdaptor>(positions_);
        index_ = std::make_unique<KDTree>(3, *adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index_->buildIndex();
    }

    std::vector<int> CameraKNN::find_neighbors(int camera_idx, int k) const {
        // k+1 because the query point itself will be the closest neighbor
        const size_t num_results = k + 1;
        std::vector<size_t> ret_indices(num_results);
        std::vector<float> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_indices[0], &out_dists_sqr[0]);

        const float* query_pt = positions_.data_ptr<float>() + camera_idx * 3;
        index_->findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));

        std::vector<int> neighbors;
        // Start from 1 to skip the query point itself
        for (size_t i = 1; i < num_results && i < ret_indices.size(); ++i) {
            neighbors.push_back(static_cast<int>(ret_indices[i]));
        }
        return neighbors;
    }

} // namespace gs
