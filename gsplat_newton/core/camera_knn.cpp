#include "gsplat_newton/camera_knn.hpp"
#include <torch/torch.h>

namespace gs {

    /**
     * @brief Construct a new CameraKNN object.
     * @param features A tensor of camera features. For standard KNN, these are 3D positions.
     * For spherical KNN, these must be 3D unit vectors (positions projected onto a sphere).
     * @param is_spherical If true, treats the features as points on a unit sphere. The standard
     * L2 distance is used, which is equivalent to spherical distance for finding
     * nearest neighbors on a sphere.
     */
    CameraKNN::CameraKNN(const torch::Tensor& features, bool is_spherical) {
        // Ensure features are on CPU and contiguous for nanoflann
        positions_ = features.to(torch::kCPU).contiguous();

        // --- Spherical Distance Safety Check ---
        // If we are using spherical distance, the input vectors must be normalized.
        // This check verifies that the L2 norm of the first vector is close to 1.
        if (is_spherical) {
            TORCH_CHECK(positions_.size(1) == 3, "Spherical KNN requires 3D feature vectors.");
            float norm = positions_[0].norm().item<float>();
            TORCH_CHECK(std::abs(norm - 1.0f) < 1e-5,
                        "For spherical distance, input features must be unit vectors (normalized). "
                        "The norm of the first vector is ", norm);
        }

        // The adaptor provides an interface for nanoflann to access our torch::Tensor data.
        adaptor_ = std::make_unique<CameraPositionAdaptor>(positions_);

        // --- KD-Tree Construction ---
        // We use the standard L2 distance KD-Tree. For points on a unit sphere (normalized vectors),
        // minimizing Euclidean distance is equivalent to minimizing the spherical angle,
        // so this correctly finds the nearest neighbors in terms of perspective.
        // The tree is built on 3D points.
        index_ = std::make_unique<KDTree>(3, *adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index_->buildIndex();
    }

    /**
     * @brief Finds the k-nearest neighbors for a given camera.
     * @param camera_idx The index of the query camera.
     * @param k The number of neighbors to find.
     * @return A vector of integer indices for the neighboring cameras.
     */
    std::vector<int> CameraKNN::find_neighbors(int camera_idx, int k) const {
        // k+1 because the query point itself will be found as the closest neighbor.
        const size_t num_results = k + 1;
        std::vector<size_t> ret_indices(num_results);
        std::vector<float> out_dists_sqr(num_results);

        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_indices[0], &out_dists_sqr[0]);

        // Get a pointer to the query point data.
        const float* query_pt = positions_.data_ptr<float>() + camera_idx * positions_.size(1);
        index_->findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));

        std::vector<int> neighbors;
        // Start from index 1 to skip the query point itself (which is always at index 0).
        for (size_t i = 1; i < num_results && i < ret_indices.size(); ++i) {
            neighbors.push_back(static_cast<int>(ret_indices[i]));
        }
        return neighbors;
    }

} // namespace gs
