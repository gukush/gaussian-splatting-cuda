#pragma once

#include "core/camera.hpp"
#include "external/nanoflann.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>

namespace gs {

    // Point cloud adaptor for nanoflann, specifically for camera positions
    struct CameraPositionAdaptor {
        const torch::Tensor& positions;

        CameraPositionAdaptor(const torch::Tensor& pos)
            : positions(pos) {
            TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "Positions must be [N, 3]");
            TORCH_CHECK(positions.is_contiguous(), "Positions tensor must be contiguous");
        }

        inline size_t kdtree_get_point_count() const { return positions.size(0); }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return positions.accessor<float, 2>()[idx][dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, CameraPositionAdaptor>,
        CameraPositionAdaptor, 3>;

    class CameraKNN {
    public:
        CameraKNN(const torch::Tensor& camera_positions, bool is_spherical);

        // Find the k nearest neighbors for a given camera index
        std::vector<int> find_neighbors(int camera_idx, int k) const;

    private:
        torch::Tensor positions_; // Stored on CPU for nanoflann
        std::unique_ptr<CameraPositionAdaptor> adaptor_;
        std::unique_ptr<KDTree> index_;
    };

} // namespace gs
