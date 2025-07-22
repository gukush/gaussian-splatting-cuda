#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "local_newton_context.hpp"

class Camera;
namespace gs {
    struct RenderOutput;
    enum class RenderMode;
}
namespace gs {
RenderOutput rasterize_newton_step(
    Camera& viewpoint_camera,
    const SplatData& gaussian_model,
    torch::Tensor& bg_color,
    const torch::Tensor& gt_image,
    float scaling_modifier,
    RenderMode render_mode,
    LocalNewtonContext* context);


}