#pragma once
#include <torch/torch.h>

// This struct holds all intermediate tensors required for the local Newton optimization.
// It is populated by the projection and rasterization kernels and consumed by the assembly and update kernels.
struct LocalNewtonContext {
    // ==================================
    // PROJECTION FORWARD OUTPUTS
    // ==================================
    torch::Tensor means2d;       // [C, N, 2] - 2D means in screen space
    torch::Tensor conics;        // [C, N, 3] - Inverse 2D covariance matrices
    torch::Tensor depths;        // [C, N]    - Z-depths in camera space
    torch::Tensor view_dirs;     // [C, N, 3] - View directions from camera to mean
    torch::Tensor radii;
    torch::Tensor viewmat;
    // ==================================
    // PROJECTION DERIVATIVES (w.r.t. 3D position p_k)
    // ==================================
    // For Mean (π_k)
    torch::Tensor d_mean2d_dp;   // [C, N, 2, 3] - Jacobian ∂π/∂p
    torch::Tensor H_mean2d_dp;   // [C, N, 2, 3, 3] - Hessian ∂²π/∂p²

    // For Covariance (Σ_k)
    torch::Tensor d_Sigma_dp;    // [C, N, 3, 2, 2] - Jacobian ∂Σ/∂p (for p_x, p_y, p_z)
    torch::Tensor H_Sigma_dp;    // [C, N, 6, 2, 2] - Compacted Hessian ∂²Σ/∂p²

    // For SH View Direction (r_k)
    torch::Tensor d_r_dp;        // [C, N, 3, 3] - Jacobian ∂r/∂p
    torch::Tensor H_r_dp;        // [C, N, 18?] - Hessian ∂²r/∂p² (or compacted)

    // ==================================
    // RASTERIZATION BACKWARD AGGREGATES
    // ==================================
    // Per-Gaussian sums of derivatives, aggregated over all pixels
    torch::Tensor dL_dc;         // [C, N, 3] - Σ(∂L/∂c) for each Gaussian
    torch::Tensor dL_dG;         // [C, N]    - Σ(∂L/∂c * ∂c/∂G)
    torch::Tensor dG_dmean2d;    // [C, N, 2] - Σ(∂G/∂π)
    torch::Tensor dG_dSigma;     // [C, N, 3] - Σ(∂G/∂Σ)
    torch::Tensor H_G_mean2d;    // [C, N, 3] - Σ(∂²G/∂π²)
    torch::Tensor H_G_Sigma;     // [C, N, 6] - Σ(∂²G/∂Σ²)
    torch::Tensor H_G_mixed;     // [C, N, 6] - Σ(∂²G/∂π∂Σ)
    torch::Tensor dc_dcSH;
    // ==================================
    // SH DERIVATIVES (w.r.t. 3D position p_k)
    // ==================================
    torch::Tensor d_c_ck;         // [C, N,  3] I think?
    torch::Tensor d_cSH_dp;      // [C, N, 3, 3] - Jacobian ∂c̃/∂p
    torch::Tensor H_cSH_dp;      // [C, N, 3, 3, 3] - Hessian ∂²c̃/∂p²

    // ==================================
    // SH Intermediates
    // ==================================
    torch::Tensor dirs;
    torch::Tensor coeffs;
    torch::Tensor masks;
    torch::Tensor colors; // output of spherical_harmonics_fwd
    int sh_degree;
    int num_bases;
    torch::Tensor campos;
    // ==================================
    // ROTATION & SCALING DERIVATIVES
    // ==================================
    torch::Tensor d_Sigma_dtheta; // [C, N, 2, 2] - ∂Σ/∂θ
    torch::Tensor H_Sigma_dtheta; // [C, N, 2, 2] - ∂²Σ/∂θ²
    torch::Tensor T_matrices;     // [C, N, 2, 3] - Transformation matrix for scaling
    // ==================================
    // OTHER IMPORTANT DATA i.e. TILE INTERSECTION
    // ==================================
    torch::Tensor flatten_ids;
    torch::Tensor last_ids;
    torch::Tensor tile_offsets;
    torch::Tensor render_alphas;
    torch::Tensor dL_d_color_img;
    torch::Tensor H_L_color_img;;
    //===================================
    // FINAL DERIVATIVES AND HESSIANS (merged due to overshoot prevention)
    //===================================
    torch::Tensor dL_d_pos;
    torch::Tensor H_L_pos;
    torch::Tensor dL_d_scale;
    torch::Tensor H_L_scale;
    torch::Tensor dL_d_rot;
    torch::Tensor H_L_rot;
    torch::Tensor dL_d_opacity;
    torch::Tensor H_L_opacity;
    torch::Tensor dL_d_color;
    torch::Tensor H_L_color;
};