#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

using namespace gsplat;

template <typename scalar_t>
__device__ void chain_rule_color_position_kernel(
    const glm::vec3 &p_k,                 // Input position
    const glm::vec3 &camera_center,       // Camera center
    const glm::vec3 &color_dir_grad,      // ∂c/∂r (gradient in direction space)
    const scalar_t *color_dir_hess,       // ∂²c/∂r² (Hessian in direction space, packed as [xx, yy, zz, xy, xz, yz])
    // Outputs
    glm::vec3 *color_pos_grad,           // ∂c/∂p (gradient in position space)
    scalar_t *color_pos_hess             // ∂²c/∂p² (Hessian in position space, packed as [xx, yy, zz, xy, xz, yz])
) {
    // 1. Compute ∂r/∂p and ∂²r/∂p²
    glm::vec3 d = p_k - camera_center;
    float L = glm::length(d);
    glm::vec3 r = d / L;

    // Jacobian ∂r/∂p = (I - r rᵀ) / |d|
    glm::mat3 I(1.0f);
    glm::mat3 J = (I - glm::outerProduct(r, r)) * (1.0f / L);

    // Extract J elements: J[i][j] = J[col=j][row=i]
    float j00 = J[0][0], j10 = J[1][0], j20 = J[2][0];
    float j01 = J[0][1], j11 = J[1][1], j21 = J[2][1];
    float j02 = J[0][2], j12 = J[1][2], j22 = J[2][2];

    // 2. Compute gradient: gc_p = gc_r · J
    glm::vec3 gc_p;
    gc_p.x = color_dir_grad.x * j00 + color_dir_grad.y * j10 + color_dir_grad.z * j20;
    gc_p.y = color_dir_grad.x * j01 + color_dir_grad.y * j11 + color_dir_grad.z * j21;
    gc_p.z = color_dir_grad.x * j02 + color_dir_grad.y * j12 + color_dir_grad.z * j22;
    *color_pos_grad = gc_p;

    if (color_pos_hess == nullptr) return;

    // 3. Compute Hessian: H_p = Jᵀ H_r J + Σ_a (gc_r[a] * H_a)

    // Term1 = Jᵀ H_r J
    // Unpack input Hessian (color_dir_hess)
    float h00 = color_dir_hess[0], h11 = color_dir_hess[1], h22 = color_dir_hess[2];
    float h01 = color_dir_hess[3], h02 = color_dir_hess[4], h12 = color_dir_hess[5];

    // Compute Term1 components
    float T1_xx = j00*(h00*j00 + h01*j10 + h02*j20)
                + j10*(h01*j00 + h11*j10 + h12*j20)
                + j20*(h02*j00 + h12*j10 + h22*j20);

    float T1_xy = j00*(h00*j01 + h01*j11 + h02*j21)
                + j10*(h01*j01 + h11*j11 + h12*j21)
                + j20*(h02*j01 + h12*j11 + h22*j21);

    float T1_xz = j00*(h00*j02 + h01*j12 + h02*j22)
                + j10*(h01*j02 + h11*j12 + h12*j22)
                + j20*(h02*j02 + h12*j12 + h22*j22);

    float T1_yy = j01*(h00*j01 + h01*j11 + h02*j21)
                + j11*(h01*j01 + h11*j11 + h12*j21)
                + j21*(h02*j01 + h12*j11 + h22*j21);

    float T1_yz = j01*(h00*j02 + h01*j12 + h02*j22)
                + j11*(h01*j02 + h11*j12 + h12*j22)
                + j21*(h02*j02 + h12*j12 + h22*j22);

    float T1_zz = j02*(h00*j02 + h01*j12 + h02*j22)
                + j12*(h01*j02 + h11*j12 + h12*j22)
                + j22*(h02*j02 + h12*j12 + h22*j22);

    // Term2 = Σ_a gc_r[a] * H_a (second derivatives of r components)
    float L2 = L*L, L3 = L2*L, L5 = L3*L2;
    auto δ = [](int i,int j){ return (i==j) ? 1.0f : 0.0f; };
    auto A = [&](int i,int j,int k) {
        float di=d[i], dj=d[j], dk=d[k];
        return - (δ(i,j)*dk + δ(i,k)*dj + δ(j,k)*di)/L3
               + 3.0f*di*dj*dk/L5;
    };

    // Compute components of H0, H1, H2 (second derivatives of r components)
    float T2_xx = color_dir_grad.x * A(0,0,0) + color_dir_grad.y * A(1,0,0) + color_dir_grad.z * A(2,0,0);
    float T2_xy = color_dir_grad.x * A(0,0,1) + color_dir_grad.y * A(1,0,1) + color_dir_grad.z * A(2,0,1);
    float T2_xz = color_dir_grad.x * A(0,0,2) + color_dir_grad.y * A(1,0,2) + color_dir_grad.z * A(2,0,2);
    float T2_yy = color_dir_grad.x * A(0,1,1) + color_dir_grad.y * A(1,1,1) + color_dir_grad.z * A(2,1,1);
    float T2_yz = color_dir_grad.x * A(0,1,2) + color_dir_grad.y * A(1,1,2) + color_dir_grad.z * A(2,1,2);
    float T2_zz = color_dir_grad.x * A(0,2,2) + color_dir_grad.y * A(1,2,2) + color_dir_grad.z * A(2,2,2);

    // Final Hessian output
    color_pos_hess[0] = T1_xx + T2_xx;  // xx
    color_pos_hess[1] = T1_yy + T2_yy;  // yy
    color_pos_hess[2] = T1_zz + T2_zz;  // zz
    color_pos_hess[3] = T1_xy + T2_xy;  // xy
    color_pos_hess[4] = T1_xz + T2_xz;  // xz
    color_pos_hess[5] = T1_yz + T2_yz;  // yz
}

template <typename scalar_t>
__global__ void chain_rule_color_position_global_kernel(
    const uint32_t       N,
    const glm::vec3     *p_k,               // [N,3]
    const glm::vec3      camera_center,     // broadcast
    const glm::vec3     *color_dir_grad,    // [N,3]
    const scalar_t      *color_dir_hess,    // [N,6]
    glm::vec3           *color_pos_grad,    // [N,3] output
    scalar_t            *color_pos_hess     // [N,6] output
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) return;

    // call your device function per‐sample
    chain_rule_color_position_kernel<scalar_t>(
        p_k[idx],
        camera_center,
        color_dir_grad[idx],
        color_dir_hess + idx * 6,
        &color_pos_grad[idx],
        color_pos_hess + idx * 6
    );
}

void launch_chain_rule_color_position_kernel(
    const at::Tensor& p_k,             // [...,3]
    const at::Tensor& camera_center,   // [3]
    const at::Tensor& color_dir_grad,  // [...,3]
    const at::Tensor& color_dir_hess,  // [...,6]
    at::Tensor&       color_pos_grad,  // [...,3]
    at::Tensor&       color_pos_hess   // [...,6]
) {
    const uint32_t N = p_k.numel() / 3;
    if (N == 0) return;
    /*{
      // return two empty tensors of shape [0,3] and [0,6]
      auto empty_grad = at::empty({0,3}, p_k_.options());
      auto empty_hess = at::empty({0,6}, p_k_.options());
      return { empty_grad, empty_hess };
    }
    auto pos_grad = at::empty({(int64_t)N, 3}, p_k_.options());
    auto pos_hess = at::empty({(int64_t)N, 6}, p_k_.options());
    */
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(p_k.scalar_type(),
        "chain_rule_color_position_global_kernel", [&] {
        // load camera_center once
        glm::vec3 cc = *reinterpret_cast<const glm::vec3*>(
            camera_center.data_ptr<scalar_t>()
        );
        chain_rule_color_position_global_kernel<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            N,
            reinterpret_cast<const glm::vec3*>(p_k.data_ptr<scalar_t>()),
            cc,
            reinterpret_cast<const glm::vec3*>(color_dir_grad.data_ptr<scalar_t>()),
            color_dir_hess.data_ptr<scalar_t>(),
            reinterpret_cast<glm::vec3*>(color_pos_grad.data_ptr<scalar_t>()),
            color_pos_hess.data_ptr<scalar_t>()
        );
    });

    //return { pos_grad, pos_hess };
}


// Wrapper kernel for batched processing
template <typename scalar_t>
__global__ void chain_rule_color_position_batched(
    const glm::vec3 *positions,         // [N]
    const glm::vec3 *camera_center,     // [1] or [N]
    const glm::vec3 *color_dir_grad,    // [N]
    const scalar_t *color_dir_hess,     // [N*6] packed as [xx, yy, zz, xy, xz, yz]
    glm::vec3 *color_pos_grad,          // [N]
    scalar_t *color_pos_hess,           // [N*6] packed as [xx, yy, zz, xy, xz, yz]
    int N
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    const glm::vec3 cam_center = camera_center[0]; // Assuming single camera center for all points

    chain_rule_color_position_kernel<scalar_t>(
        positions[k],
        cam_center,
        color_dir_grad[k],
        color_dir_hess ? &color_dir_hess[k*6] : nullptr,
        &color_pos_grad[k],
        color_pos_hess ? &color_pos_hess[k*6] : nullptr
    );
}

template <typename scalar_t>
__device__ void sh_coeffs_to_color_fast_LN(
    const uint32_t degree,    // degree of SH to be evaluated
    const uint32_t c,         // color channel
    const vec3 &dir,          // [3]
    const scalar_t *coeffs,   // [K, 3]
    const scalar_t *v_colors, // [3]
    // output
    scalar_t *v_coeffs, // [K, 3]
    vec3 *v_dir,        // [3] optional (in LN this is dc~/drk)
    scalar_t* H_dir     // [6] (in LN this is d2c~/drk2) stored like: [Hxx, Hyy, Hzz, Hxy, Hxz, Hyz]
) {
    float v_colors_local = v_colors[c];
    if (c == 0) { // Only zero out on the first channel call
        if (v_dir != nullptr) { v_dir->x = 0.f; v_dir->y = 0.f; v_dir->z = 0.f; }
        if (H_dir != nullptr) {
            H_dir[0] = 0.f; H_dir[1] = 0.f; H_dir[2] = 0.f;
            H_dir[3] = 0.f; H_dir[4] = 0.f; H_dir[5] = 0.f;
        }
    }

    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x = dir.x * inorm;
    float y = dir.y * inorm;
    float z = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    // --- Degree 0 ---
    v_coeffs[c] = 0.2820947917738781f * v_colors_local;
    if (degree < 1) return;

    // --- Degree 1 ---
    v_coeffs[1 * 3 + c] = -0.48860251190292f * y * v_colors_local;
    v_coeffs[2 * 3 + c] = 0.48860251190292f * z * v_colors_local;
    v_coeffs[3 * 3 + c] = -0.48860251190292f * x * v_colors_local;

    if (v_dir != nullptr) {
        v_x += -0.48860251190292f * coeffs[3 * 3 + c];
        v_y += -0.48860251190292f * coeffs[1 * 3 + c];
        v_z += 0.48860251190292f * coeffs[2 * 3 + c];
    }

    // --- Degree 2 ---
    if (degree >= 2) {
        float z2 = z * z;
        float x2 = x * x, y2 = y * y;
        float fTmp0B = -1.092548430592079f * z;
        float fC1 = x2 - y2;
        float fS1 = 2.f * x * y;
        v_coeffs[4 * 3 + c] = (0.5462742152960395f * fS1) * v_colors_local;
        v_coeffs[5 * 3 + c] = (fTmp0B * y) * v_colors_local;
        v_coeffs[6 * 3 + c] = (0.9461746957575601f * z2 - 0.3153915652525201f) * v_colors_local;
        v_coeffs[7 * 3 + c] = (fTmp0B * x) * v_colors_local;
        v_coeffs[8 * 3 + c] = (0.5462742152960395f * fC1) * v_colors_local;

        if (v_dir != nullptr) {
            float fC1_x = 2.f*x, fC1_y = -2.f*y;
            float fS1_x = 2.f*y, fS1_y = 2.f*x;
            v_x += (0.5462742152960395f * fS1_x) * coeffs[4*3+c] + (fTmp0B) * coeffs[7*3+c] + (0.5462742152960395f * fC1_x) * coeffs[8*3+c];
            v_y += (0.5462742152960395f * fS1_y) * coeffs[4*3+c] + (fTmp0B) * coeffs[5*3+c] + (0.5462742152960395f * fC1_y) * coeffs[8*3+c];
            v_z += (-1.092548430592079f * y) * coeffs[5*3+c] + (2.f * 0.9461746957575601f * z) * coeffs[6*3+c] + (-1.092548430592079f * x) * coeffs[7*3+c];

            float w = v_colors_local;
            H_dir[0] += w * (2.f * 0.5462742152960395f) * coeffs[8*3+c];
            H_dir[1] += w * (-2.f * 0.5462742152960395f) * coeffs[8*3+c];
            H_dir[2] += w * (2.f * 0.9461746957575601f) * coeffs[6*3+c];
            H_dir[3] += w * (2.f * 0.5462742152960395f) * coeffs[4*3+c];
            H_dir[4] += w * (-1.092548430592079f) * coeffs[7*3+c];
            H_dir[5] += w * (-1.092548430592079f) * coeffs[5*3+c];
        }

        // --- Degree 3 ---
        if (degree >= 3) {
            const float alpha3 = -0.5900435899266435f;
            const float beta3  =  1.445305721320277f;
            const float gamma3 = -2.285228997322329f;
            float fTmp0C = gamma3 * z2 + 0.4570457994644658f;
            float fTmp1B = beta3 * z;
            float fC2 = x * fC1 - y * fS1;
            float fS2 = x * fS1 + y * fC1;
            v_coeffs[9 * 3 + c]  = (alpha3 * fS2) * v_colors_local;
            v_coeffs[10 * 3 + c] = (fTmp1B * fS1) * v_colors_local;
            v_coeffs[11 * 3 + c] = (fTmp0C * y) * v_colors_local;
            v_coeffs[12 * 3 + c] = (z * (1.865881662950577f * z2 - 1.119528997770346f)) * v_colors_local;
            v_coeffs[13 * 3 + c] = (fTmp0C * x) * v_colors_local;
            v_coeffs[14 * 3 + c] = (fTmp1B * fC1) * v_colors_local;
            v_coeffs[15 * 3 + c] = (alpha3 * fC2) * v_colors_local;

            if (v_dir != nullptr) {
                v_x += alpha3 * (6*x*y) * coeffs[9*3+c] + (beta3*2*y*z) * coeffs[10*3+c] + fTmp0C * coeffs[13*3+c] + (beta3*2*x*z) * coeffs[14*3+c] + alpha3 * (3*x2-3*y2) * coeffs[15*3+c];
                v_y += alpha3 * (3*x2-3*y2) * coeffs[9*3+c] + (beta3*2*x*z) * coeffs[10*3+c] + fTmp0C * coeffs[11*3+c] + (beta3*-2*y*z) * coeffs[14*3+c] + alpha3 * (-6*x*y) * coeffs[15*3+c];
                v_z += (beta3*fS1) * coeffs[10*3+c] + (gamma3*2*z*y) * coeffs[11*3+c] + (3*1.865881662950577f*z2 - 1.119528997770346f) * coeffs[12*3+c] + (gamma3*2*z*x) * coeffs[13*3+c] + (beta3*fC1) * coeffs[14*3+c];

                float w = v_colors_local;
                H_dir[0] += w * (alpha3*6*y*coeffs[9*3+c] + beta3*2*z*coeffs[14*3+c] + alpha3*6*x*coeffs[15*3+c]);
                H_dir[1] += w * (alpha3*-6*y*coeffs[9*3+c] + beta3*-2*z*coeffs[14*3+c] + alpha3*-6*x*coeffs[15*3+c]);
                H_dir[2] += w * (gamma3*2*y*coeffs[11*3+c] + 6*1.865881662950577f*z*coeffs[12*3+c] + gamma3*2*x*coeffs[13*3+c]);
                H_dir[3] += w * (alpha3*6*x*coeffs[9*3+c] + beta3*2*z*coeffs[10*3+c] + alpha3*-6*y*coeffs[15*3+c]);
                H_dir[4] += w * (beta3*2*y*coeffs[10*3+c] + gamma3*2*z*coeffs[13*3+c] + beta3*2*x*coeffs[14*3+c]);
                H_dir[5] += w * (beta3*2*x*coeffs[10*3+c] + gamma3*2*z*coeffs[11*3+c] + beta3*-2*y*coeffs[14*3+c]);
            }

            // --- Degree 4 ---
            if (degree >= 4) {
                float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                float fTmp2B = -1.770130769779931f * z;
                float fC3 = x * fC2 - y * fS2;
                float fS3 = x * fS2 + y * fC2;
                float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f); // Re-use from degree 3
                float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);   // Re-use from degree 2
                v_coeffs[16 * 3 + c] = (0.6258357354491763f * fS3) * v_colors_local;
                v_coeffs[17 * 3 + c] = (fTmp2B * fS2) * v_colors_local;
                v_coeffs[18 * 3 + c] = (fTmp1C * fS1) * v_colors_local;
                v_coeffs[19 * 3 + c] = (fTmp0D * y) * v_colors_local;
                v_coeffs[20 * 3 + c] = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6) * v_colors_local;
                v_coeffs[21 * 3 + c] = (fTmp0D * x) * v_colors_local;
                v_coeffs[22 * 3 + c] = (fTmp1C * fC1) * v_colors_local;
                v_coeffs[23 * 3 + c] = (fTmp2B * fC2) * v_colors_local;
                v_coeffs[24 * 3 + c] = (0.6258357354491763f * fC3) * v_colors_local;

                if (v_dir != nullptr) {
                    // First derivative helpers
                    float fC1_x = 2.f*x, fC1_y = -2.f*y, fS1_x = 2.f*y, fS1_y = 2.f*x; // From degree 2
                    float fC2_x = fC1+x*fC1_x-y*fS1_x, fC2_y = x*fC1_y-fS1-y*fS1_y;
                    float fS2_x = fS1+x*fS1_x+y*fC1_x, fS2_y = x*fS1_y+fC1+y*fC1_y; // From degree 3
                    float fC3_x = fC2+x*fC2_x-y*fS2_x, fC3_y = x*fC2_y-fS2-y*fS2_y;
                    float fS3_x = fS2+x*fS2_x+y*fC2_x, fS3_y = x*fS2_y+fC2+y*fC2_y;
                    float fTmp0D_z = -14.049977414703075f*z2 + 2.007139630671868f;
                    float fTmp1C_z = 6.62322287030292f * z;
                    float fTmp2B_z = -1.770130769779931f;
                    float pSH12_z = 5.597645000085131f*z2 - 1.119528997770346f;
                    float pSH6_z = 1.8923493915151202f * z;
                    float pSH20_z = 1.984313483298443f*(pSH12 + z*pSH12_z) - 1.006230589874905f*pSH6_z;

                    // Accumulate first derivatives for degree 4
                    v_x += (0.6258357354491763f*fS3_x)*coeffs[16*3+c] + (fTmp2B*fS2_x)*coeffs[17*3+c] + (fTmp1C*fS1_x)*coeffs[18*3+c] + fTmp0D*coeffs[21*3+c] + (fTmp1C*fC1_x)*coeffs[22*3+c] + (fTmp2B*fC2_x)*coeffs[23*3+c] + (0.6258357354491763f*fC3_x)*coeffs[24*3+c];
                    v_y += (0.6258357354491763f*fS3_y)*coeffs[16*3+c] + (fTmp2B*fS2_y)*coeffs[17*3+c] + (fTmp1C*fS1_y)*coeffs[18*3+c] + fTmp0D*coeffs[19*3+c] + (fTmp1C*fC1_y)*coeffs[22*3+c] + (fTmp2B*fC2_y)*coeffs[23*3+c] + (0.6258357354491763f*fC3_y)*coeffs[24*3+c];
                    v_z += (fTmp2B_z*fS2)*coeffs[17*3+c] + (fTmp1C_z*fS1)*coeffs[18*3+c] + (fTmp0D_z*y)*coeffs[19*3+c] + pSH20_z*coeffs[20*3+c] + (fTmp0D_z*x)*coeffs[21*3+c] + (fTmp1C_z*fC1)*coeffs[22*3+c] + (fTmp2B_z*fC2)*coeffs[23*3+c];

                    // Accumulate Hessian (second derivatives) for degree 4
                    float w = v_colors_local;
                    float xy = x*y, xz = x*z, yz = y*z;
                    const float c16 = 0.6258357354491763f;
                    const float c17 = -1.770130769779931f;
                    const float c18_a = 3.31161143515146f;
                    const float c19_a = -4.683325804901025f, c19_b = 2.007139630671868f;
                    const float c20_a = 1.984313483298443f, c20_b = -1.006230589874905f, c20_c = 1.865881662950577f, c20_d = -1.119528997770346f, c20_e = 0.9461746957575601f;

                    H_dir[0] += w * (c16*(24.f*xy)*coeffs[16*3+c] + c17*(6.f*yz)*coeffs[17*3+c] + (c18_a*z2-c19_b)*2.f*coeffs[22*3+c] + c17*6.f*xz*coeffs[23*3+c] + c16*(12.f*x2-12.f*y2)*coeffs[24*3+c]);
                    H_dir[1] += w * (c16*(-24.f*xy)*coeffs[16*3+c] + c17*(-6.f*yz)*coeffs[17*3+c] + (c18_a*z2-c19_b)*-2.f*coeffs[22*3+c] + c17*(-6.f*xz)*coeffs[23*3+c] + c16*(-12.f*x2+12.f*y2)*coeffs[24*3+c]);
                    H_dir[2] += w * (c19_a*(-6.f*z)*y*coeffs[19*3+c] + (12.f*c20_a*c20_c*z2+2.f*(c20_a*c20_d+c20_b*c20_e))*coeffs[20*3+c] + c19_a*(-6.f*z)*x*coeffs[21*3+c] + c18_a*2.f*(x2-y2)*coeffs[22*3+c]);
                    H_dir[3] += w * (c16*(12.f*x2-12.f*y2)*coeffs[16*3+c] + c17*(6.f*xz)*coeffs[17*3+c] + (c18_a*z2-c19_b)*2.f*coeffs[18*3+c] + c17*(-6.f*yz)*coeffs[23*3+c] + c16*(-24.f*xy)*coeffs[24*3+c]);
                    H_dir[4] += w * (c17*(6.f*xy)*coeffs[17*3+c] + c18_a*4.f*yz*coeffs[18*3+c] + (c19_a*(-3.f*z2)+c19_b)*coeffs[21*3+c] + c18_a*4.f*xz*coeffs[22*3+c] + c17*(3.f*x2-3.f*y2)*coeffs[23*3+c]);
                    H_dir[5] += w * (c17*(3.f*x2-3.f*y2)*coeffs[17*3+c] + c18_a*4.f*xz*coeffs[18*3+c] + (c19_a*(-3.f*z2)+c19_b)*coeffs[19*3+c] + c18_a*(-4.f*yz)*coeffs[22*3+c] + c17*(-6.f*xy)*coeffs[23*3+c]);
                }
            }
        }
    }

    // Final projection of derivatives
    if (v_dir != nullptr) {
        vec3 dir_n = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x * v_colors_local, v_y * v_colors_local, v_z * v_colors_local);
        vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;
        // Accumulate results from this channel
        v_dir->x += v_d.x;
        v_dir->y += v_d.y;
        v_dir->z += v_d.z;
    }
}


template <typename scalar_t>
__global__ void spherical_harmonics_LN_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3   *__restrict__ dirs,        // [N, 3]
    const scalar_t *__restrict__ coeffs,    // [N, K, 3]
    const scalar_t *__restrict__ v_colors,  // [N, 3]
    // outputs
    scalar_t       *__restrict__ out_v_coeffs, // [N, K, 3]
    vec3           *__restrict__ out_v_dir,    // [N, 3]
    scalar_t       *__restrict__ out_H_dir     // [N, 6]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) return;

    // pointers for this sample
    const vec3    dir         = dirs[idx];
    const scalar_t* coeffs_ptr = coeffs    + idx * K * 3;
    const scalar_t* vc_ptr     = v_colors  + idx * 3;
          scalar_t* vc_out_ptr = out_v_coeffs + idx * K * 3;
          vec3*     vd_ptr     = out_v_dir    ? &out_v_dir[idx] : nullptr;
          scalar_t* h_ptr      = out_H_dir    ? out_H_dir + idx * 6 : nullptr;

    // run all 3 channels in one thread
    for (uint32_t c = 0; c < 3; ++c) {
        sh_coeffs_to_color_fast_LN<scalar_t>(
            degrees_to_use,
            c,
            dir,
            coeffs_ptr,
            vc_ptr,
            vc_out_ptr,
            vd_ptr,
            h_ptr
        );
    }
}

void launch_spherical_harmonics_LN_kernel(
    const uint32_t      degrees_to_use,
    const at::Tensor&   dirs,       // [..., 3]
    const at::Tensor&   coeffs,     // [..., K, 3]
    const at::Tensor&   v_colors,   // dc_RAST / dc_SH
    //outputs
    at::Tensor&         v_coeffs,   // dc_RAST / dc_attribute
    at::Tensor&         v_dir,      // [..., 3]     (dc_RAST / dr)
    at::Tensor&         H_dir       // [..., 6]     (d2c_RAST / dr2)
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;
    if (N == 0) return;
    /*
    if (N == 0) {
      // return three empty tensors with the right shape
      return {
        at::empty({0, K, 3}, dirs_.options()),
        at::empty({0,   3}, dirs_.options()),
        at::empty({0,   6}, dirs_.options())
      };
    }
    */

    // create outputs
    //auto v_coeffs = at::empty({(int64_t)N, (int64_t)K, 3}, dirs_.options());
    //auto v_dir    = at::empty({(int64_t)N,          3}, dirs_.options());
    //auto H_dir    = at::empty({(int64_t)N,          6}, dirs_.options());

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    //at::Tensor d_color_d_dir = at::empty(grad_shape, dirs.options());
    AT_DISPATCH_FLOATING_TYPES(dirs.scalar_type(), "spherical_harmonics_LN_kernel", [&] {
        spherical_harmonics_LN_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            N,
            K,
            degrees_to_use,
            reinterpret_cast<const vec3*>(dirs.data_ptr<scalar_t>()),
            coeffs.data_ptr<scalar_t>(),
            v_colors.data_ptr<scalar_t>(),
            v_coeffs.data_ptr<scalar_t>(),
            reinterpret_cast<vec3*>(v_dir.data_ptr<scalar_t>()),
            H_dir.data_ptr<scalar_t>()
        );
    });
    //return { out_v_coeffs, out_v_dir, out_H_dir };
}


template<typename scalar_t>
__global__ void color_solve_fwd_kernel(
    const uint32_t N,               // number of pixels
    const uint32_t K,               // number of SH‐bases per pixel
    const scalar_t* __restrict__ dc_dcSH,  // [N, K]    pre‐computed per‐pixel red‐term (and green, blue if you pack them)
    const scalar_t* __restrict__ B,        // [N, K, 3] SH‐basis rows for R,G,B
    scalar_t* __restrict__ grad_c          // [N, 3]   ∂c/∂c_k summed over k, for R,G,B
) {
  // one thread per (pixel,channel)
  uint32_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= N*3) return;

  uint32_t pix = idx / 3;       // which pixel
  uint32_t c   = idx % 3;       // which channel (0=R,1=G,2=B)

  // pointer‐offset into dc_dcSH and B
  // we assume you have packed your three channels of dc_dcSH
  // into a single array [N,K,3] in the same layout as B;
  // if you actually have three separate [N,K] arrays, just
  // load the correct one here.
  const scalar_t* term_base = dc_dcSH + pix*K*3;
  const scalar_t* B_base    = B      + pix*K*3;

  scalar_t sum = 0;
  for (uint32_t k = 0; k < K; ++k) {
    // dc_dcSH[pix,k,c] * B[pix,k,c]
    sum += term_base[k*3 + c] * B_base[k*3 + c];
  }

  grad_c[idx] = sum;
}

//
// Host‐side launcher (similar style to your SH‐kernels)
//
void launch_color_solve_fwd(
    const uint32_t N,
    const uint32_t K,
    const at::Tensor& dc_dcSH,   // [..., K, 3]
    const at::Tensor& B,         // [..., K, 3]
    at::Tensor& grad_c           // [..., 3]
) {
  const auto n_elements = N*3;
  const dim3 threads(256);
  const dim3 grid((n_elements + threads.x - 1) / threads.x);

  AT_DISPATCH_FLOATING_TYPES(
    dc_dcSH.scalar_type(),
    "color_solve_fwd_kernel",
    [&] {
      color_solve_fwd_kernel<scalar_t><<<
          grid, threads, 0,
          at::cuda::getCurrentCUDAStream()>>>(
        N,
        K,
        dc_dcSH.data_ptr<scalar_t>(),
        B.data_ptr<scalar_t>(),
        grad_c.data_ptr<scalar_t>()
      );
    }
  );
}