#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

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
    const scalar_t *v_colors, // [3] to avoid ta
    // output
    scalar_t *v_coeffs, // [K, 3]
    vec3 *v_dir         // [3] optional (in LN this is dc~/drk)
    scalar_t* H_dir, // [6] (in LN this is d2c~/drk2) stored like: [Hxx, Hyy, Hzz, Hxy, Hxz, Hyz]
) {
    float v_colors_local = v_colors[c];
    if (H_dir != nullptr) {
        H_dir[0] = 0.f;
        H_dir[1] = 0.f;
        H_dir[2] = 0.f;
        H_dir[3] = 0.f;
        H_dir[4] = 0.f;
        H_dir[5] = 0.f;
    }

    v_coeffs[c] = 0.2820947917738781f * v_colors_local;
    if (degree < 1) {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x = dir.x * inorm;
    float y = dir.y * inorm;
    float z = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    v_coeffs[1 * 3 + c] = -0.48860251190292f * y * v_colors_local;
    v_coeffs[2 * 3 + c] = 0.48860251190292f * z * v_colors_local;
    v_coeffs[3 * 3 + c] = -0.48860251190292f * x * v_colors_local;

    if (v_dir != nullptr) {
        v_x += -0.48860251190292f * coeffs[3 * 3 + c] * v_colors_local;
        v_y += -0.48860251190292f * coeffs[1 * 3 + c] * v_colors_local;
        v_z += 0.48860251190292f * coeffs[2 * 3 + c] * v_colors_local;
    }
    if (degree < 2) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = 0.5462742152960395f * fC1;
    float pSH4 = 0.5462742152960395f * fS1;

    const float S4    = 0.5462742152960395f * 2.f;       // ∂² pSH4 / ∂x ∂y
    const float S5    = -1.092548430592079f;             // ∂² pSH5 / ∂y ∂z
    const float S6    = 2.f * 0.9461746957575601f;       // ∂² pSH6 / ∂z²
    const float S7    = -1.092548430592079f;             // ∂² pSH7 / ∂x ∂z
    const float S8_xx = 2.f * 0.5462742152960395f;       // ∂² pSH8 / ∂x²
    const float S8_yy = -2.f * 0.5462742152960395f;      // ∂² pSH8 / ∂y²

    v_coeffs[4 * 3 + c] = pSH4 * v_colors_local;
    v_coeffs[5 * 3 + c] = pSH5 * v_colors_local;
    v_coeffs[6 * 3 + c] = pSH6 * v_colors_local;
    v_coeffs[7 * 3 + c] = pSH7 * v_colors_local;
    v_coeffs[8 * 3 + c] = pSH8 * v_colors_local;

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
        pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if (v_dir != nullptr) {
        fTmp0B_z = -1.092548430592079f;
        fC1_x = 2.f * x;
        fC1_y = -2.f * y;
        fS1_x = 2.f * y;
        fS1_y = 2.f * x;
        pSH6_z = 2.f * 0.9461746957575601f * z;
        pSH7_x = fTmp0B;
        pSH7_z = fTmp0B_z * x;
        pSH5_y = fTmp0B;
        pSH5_z = fTmp0B_z * y;
        pSH8_x = 0.5462742152960395f * fC1_x;
        pSH8_y = 0.5462742152960395f * fC1_y;
        pSH4_x = 0.5462742152960395f * fS1_x;
        pSH4_y = 0.5462742152960395f * fS1_y;

        v_x += v_colors_local *
               (pSH4_x * coeffs[4 * 3 + c] + pSH8_x * coeffs[8 * 3 + c] +
                pSH7_x * coeffs[7 * 3 + c]);
        v_y += v_colors_local *
               (pSH4_y * coeffs[4 * 3 + c] + pSH8_y * coeffs[8 * 3 + c] +
                pSH5_y * coeffs[5 * 3 + c]);
        v_z += v_colors_local *
               (pSH6_z * coeffs[6 * 3 + c] + pSH7_z * coeffs[7 * 3 + c] +
                pSH5_z * coeffs[5 * 3 + c]);
        float w = v_colors_local;
        H_dir[0] += w * S8_xx * coeffs[8*3 + c];  // Hxx
        H_dir[1] += w * S8_yy * coeffs[8*3 + c];  // Hyy
        H_dir[2] += w * S6    * coeffs[6*3 + c];  // Hzz
        H_dir[3] += w * S4    * coeffs[4*3 + c];  // Hxy
        H_dir[4] += w * S7    * coeffs[7*3 + c];  // Hxz
        H_dir[5] += w * S5    * coeffs[5*3 + c];  // Hyz
    }

    if (degree < 3) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }
    const float alpha_ = -0.5900435899266435f;
    const float beta_ =  1.445305721320277f;
    const float gamma_ =  2.285228997322329f;

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = -0.5900435899266435f * fC2;
    float pSH9 = -0.5900435899266435f * fS2;
    v_coeffs[9 * 3 + c] = pSH9 * v_colors_local;
    v_coeffs[10 * 3 + c] = pSH10 * v_colors_local;
    v_coeffs[11 * 3 + c] = pSH11 * v_colors_local;
    v_coeffs[12 * 3 + c] = pSH12 * v_colors_local;
    v_coeffs[13 * 3 + c] = pSH13 * v_colors_local;
    v_coeffs[14 * 3 + c] = pSH14 * v_colors_local;
    v_coeffs[15 * 3 + c] = pSH15 * v_colors_local;

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
        pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
        pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (v_dir != nullptr) {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x = fC1 + x * fC1_x - y * fS1_x;
        fC2_y = x * fC1_y - fS1 - y * fS1_y;
        fS2_x = fS1 + x * fS1_x + y * fC1_x;
        fS2_y = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x = fTmp0C;
        pSH13_z = fTmp0C_z * x;
        pSH11_y = fTmp0C;
        pSH11_z = fTmp0C_z * y;
        pSH14_x = fTmp1B * fC1_x;
        pSH14_y = fTmp1B * fC1_y;
        pSH14_z = fTmp1B_z * fC1;
        pSH10_x = fTmp1B * fS1_x;
        pSH10_y = fTmp1B * fS1_y;
        pSH10_z = fTmp1B_z * fS1;
        pSH15_x = -0.5900435899266435f * fC2_x;
        pSH15_y = -0.5900435899266435f * fC2_y;
        pSH9_x = -0.5900435899266435f * fS2_x;
        pSH9_y = -0.5900435899266435f * fS2_y;

        v_x += v_colors_local *
               (pSH9_x * coeffs[9 * 3 + c] + pSH15_x * coeffs[15 * 3 + c] +
                pSH10_x * coeffs[10 * 3 + c] + pSH14_x * coeffs[14 * 3 + c] +
                pSH13_x * coeffs[13 * 3 + c]);

        v_y += v_colors_local *
               (pSH9_y * coeffs[9 * 3 + c] + pSH15_y * coeffs[15 * 3 + c] +
                pSH10_y * coeffs[10 * 3 + c] + pSH14_y * coeffs[14 * 3 + c] +
                pSH11_y * coeffs[11 * 3 + c]);

        v_z += v_colors_local *
               (pSH12_z * coeffs[12 * 3 + c] + pSH13_z * coeffs[13 * 3 + c] +
                pSH11_z * coeffs[11 * 3 + c] + pSH14_z * coeffs[14 * 3 + c] +
                pSH10_z * coeffs[10 * 3 + c]);
        // ℓ=3 SECOND‐PARTIALS
        const float S9_xx =  6.f * alpha_ * y;
        const float S9_xy =  6.f * alpha_ * x;
        const float S9_yy = -6.f * alpha_ * y;

        const float S10_xy = 2.f * beta_ * z;
        const float S10_xz = 2.f * beta_ * y;
        const float S10_yz = 2.f * beta_ * x;

        const float S11_yz = -2.f * gamma_ * z;
        const float S11_zz = -2.f * gamma_ * y;

        const float S12_zz =  6.f * 1.865881662950577f * z;

        const float S13_xz = -2.f * gamma_ * z;
        const float S13_zz = -2.f * gamma_ * x;

        const float S14_xx =  2.f * beta_ * z;
        const float S14_yy = -2.f * beta_ * z;
        const float S14_xz =  2.f * beta_ * x;
        const float S14_yz = -2.f * beta_ * y;

        const float S15_xx =  6.f * alpha_ * x;
        const float S15_xy = -6.f * alpha_ * y;
        const float S15_yy = -6.f * alpha_ * x;

        // accumulate ℓ=3
        H_dir[0]+=w_color*(S9_xx*coeffs[9*3+c]  + S14_xx*coeffs[14*3+c] + S15_xx*coeffs[15*3+c]);
        H_dir[1]+=w_color*(S9_yy*coeffs[9*3+c]  + S14_yy*coeffs[14*3+c] + S15_yy*coeffs[15*3+c]);
        H_dir[2]+=w_color*(S11_zz*coeffs[11*3+c]+ S12_zz*coeffs[12*3+c] + S13_zz*coeffs[13*3+c]);
        H_dir[3]+=w_color*(S9_xy*coeffs[9*3+c]  + S10_xy*coeffs[10*3+c] + S15_xy*coeffs[15*3+c]);
        H_dir[4]+=w_color*(S10_xz*coeffs[10*3+c]+ S11_yz*coeffs[11*3+c] + S13_xz*coeffs[13*3+c]);
        H_dir[5]+=w_color*(S10_yz*coeffs[10*3+c]+ S14_yz*coeffs[14*3+c] + S11_yz*coeffs[11*3+c]);
    }

    if (degree < 4) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }
    // auto‐generated second‐partials ℓ=4 (cleaned):
    const float S16_xx = 15.020057650780231f * x * y;
    const float S16_yy =-15.020057650780231f * x * y;
    const float S16_xy =  7.5100288253901155f * (x*x - y*y);

    const float S17_xx =-10.620784618679586f * y * z;
    const float S17_yy = 10.620784618679586f * y * z;
    const float S17_xy =-10.620784618679586f * x * z;
    const float S17_xz =-10.620784618679586f * x * y;
    const float S17_yz =-5.3103923093397931f * (x*x - y*y);

    const float S18_xx = 6.6232228703029197f * z*z - 0.94617469575756f;
    const float S18_yy =-S18_xx;
    const float S18_xy = 6.6232228703029197f * z*z - 0.94617469575756f;
    const float S18_xz = 13.246445740605839f * y * z;
    const float S18_yz = 13.246445740605839f * x * z;
    const float S18_zz = 13.246445740605839f * x * y;

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = 0.6258357354491763f * fC3;
    float pSH16 = 0.6258357354491763f * fS3;
    v_coeffs[16 * 3 + c] = pSH16 * v_colors_local;
    v_coeffs[17 * 3 + c] = pSH17 * v_colors_local;
    v_coeffs[18 * 3 + c] = pSH18 * v_colors_local;
    v_coeffs[19 * 3 + c] = pSH19 * v_colors_local;
    v_coeffs[20 * 3 + c] = pSH20 * v_colors_local;
    v_coeffs[21 * 3 + c] = pSH21 * v_colors_local;
    v_coeffs[22 * 3 + c] = pSH22 * v_colors_local;
    v_coeffs[23 * 3 + c] = pSH23 * v_colors_local;
    v_coeffs[24 * 3 + c] = pSH24 * v_colors_local;

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
        pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
        pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (v_dir != nullptr) {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x = fC2 + x * fC2_x - y * fS2_x;
        fC3_y = x * fC2_y - fS2 - y * fS2_y;
        fS3_x = fS2 + y * fC2_x + x * fS2_x;
        fS3_y = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
                  -1.006230589874905f * pSH6_z;
        pSH21_x = fTmp0D;
        pSH21_z = fTmp0D_z * x;
        pSH19_y = fTmp0D;
        pSH19_z = fTmp0D_z * y;
        pSH22_x = fTmp1C * fC1_x;
        pSH22_y = fTmp1C * fC1_y;
        pSH22_z = fTmp1C_z * fC1;
        pSH18_x = fTmp1C * fS1_x;
        pSH18_y = fTmp1C * fS1_y;
        pSH18_z = fTmp1C_z * fS1;
        pSH23_x = fTmp2B * fC2_x;
        pSH23_y = fTmp2B * fC2_y;
        pSH23_z = fTmp2B_z * fC2;
        pSH17_x = fTmp2B * fS2_x;
        pSH17_y = fTmp2B * fS2_y;
        pSH17_z = fTmp2B_z * fS2;
        pSH24_x = 0.6258357354491763f * fC3_x;
        pSH24_y = 0.6258357354491763f * fC3_y;
        pSH16_x = 0.6258357354491763f * fS3_x;
        pSH16_y = 0.6258357354491763f * fS3_y;

        v_x += v_colors_local *
               (pSH16_x * coeffs[16 * 3 + c] + pSH24_x * coeffs[24 * 3 + c] +
                pSH17_x * coeffs[17 * 3 + c] + pSH23_x * coeffs[23 * 3 + c] +
                pSH18_x * coeffs[18 * 3 + c] + pSH22_x * coeffs[22 * 3 + c] +
                pSH21_x * coeffs[21 * 3 + c]);
        v_y += v_colors_local *
               (pSH16_y * coeffs[16 * 3 + c] + pSH24_y * coeffs[24 * 3 + c] +
                pSH17_y * coeffs[17 * 3 + c] + pSH23_y * coeffs[23 * 3 + c] +
                pSH18_y * coeffs[18 * 3 + c] + pSH22_y * coeffs[22 * 3 + c] +
                pSH19_y * coeffs[19 * 3 + c]);
        v_z += v_colors_local *
               (pSH20_z * coeffs[20 * 3 + c] + pSH21_z * coeffs[21 * 3 + c] +
                pSH19_z * coeffs[19 * 3 + c] + pSH22_z * coeffs[22 * 3 + c] +
                pSH18_z * coeffs[18 * 3 + c] + pSH23_z * coeffs[23 * 3 + c] +
                pSH17_z * coeffs[17 * 3 + c]);

        vec3 dir_n = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x, v_y, v_z);
        vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

        v_dir->x = v_d.x;
        v_dir->y = v_d.y;
        v_dir->z = v_d.z;

            float w = w_color;

        // pSH16 (idx 16)
        H_dir[0] += w * S16_xx * coeffs[16*3 + c];  // Hxx
        H_dir[1] += w * S16_yy * coeffs[16*3 + c];  // Hyy
        H_dir[3] += w * S16_xy * coeffs[16*3 + c];  // Hxy

        // pSH17 (idx 17)
        H_dir[0] += w * S17_xx * coeffs[17*3 + c];  // Hxx
        H_dir[1] += w * S17_yy * coeffs[17*3 + c];  // Hyy
        H_dir[3] += w * S17_xy * coeffs[17*3 + c];  // Hxy
        H_dir[4] += w * S17_xz * coeffs[17*3 + c];  // Hxz
        H_dir[5] += w * S17_yz * coeffs[17*3 + c];  // Hyz

        // pSH18 (idx 18)
        H_dir[0] += w * S18_xx * coeffs[18*3 + c];  // Hxx
        H_dir[1] += w * S18_yy * coeffs[18*3 + c];  // Hyy
        H_dir[2] += w * S18_zz * coeffs[18*3 + c];  // Hzz
        H_dir[3] += w * S18_xy * coeffs[18*3 + c];  // Hxy
        H_dir[4] += w * S18_xz * coeffs[18*3 + c];  // Hxz
        H_dir[5] += w * S18_yz * coeffs[18*3 + c];  // Hyz

        // pSH19 (idx 19)
        H_dir[2] += w * S19_zz * coeffs[19*3 + c];  // Hzz
        H_dir[5] += w * S19_yz * coeffs[19*3 + c];  // Hyz

        // pSH20 (idx 20)
        H_dir[2] += w * S20_zz * coeffs[20*3 + c];  // Hzz

        // pSH21 (idx 21)
        H_dir[2] += w * S21_zz * coeffs[21*3 + c];  // Hzz
        H_dir[4] += w * S21_xz * coeffs[21*3 + c];  // Hxz

        // pSH22 (idx 22)
        H_dir[0] += w * S22_xx * coeffs[22*3 + c];  // Hxx
        H_dir[1] += w * S22_yy * coeffs[22*3 + c];  // Hyy
        H_dir[2] += w * S22_zz * coeffs[22*3 + c];  // Hzz
        H_dir[4] += w * S22_xz * coeffs[22*3 + c];  // Hxz
        H_dir[5] += w * S22_yz * coeffs[22*3 + c];  // Hyz

        // pSH23 (idx 23)
        H_dir[0] += w * S23_xx * coeffs[23*3 + c];  // Hxx
        H_dir[1] += w * S23_yy * coeffs[23*3 + c];  // Hyy
        H_dir[3] += w * S23_xy * coeffs[23*3 + c];  // Hxy
        H_dir[4] += w * S23_xz * coeffs[23*3 + c];  // Hxz
        H_dir[5] += w * S23_yz * coeffs[23*3 + c];  // Hyz

        // pSH24 (idx 24)
        H_dir[0] += w * S24_xx * coeffs[24*3 + c];  // Hxx
        H_dir[1] += w * S24_yy * coeffs[24*3 + c];  // Hyy
        H_dir[3] += w * S24_xy * coeffs[24*3 + c];  // Hxy
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
    const at::Tensor&   v_colors,   // [..., 3]
    at::Tensor&         v_coeffs,   // [..., K, 3]  (output)
    at::Tensor&         v_dir,      // [..., 3]     (output)
    at::Tensor&         H_dir       // [..., 6]     (output)
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;
    if (N == 0) return;

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

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
}