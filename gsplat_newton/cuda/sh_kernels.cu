#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

using namespace gsplat;

namespace gsplat_newton {
// TODO: add CDIM as template parameter
template <typename scalar_t>
__global__ void spherical_harmonics_LN_kernel(
    const uint32_t N,
    const uint32_t K,
    const uint32_t degrees_to_use,
    const vec3   *__restrict__ dirs,        // [N, 3]
    const scalar_t *__restrict__ coeffs,    // [N, K, 3]
    const scalar_t *__restrict__ dL_dcSH,  // [N, 1] because value is the same for each channel
    // outputs
    scalar_t       *__restrict__ out_v_coeffs, // [N, K, 3]
    scalar_t       *__restrict__ out_H_coeffs, // [N, K ,3]
    vec3           *__restrict__ out_v_dir,    // [N, CDIM, 3]
    scalar_t       *__restrict__ out_H_dir     // [N, CDIM, 6]
) {
    uint32_t total_idx = cg::this_grid().thread_rank();
    uint32_t sample_idx = total_idx / 3;  // Which sample (0 to N-1)
    uint32_t channel = total_idx % 3;     // Which channel (0, 1, 2)

    if (sample_idx >= N) return;

    // Create a cooperative group of 4 threads processing the same sample (only 3 do the work)
    auto tile = cg::tiled_partition<4>(cg::this_thread_block());
    bool active = (tile.thread_rank() < 3);

    // Get inputs for this sample and channel
    const vec3 dir = dirs[sample_idx];
    const scalar_t* coeffs_ptr = coeffs + sample_idx * K * 3;
    const scalar_t* vc_ptr = dL_dcSH + sample_idx * 3;
    scalar_t* vc_out_ptr = out_v_coeffs + sample_idx * K * 3;
    scalar_t* hc_out_ptr = out_H_coeffs ? out_H_coeffs + sample_idx * K * 3 : nullptr;
    vec3* vd_ptr = out_v_dir ? &out_v_dir[sample_idx] : nullptr;
    scalar_t* h_ptr = out_H_dir ? out_H_dir + sample_idx * 6 : nullptr;

    float v_colors_local = vc_ptr[channel];  // Same for all channels

    // Local accumulators for Hessian and gradient
    float local_H_dir[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    vec3 local_v_dir = vec3(0.f, 0.f, 0.f);
    if (active) {
    // Assume output tensors are pre-initialized to zeros

    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x = dir.x * inorm;
    float y = dir.y * inorm;
    float z = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    // --- Degree 0 ---
    float basis_0 = 0.2820947917738781f;
    vc_out_ptr[channel] = basis_0 * v_colors_local;
    if (hc_out_ptr != nullptr) {
        hc_out_ptr[channel] = basis_0 * basis_0 * v_colors_local;
    }
    if (degrees_to_use < 1) return;

    // --- Degree 1 ---
    float basis_1 = -0.48860251190292f * y;
    float basis_2 = 0.48860251190292f * z;
    float basis_3 = -0.48860251190292f * x;

    vc_out_ptr[1 * 3 + channel] = basis_1 * v_colors_local;
    vc_out_ptr[2 * 3 + channel] = basis_2 * v_colors_local;
    vc_out_ptr[3 * 3 + channel] = basis_3 * v_colors_local;

    if (hc_out_ptr != nullptr) {
        hc_out_ptr[1 * 3 + channel] = basis_1 * basis_1 * v_colors_local;
        hc_out_ptr[2 * 3 + channel] = basis_2 * basis_2 * v_colors_local;
        hc_out_ptr[3 * 3 + channel] = basis_3 * basis_3 * v_colors_local;
    }

    if (vd_ptr != nullptr) {
        v_x += -0.48860251190292f * coeffs_ptr[3 * 3 + channel];
        v_y += -0.48860251190292f * coeffs_ptr[1 * 3 + channel];
        v_z += 0.48860251190292f * coeffs_ptr[2 * 3 + channel];
    }

    // --- Degree 2 ---
    if (degrees_to_use >= 2) {
        float z2 = z * z;
        float x2 = x * x, y2 = y * y;
        float fTmp0B = -1.092548430592079f * z;
        float fC1 = x2 - y2;
        float fS1 = 2.f * x * y;

        float basis_4 = 0.5462742152960395f * fS1;
        float basis_5 = fTmp0B * y;
        float basis_6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
        float basis_7 = fTmp0B * x;
        float basis_8 = 0.5462742152960395f * fC1;

        vc_out_ptr[4 * 3 + channel] = basis_4 * v_colors_local;
        vc_out_ptr[5 * 3 + channel] = basis_5 * v_colors_local;
        vc_out_ptr[6 * 3 + channel] = basis_6 * v_colors_local;
        vc_out_ptr[7 * 3 + channel] = basis_7 * v_colors_local;
        vc_out_ptr[8 * 3 + channel] = basis_8 * v_colors_local;

        if (hc_out_ptr != nullptr) {
            hc_out_ptr[4 * 3 + channel] = basis_4 * basis_4 * v_colors_local;
            hc_out_ptr[5 * 3 + channel] = basis_5 * basis_5 * v_colors_local;
            hc_out_ptr[6 * 3 + channel] = basis_6 * basis_6 * v_colors_local;
            hc_out_ptr[7 * 3 + channel] = basis_7 * basis_7 * v_colors_local;
            hc_out_ptr[8 * 3 + channel] = basis_8 * basis_8 * v_colors_local;
        }

        if (vd_ptr != nullptr) {
            float fC1_x = 2.f*x, fC1_y = -2.f*y;
            float fS1_x = 2.f*y, fS1_y = 2.f*x;
            v_x += (0.5462742152960395f * fS1_x) * coeffs_ptr[4*3+channel] + (fTmp0B) * coeffs_ptr[7*3+channel] + (0.5462742152960395f * fC1_x) * coeffs_ptr[8*3+channel];
            v_y += (0.5462742152960395f * fS1_y) * coeffs_ptr[4*3+channel] + (fTmp0B) * coeffs_ptr[5*3+channel] + (0.5462742152960395f * fC1_y) * coeffs_ptr[8*3+channel];
            v_z += (-1.092548430592079f * y) * coeffs_ptr[5*3+channel] + (2.f * 0.9461746957575601f * z) * coeffs_ptr[6*3+channel] + (-1.092548430592079f * x) * coeffs_ptr[7*3+channel];

            float w = v_colors_local;
            local_H_dir[0] += w * (2.f * 0.5462742152960395f) * coeffs_ptr[8*3+channel];
            local_H_dir[1] += w * (-2.f * 0.5462742152960395f) * coeffs_ptr[8*3+channel];
            local_H_dir[2] += w * (2.f * 0.9461746957575601f) * coeffs_ptr[6*3+channel];
            local_H_dir[3] += w * (2.f * 0.5462742152960395f) * coeffs_ptr[4*3+channel];
            local_H_dir[4] += w * (-1.092548430592079f) * coeffs_ptr[7*3+channel];
            local_H_dir[5] += w * (-1.092548430592079f) * coeffs_ptr[5*3+channel];
        }

        // --- Degree 3 ---
        if (degrees_to_use >= 3) {
            const float alpha3 = -0.5900435899266435f;
            const float beta3  =  1.445305721320277f;
            const float gamma3 = -2.285228997322329f;
            float fTmp0C = gamma3 * z2 + 0.4570457994644658f;
            float fTmp1B = beta3 * z;
            float fC2 = x * fC1 - y * fS1;
            float fS2 = x * fS1 + y * fC1;

            float basis_9  = alpha3 * fS2;
            float basis_10 = fTmp1B * fS1;
            float basis_11 = fTmp0C * y;
            float basis_12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
            float basis_13 = fTmp0C * x;
            float basis_14 = fTmp1B * fC1;
            float basis_15 = alpha3 * fC2;

            vc_out_ptr[9 * 3 + channel]  = basis_9 * v_colors_local;
            vc_out_ptr[10 * 3 + channel] = basis_10 * v_colors_local;
            vc_out_ptr[11 * 3 + channel] = basis_11 * v_colors_local;
            vc_out_ptr[12 * 3 + channel] = basis_12 * v_colors_local;
            vc_out_ptr[13 * 3 + channel] = basis_13 * v_colors_local;
            vc_out_ptr[14 * 3 + channel] = basis_14 * v_colors_local;
            vc_out_ptr[15 * 3 + channel] = basis_15 * v_colors_local;

            if (hc_out_ptr != nullptr) {
                hc_out_ptr[9 * 3 + channel]  = basis_9 * basis_9 * v_colors_local;
                hc_out_ptr[10 * 3 + channel] = basis_10 * basis_10 * v_colors_local;
                hc_out_ptr[11 * 3 + channel] = basis_11 * basis_11 * v_colors_local;
                hc_out_ptr[12 * 3 + channel] = basis_12 * basis_12 * v_colors_local;
                hc_out_ptr[13 * 3 + channel] = basis_13 * basis_13 * v_colors_local;
                hc_out_ptr[14 * 3 + channel] = basis_14 * basis_14 * v_colors_local;
                hc_out_ptr[15 * 3 + channel] = basis_15 * basis_15 * v_colors_local;
            }

            if (vd_ptr != nullptr) {
                v_x += alpha3 * (6*x*y) * coeffs_ptr[9*3+channel] + (beta3*2*y*z) * coeffs_ptr[10*3+channel] + fTmp0C * coeffs_ptr[13*3+channel] + (beta3*2*x*z) * coeffs_ptr[14*3+channel] + alpha3 * (3*x2-3*y2) * coeffs_ptr[15*3+channel];
                v_y += alpha3 * (3*x2-3*y2) * coeffs_ptr[9*3+channel] + (beta3*2*x*z) * coeffs_ptr[10*3+channel] + fTmp0C * coeffs_ptr[11*3+channel] + (beta3*-2*y*z) * coeffs_ptr[14*3+channel] + alpha3 * (-6*x*y) * coeffs_ptr[15*3+channel];
                v_z += (beta3*fS1) * coeffs_ptr[10*3+channel] + (gamma3*2*z*y) * coeffs_ptr[11*3+channel] + (3*1.865881662950577f*z2 - 1.119528997770346f) * coeffs_ptr[12*3+channel] + (gamma3*2*z*x) * coeffs_ptr[13*3+channel] + (beta3*fC1) * coeffs_ptr[14*3+channel];

                float w = v_colors_local;
                local_H_dir[0] += w * (alpha3*6*y*coeffs_ptr[9*3+channel] + beta3*2*z*coeffs_ptr[14*3+channel] + alpha3*6*x*coeffs_ptr[15*3+channel]);
                local_H_dir[1] += w * (alpha3*-6*y*coeffs_ptr[9*3+channel] + beta3*-2*z*coeffs_ptr[14*3+channel] + alpha3*-6*x*coeffs_ptr[15*3+channel]);
                local_H_dir[2] += w * (gamma3*2*y*coeffs_ptr[11*3+channel] + 6*1.865881662950577f*z*coeffs_ptr[12*3+channel] + gamma3*2*x*coeffs_ptr[13*3+channel]);
                local_H_dir[3] += w * (alpha3*6*x*coeffs_ptr[9*3+channel] + beta3*2*z*coeffs_ptr[10*3+channel] + alpha3*-6*y*coeffs_ptr[15*3+channel]);
                local_H_dir[4] += w * (beta3*2*y*coeffs_ptr[10*3+channel] + gamma3*2*z*coeffs_ptr[13*3+channel] + beta3*2*x*coeffs_ptr[14*3+channel]);
                local_H_dir[5] += w * (beta3*2*x*coeffs_ptr[10*3+channel] + gamma3*2*z*coeffs_ptr[11*3+channel] + beta3*-2*y*coeffs_ptr[14*3+channel]);
            }

            // --- Degree 4 ---
            if (degrees_to_use >= 4) {
                float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                float fTmp2B = -1.770130769779931f * z;
                float fC3 = x * fC2 - y * fS2;
                float fS3 = x * fS2 + y * fC2;
                float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);

                float basis_16 = 0.6258357354491763f * fS3;
                float basis_17 = fTmp2B * fS2;
                float basis_18 = fTmp1C * fS1;
                float basis_19 = fTmp0D * y;
                float basis_20 = 1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6;
                float basis_21 = fTmp0D * x;
                float basis_22 = fTmp1C * fC1;
                float basis_23 = fTmp2B * fC2;
                float basis_24 = 0.6258357354491763f * fC3;

                vc_out_ptr[16 * 3 + channel] = basis_16 * v_colors_local;
                vc_out_ptr[17 * 3 + channel] = basis_17 * v_colors_local;
                vc_out_ptr[18 * 3 + channel] = basis_18 * v_colors_local;
                vc_out_ptr[19 * 3 + channel] = basis_19 * v_colors_local;
                vc_out_ptr[20 * 3 + channel] = basis_20 * v_colors_local;
                vc_out_ptr[21 * 3 + channel] = basis_21 * v_colors_local;
                vc_out_ptr[22 * 3 + channel] = basis_22 * v_colors_local;
                vc_out_ptr[23 * 3 + channel] = basis_23 * v_colors_local;
                vc_out_ptr[24 * 3 + channel] = basis_24 * v_colors_local;

                if (hc_out_ptr != nullptr) {
                    hc_out_ptr[16 * 3 + channel] = basis_16 * basis_16 * v_colors_local;
                    hc_out_ptr[17 * 3 + channel] = basis_17 * basis_17 * v_colors_local;
                    hc_out_ptr[18 * 3 + channel] = basis_18 * basis_18 * v_colors_local;
                    hc_out_ptr[19 * 3 + channel] = basis_19 * basis_19 * v_colors_local;
                    hc_out_ptr[20 * 3 + channel] = basis_20 * basis_20 * v_colors_local;
                    hc_out_ptr[21 * 3 + channel] = basis_21 * basis_21 * v_colors_local;
                    hc_out_ptr[22 * 3 + channel] = basis_22 * basis_22 * v_colors_local;
                    hc_out_ptr[23 * 3 + channel] = basis_23 * basis_23 * v_colors_local;
                    hc_out_ptr[24 * 3 + channel] = basis_24 * basis_24 * v_colors_local;
                }

                if (vd_ptr != nullptr) {
                    // First derivative helpers
                    float fC1_x = 2.f*x, fC1_y = -2.f*y, fS1_x = 2.f*y, fS1_y = 2.f*x;
                    float fC2_x = fC1+x*fC1_x-y*fS1_x, fC2_y = x*fC1_y-fS1-y*fS1_y;
                    float fS2_x = fS1+x*fS1_x+y*fC1_x, fS2_y = x*fS1_y+fC1+y*fC1_y;
                    float fC3_x = fC2+x*fC2_x-y*fS2_x, fC3_y = x*fC2_y-fS2-y*fS2_y;
                    float fS3_x = fS2+x*fS2_x+y*fC2_x, fS3_y = x*fS2_y+fC2+y*fC2_y;
                    float fTmp0D_z = -14.049977414703075f*z2 + 2.007139630671868f;
                    float fTmp1C_z = 6.62322287030292f * z;
                    float fTmp2B_z = -1.770130769779931f;
                    float pSH12_z = 5.597645000085131f*z2 - 1.119528997770346f;
                    float pSH6_z = 1.8923493915151202f * z;
                    float pSH20_z = 1.984313483298443f*(pSH12 + z*pSH12_z) - 1.006230589874905f*pSH6_z;

                    // Accumulate first derivatives for degree 4
                    v_x += (0.6258357354491763f*fS3_x)*coeffs_ptr[16*3+channel] + (fTmp2B*fS2_x)*coeffs_ptr[17*3+channel] + (fTmp1C*fS1_x)*coeffs_ptr[18*3+channel] + fTmp0D*coeffs_ptr[21*3+channel] + (fTmp1C*fC1_x)*coeffs_ptr[22*3+channel] + (fTmp2B*fC2_x)*coeffs_ptr[23*3+channel] + (0.6258357354491763f*fC3_x)*coeffs_ptr[24*3+channel];
                    v_y += (0.6258357354491763f*fS3_y)*coeffs_ptr[16*3+channel] + (fTmp2B*fS2_y)*coeffs_ptr[17*3+channel] + (fTmp1C*fS1_y)*coeffs_ptr[18*3+channel] + fTmp0D*coeffs_ptr[19*3+channel] + (fTmp1C*fC1_y)*coeffs_ptr[22*3+channel] + (fTmp2B*fC2_y)*coeffs_ptr[23*3+channel] + (0.6258357354491763f*fC3_y)*coeffs_ptr[24*3+channel];
                    v_z += (fTmp2B_z*fS2)*coeffs_ptr[17*3+channel] + (fTmp1C_z*fS1)*coeffs_ptr[18*3+channel] + (fTmp0D_z*y)*coeffs_ptr[19*3+channel] + pSH20_z*coeffs_ptr[20*3+channel] + (fTmp0D_z*x)*coeffs_ptr[21*3+channel] + (fTmp1C_z*fC1)*coeffs_ptr[22*3+channel] + (fTmp2B_z*fC2)*coeffs_ptr[23*3+channel];

                    // Accumulate Hessian (second derivatives) for degree 4
                    float w = v_colors_local;
                    float xy = x*y, xz = x*z, yz = y*z;
                    const float c16 = 0.6258357354491763f;
                    const float c17 = -1.770130769779931f;
                    const float c18_a = 3.31161143515146f;
                    const float c19_a = -4.683325804901025f, c19_b = 2.007139630671868f;
                    const float c20_a = 1.984313483298443f, c20_b = -1.006230589874905f, c20_c = 1.865881662950577f, c20_d = -1.119528997770346f, c20_e = 0.9461746957575601f;

                    local_H_dir[0] += w * (c16*(24.f*xy)*coeffs_ptr[16*3+channel] + c17*(6.f*yz)*coeffs_ptr[17*3+channel] + (c18_a*z2-c19_b)*2.f*coeffs_ptr[22*3+channel] + c17*6.f*xz*coeffs_ptr[23*3+channel] + c16*(12.f*x2-12.f*y2)*coeffs_ptr[24*3+channel]);
                    local_H_dir[1] += w * (c16*(-24.f*xy)*coeffs_ptr[16*3+channel] + c17*(-6.f*yz)*coeffs_ptr[17*3+channel] + (c18_a*z2-c19_b)*-2.f*coeffs_ptr[22*3+channel] + c17*(-6.f*xz)*coeffs_ptr[23*3+channel] + c16*(-12.f*x2+12.f*y2)*coeffs_ptr[24*3+channel]);
                    local_H_dir[2] += w * (c19_a*(-6.f*z)*y*coeffs_ptr[19*3+channel] + (12.f*c20_a*c20_c*z2+2.f*(c20_a*c20_d+c20_b*c20_e))*coeffs_ptr[20*3+channel] + c19_a*(-6.f*z)*x*coeffs_ptr[21*3+channel] + c18_a*2.f*(x2-y2)*coeffs_ptr[22*3+channel]);
                    local_H_dir[3] += w * (c16*(12.f*x2-12.f*y2)*coeffs_ptr[16*3+channel] + c17*(6.f*xz)*coeffs_ptr[17*3+channel] + (c18_a*z2-c19_b)*2.f*coeffs_ptr[18*3+channel] + c17*(-6.f*yz)*coeffs_ptr[23*3+channel] + c16*(-24.f*xy)*coeffs_ptr[24*3+channel]);
                    local_H_dir[4] += w * (c17*(6.f*xy)*coeffs_ptr[17*3+channel] + c18_a*4.f*yz*coeffs_ptr[18*3+channel] + (c19_a*(-3.f*z2)+c19_b)*coeffs_ptr[21*3+channel] + c18_a*4.f*xz*coeffs_ptr[22*3+channel] + c17*(3.f*x2-3.f*y2)*coeffs_ptr[23*3+channel]);
                    local_H_dir[5] += w * (c17*(3.f*x2-3.f*y2)*coeffs_ptr[17*3+channel] + c18_a*4.f*xz*coeffs_ptr[18*3+channel] + (c19_a*(-3.f*z2)+c19_b)*coeffs_ptr[19*3+channel] + c18_a*(-4.f*yz)*coeffs_ptr[22*3+channel] + c17*(-6.f*xy)*coeffs_ptr[23*3+channel]);
                }
            }
        }
    }

    // Final projection of derivatives
    if (vd_ptr != nullptr) {
        vec3 dir_n = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x * v_colors_local, v_y * v_colors_local, v_z * v_colors_local);
        vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

        // Store in local accumulator
        local_v_dir.x += v_d.x;
        local_v_dir.y += v_d.y;
        local_v_dir.z += v_d.z;
    }
    }
    // Reduce across the 3 channels using warp reduction
    if (vd_ptr != nullptr) {
        warpSum(local_v_dir, tile);
        // Only thread 0 of each 3-thread tile writes to global memory
        if (tile.thread_rank() == 0) {
            atomicAdd(&vd_ptr->x, local_v_dir.x);
            atomicAdd(&vd_ptr->y, local_v_dir.y);
            atomicAdd(&vd_ptr->z, local_v_dir.z);
        }
    }

    if (h_ptr != nullptr) {
        warpSum<6>(local_H_dir, tile);
        // Only thread 0 of each 3-thread tile writes to global memory
        if (tile.thread_rank() == 0) {
            atomicAdd(&h_ptr[0], local_H_dir[0]);
            atomicAdd(&h_ptr[1], local_H_dir[1]);
            atomicAdd(&h_ptr[2], local_H_dir[2]);
            atomicAdd(&h_ptr[3], local_H_dir[3]);
            atomicAdd(&h_ptr[4], local_H_dir[4]);
            atomicAdd(&h_ptr[5], local_H_dir[5]);
        }
    }
}

void launch_spherical_harmonics_LN_kernel(
    const uint32_t      degrees_to_use,
    const at::Tensor   dirs,       // [..., 3]
    const at::Tensor   coeffs,     // [..., K, 3]
    const at::Tensor   v_colors,   // dL / dc_SH
    //outputs
    at::Tensor         v_coeffs,   // dc_RAST / dc_attribute
    at::Tensor         H_coeffs,   // d2c_RAST / dc_attribute^2 (diagonal terms only)
    at::Tensor         v_dir,      // [..., 3]     (dL / dr)
    at::Tensor         H_dir       // [..., 6]     (d2L / dr2)
) {
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;
    if (N == 0) return;

    const int threads = 256;
    const int total_work = N * 4;  // N samples × 3 channels (4 tiles)
    const int blocks = (total_work + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(dirs.scalar_type(), "spherical_harmonics_LN_kernel", [&] {
        spherical_harmonics_LN_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            N,
            K,
            degrees_to_use,
            reinterpret_cast<const vec3*>(dirs.data_ptr<scalar_t>()),
            coeffs.data_ptr<scalar_t>(),
            v_colors.data_ptr<scalar_t>(),
            v_coeffs.data_ptr<scalar_t>(),
            H_coeffs.data_ptr<scalar_t>(),
            reinterpret_cast<vec3*>(v_dir.data_ptr<scalar_t>()),
            H_dir.data_ptr<scalar_t>()
        );
    });
}



template <typename scalar_t>
__global__ void chain_rule_color_position_kernel(
    const uint32_t       N,
    const int32_t* __restrict__ radii,
    const glm::vec3     *p_k,               // [N,3]
    const glm::vec3     *camera_center,     // broadcast
    const glm::vec3     *color_dir_grad,    // [N,3]
    const scalar_t      *color_dir_hess,    // [N,6]
    glm::vec3           *color_pos_grad,    // [N,3] output
    scalar_t            *color_pos_hess     // [N,6] output
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) return;

    // Get inputs for this thread
    const glm::vec3& position = p_k[idx];
    const glm::vec3& cam_center = camera_center[0];
    const glm::vec3& color_grad = color_dir_grad[idx];
    const scalar_t* input_hess = color_dir_hess + idx * 6;

    // 1. Compute ∂r/∂p and ∂²r/∂p²
    glm::vec3 d = position - cam_center;
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
    gc_p.x = color_grad.x * j00 + color_grad.y * j10 + color_grad.z * j20;
    gc_p.y = color_grad.x * j01 + color_grad.y * j11 + color_grad.z * j21;
    gc_p.z = color_grad.x * j02 + color_grad.y * j12 + color_grad.z * j22;
    color_pos_grad[idx] = gc_p;

    if (color_pos_hess == nullptr) return;

    // 3. Compute Hessian: H_p = Jᵀ H_r J + Σ_a (gc_r[a] * H_a)

    // Term1 = Jᵀ H_r J
    // Unpack input Hessian
    float h00 = input_hess[0], h11 = input_hess[1], h22 = input_hess[2];
    float h01 = input_hess[3], h02 = input_hess[4], h12 = input_hess[5];

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
    float T2_xx = color_grad.x * A(0,0,0) + color_grad.y * A(1,0,0) + color_grad.z * A(2,0,0);
    float T2_xy = color_grad.x * A(0,0,1) + color_grad.y * A(1,0,1) + color_grad.z * A(2,0,1);
    float T2_xz = color_grad.x * A(0,0,2) + color_grad.y * A(1,0,2) + color_grad.z * A(2,0,2);
    float T2_yy = color_grad.x * A(0,1,1) + color_grad.y * A(1,1,1) + color_grad.z * A(2,1,1);
    float T2_yz = color_grad.x * A(0,1,2) + color_grad.y * A(1,1,2) + color_grad.z * A(2,1,2);
    float T2_zz = color_grad.x * A(0,2,2) + color_grad.y * A(1,2,2) + color_grad.z * A(2,2,2);

    // Final Hessian output
    scalar_t* output_hess = color_pos_hess + idx * 6;
    output_hess[0] = T1_xx + T2_xx;  // xx
    output_hess[1] = T1_yy + T2_yy;  // yy
    output_hess[2] = T1_zz + T2_zz;  // zz
    output_hess[3] = T1_xy + T2_xy;  // xy
    output_hess[4] = T1_xz + T2_xz;  // xz
    output_hess[5] = T1_yz + T2_yz;  // yz
}

void launch_chain_rule_color_position_kernel(
    const at::Tensor p_k,             // [...,3]
    const at::Tensor radii,
    const at::Tensor camera_center,   // [3]
    const at::Tensor color_dir_grad,  // [...,3]
    const at::Tensor color_dir_hess,  // [...,6]
    at::Tensor       color_pos_grad,  // [...,3]
    at::Tensor       color_pos_hess   // [...,6]
) {
    const uint32_t N = p_k.numel() / 3;
    if (N == 0) return;

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(p_k.scalar_type(),
        "chain_rule_color_position_kernel", [&] {
        chain_rule_color_position_kernel<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()
        >>>(
            N,
            radii.data_ptr<int32_t>(),
            reinterpret_cast<const glm::vec3*>(p_k.data_ptr<scalar_t>()),
            reinterpret_cast<const glm::vec3*>(camera_center.data_ptr<scalar_t>()),
            reinterpret_cast<const glm::vec3*>(color_dir_grad.data_ptr<scalar_t>()),
            color_dir_hess.data_ptr<scalar_t>(),
            reinterpret_cast<glm::vec3*>(color_pos_grad.data_ptr<scalar_t>()),
            color_pos_hess.data_ptr<scalar_t>()
        );
    });
}
}
 // namespace gsplat_newton