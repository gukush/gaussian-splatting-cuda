#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "gsplat/Common.h"
#include "gsplat/Rasterization.h"
#include "gsplat/Utils.cuh"


#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"
#include <cooperative_groups.h>
/**
 * @brief Kernel 1: Computes and aggregates intermediate derivatives for the
 * Newton-Raphson solver.
 *
 * This kernel performs the backward pass of the 3DGS rasterizer, calculating
 * the sum of all pixel-dependent derivatives for each Gaussian. It uses atomic
 * adds to aggregate contributions from different tiles and warps. The outputs of
 * this kernel serve as the direct inputs for the second assembly kernel.
 */
template <uint32_t CDIM, typename scalar_t>
__global__ void compute_intermediate_derivatives_kernel(
    const uint32_t C, const uint32_t N, const uint32_t n_isects, const bool packed,
    // --- Forward Pass Inputs ---
    const vec2 *__restrict__ means2d,
    const vec3 *__restrict__ conics,
    const scalar_t *__restrict__ colors,
    const scalar_t *__restrict__ opacities,
    const scalar_t *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // --- Forward Pass Outputs ---
    const scalar_t *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    // --- Backward Pass Inputs (Output Gradients) ---
    const scalar_t *__restrict__ v_render_colors,
    const scalar_t *__restrict__ v_render_alphas,

    // --- INTERMEDIATE OUTPUTS (per-Gaussian) ---
    scalar_t *__restrict__ dc_dcSH,      // Σ(∂c/∂c̃ₖ)
    scalar_t *__restrict__ dc_dG,        // Σ(∂c/∂Gₖ)
    vec2 *__restrict__ dG_dmean2d,       // Σ(∂Gₖ/∂πₖ)
    vec3 *__restrict__ dG_dSigma,        // Σ(∂Gₖ/∂Σₖ)
    vec3 *__restrict__ H_G_mean2d,       // Σ(∂²Gₖ/∂πₖ²)
    scalar_t *__restrict__ H_G_sigma,    // Σ(∂²Gₖ/∂Σₖ²), shape [N, 6]
    scalar_t *__restrict__ H_G_mixed,    // Σ(∂²Gₖ/∂πₖ∂Σₖ), shape [N, 6]
    scalar_t *__restrict__ dc_opac        // Σ(∂c/∂σₖ) hopefully... ???
) {
    // --- Boilerplate: Thread and memory indexing (from original kernel) ---
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // --- Pointer arithmetic for camera-specific data ---
    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * CDIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    // --- Tile Masking ---
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    // --- Pixel Coordinates and ID ---
    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // --- Check if thread is inside image bounds ---
    bool inside = (i < image_height && j < image_width);

    // --- Backward Pass State Initialization ---
    const float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // --- Main Loop over Gaussian Batches ---
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    // --- Shared Memory for Batched Data ---
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec3 *conic_batch = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);
    float *rgbs_batch = (float *)&conic_batch[block_size];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // --- Load batch data from global to shared memory (back to front) ---
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, (uint32_t)(batch_end + 1 - range_start));
        const int32_t idx = batch_end - tr;

        if (idx >= range_start) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        block.sync();

        // --- Process Gaussians in the current batch ---
        for (int32_t t = 0; t < batch_size; ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = false;
            }

            float alpha = 0.f, sigma = 0.f;
            if (valid) {
                const vec3 xy_opac = xy_opacity_batch[t];
                const float opac = xy_opac.z;
                const vec2 mean2d_val = {xy_opac.x, xy_opac.y};
                const vec3 conic_val = conic_batch[t];
                const vec2 delta_val = {mean2d_val.x - px, mean2d_val.y - py};

                sigma = 0.5f * (conic_val.x * delta_val.x * delta_val.x +
                                conic_val.z * delta_val.y * delta_val.y) +
                                conic_val.y * delta_val.x * delta_val.y;

                const float vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);

                if (sigma < 0.f || alpha < 1e-6f) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            // Initialize local derivatives to zero for this thread
            float dc_dcSH_local = 0.f;
            float dc_dG_local = 0.f;
            float dc_opac_local = 0.f;
            vec2 dG_dmean2d_local = {0.f, 0.f};
            vec3 dG_dSigma_local = {0.f, 0.f, 0.f};
            vec3 H_G_mean2d_local = {0.f, 0.f, 0.f};
            float H_G_sigma_local[6] = {0.f};
            float H_G_mixed_local[6] = {0.f};

            if (valid) {
                const int32_t g = id_batch[t];
                const vec3 xy_opac = xy_opacity_batch[t];
                const float opac = xy_opac.z;
                const vec2 mean2d = {xy_opac.x, xy_opac.y};
                const vec3 conic = conic_batch[t];
                const vec2 delta = {mean2d.x - px, mean2d.y - py};
                const float vis = __expf(-sigma);

                // --- Backpropagate loss to get v_alpha = ∂L/∂α ---
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }
                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                // --- Calculate this pixel's contribution to intermediate derivatives ---
                dc_dcSH_local = fac;
                if (opac * vis <= 0.999f) {
                    dc_dG_local = opac * v_alpha;
                }
                const float Gk_T_prefix = vis*T;
                float dc_dsigma_k[CDIM]; // sigma means opacity here, not conic
                #pragma unroll
                for (uint32_t k_chan = 0; k_chan < CDIM; ++k_chan) {
                    float color_term = rgbs_batch[t*CDIM + k_chan] - buffer[k_chan];
                    dc_dsigma_k[k_chan] = Gk_T_prefix* color_term;
                }
                dc_opac_local = 0.f;
                #pragma unroll
                for (uint32_t k_chan = 0; k_chan < CDIM; ++k_chan) {
                    dc_opac_local += v_render_c[k_chan] * dc_dsigma_k[k_chan];
                }
                const vec2 v_grad = {conic.x * delta.x + conic.y * delta.y, conic.y * delta.x + conic.z * delta.y};

                dG_dmean2d_local = {-vis * v_grad.x, -vis * v_grad.y};
                H_G_mean2d_local = {
                    vis * (v_grad.x * v_grad.x - conic.x),
                    vis * (v_grad.x * v_grad.y - conic.y),
                    vis * (v_grad.y * v_grad.y - conic.z)
                };

                dG_dSigma_local = {
                    -0.5f * vis * delta.x * delta.x,
                    -vis * delta.x * delta.y,
                    -0.5f * vis * delta.y * delta.y
                };

                const float dx2 = delta.x * delta.x, dy2 = delta.y * delta.y, dxdy = delta.x * delta.y;
                H_G_sigma_local[0] = vis * 0.25f * dx2 * dx2;
                H_G_sigma_local[1] = vis * 0.25f * dy2 * dy2;
                H_G_sigma_local[2] = vis * dx2 * dy2;
                H_G_sigma_local[3] = vis * 0.5f * dx2 * dxdy;
                H_G_sigma_local[4] = vis * 0.5f * dy2 * dxdy;
                H_G_sigma_local[5] = vis * 0.25f * dx2 * dy2;

                H_G_mixed_local[0] = vis * (0.5f * dx2 * v_grad.x - delta.x);
                H_G_mixed_local[1] = vis * (dxdy * v_grad.x - delta.y);
                H_G_mixed_local[2] = vis * (0.5f * dy2 * v_grad.x);
                H_G_mixed_local[3] = vis * (0.5f * dx2 * v_grad.y);
                H_G_mixed_local[4] = vis * (dxdy * v_grad.y - delta.x);
                H_G_mixed_local[5] = vis * (0.5f * dy2 * v_grad.y - delta.y);

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }

            // --- Aggregate contributions within warp using warpSum ---
            warpSum(dc_dcSH_local, warp);
            warpSum(dc_dG_local, warp);
            warpSum(dG_dmean2d_local, warp);
            warpSum(dG_dSigma_local, warp);
            warpSum(H_G_mean2d_local, warp);
            warpSum(dc_opac_local, warp);
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                warpSum(H_G_sigma_local[k], warp);
            }
            #pragma unroll
            for (int k = 0; k < 6; ++k) {
                warpSum(H_G_mixed_local[k], warp);
            }

            // --- Atomically add to global memory ---
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                gpuAtomicAdd(dc_dcSH + g, dc_dcSH_local);
                gpuAtomicAdd(dc_dG + g, dc_dG_local);
                gpuAtomicAdd(dc_opac + g, dc_opac_local);

                gpuAtomicAdd(&(dG_dmean2d[g].x), dG_dmean2d_local.x);
                gpuAtomicAdd(&(dG_dmean2d[g].y), dG_dmean2d_local.y);

                gpuAtomicAdd(&(dG_dSigma[g].x), dG_dSigma_local.x);
                gpuAtomicAdd(&(dG_dSigma[g].y), dG_dSigma_local.y);
                gpuAtomicAdd(&(dG_dSigma[g].z), dG_dSigma_local.z);

                gpuAtomicAdd(&(H_G_mean2d[g].x), H_G_mean2d_local.x);
                gpuAtomicAdd(&(H_G_mean2d[g].y), H_G_mean2d_local.y);
                gpuAtomicAdd(&(H_G_mean2d[g].z), H_G_mean2d_local.z);

                float* H_G_sigma_ptr = H_G_sigma + 6 * g;
                #pragma unroll
                for(int k=0; k<6; ++k) gpuAtomicAdd(H_G_sigma_ptr + k, H_G_sigma_local[k]);

                float* H_G_mixed_ptr = H_G_mixed + 6 * g;
                #pragma unroll
                for(int k=0; k<6; ++k) gpuAtomicAdd(H_G_mixed_ptr + k, H_G_mixed_local[k]);
            }
        }
    }
}

