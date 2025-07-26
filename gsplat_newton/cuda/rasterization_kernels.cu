#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"


#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"
#include <cooperative_groups.h>

using namespace gsplat;
/**
 * @brief Kernel 1: Computes and aggregates intermediate derivatives for the
 * Newton-Raphson solver.
 *
 * This kernel performs the backward pass of the 3DGS rasterizer, calculating
 * the sum of all pixel-dependent derivatives for each Gaussian. It uses atomic
 * adds to aggregate contributions from different tiles and warps. The outputs of
 * this kernel serve as the direct inputs for the second assembly kernel.
 */

 namespace gsplat_newton {

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
    // --- Loss Gradients ---
    const scalar_t *__restrict__ dL_dcIMG,
    const scalar_t *__restrict__ H_L_dcIMG,

    // --- FINAL DERIVATIVES (OUTPUT) ---
    scalar_t *__restrict__ dL_dcSH,      // Σ(∂L/∂c̃ₖ) [N, 3]
    scalar_t *__restrict__ H_L_dcsh,     // [N x 3 ???]
    scalar_t *__restrict__ dL_dG,        // [N, 3]
    vec2 *__restrict__ dL_dmean2d,       // [N, 2]
    vec3 *__restrict__ dL_dconic,        // [N, 3]
    vec3 *__restrict__ H_L_mean2d,       // [N, 3]
    scalar_t *__restrict__ H_L_conic,    // [N, 6]
    scalar_t *__restrict__ H_L_mixed,    // [N, 6]
    scalar_t *__restrict__ dL_dopac,        // Σ(∂L/∂σₖ) hopefully... ??? [N, 3]
    scalar_t *__restrict__ H_L_dopac     // [N, ???]
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
            float dc_dG_local[CDIM]  = {0.f};
            float dc_opac_local[CDIM]  = {0.f};
            vec2 dG_dmean2d_local = {0.f, 0.f};
            vec3 dG_dconic_local = {0.f, 0.f, 0.f};
            vec3 H_G_mean2d_local = {0.f, 0.f, 0.f};
            float H_G_conic_local[6] = {0.f};
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
                const float T_before = T * ra;
                const float GkT = vis * T_before;
                float Dk [CDIM];
                #pragma unroll
                for (uint32_t k=0;k<CDIM;++k)
                    Dk[k] = buffer[k] / T_before;   // safe (T>0)

                /* colour difference  (ĉ_k - D_k)  */
                float color_term[CDIM];
                #pragma unroll
                for (uint32_t k=0;k<CDIM;++k)
                    color_term[k] = rgbs_batch[t*CDIM+k] - Dk[k];

                // --- Calculate this pixel's contribution to intermediate derivatives ---
                dc_dcSH_local = alpha * T_before; // value is the same for each channel from what I infer.
                #pragma unroll
                for (uint32_t k=0;k<CDIM;++k){
                    dc_dG_local  [k] = opac * T_before * color_term[k]; // eq.(2)
                    dc_opac_local[k] = GkT            * color_term[k];      // eq.(2)
                }
                const vec2 v_grad = {conic.x * delta.x + conic.y * delta.y, conic.y * delta.x + conic.z * delta.y};

                dG_dmean2d_local = {-vis * v_grad.x, -vis * v_grad.y};
                H_G_mean2d_local = {
                    vis * (v_grad.x * v_grad.x - conic.x),
                    vis * (v_grad.x * v_grad.y - conic.y),
                    vis * (v_grad.y * v_grad.y - conic.z)
                };

                dG_dconic_local = {
                    -0.5f * vis * delta.x * delta.x,
                    -vis * delta.x * delta.y,
                    -0.5f * vis * delta.y * delta.y
                };

                const float dx2 = delta.x * delta.x, dy2 = delta.y * delta.y, dxdy = delta.x * delta.y;
                H_G_conic_local[0] = vis * 0.25f * dx2 * dx2;
                H_G_conic_local[1] = vis * 0.25f * dy2 * dy2;
                H_G_conic_local[2] = vis * 0.25f * dx2 * dy2;
                H_G_conic_local[3] = vis * 0.5f * dx2 * dxdy;
                H_G_conic_local[4] = vis * 0.5f * dy2 * dxdy;
                H_G_conic_local[5] = vis        * dx2 * dy2;

                H_G_mixed_local[0] = vis * (0.5f * dx2 * v_grad.x - delta.x);
                H_G_mixed_local[1] = vis * (dxdy * v_grad.x - delta.y);
                H_G_mixed_local[2] = vis * (0.5f * dy2 * v_grad.x);
                H_G_mixed_local[3] = vis * (0.5f * dx2 * v_grad.y);
                H_G_mixed_local[4] = vis * (dxdy * v_grad.y - delta.x);
                H_G_mixed_local[5] = vis * (0.5f * dy2 * v_grad.y - delta.y);

                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * alpha * T_before;
                }
                T = T_before;

                // getting correct dL/dc and H_L_dc
                float dL_dc[CDIM], Hc_diag[CDIM];
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    dL_dc[k]  =  dL_dcIMG [pix_id*CDIM + k];
                    Hc_diag[k] = H_L_dcIMG[pix_id*CDIM + k];
                }

                float dL_dcSH_local[CDIM];
                float dL_dG_local = 0.0f, dL_opac_scalar = 0.0f;
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    dL_dcSH_local[k]  = dL_dc[k] * dc_dcSH_local;          // ∂L/∂c̃_k
                    dL_dG_local      += dL_dc[k] * dc_dG_local[k];         // g_G
                    dL_opac_scalar   += dL_dc[k] * dc_opac_local[k];       // ∂L/∂σ_k
                }
                float H_L_G_local = 0.0f;

                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    H_L_G_local += Hc_diag[k] * dc_dG_local[k] * dc_dG_local[k];
                } //  + 0  (c is linear in G ⇒ no “intrinsic’’ term)
                // dL_dmean2d. dL_dconic
                vec2 dL_dmean2d_local = {
                    dL_dG_local * dG_dmean2d_local.x,
                    dL_dG_local * dG_dmean2d_local.y };

                }
                vec3 dL_dconic_local = {
                    dL_dG_local * dG_dconic_local.x,
                    dL_dG_local * dG_dconic_local.y,
                    dL_dG_local * dG_dconic_local.z };
                // -- Hessian  Σμ  (xx,xy,yy) -
                vec3 H_L_mean2d_local = {
                    H_L_G_local * dG_dmean2d_local.x * dG_dmean2d_local.x +
                        dL_dG_local * H_G_mean2d_local.x,                // xx
                    H_L_G_local * dG_dmean2d_local.x * dG_dmean2d_local.y +
                        dL_dG_local * H_G_mean2d_local.y,                // xy
                    H_L_G_local * dG_dmean2d_local.y * dG_dmean2d_local.y +
                        dL_dG_local * H_G_mean2d_local.z };              // yy

                // -- Hessian  Σa  (A,A) (C,C) (A,C) (A,B) (B,C) (B,B) in that order --
                float H_L_conic_local[6];
                H_L_conic_local[0] = H_L_G_local * dG_dconic_local.x * dG_dconic_local.x +
                                    dL_dG_local * H_G_conic_local[0];          // (A,A)
                H_L_conic_local[1] = H_L_G_local * dG_dconic_local.z * dG_dconic_local.z +
                                    dL_dG_local * H_G_conic_local[1];          // (C,C)
                H_L_conic_local[2] = H_L_G_local * dG_dconic_local.x * dG_dconic_local.z +
                                    dL_dG_local * H_G_conic_local[2];          // (A,C)
                H_L_conic_local[3] = H_L_G_local * dG_dconic_local.x * dG_dconic_local.y +
                                    dL_dG_local * H_G_conic_local[3];          // (A,B)
                H_L_conic_local[4] = H_L_G_local * dG_dconic_local.y * dG_dconic_local.z +
                                    dL_dG_local * H_G_conic_local[4];          // (B,C)
                H_L_conic_local[5] = H_L_G_local * dG_dconic_local.y * dG_dconic_local.y +
                                    dL_dG_local * H_G_conic_local[5];          // (B,B)

                /* -- mixed Hessian  Σμa  (flattened in exactly your order) --------- */
                float H_L_mixed_local[6];
                H_L_mixed_local[0] = H_L_G_local * dG_dmean2d_local.x * dG_dconic_local.x +
                                    dL_dG_local * H_G_mixed_local[0];          // μx–A
                H_L_mixed_local[1] = H_L_G_local * dG_dmean2d_local.y * dG_dconic_local.x +
                                    dL_dG_local * H_G_mixed_local[1];          // μy–A
                H_L_mixed_local[2] = H_L_G_local * dG_dmean2d_local.x * dG_dconic_local.z +
                                    dL_dG_local * H_G_mixed_local[2];          // μx–C
                H_L_mixed_local[3] = H_L_G_local * dG_dmean2d_local.x * dG_dconic_local.y +
                                    dL_dG_local * H_G_mixed_local[3];          // μx–B
                H_L_mixed_local[4] = H_L_G_local * dG_dmean2d_local.y * dG_dconic_local.y +
                                    dL_dG_local * H_G_mixed_local[4];          // μy–B
                H_L_mixed_local[5] = H_L_G_local * dG_dmean2d_local.y * dG_dconic_local.z +
                                    dL_dG_local * H_G_mixed_local[5];          // μy–C

                float H_L_dcSH_local[CDIM];
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    H_L_dcSH_local[k] = Hc_diag[k] * dc_dcSH_local * dc_dcSH_local;
                }
                float H_L_dopac_local = 0.0f;          // sandwich part
                float intrinsic_opac = 0.0f;          // ∑ dL/∂c · ∂²c/∂p²

                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    H_L_opac_local += Hc_diag[k] * dc_opac_local[k] * dc_opac_local[k];
                    // assumption is that d2c/dopac2 is 0
                    //intrinsic_opac += dL_dc[k]  * (vis * vis * T_before * ra
                    //                           * rgbs_batch[t*CDIM + k]);
                }
                //H_L_dopac_local += intrinsic_opac;     // full Hessian entry

            // --- Aggregate contributions within warp using warpSum ---
            #pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                warpSum(dL_dcSH_local[k], warp);
            }

            warpSum(dL_dG_local      , warp);
            warpSum(dL_opac_scalar    , warp);
            warpSum(dL_dmean2d_local  , warp);
            warpSum(dL_dconic_local   , warp);
            warpSum(H_L_mean2d_local  , warp);

            #pragma unroll
            for (int k=0;k<6;++k){
                warpSum(H_L_conic_local [k], warp);
                warpSum(H_L_mixed_local [k], warp);
            }
            #pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k)
                warpSum(H_L_dcSH_local[k], warp);
            warpSum(H_L_dopac_local, warp);

            // --- Atomically add to global memory ---
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];

                float* dCptr  = dL_dcSH  + g*CDIM;        // [N,CDIM]
                #pragma unroll
                for (uint32_t k=0;k<CDIM;++k)
                    gpuAtomicAdd(dCptr+k, dL_dcSH_local[k]);
                gpuAtomicAdd(dL_dcSH + g, dc_dcSH_local);
                gpuAtomicAdd(dL_dG       + g,  dL_dG_local);          // [N]
                gpuAtomicAdd(&(dL_opac   [g]), dL_opac_scalar);       // [N]

                gpuAtomicAdd(&(dL_dmean2d[g].x), dL_dmean2d_local.x); // [N,2]
                gpuAtomicAdd(&(dL_dmean2d[g].y), dL_dmean2d_local.y);
                gpuAtomicAdd(&(dL_dconic[g].x ), dL_dconic_local.x ); // [N,3]
                gpuAtomicAdd(&(dL_dconic[g].y ), dL_dconic_local.y );
                gpuAtomicAdd(&(dL_dconic[g].z ), dL_dconic_local.z );
                // ---- second‑order ----------------------------------------------
                gpuAtomicAdd(&(H_L_mean2d[g].x), H_L_mean2d_local.x); // [N,3]
                gpuAtomicAdd(&(H_L_mean2d[g].y), H_L_mean2d_local.y);
                gpuAtomicAdd(&(H_L_mean2d[g].z), H_L_mean2d_local.z);
                float* HcshPtr = H_L_dcSH + g*CDIM;
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k)
                    gpuAtomicAdd(HcshPtr + k, H_L_dcSH_local[k]);
                gpuAtomicAdd(H_L_opac + g, H_L_opac_local);
                float* HcPtr   = H_L_conic  + 6*g;                    // [N,6]
                float* HmxPtr  = H_L_mixed  + 6*g;                    // [N,6]
                #pragma unroll
                for (int k=0;k<6;++k){
                    gpuAtomicAdd(HcPtr +k, H_L_conic_local [k]);
                    gpuAtomicAdd(HmxPtr+k, H_L_mixed_local[k]);
                }
        }
    }
}
template <uint32_t CDIM>
void launch_compute_intermediate_derivatives_kernel(
    const bool packed,
    // --- Forward Pass Inputs ---
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    // --- Forward Pass Outputs ---
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    // --- Loss Gradients (ADDED) ---
    const at::Tensor dL_dcIMG,
    const at::Tensor H_L_dcIMG,
    // --- OUTPUTS (per-Gaussian) ---
    at::Tensor dL_dcSH,
    at::Tensor H_L_dcSH,  // Fixed name from H_L_cSH
    at::Tensor dL_dG,
    at::Tensor dL_dmean2d,
    at::Tensor dL_dconic,
    at::Tensor H_L_mean2d,
    at::Tensor H_L_conic,   // Fixed name from H_L_sigma
    at::Tensor H_L_mixed,
    at::Tensor dL_dopac,
    at::Tensor H_L_dopac    // ADDED
) {
    uint32_t C = tile_offsets.size(0);
    uint32_t N = means2d.size(0);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    if (n_isects == 0) {
        return;
    }

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    const int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);

    if (cudaFuncSetAttribute(
            compute_intermediate_derivatives_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    compute_intermediate_derivatives_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C, N, n_isects, packed,
            // --- Forward Pass Inputs ---
            reinterpret_cast<const vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<const vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width, image_height,
            tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            // --- Forward Pass Outputs ---
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            // --- Loss Gradients (FIXED) ---
            dL_dcIMG.data_ptr<float>(),
            H_L_dcIMG.data_ptr<float>(),
            // --- OUTPUTS (per-Gaussian) ---
            dL_dcSH.data_ptr<float>(),
            H_L_dcSH.data_ptr<float>(),  // Fixed
            dL_dG.data_ptr<float>(),
            reinterpret_cast<vec2 *>(dL_dmean2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(dL_dconic.data_ptr<float>()),
            reinterpret_cast<vec3 *>(H_L_mean2d.data_ptr<float>()),
            H_L_conic.data_ptr<float>(),
            H_L_mixed.data_ptr<float>(),
            dL_dopac.data_ptr<float>(),
            H_L_dopac.data_ptr<float>()  // Added
        );
}

// Explicit Instantiation for various color dimensions (CDIM)
#define __INS__(CDIM)                                                                   \
    template void launch_compute_intermediate_derivatives_kernel<CDIM>(                 \
        const bool packed,                                                              \
        const at::Tensor means2d,                                                       \
        const at::Tensor conics,                                                        \
        const at::Tensor colors,                                                        \
        const at::Tensor opacities,                                                     \
        const at::optional<at::Tensor> backgrounds,                                     \
        const at::optional<at::Tensor> masks,                                           \
        const uint32_t image_width,                                                     \
        const uint32_t image_height,                                                    \
        const uint32_t tile_size,                                                       \
        const at::Tensor tile_offsets,                                                  \
        const at::Tensor flatten_ids,                                                   \
        const at::Tensor render_alphas,                                                 \
        const at::Tensor last_ids,                                                      \
        const at::Tensor dL_dcIMG,                                                      \
        const at::Tensor H_L_dcIMG,                                                     \
        at::Tensor dL_dcSH,                                                             \
        at::Tensor dL_dG,                                                               \
        at::Tensor dL_dmean2d,                                                          \
        at::Tensor dL_dconic,                                                           \
        at::Tensor H_L_mean2d,                                                          \
        at::Tensor H_L_conic,                                                           \
        at::Tensor H_L_mixed,                                                           \
        at::Tensor dL_opac,                                                             \
        at::Tensor H_L_dopac);                                                          \

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)

#undef __INS__

} //namespace gsplat_newton
