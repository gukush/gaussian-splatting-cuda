#include <cuda_runtime.h>
#include <math_constants.h>



#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>


#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace cg = cooperative_groups;
using namespace gsplat;
//-----------------------------------------------------------------------------
// (1) Accumulate per-gaussian ∂L/∂y_k  and  ∂²L/∂y_k²
//-----------------------------------------------------------------------------
template<uint32_t CDIM, typename scalar_t>
__global__ void accumulate_y_2nd_order_kernel(
    const uint32_t C,                 // # cameras / batches
    const uint32_t n_isects,          // total length of flatten_ids[]
    const bool     packed,            // (unused here, but you may need it)

    // --- tile↔flatten index lists (exactly as in your existing kernel) ---
    const bool   *__restrict__ masks,         // [C, tile_h, tile_w]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_h, tile_w]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    // --- “last Gaussian” per pixel = bin_final ---
    const int32_t *__restrict__ last_ids,     // [C, H, W]

    // --- image-space derivatives at each pixel ---
    const scalar_t *__restrict__ dL_dc,       // [C, H, W, CDIM]
    const scalar_t *__restrict__ d2L_dc2,     // [C, H, W, CDIM]

    // --- your analytic derivatives per Gaussian (already computed!) ---
    //    dcdy[g][c][0..1],   d2cdy2[g][c][0..2]
    const scalar_t *__restrict__ dcdy,        // [n_isects, CDIM, 2]
    const scalar_t *__restrict__ d2cdy2,      // [n_isects, CDIM, 3]

    // --- outputs: accumulate into these via atomicAdd() ---
    scalar_t *__restrict__ grad_y,            // [n_isects, 2]
    scalar_t *__restrict__ hess_y             // [n_isects, 3]
) {
  auto block = cg::this_thread_block();

  // map block → (camera, tileY, tileX)
  uint32_t cam   = block.group_index().x;
  uint32_t tileY = block.group_index().y;
  uint32_t tileX = block.group_index().z;

  // map thread → pixel within that tile
  uint32_t i = tileY*tile_size + block.thread_index().y;
  uint32_t j = tileX*tile_size + block.thread_index().x;
  bool     inside = (i<image_height && j<image_width);

  // flatten to one pixel‐id (clamped)
  uint32_t pix_id = min( i*image_width + j,
                         image_width*image_height - 1u );

  // advance all camera‐strided pointers
  if (masks)       masks     += cam*tile_height*tile_width;
  const int32_t *  tofs      = tile_offsets + cam*tile_height*tile_width;
  const int32_t *  lst_ids   = last_ids    + cam*image_height*image_width;
  dL_dc             += (size_t)cam*image_height*image_width*CDIM;
  d2L_dc2           += (size_t)cam*image_height*image_width*CDIM;

  // skip masked‐out tiles
  uint32_t tile_id = tileY*tile_width + tileX;
  if (masks && !masks[tile_id]) return;

  // how many Gaussians touch *this* tile?
  int32_t range_start = tofs[tile_id];
  int32_t range_end   = (cam==C-1 && tile_id==tile_width*tile_height-1)
                        ? int32_t(n_isects)
                        : tofs[tile_id+1];

  // we’ll walk them in “reverse‐front” batches of size block.size()
  const uint32_t BS = block.size();
  const uint32_t nB = (range_end - range_start + BS - 1)/BS;

  // shared scratch for one batch of up to BS analytics
  extern __shared__ int8_t  _s[];
  int32_t *   id_batch    = (int32_t*)_s;                                        // [BS]
  // dcdy_batch: BS × CDIM × 2
  scalar_t *  dcdy_batch  = (scalar_t*)&id_batch[BS];                             // [BS*CDIM*2]
  // d2cdy2_batch: BS × CDIM × 3
  scalar_t *  d2cdy2_batch= (scalar_t*)&dcdy_batch[BS*CDIM*2];                    // [BS*CDIM*3]

  // pixel’s “last index” for early‐out
  int32_t bin_final = inside ? lst_ids[pix_id] : 0;

  cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
  uint32_t tr = block.thread_rank();
  uint32_t warp_bin_final = 0;

  for (uint32_t b = 0; b < nB; ++b) {
    block.sync();

    // load one batch of at most BS Gaussians, back→front
    int32_t batch_end = range_end - 1 - int32_t(BS*b);
    int32_t batch_len = max(0, min(int32_t(BS), batch_end+1-range_start));
    int32_t idx       = batch_end - int32_t(tr);

    if (idx >= range_start && idx <= batch_end) {
      int32_t g = flatten_ids[idx];
      id_batch[tr] = g;

      // copy in precomputed dc/dy and d2c/dy2
      // layout: [ g,chan, * ]
      #pragma unroll
      for (uint32_t c=0; c<CDIM; ++c) {
        dcdy_batch[(tr*CDIM+c)*2 + 0] = dcdy[(g*CDIM + c)*2 + 0];
        dcdy_batch[(tr*CDIM+c)*2 + 1] = dcdy[(g*CDIM + c)*2 + 1];
        d2cdy2_batch[(tr*CDIM+c)*3 + 0] = d2cdy2[(g*CDIM + c)*3 + 0];
        d2cdy2_batch[(tr*CDIM+c)*3 + 1] = d2cdy2[(g*CDIM + c)*3 + 1];
        d2cdy2_batch[(tr*CDIM+c)*3 + 2] = d2cdy2[(g*CDIM + c)*3 + 2];
      }
    }

    block.sync();
    // warp‐max of bin_final for this batch
    warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int32_t>());

    // each batch entry t in [0..batch_len)
    for (int32_t t = max(0,int(batch_end - warp_bin_final));
         t < batch_len;
         ++t)
    {
      bool valid = inside && (batch_end - t <= bin_final);
      if (!warp.any(valid)) continue;

      // local accumulators
      float grad0 = 0.0f, grad1 = 0.0f;
      float Hxx   = 0.0f, Hxy   = 0.0f, Hyy = 0.0f;

      if (valid) {
        // load image‐space dL/dc , d2L/dc2
        float dL_loc [CDIM], d2L_loc[CDIM];
        #pragma unroll
        for (uint32_t c=0; c<CDIM; ++c) {
          dL_loc[c]   = dL_dc[ pix_id*CDIM + c ];
          d2L_loc[c] = d2L_dc2[ pix_id*CDIM + c ];
        }

        // chain‐rule across CDIM channels
        #pragma unroll
        for (uint32_t c=0; c<CDIM; ++c) {
          float dc0 = dcdy_batch[(t*CDIM+c)*2 + 0];
          float dc1 = dcdy_batch[(t*CDIM+c)*2 + 1];
          float d200= d2cdy2_batch[(t*CDIM+c)*3 + 0];
          float d201= d2cdy2_batch[(t*CDIM+c)*3 + 1];
          float d211= d2cdy2_batch[(t*CDIM+c)*3 + 2];

          // grad
          grad0 += dL_loc[c] * dc0;
          grad1 += dL_loc[c] * dc1;

          // Hessian: (dcdy)^T·d²L·(dcdy)  +  dL·(d²cdy2)
          Hxx += dc0 * d2L_loc[c] * dc0 + dL_loc[c]*d200;
          Hxy += dc0 * d2L_loc[c] * dc1 + dL_loc[c]*d201;
          Hyy += dc1 * d2L_loc[c] * dc1 + dL_loc[c]*d211;
        }
      }

      // warp‐sum
      grad0 = cg::reduce(warp, grad0, cg::plus<float>());
      grad1 = cg::reduce(warp, grad1, cg::plus<float>());
      Hxx   = cg::reduce(warp, Hxx,   cg::plus<float>());
      Hxy   = cg::reduce(warp, Hxy,   cg::plus<float>());
      Hyy   = cg::reduce(warp, Hyy,   cg::plus<float>());

      // lane-0 writes back
      if (warp.thread_rank()==0) {
        int32_t g = id_batch[t];
        atomicAdd(&grad_y[2*g + 0], grad0);
        atomicAdd(&grad_y[2*g + 1], grad1);
        atomicAdd(&hess_y[3*g + 0], Hxx);
        atomicAdd(&hess_y[3*g + 1], Hxy);
        atomicAdd(&hess_y[3*g + 2], Hyy);
      }
    }
  }
}

//-----------------------------------------------------------------------------
// (2) Invert each 2×2 Hessian → Δy_k = −H⁻¹ · grad_y
//-----------------------------------------------------------------------------
template<typename scalar_t>
__global__ void
compute_y_updates_kernel(
    const uint32_t n_isects,
    // from the first pass:
    const scalar_t* __restrict__ grad_y,   // [n_isects, 2]
    const scalar_t* __restrict__ hess_y,   // [n_isects, 3]
    // optional regularizer:
    const bool      do_reg,                // whether to add 2·λ to Hessian & λ·y to grad
    const scalar_t  lambda,                // your λ
    const vec2*     __restrict__ yk,       // the current y_k values, [n_isects]
    // outputs:
    vec2*           __restrict__ delta_y   // [n_isects]
) {
  uint32_t g = blockIdx.x*blockDim.x + threadIdx.x;
  if (g >= n_isects) return;

  // load pure‐data
  float gx  = grad_y[2*g + 0],
        gy  = grad_y[2*g + 1];
  float Hxx = hess_y[3*g + 0],
        Hxy = hess_y[3*g + 1],
        Hyy = hess_y[3*g + 2];

  // --- apply L2 penalty:  ∂/∂y [ ½λ‖y‖² ] =  λ y
  //                         ∂²/∂y² [ ½λ‖y‖² ] =  λ I
  if (do_reg) {
    gx  += lambda * yk[g].x;
    gy  += lambda * yk[g].y;
    Hxx += lambda;
    Hyy += lambda;
  }

  // invert 2×2:
  float det = (Hxx*Hyy - Hxy*Hxy);
  float inv = 1.0f / det;

  // Δy = −H⁻¹·grad
  delta_y[g].x = -inv * ( Hyy*gx - Hxy*gy );
  delta_y[g].y = -inv * ( Hxx*gy - Hxy*gx );
}


template <uint32_t CDIM, typename scalar_t>
void launch_accumulate_y_2nd_order_kernel(
    const uint32_t C,
    const uint32_t n_isects,
    const bool packed,
    const bool* masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t* tile_offsets,
    const int32_t* flatten_ids,
    const int32_t* last_ids,
    const scalar_t* dL_dc,
    const scalar_t* d2L_dc2,
    const scalar_t* dcdy,
    const scalar_t* d2cdy2,
    scalar_t* grad_y,
    scalar_t* hess_y,
    size_t shmem_size
) {
    // Configure kernel launch
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    // Set shared memory configuration
    cudaFuncSetAttribute(
        accumulate_y_2nd_order_kernel<CDIM, scalar_t>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size
    );

    // Launch kernel
    accumulate_y_2nd_order_kernel<CDIM, scalar_t><<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
        C,
        n_isects,
        packed,
        masks,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        tile_offsets,
        flatten_ids,
        last_ids,
        dL_dc,
        d2L_dc2,
        dcdy,
        d2cdy2,
        grad_y,
        hess_y
    );

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


void launch_compute_y_updates_kernel(
    const uint32_t n_isects,
    const float* grad_y,
    const float* hess_y,
    const bool do_reg,
    const float lambda,
    const vec2* yk,
    vec2* delta_y
) {
    // Configure kernel launch
    const int threads = 256;
    const int blocks = (n_isects + threads - 1) / threads;

    compute_y_updates_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_isects,
        grad_y,
        hess_y,
        do_reg,
        lambda,
        yk,
        delta_y
    );

    // Check for errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}