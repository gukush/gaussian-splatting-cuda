#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <iostream>
#include <torch/extension.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
namespace {
__constant__ float cGauss[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f};
}

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ float get_pix_value(
    const float* img,
    int b, int c, int y, int x,
    int CH, int H, int W) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
__global__ void fusedssimCUDA_LN(
    int H,
    int W,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ mu1_map,
    float* __restrict__ mu2_map,
    float* __restrict__ sigma1_sq_map,
    float* __restrict__ sigma2_sq_map,
    float* __restrict__ sigma12_map) {
    auto block = cg::this_thread_block();
    const int bIdx = block.group_index().z; // batch index
    const int pix_y = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    // Shared memory for the tile (img1, img2)
    __shared__ float sTile[SHARED_Y][SHARED_X][2];
    // After horizontal pass, store partial sums here
    // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    __shared__ float xconv[CONV_Y][CONV_X][5];

    // Each block processes B x C sub-batches. We loop over channels:
    for (int c = 0; c < CH; ++c) {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int tileSize = SHARED_Y * SHARED_X;
            const int threads = BLOCK_X * BLOCK_Y;
            const int steps = (tileSize + threads - 1) / threads;

            const int tileStartY = block.group_index().y * BLOCK_Y;
            const int tileStartX = block.group_index().x * BLOCK_X;

            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / SHARED_X;
                    int local_x = tid % SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;

                    float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                    float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                    sTile[local_y][local_x][0] = X;
                    sTile[local_y][local_x][1] = Y;
                }
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO; // skip left halo

            float sumX = 0.f;
            float sumX2 = 0.f;
            float sumY = 0.f;
            float sumY2 = 0.f;
            float sumXY = 0.f;

            // #pragma unroll for those 5 pairs
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float Xleft = sTile[ly][lx - d][0];
                float Yleft = sTile[ly][lx - d][1];
                float Xright = sTile[ly][lx + d][0];
                float Yright = sTile[ly][lx + d][1];

                sumX += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            // center
            {
                float centerX = sTile[ly][lx][0];
                float centerY = sTile[ly][lx][1];
                float wc = cGauss[HALO];
                sumX += centerX * wc;
                sumX2 += (centerX * centerX) * wc;
                sumY += centerY * wc;
                sumY2 += (centerY * centerY) * wc;
                sumXY += (centerX * centerY) * wc;
            }

            // Write out partial sums
            xconv[ly][threadIdx.x][0] = sumX;
            xconv[ly][threadIdx.x][1] = sumX2;
            xconv[ly][threadIdx.x][2] = sumY;
            xconv[ly][threadIdx.x][3] = sumY2;
            xconv[ly][threadIdx.x][4] = sumXY;

            // Possibly handle second row in same warp
            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX = 0.f;
                sumX2 = 0.f;
                sumY = 0.f;
                sumY2 = 0.f;
                sumXY = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft = sTile[ly2][lx - d][0];
                    float Yleft = sTile[ly2][lx - d][1];
                    float Xright = sTile[ly2][lx + d][0];
                    float Yright = sTile[ly2][lx + d][1];

                    sumX += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float cx = sTile[ly2][lx][0];
                    float cy = sTile[ly2][lx][1];
                    float wc = cGauss[HALO];
                    sumX += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                }
                xconv[ly2][threadIdx.x][0] = sumX;
                xconv[ly2][threadIdx.x][1] = sumX2;
                xconv[ly2][threadIdx.x][2] = sumY;
                xconv[ly2][threadIdx.x][3] = sumY2;
                xconv[ly2][threadIdx.x][4] = sumXY;
            }
        }
        block.sync();

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            int ly = threadIdx.y + HALO;
            int lx = threadIdx.x;

            float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = xconv[ly - d][lx];
                float* bot = xconv[ly + d][lx];

                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
                out3 += (top[3] + bot[3]) * w;
                out4 += (top[4] + bot[4]) * w;
            }
            // center
            {
                float wC = cGauss[HALO];
                float* ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H) {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12 = out4 - mu1 * mu2;

                float A = mu1_sq * mu2_sq + C1;
                float B = sigma1_sq * sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                int global_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;
                if(mu1_map) {
                    mu1_map[global_idx] = out0;
                    mu2_map[global_idx] = out2;
                    sigma1_sq_map[global_idx] = sigma1_sq;
                    sigma2_sq_map[global_idx] = sigma2_sq;
                    sigma12_map[global_idx] = sigma12;
                }
            }
        }
    }
}

// ------------------------------------------
// Backward Kernel: Apply chain rule to get
//    dL/d(img1) from partial derivatives
//    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
//    and dL/dmap (the gradient from above).
// ------------------------------------------
// now returns BOTH first and second derivatives
__global__ void fusedssim_color_backwardCUDA_LN(
    int    H, int    W, int    CH,
    float  C1, float  C2,
    const float* __restrict__ img1,     // [B,C,H,W]
    const float* __restrict__ img2,     // [B,C,H,W]
    const float* __restrict__ dL_dmap,  // [B,1,H,W]
    const float* __restrict__ mu1_map,  // [B,C,H,W]
    const float* __restrict__ mu2_map,  // [B,C,H,W]
    const float* __restrict__ s1_map,   // [B,C,H,W]
    const float* __restrict__ s2_map,   // [B,C,H,W]
    const float* __restrict__ s12_map,  // [B,C,H,W]
    float* __restrict__ dL_dimg1,       // [B,C,H,W]
    float* __restrict__ d2L_dimg1       // [B,C,H,W]   ← new
) {
  int b = blockIdx.z;
  int y = blockIdx.y*BLOCK_Y + threadIdx.y;
  int x = blockIdx.x*BLOCK_X + threadIdx.x;
  if (x>=W || y>=H) return;

  constexpr int h = HALO, w = HALO;
  const int   pixM = ((b*1 +0)*H + y)*W + x;
  const float upstream = dL_dmap[pixM];

  for(int c=0; c<CH; ++c) {
    float grad1 = 0.0f;    // will hold   ∂L/∂c_k
    float grad2 = 0.0f;    // will hold ∂²L/∂c_k²

    // slide 11×11 window
    #pragma unroll
    for(int dy=-h; dy<=+h; ++dy){
    #pragma unroll
      for(int dx=-w; dx<=+w; ++dx){
        int xx = x+dx, yy = y+dy;
        if (xx<0||yy<0||xx>=W||yy>=H) continue;

        int   m   = ((b*1+0)*H + yy)*W + xx;
        int idxC = ((b*CH +c)*H + yy)*W + xx;

        // 1) read precomputed scalars:
        float mu1   = mu1_map [m];
        float mu2   = mu2_map [m];
        float var1  =  s1_map [m];
        float var2  =  s2_map [m];
        float cov12 = s12_map[m];

        float I1 = img1[idxC];
        float I2 = img2[idxC];
        float M  = upstream;   // ∂L/∂M at (m)

        // 2) compute the four f-helpers (eq 9)
        float f0 = 2.f*mu1*mu2 + C1;
        float f1 = 2.f*cov12 + C2;
        float f2 = mu1*mu1 + C1;
        float f3 = var1    + C2;

        // 3) compute the four g-helpers (below eq 13)
        //    note the window weight w_{ij} comes from your cGauss[]
        float w_ij = cGauss[h + dx];   // but in 2D you'd multiply cGauss[h+dx]*cGauss[h+dy]
        //    here we assume you stored a full 2D Gaussian in w2D[dy+HALO][dx+HALO]
        //    so replace the above line with:
        //      float w_ij = w2D[dy+HALO][dx+HALO];
        float g0 = 2.f * w_ij * mu2;
        float g1 = 2.f * w_ij * (I2 - mu2);
        float g2 = 2.f * w_ij * mu1;
        float g3 = 2.f * w_ij * (I1 - mu1);

        // 4) first derivative ∂f/∂I1  (eq 12)
        float df =
             ( f1/(f2*f3) ) * g0
           + ( f0/(f2*f3) ) * g1
           - ( f0*f1/(f2*f2*f3) ) * g2
           - ( f0*f1/(f2*f3*f3) ) * g3;

        // 5) second derivative ∂²f/∂I1²  (eq 13)
        //    we only show the g0^2 and g1^2 terms; you must add the g2^2 and g3^2 terms
        float g2_num = (2.0f * f2 * f3 * g2 + f2 * f2 * g3) * f0 * f1;
        float g2_den = (f2 * f2 * f3) * (f2 * f2 * f3);
        float g3_num = (2.0f * f2 * f3 * g3 + f3 * f3 * g2) * f0 * f1;
        float g3_den = (f2 * f3 * f3) * (f2 * f3 * f3);
        float d2f =
           /* from g0^2 term */
           ( ( g1/(f2*f3)   -  ((f2*g3+f3*g2)/(f2*f2*f3*f3))*f1 ) * g0*g0 )
         + /* from g1^2 term */
           ( ( g0/(f2*f3)   -  ((f2*g3+f3*g2)/(f2*f2*f3*f3))*f0 ) * g1*g1 )
         /* + analogous terms for g2^2 and g3^2 from eq (13) */
         + ( (-(f0 * g1 + f1 * g0) / (f2 * f2 * f3)) + (g2_num / g2_den) ) * g2 * g2;
         + ( (-(f0 * g1 + f1 * g0) / (f2 * f3 * f3)) + (g3_num / g3_den) ) * g3 * g3
         ;

        // 6) accumulate, remember the outer averaging factor 1/(3·|I|)
        grad1 += df * M;
        grad2 += d2f         * 1.f;
        // note: second derivative of the loss is simply
        //   (1/(3|I|)) sum_{i,j} w_{ij} ∂²f/∂c²
      }
    }

    // write out
    int outC = ((b*CH +c)*H + y)*W + x;
    dL_dimg1[outC]  = grad1 / (3.f * H * W);
    d2L_dimg1[outC] = grad2 / (3.f * H * W);
  }
}


void launch_fusedssim_LN_kernel(
    int64_t B, int64_t CH, int64_t H, int64_t W,
    float C1, float C2,
    const float* img1,
    const float* img2,
    float* ssim_map,
    float* mu1_map,
    float* mu2_map,
    float* s1_map,
    float* s2_map,
    float* s12_map,
    bool train,
    cudaStream_t stream
) {
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    fusedssimCUDA_LN<<<grid, block, 0, stream>>>(
        H, W, CH,
        C1, C2,
        img1,
        img2,
        ssim_map,
        train ? mu1_map : nullptr,
        train ? mu2_map : nullptr,
        train ? s1_map  : nullptr,
        train ? s2_map  : nullptr,
        train ? s12_map : nullptr
    );
}

// Backward-launcher
void launch_fusedssim_backward_LN_kernel(
    int64_t B, int64_t CH, int64_t H, int64_t W,
    float C1, float C2,
    const float* img1,
    const float* img2,
    const float* dL_dmap,
    const float* mu1_map,
    const float* mu2_map,
    const float* s1_map,
    const float* s2_map,
    const float* s12_map,
    float* dL_dimg1,
    float* d2L_dimg1,
    cudaStream_t stream
) {
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    fusedssim_color_backwardCUDA_LN<<<grid, block, 0, stream>>>(
        H, W, CH,
        C1, C2,
        img1,
        img2,
        dL_dmap,
        mu1_map,
        mu2_map,
        s1_map,
        s2_map,
        s12_map,
        dL_dimg1,
        d2L_dimg1
    );
}
/*
// ------------------------------------------
//  C++ Interface (Forward)
//   Returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12).
//   If train=false, derivative Tensors are empty.
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_LN(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    bool train) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B = img1.size(0);
    int CH = img1.size(1);
    int H = img1.size(2);
    int W = img1.size(3);

    // Launch config
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    // Output SSIM map
    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate intermediate Tensors
    auto mu1_map = torch::zeros({B, CH, H, W}, img1.options()).contiguous();
    auto mu2_map = torch::zeros({B, CH, H, W}, img1.options()).contiguous();
    auto s1_map = torch::zeros({B, CH, H, W}, img1.options()).contiguous();
    auto s2_map = torch::zeros({B, CH, H, W}, img1.options()).contiguous();
    auto s12_map = torch::zeros({B, CH, H, W}, img1.options()).contiguous();

    fusedssimCUDA_LN<<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? mu1_map.data_ptr<float>() : nullptr,
        train ? mu2_map.data_ptr<float>() : nullptr,
        train ? s1_map.data_ptr<float>() : nullptr,
        train ? s2_map.data_ptr<float>() : nullptr,
        train ? s12_map.data_ptr<float>() : nullptr);

    return std::make_tuple(ssim_map, mu1_map, mu2_map, s1_map, s2_map, s12_map);
}

// ------------------------------------------
// C++ Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
fusedssim_backward_LN(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    torch::Tensor& dL_dmap,
    torch::Tensor& mu1_map,
    torch::Tensor& mu2_map,
    torch::Tensor& s1_map,
    torch::Tensor& s2_map,
    torch::Tensor& s12_map) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B = img1.size(0);
    int CH = img1.size(1);
    int H = img1.size(2);
    int W = img1.size(3);

    auto dL_dimg1 = torch::zeros_like(img1);
    auto d2L_dimg1 = torch::zeros_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    fusedssim_color_backwardCUDA_LN<<<grid, block>>>(
        H, W, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        dL_dmap.contiguous().data_ptr<float>(),
        mu1_map.contiguous().data_ptr<float>(),
        mu2_map.contiguous().data_ptr<float>(),
        s1_map.contiguous().data_ptr<float>(),
        s2_map.contiguous().data_ptr<float>(),
        s12_map.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        d2L_dimg1.data_ptr<float>() );

    return std::make_tuple(dL_dimg1,d2L_dimg1);
}

*/