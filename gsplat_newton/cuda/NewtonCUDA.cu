#include "gsplat_newton/kernels.hpp"
#include "Common.h"
#include "Utils.cuh" // For matrix operations
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAStream.h>

using namespace gsplat;

namespace gsplat_newton {


// -----------------------------------------------------------------------------
// Constants for regularisation ------------------------------------------------
// -----------------------------------------------------------------------------
__constant__ float kMatInvEps = 1e-6f;
__constant__ float kLambdaOpacity = 1e-3f;   // smooth‑L1 λ
__constant__ float kL1Eps        = 1e-4f;    // smooth corner ε

// -----------------------------------------------------------------------------
// Small helpers ---------------------------------------------------------------
// -----------------------------------------------------------------------------


// Helper function for 2x2 matrix regularization
__device__ __forceinline__ void regularize_2x2_if_needed(mat2& M, float fixed_eps = 1e-6f, float det_threshold = 1e-8f) {
    // Compute determinant
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];

    if (fabsf(det) < det_threshold) {
        // Matrix is near singular - use adaptive regularization
        float max_diag = fmaxf(fabsf(M[0][0]), fabsf(M[1][1]));
        float adaptive_eps = fmaxf(fixed_eps, max_diag * 1e-4f);

        M[0][0] += adaptive_eps;
        M[1][1] += adaptive_eps;
    } else {
        // Matrix is well-conditioned - use simple fixed regularization
        M[0][0] += fixed_eps;
        M[1][1] += fixed_eps;
    }
}


// Helper function for 3x3 matrix regularization
__device__ __forceinline__ void regularize_3x3_if_needed(mat3& M, float fixed_eps = 1e-6f, float det_threshold = 1e-8f) {
    // Compute determinant for 3x3
    float det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
              - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
              + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);

    if (fabsf(det) < det_threshold) {
        // Matrix is near singular - use adaptive regularization
        float max_diag = fmaxf(fmaxf(fabsf(M[0][0]), fabsf(M[1][1])), fabsf(M[2][2]));
        float adaptive_eps = fmaxf(fixed_eps, max_diag * 1e-4f);

        M[0][0] += adaptive_eps;
        M[1][1] += adaptive_eps;
        M[2][2] += adaptive_eps;
    } else {
        // Matrix is well-conditioned - use simple fixed regularization
        M[0][0] += fixed_eps;
        M[1][1] += fixed_eps;
        M[2][2] += fixed_eps;
    }
}


__device__ __forceinline__ vec3 mul_mat3x2_vec2(const mat3x2 &A, const vec2 &v)
{
    // A is column‑major (3 columns, 2 rows)
    return vec3(
        A[0][0] * v.x + A[0][1] * v.y,
        A[1][0] * v.x + A[1][1] * v.y,
        A[2][0] * v.x + A[2][1] * v.y);
}

__device__ __forceinline__ mat3 mul_mat3x2_mat2x3(const mat3x2 &A, const mat2x3 &B)
{
    // Compute C = A * B  (3×2)·(2×3) = (3×3)
    mat3 C(0.0f);
    #pragma unroll
    for (int c = 0; c < 3; ++c) {         // column of C
        #pragma unroll
        for (int r = 0; r < 3; ++r) {     // row   of C
            C[c][r] = A[c][0] * B[0][r]   // col0·row0
                      + A[c][1] * B[1][r];// col1·row1
        }
    }
    return C;
}

__device__ __forceinline__ vec3 mul_mat3_vec3(const mat3 &M, const vec3 &v)
{
    // Column‑major convention: M * v = Σ (col_i * v_i)
    return v.x * vec3(M[0]) + v.y * vec3(M[1]) + v.z * vec3(M[2]);
}

// Quaternion multiply written out to avoid <glm/gtx/quaternion.hpp> ambiguity
__device__ __forceinline__ vec4 quat_mul(const vec4 &q, const vec4 &p)
{
    // components are (x y z w)
    return vec4(
        q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y,
        q.w * p.y - q.x * p.z + q.y * p.w + q.z * p.x,
        q.w * p.z + q.x * p.y - q.y * p.x + q.z * p.w,
        q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z);
}


__device__ inline vec3 solve_sym3x3_cholesky(const mat3 &M, const vec3 &b) {
    // small ridge to guarantee positive–definite
    const float eps = 1e-6f;

    // load + regularize diagonal
    float m00 = M[0][0] + eps;
    float m01 = M[0][1];
    float m02 = M[0][2];
    float m11 = M[1][1] + eps;
    float m12 = M[1][2];
    float m22 = M[2][2] + eps;

    // --- Cholesky factorization M+eps·I = L * L^T ---
    float l00 = sqrtf(m00);
    float l10 = m01 / l00;
    float l20 = m02 / l00;
    float l11 = sqrtf(m11 - l10*l10);
    float l21 = (m12 - l20*l10) / l11;
    float l22 = sqrtf(m22 - l20*l20 - l21*l21);

    // --- forward substitution L * y = b ---
    float y0 = b.x / l00;
    float y1 = (b.y - l10*y0) / l11;
    float y2 = (b.z - l20*y0 - l21*y1) / l22;

    // --- back substitution L^T * x = y ---
    vec3 x;
    x.z = y2 / l22;
    x.y = (y1 - l21*x.z) / l11;
    x.x = (y0 - l10*x.y - l20*x.z) / l00;
    return x;
}

// TODO add CDIM as template parameter
// This single kernel handles solving and updating for all attributes for one Gaussian.
// Each thread processes one Gaussian.
__global__ void solve_updates_and_backproject_kernel_impl(
    const uint32_t N,
    const int K,
    const int32_t* __restrict__ radii,
    // Gradients and Hessians
    const float* __restrict__ dL_d_pos,      // [N, 2]
    const float* __restrict__ H_L_pos,       // [N, 2, 2]
    const float* __restrict__ dL_d_scale,    // [N, 2]
    const float* __restrict__ H_L_scale,     // [N, 2, 2]
    const float* __restrict__ dL_d_rot,      // [N]
    const float* __restrict__ H_L_rot,       // [N]
    const float* __restrict__ dL_d_opacity,  // [N]
    const float* __restrict__ H_L_opacity,   // [N]
    const float* __restrict__ dL_d_color,    // [N, 3]
    const float* __restrict__ H_L_color,     // [N, 3] only diagonal elements
    // Context tensors
    const float* __restrict__ U_k_bases,     // [N, 2, 3]
    const float* __restrict__ T_k_matrices,  // [N, 2, 3]
    const float* __restrict__ view_dirs,     // [N, 3]
    // Params (in‑place)
    float* __restrict__ means,               // [N, 3]
    float* __restrict__ scales,              // [N, 3]
    float* __restrict__ quats,               // [N, 4]  (x y z w)
    float* __restrict__ opacities,           // [N]
    float* __restrict__ sh_coeffs            // [N, K, 3]
) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    if(radii[gid*2] <= 0 || radii[gid*2+1] <= 0) return;
    vec2  g_pos = glm::make_vec2(dL_d_pos + gid * 2);
    vec2  g_scale = glm::make_vec2(dL_d_scale + gid * 2);
    float g_rot      = dL_d_rot[gid];

    const float* gc_ptr = dL_d_color + gid * 3;
    // ---------------------------------------------------------------------
    // Position update (Δp_k = U^T Δv_k) -----------------------------------
    // ---------------------------------------------------------------------
    mat2  H_pos = glm::make_mat2(H_L_pos  + gid * 4);
    regularize_2x2_if_needed(H_pos);
    vec2  delta_vk = -glm::inverse(H_pos) * g_pos;
    mat2x3 U_k  = glm::make_mat2x3(U_k_bases + gid * 6);
    mat3x2 U_k_T = glm::transpose(U_k);
    vec3  delta_p = mul_mat3x2_vec2(U_k_T, delta_vk);

    means[gid * 3 + 0] += delta_p.x;
    means[gid * 3 + 1] += delta_p.y;
    means[gid * 3 + 2] += delta_p.z;

    // ---------------------------------------------------------------------
    // Scale update  (Δs_k = (T^T T)^{-1} T^T Δλ) ---------------------------
    // ---------------------------------------------------------------------
    mat2  H_scale = glm::make_mat2(H_L_scale  + gid * 4);

    regularize_2x2_if_needed(H_scale);
    vec2  delta_lambda = -glm::inverse(H_scale) * g_scale;
    if (gid % 1000 == 0) {
        printf("Delta_lambda: %f %f \n", delta_lambda.x, delta_lambda.y);
    } float dlambda_min = 1e-5f;
    if(fabsf(delta_lambda.x) < dlambda_min && fabs(delta_lambda.y) < dlambda_min)
    {
        // do nothing, but dont return, we still need other updates!

        //printf("lambda too small; Skipping id: %d\n", gid);
    }
    else
    {
        // proceed with update
        mat2x3 T_k   = glm::make_mat2x3(T_k_matrices + gid * 6);
        /*
        if (gid % 1000 == 0) {
            printf(
                "T_k:\n %f %f %f \n %f %f %f\n", T_k[0][0], T_k[0][1], T_k[0][2],
                                                T_k[1][0], T_k[1][1], T_k[1][2]);
        }
        */
        mat3x2 T_k_T = glm::transpose(T_k);

        // (3×2)*(2×3) manual
        mat3 TtT = mul_mat3x2_mat2x3(T_k_T, T_k);
        vec3 rhs = mul_mat3x2_vec2(T_k_T, delta_lambda);
        vec3 delta_s = solve_sym3x3_cholesky(TtT, rhs);
        //regularize_3x3_if_needed(TtT);
        //mat3 TtT_inv = glm::inverse(TtT);
        if (gid % 1000 == 0)
        {
            printf("delta_s: %f %f %f\n",delta_s.x, delta_s.y, delta_s.z);
        }
        /*
        if (gid % 1000 == 0) {
            printf(
                "TtT_inv:\n %f %f %f \n %f %f %f \n %f %f %f\n", TtT_inv[0][0], TtT_inv[1][0], TtT_inv[2][0],
                                                            TtT_inv[0][1], TtT_inv[1][1], TtT_inv[2][1],
                                                            TtT_inv[2][0],TtT_inv[2][1],TtT_inv[2][2]);
        }*/
        //vec3 temp_vec = mul_mat3x2_vec2(T_k_T, delta_lambda);
        //vec3 delta_s  = mul_mat3_vec3(TtT_inv, temp_vec);
        scales[gid * 3 + 0] = fmaxf(1e-4f, scales[gid * 3 + 0] + delta_s.x);
        scales[gid * 3 + 1] = fmaxf(1e-4f, scales[gid * 3 + 1] + delta_s.y);
        scales[gid * 3 + 2] = fmaxf(1e-4f, scales[gid * 3 + 2] + delta_s.z);
    }
    // ---------------------------------------------------------------------
    // Rotation update ------------------------------------------------------
    // ---------------------------------------------------------------------
    float H_rot = H_L_rot[gid];
    if (fabsf(H_rot) < 1e-8f) {
    // Near zero Hessian - use adaptive regularization
        H_rot += fmaxf(kMatInvEps, fabsf(H_rot) * 1e-4f);
    } else {
        H_rot += kMatInvEps;
    }

    float delta_theta = -g_rot / H_rot;  // small angle
        if(gid % 1000 == 0){
        printf("Delta theta: %f\n",delta_theta);
    }
    // axis = view_dir (already world‑space unit)
    vec3 axis = glm::normalize(glm::make_vec3(view_dirs + gid * 3));
    float half = 0.5f * delta_theta;
    float s = sinf(half);
    vec4 dq(axis.x * s, axis.y * s, axis.z * s, cosf(half));

    vec4 q = glm::make_vec4(quats + gid * 4); // (x y z w)
    vec4 q_new = glm::normalize(quat_mul(dq, q));
    quats[gid * 4 + 0] = q_new.x;
    quats[gid * 4 + 1] = q_new.y;
    quats[gid * 4 + 2] = q_new.z;
    quats[gid * 4 + 3] = q_new.w;


    // ---------------------------------------------------------------------
    // Opacity update  (barrier + smooth‑L1) --------------------------------
    // ---------------------------------------------------------------------
    float H_opac = H_L_opacity[gid];
    float sigma  = opacities[gid];

    // Barrier gradient & Hessian
    float g_bar  = -(1.f / sigma) + (1.f / (1.f - sigma));
    float H_bar  =  1.f / (sigma * sigma) + 1.f / ((1.f - sigma) * (1.f - sigma));

    // Smooth‑L1 ≈ |σ| with C2 continuity
    float abs_sig = fabsf(sigma);
    float g_l1    = kLambdaOpacity * (abs_sig > kL1Eps ? copysignf(1.f, sigma)
                                                       : sigma / kL1Eps);
    float H_l1    = kLambdaOpacity * (abs_sig > kL1Eps ? 0.f : 1.f / kL1Eps);

    float g_opac  = dL_d_opacity[gid] + g_bar + g_l1;
    float H_total = H_opac + H_bar + H_l1;
    H_total += kMatInvEps;
    if (fabsf(H_total) > 1e-6f) {
        float delta_sigma = -g_opac / H_total;
        float new_sig = fminf(1.f - 1e-6f, fmaxf(1e-6f, sigma + delta_sigma));
        opacities[gid] = new_sig;
    }

    // ---------------------------------------------------------------------
    // Colour (SH DC coefficient only, diagonal Hessian) --------------------
    // ---------------------------------------------------------------------
    const float* Hc_ptr = H_L_color + gid * 9;  // 3×3 but we use diag only

    for (int ch = 0; ch < 3; ++ch) {
        float g = gc_ptr[ch];
        float h = Hc_ptr[ch * 4] + kMatInvEps; // diag indices 0,4,8
        float delta = -g / h;
        sh_coeffs[(gid * K + 0) * 3 + ch] += delta; // DC component (band 0)
    }

    // Higher SH coefficients (band 1..K‑1) --------------------------------
    // Using diagonal Hessian approximation as per your simplification.
    if (K > 1) {
        for (int k = 1; k < K; ++k) {
            for (int ch = 0; ch < 3; ++ch) {
                // grad & Hessian pointers advance by K*3; assume external fill
                int idx = (gid * K + k) * 3 + ch;
                float g = dL_d_color[idx];             // provided grad
                float h = H_L_color[idx] + kMatInvEps; // diag only
                sh_coeffs[idx] += -g / h;
            }
        }
    }
}




void launch_solve_and_update_all_attributes_kernel(
    const at::Tensor dL_d_pos, const at::Tensor H_L_pos,
    const at::Tensor dL_d_scale, const at::Tensor H_L_scale,
    const at::Tensor dL_d_rot, const at::Tensor H_L_rot,
    const at::Tensor dL_d_opacity, const at::Tensor H_L_opacity,
    const at::Tensor dL_d_color, const at::Tensor H_L_color,
    const at::Tensor U_k_bases, const at::Tensor T_k_matrices,
    const at::Tensor view_dirs,
    const at::Tensor radii,
    at::Tensor means, at::Tensor scales, at::Tensor quats,
    at::Tensor opacities, at::Tensor sh_coeffs
) {
    const uint32_t N = means.size(0);
    if (N == 0) return;
    const int K = sh_coeffs.size(1);
    const dim3 threads(256);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    solve_updates_and_backproject_kernel_impl<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        K,
        radii.data_ptr<int32_t>(),
        dL_d_pos.data_ptr<float>(), H_L_pos.data_ptr<float>(),
        dL_d_scale.data_ptr<float>(), H_L_scale.data_ptr<float>(),
        dL_d_rot.data_ptr<float>(), H_L_rot.data_ptr<float>(),
        dL_d_opacity.data_ptr<float>(), H_L_opacity.data_ptr<float>(),
        dL_d_color.data_ptr<float>(), H_L_color.data_ptr<float>(),
        U_k_bases.data_ptr<float>(), T_k_matrices.data_ptr<float>(),
        view_dirs.data_ptr<float>(),
        means.data_ptr<float>(), scales.data_ptr<float>(), quats.data_ptr<float>(),
        opacities.data_ptr<float>(), sh_coeffs.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}




#include <cmath>
////////////////////////////////////////////////////////////////////////////////
// General‑axis version of compute_covariance_derivatives_kernel_impl
////////////////////////////////////////////////////////////////////////////////
__global__ void compute_covariance_derivatives_kernel_impl(
    const uint32_t  N,
    const float* __restrict__ quat,        // [N,4]
    const float* __restrict__ scale,       // [N,3]
    const float* __restrict__ view_matrix, // [4,4]  row‑major
    const float* __restrict__ position,    // [N,3]
    float* __restrict__ dSigma_dtheta,     // [N,2,2]
    float* __restrict__ d2Sigma_dtheta2)   // [N,2,2]
{
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;

    //----------------------------------------------------------------------
    // 0.  Pointers to this Gaussian’s data
    //----------------------------------------------------------------------
    const float* q  = quat     + gid*4;
    const float* s  = scale    + gid*3;
    const float* pW = position + gid*3;    // world

    float* dS  = dSigma_dtheta    + gid*4; // 2×2 = 4
    float* dS2 = d2Sigma_dtheta2  + gid*4;

    //----------------------------------------------------------------------
    // 1.  Covariance in world space :  M_W = R  S  Rᵀ
    //----------------------------------------------------------------------
    float R[3][3];
    {
        const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
        R[0][0] = 1.f - 2.f*(qy*qy + qz*qz);
        R[0][1] = 2.f*(qx*qy - qz*qw);
        R[0][2] = 2.f*(qx*qz + qy*qw);
        R[1][0] = 2.f*(qx*qy + qz*qw);
        R[1][1] = 1.f - 2.f*(qx*qx + qz*qz);
        R[1][2] = 2.f*(qy*qz - qx*qw);
        R[2][0] = 2.f*(qx*qz - qy*qw);
        R[2][1] = 2.f*(qy*qz + qx*qw);
        R[2][2] = 1.f - 2.f*(qx*qx + qy*qy);
    }
    const float S[3][3] = { {s[0]*s[0], 0.f, 0.f},
                            {0.f, s[1]*s[1], 0.f},
                            {0.f, 0.f, s[2]*s[2]} };

    float RS[3][3], M_W[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += R[i][k]*S[k][j];
            RS[i][j]=v;
        }
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += RS[i][k]*R[j][k];
            M_W[i][j]=v;
        }

    //----------------------------------------------------------------------
    // 2.  World position -> camera space  &  view direction r_k
    //----------------------------------------------------------------------
    float pC[3];                     // camera‑space position
    {
        const float ph[4] = {pW[0], pW[1], pW[2], 1.f};
        #pragma unroll
        for (int i=0;i<3;++i)
        {
            float v=0.f;
            #pragma unroll
            for (int j=0;j<4;++j) v += view_matrix[i*4+j]*ph[j];
            pC[i]=v;
        }
    }
    const float z      = pC[2];
    const float invL   = rsqrtf(pC[0]*pC[0] + pC[1]*pC[1] + pC[2]*pC[2] + 1e-20f); // |pC|⁻¹
    const float rx     = pC[0]*invL;
    const float ry     = pC[1]*invL;
    const float rz     = pC[2]*invL;

    //----------------------------------------------------------------------
    // 3.  [r]ₓ   and   [r]ₓ²  (skew‑symm matrix and its square)
    //----------------------------------------------------------------------
    float rX[3][3]  = { {   0.f, -rz ,  ry },
                        {  rz  ,  0.f, -rx },
                        { -ry  ,  rx ,  0.f} };

    float rX2[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += rX[i][k]*rX[k][j];
            rX2[i][j]=v;
        }

    //----------------------------------------------------------------------
    // 4.  Covariance in camera space :  M0 = R_cam M_W R_camᵀ
    //----------------------------------------------------------------------
    float M0[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)
            {
                float t=0.f;
                #pragma unroll
                for (int l=0;l<3;++l)
                    t += view_matrix[i*4+l]*M_W[l][k];
                v += t*view_matrix[j*4+k];
            }
            M0[i][j]=v;
        }

    //----------------------------------------------------------------------
    // 5.  First & second derivative of  M(θ)  at θ=0
    //----------------------------------------------------------------------
    float dM[3][3], d2M[3][3];

    // dM = rX*M0 + M0*rXᵀ
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float a=0.f, b=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)  a += rX[i][k]*M0[k][j];
            #pragma unroll
            for (int k=0;k<3;++k)  b += M0[i][k]*rX[j][k];   // rXᵀ
            dM[i][j] = a + b;
        }

    // temp = rX*M0
    float temp[3][3];
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v=0.f;
            #pragma unroll
            for (int k=0;k<3;++k) v += rX[i][k]*M0[k][j];
            temp[i][j]=v;
        }

    // d2M = rX2*M0 + M0*rX2ᵀ + 2*temp*rXᵀ
    #pragma unroll
    for (int i=0;i<3;++i)
        #pragma unroll
        for (int j=0;j<3;++j)
        {
            float v1=0.f, v2=0.f, v3=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)  v1 += rX2[i][k]*M0[k][j];
            #pragma unroll
            for (int k=0;k<3;++k)  v2 += M0[i][k]*rX2[j][k];
            #pragma unroll
            for (int k=0;k<3;++k)  v3 += temp[i][k]*rX[j][k];
            d2M[i][j] = v1 + v2 + 2.f*v3;
        }

    //----------------------------------------------------------------------
    // 6.  Jacobian of the perspective projection  J
    //----------------------------------------------------------------------
    if (z < 1e-6f) {
        // either set your dS/dS2 to zero or some safe fallback
        dS[0]=dS[1]=dS[2]=dS[3]= 0.0f;
        dS2[0]=dS2[1]=dS2[2]=dS2[3]=0.0f;
        return;
        }

    const float invZ  = 1.f / z;
    const float invZ2 = invZ * invZ;
    const float J[2][3] = { { invZ, 0.f  , -pC[0]*invZ2 },
                            { 0.f , invZ , -pC[1]*invZ2 } };

    //----------------------------------------------------------------------
    // 7.  Project derivatives :  Σ = J M Jᵀ
    //----------------------------------------------------------------------
    float JdM [2][3], Jd2M[2][3];
    #pragma unroll
    for (int i=0;i<2;++i)
        #pragma unroll
        for (int k=0;k<3;++k)
        {
            float a=0.f, b=0.f;
            #pragma unroll
            for (int j=0;j<3;++j)
            {
                a += J[i][j]*dM [j][k];
                b += J[i][j]*d2M[j][k];
            }
            JdM [i][k]=a;
            Jd2M[i][k]=b;
        }

    //----------------------------------------------------------------------
    // 8.  Final 2×2 blocks (row‑major)
    //----------------------------------------------------------------------
    #pragma unroll
    for (int i=0;i<2;++i)
        #pragma unroll
        for (int j=0;j<2;++j)
        {
            float s1=0.f, s2=0.f;
            #pragma unroll
            for (int k=0;k<3;++k)
            {
                s1 += JdM [i][k]*J[j][k];
                s2 += Jd2M[i][k]*J[j][k];
            }
            dS [i*2+j] = s1;
            dS2[i*2+j] = s2;
        }
}


// Launcher function following the same pattern as your existing code
void launch_compute_covariance_derivatives_kernel(
    const at::Tensor quat,        // [N, 4] quaternions
    const at::Tensor scale,       // [N, 3] scales
    const at::Tensor view_matrix, // [4, 4] view matrix
    const at::Tensor position,    // [N, 3] positions
    at::Tensor dSigma_dtheta,     // [N, 2, 2] output first derivatives
    at::Tensor d2Sigma_dtheta2    // [N, 2, 2] output second derivatives
) {
    const uint32_t N = position.size(0);
    if (N == 0) return;

    const dim3 threads(256);
    const dim3 blocks((N + threads.x - 1) / threads.x);

    compute_covariance_derivatives_kernel_impl<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N,
        quat.data_ptr<float>(),
        scale.data_ptr<float>(),
        view_matrix.data_ptr<float>(),
        position.data_ptr<float>(),
        dSigma_dtheta.data_ptr<float>(),
        d2Sigma_dtheta2.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat_newton