
template <typename scalar_t>
__device__ void sh_coeffs_to_color_fast_LN_old(
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
    if (degree < 2) goto final_projection;

    // --- Degree 2 ---
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
    if (degree < 3) goto final_projection;

    // --- Degree 3 ---
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
    if (degree < 4) goto final_projection;

    // --- Degree 4 ---
    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f); // Re-use from degree 3
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);    // Re-use from degree 2
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

//final_projection:
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


torch::Tensor sh_fwd_with_derivatives(int degree,
                                      const torch::Tensor &view_dirs,
                                      const torch::Tensor &sh_coeffs,
                                      torch::Tensor &d_color_d_dir,
                                      torch::Tensor &H_color_d_dir) {
    const uint32_t N = view_dirs.size(1);
    const uint32_t C = view_dirs.size(0);
    TORCH_CHECK(C == 1, "Only single camera supported for now in SH kernels.");

    auto colors = torch::zeros({C, N, 3}, view_dirs.options());
    if (N == 0)
        return colors;

    const dim3 threads(256, 1, 1);
    const dim3 blocks(GET_BLOCKS(N, threads.x), 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        view_dirs.scalar_type(), "sh_fwd_with_derivatives_kernel", ([&] {
            sh_fwd_with_derivatives_kernel<scalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    N, degree, view_dirs.data_ptr<scalar_t>(),
                    sh_coeffs.data_ptr<scalar_t>(),
                    colors.data_ptr<scalar_t>(),
                    d_color_d_dir.data_ptr<scalar_t>(),
                    H_color_d_dir.data_ptr<scalar_t>());
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return colors;
}
