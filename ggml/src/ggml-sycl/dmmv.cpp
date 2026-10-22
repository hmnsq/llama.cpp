#ضمن "convert.hpp"
#ضمن "dmmv.hpp"
#ضمن "dequantize.hpp"
#ضمن "presets.hpp"

#إذا defined(__INTEL_LLVM_COMPILER)
    #إذا __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #ضمن <sycl/ext/oneapi/bfloat16.hpp>
        #تعریف_کن GGML_SYCL_DMMV_HAS_BF16
    #پایان_اگر
    #ضمن <sycl/ext/intel/esimd.hpp>
    #ضمن "esimd.hpp"
#پایان_اگر

ساكن فراغ convert_f16(ثابت فراغ * vx, ثابت int64_t ib, ثابت صحيح iqs, dfloat2 & v){
    ثابت sycl::half *x = (ثابت sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_BF16
ساكن فراغ convert_bf16(ثابت فراغ * vx, ثابت int64_t ib, ثابت صحيح iqs, dfloat2 & v){
    ثابت sycl::ext::oneapi::bfloat16 *x = (ثابت sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#پایان_اگر

ساكن فراغ convert_f32(ثابت فراغ * vx, ثابت int64_t ib, ثابت صحيح iqs, dfloat2 & v){
    ثابت عائم * x = (ثابت عائم *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

قالب <صحيح qk, صحيح qr, dequantize_kernel_t dequantize_kernel>
ساكن فراغ dequantize_mul_mat_vec(ثابت فراغ * __restrict__ vx, ثابت dfloat * __restrict__ y, عائم * __restrict__ dst, ثابت صحيح ncols, ثابت صحيح nrows,
                                   ثابت sycl::nd_قلم<3> &قلم_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);

    إذا (ردیف >= nrows) {
        أرجع;
    }

    ثابت صحيح tid = قلم_ct1.get_local_id(2);

    ثابت صحيح iter_stride = 2*GGML_SYCL_DMMV_X;
    ثابت صحيح vals_در_هر_iter = iter_stride / WARP_اندازه; // num quantized vals per thread and i iter
    ثابت صحيح y_جابه_جایی = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#اگر_تعریف_شده GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#وإلا
    عائم tmp = 0.0f;
#پایان_اگر // GGML_SYCL_F16

    لكل (صحيح i = 0; i < ncols; i += iter_stride) {
        ثابت صحيح ستون = i + vals_در_هر_iter*tid;
        إذا (ستون >= ncols) {
            استمر;
        }
        ثابت صحيح ib = (ردیف*ncols + ستون)/qk; // x block index
        ثابت صحيح iqs = (ستون%qk)/qr; // x quant index
        ثابت صحيح iybs = ستون - ستون%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#رهنمود unroll
        لكل (صحيح j = 0; j < vals_در_هر_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#اگر_تعریف_شده GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_جابه_جایی]};

            tmp += v * t1;
#وإلا
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_جابه_جایی];
#پایان_اگر // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ثابت صحيح ماسک_شروع = ncols > GGML_SYCL_DMMV_X ? WARP_اندازه >> 1 : WARP_اندازه >> 2;
    لكل (صحيح ماسک = ماسک_شروع; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (tid == 0) {
#اگر_تعریف_شده GGML_SYCL_F16
        dst[ردیف] = tmp.x() + tmp.y();
#وإلا
        dst[ردیف] = tmp;
#پایان_اگر // GGML_SYCL_F16
    }
}

قالب <صحيح qk, صحيح qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
ساكن فراغ dequantize_mul_mat_vec_reorder(ثابت فراغ * __restrict__ vx, ثابت dfloat * __restrict__ y, عائم * __restrict__ dst, ثابت صحيح ncols, ثابت صحيح nrows,
                                   ثابت sycl::nd_قلم<3> &قلم_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);

    إذا (ردیف >= nrows) {
        أرجع;
    }

    ثابت صحيح tid = قلم_ct1.get_local_id(2);


    ثابت صحيح ncols_چپ = ncols % (QK4_0*WARP_اندازه);
    ثابت صحيح ncols_align = ncols - ncols_چپ;
    ثابت صحيح iter_stride = 8*2*GGML_SYCL_DMMV_X;
    ثابت صحيح vals_در_هر_iter = iter_stride / WARP_اندازه; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    ثابت صحيح y_جابه_جایی = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#اگر_تعریف_شده GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#وإلا
    عائم tmp = 0.0f;
#پایان_اگر // GGML_SYCL_F16
    ثابت حرف *d_ptr = (ثابت حرف*)vx+ncols*nrows/2;
    صحيح i=0;
    لكل (i = 0; i < ncols_align; i += iter_stride) {
        ثابت صحيح ستون = i + vals_در_هر_iter*tid;
        ثابت صحيح ib = (ردیف*ncols + ستون)/qk; // x block index
        ثابت صحيح iqs = (ستون%qk)/qr; // x quant index
        ثابت صحيح iybs = ستون - ستون%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#رهنمود unroll
        لكل (صحيح j = 0; j < vals_در_هر_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ثابت فراغ *)d_ptr, ib, (ثابت فراغ *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#اگر_تعریف_شده GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_جابه_جایی]};

            tmp += v * t1;
#وإلا
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_جابه_جایی];
#پایان_اگر // GGML_SYCL_F16
        }
    }

    لكل (; i < ncols; i += iter_stride) {
        إذا (tid>=ncols_چپ/QK4_0) استمر;
        ثابت صحيح ستون = i + vals_در_هر_iter*tid;
        ثابت صحيح ib = (ردیف*ncols + ستون)/qk; // x block index
        ثابت صحيح iqs = (ستون%qk)/qr; // x quant index
        ثابت صحيح iybs = ستون - ستون%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#رهنمود unroll
        لكل (صحيح j = 0; j < vals_در_هر_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ثابت فراغ *)d_ptr, ib, (ثابت فراغ *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#اگر_تعریف_شده GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_جابه_جایی]};

            tmp += v * t1;
#وإلا
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_جابه_جایی];
#پایان_اگر // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ثابت صحيح ماسک_شروع = ncols > GGML_SYCL_DMMV_X ? WARP_اندازه >> 1 : WARP_اندازه >> 2;
    لكل (صحيح ماسک = ماسک_شروع; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (tid == 0) {
#اگر_تعریف_شده GGML_SYCL_F16
        dst[ردیف] = tmp.x() + tmp.y();
#وإلا
        dst[ردیف] = tmp;
#پایان_اگر // GGML_SYCL_F16
    }
}

ساكن فراغ convert_mul_mat_vec_f16_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                         عائم *dst, ثابت صحيح ncols,
                                         ثابت صحيح nrows,
                                         dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, قلم_ct1);
            });
    }
}

#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_BF16
ساكن فراغ convert_mul_mat_vec_bf16_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                          عائم *dst, ثابت صحيح ncols,
                                          ثابت صحيح nrows,
                                          dpct::صف_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, قلم_ct1);
            });
    }
}
#پایان_اگر

ساكن فراغ dequantize_mul_mat_vec_q2_k(ثابت فراغ *__restrict__ vx,
                                        ثابت عائم *__restrict__ yy,
                                        عائم *__restrict__ dst,
                                        ثابت صحيح ncols, صحيح nrows,
                                        ثابت sycl::nd_قلم<3> &قلم_ct1) {

    static_assert(16%K_QUANTS_در_هر_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;

    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    ثابت بلوک_q2_K * x = (ثابت بلوک_q2_K *)vx + ib0;

    عائم tmp = 0; // partial sum for thread in warp

#إذا QK_K == 256
    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0,1

    ثابت صحيح گام = 16/K_QUANTS_در_هر_ITERATION;

    ثابت صحيح درون = tid % گام;                           // 0...15 or 0...7

    ثابت صحيح l0 = K_QUANTS_در_هر_ITERATION*درون;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ثابت uint8_t * d = (ثابت uint8_t *)aux;
    ثابت uint8_t * m = (ثابت uint8_t *)(aux + 2);

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {

        ثابت عائم dall = x[i].dm[0];
        ثابت عائم dmin = x[i].dm[1];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی = 32*im + l0;
            ثابت صحيح s_جابه_جایی = 8*im;
            ثابت صحيح y_جابه_جایی = 128*im + l0;

            ثابت عائم   * y = yy + i * QK_K + y_جابه_جایی;
            ثابت uint8_t * q = x[i].qs + q_جابه_جایی;

            ثابت uint32_t * a = (ثابت uint32_t *)(x[i].scales + s_جابه_جایی);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            عائم sum1 = 0, sum2 = 0;
            لكل (صحيح l = 0; l < K_QUANTS_در_هر_ITERATION; ++l) {
                sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                      + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                      + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                      + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                      + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                      + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                      + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                      +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
                sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                      + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

            }
            tmp += dall * sum1 - dmin * sum2;
        }

    }
#وإلا
    ثابت صحيح tid = قلم_ct1.get_local_id(2) /
                    (2 * K_QUANTS_در_هر_ITERATION); // 0...15 or 0...7
    ثابت صحيح ix = قلم_ct1.get_local_id(2) %
                   (2 * K_QUANTS_در_هر_ITERATION); // 0....1 or 0...3
    ثابت صحيح جابه_جایی = tid * K_QUANTS_در_هر_ITERATION;

    uint32_t uaux[2];
    ثابت uint8_t * d = (ثابت uint8_t *)uaux;


    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {

        ثابت عائم   * y = yy + i * QK_K + جابه_جایی;
        ثابت uint8_t * q = x[i].qs + جابه_جایی;
        ثابت uint32_t * s = (ثابت uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        ثابت sycl::float2 dall =
            x[i].dm.convert<عائم, sycl::rounding_نما::automatic>();

        عائم sum1 = 0, sum2 = 0;
        لكل (صحيح l = 0; l < K_QUANTS_در_هر_ITERATION; ++l) {
            ثابت uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (قلم_ct1.get_local_id(2) == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q2_k_reorder(ثابت فراغ *__restrict__ vx,
                                                ثابت عائم *__restrict__ yy,
                                                عائم *__restrict__ dst,
                                                ثابت صحيح ncols, صحيح nrows,
                                                ثابت sycl::nd_قلم<3> &قلم_ct1) {

    static_assert(16%K_QUANTS_در_هر_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;

    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    ثابت صحيح nb = nrows * رقم_بلوکها_در_هر_ردیف;
    ثابت uint8_t     * qs_پایه     = (ثابت uint8_t *)vx;
    ثابت uint8_t     * scales_پایه = qs_پایه + (size_t)nb * (QK_K / 4);
    ثابت sycl::half2 * dm_پایه     = (ثابت sycl::half2 *)(scales_پایه + (size_t)nb * (QK_K / 16));

    عائم tmp = 0; // partial sum for thread in warp

#إذا QK_K == 256
    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0,1

    ثابت صحيح گام = 16/K_QUANTS_در_هر_ITERATION;

    ثابت صحيح درون = tid % گام;                           // 0...15 or 0...7

    ثابت صحيح l0 = K_QUANTS_در_هر_ITERATION*درون;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ثابت uint8_t * d = (ثابت uint8_t *)aux;
    ثابت uint8_t * m = (ثابت uint8_t *)(aux + 2);

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {
        ثابت صحيح bi = ib0 + i;

        ثابت sycl::half2 dm_val = dm_پایه[bi];
        ثابت عائم dall = dm_val[0];
        ثابت عائم dmin = dm_val[1];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی = 32*im + l0;
            ثابت صحيح s_جابه_جایی = 8*im;
            ثابت صحيح y_جابه_جایی = 128*im + l0;

            ثابت عائم   * y = yy + i * QK_K + y_جابه_جایی;
            ثابت uint8_t * q = qs_پایه + bi * (QK_K / 4) + q_جابه_جایی;

            ثابت uint32_t * a = (ثابت uint32_t *)(scales_پایه + bi * (QK_K / 16) + s_جابه_جایی);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            عائم sum1 = 0, sum2 = 0;
            لكل (صحيح l = 0; l < K_QUANTS_در_هر_ITERATION; ++l) {
                sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                      + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                      + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                      + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                      + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                      + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                      + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                      +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
                sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                      + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

            }
            tmp += dall * sum1 - dmin * sum2;
        }
    }
#وإلا
    GGML_استفادهنشده(vx);
    GGML_استفادهنشده(yy);
    GGML_استفادهنشده(ncols);
    GGML_استفادهنشده(قلم_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (قلم_ct1.get_local_id(2) == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q3_k(ثابت فراغ *__restrict__ vx,
                                        ثابت عائم *__restrict__ yy,
                                        عائم *__restrict__ dst,
                                        ثابت صحيح ncols, صحيح nrows,
                                        ثابت sycl::nd_قلم<3> &قلم_ct1) {

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;

    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    ثابت بلوک_q3_K * x = (ثابت بلوک_q3_K *)vx + ib0;

    عائم tmp = 0; // partial sum for thread in warp

#إذا QK_K == 256

    ثابت uint16_t kmask1 = 0x0303;
    ثابت uint16_t kmask2 = 0x0f0f;

    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0,1

    ثابت صحيح n  = K_QUANTS_در_هر_ITERATION;               // iterations in the inner loop
    ثابت صحيح گام = 16/K_QUANTS_در_هر_ITERATION;
    ثابت صحيح درون = tid % گام;                           // 0...15 or 0...7

    ثابت صحيح l0 = n*درون;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ثابت int8_t * s = (ثابت int8_t *)utmp;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {

        ثابت uint8_t * h = x[i].hmask + l0;
        ثابت عائم d = x[i].d;

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی =  32*im + l0;
            ثابت صحيح y_جابه_جایی = 128*im + l0;
            ثابت uint16_t s_لغزش = 4*im;
            ثابت uint8_t m = 1 << (4*im);

            ثابت عائم   * y  = yy + i * QK_K + y_جابه_جایی;
            ثابت uint8_t * q = x[i].qs + q_جابه_جایی;

            ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_لغزش) & kmask2) | (((a[4] >> (s_لغزش + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_لغزش) & kmask2) | (((a[5] >> (s_لغزش + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_لغزش) & kmask2) | (((a[4] >> (s_لغزش + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_لغزش) & kmask2) | (((a[5] >> (s_لغزش + 2)) & kmask1) << 4);

            عائم جمع = 0;
            لكل (صحيح l = 0; l < n; ++l) {
                جمع += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                جمع += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * جمع;
        }

    }
#وإلا

    ثابت صحيح tid = قلم_ct1.get_local_id(2)/(2*K_QUANTS_در_هر_ITERATION);  // 0...15 or 0...7
    ثابت صحيح ix  = قلم_ct1.get_local_id(2)%(2*K_QUANTS_در_هر_ITERATION);  // 0....1 or 0...3
    ثابت صحيح جابه_جایی = tid * K_QUANTS_در_هر_ITERATION;         // 0...15 or 0...14
    ثابت صحيح درون = جابه_جایی/8;                                 // 0 or 1
    ثابت صحيح im = جابه_جایی%8;                                 // 0...7

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {

        ثابت عائم   * y = yy + i * QK_K + جابه_جایی;
        ثابت uint8_t * q = x[i].qs + جابه_جایی;
        ثابت uint8_t * s = x[i].scales;

        ثابت عائم dall = (عائم)x[i].d;

        عائم جمع = 0;
        لكل (صحيح l = 0; l < K_QUANTS_در_هر_ITERATION; ++l) {
            ثابت uint8_t hl = x[i].hmask[im+l] >> درون;
            ثابت uint8_t ql = q[l];
            جمع += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += جمع;
    }
#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (قلم_ct1.get_local_id(2) == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q3_k_reorder(ثابت فراغ *__restrict__ vx,
                                                ثابت عائم *__restrict__ yy,
                                                عائم *__restrict__ dst,
                                                ثابت صحيح ncols, صحيح nrows,
                                                ثابت sycl::nd_قلم<3> &قلم_ct1) {

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;

    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    ثابت صحيح nb = nrows * رقم_بلوکها_در_هر_ردیف;
    ثابت uint8_t   * qs_پایه     = (ثابت uint8_t *)vx;
    ثابت uint8_t   * hmask_پایه  = qs_پایه + (size_t)nb * (QK_K / 4);
    ثابت uint8_t   * scales_پایه = hmask_پایه + (size_t)nb * (QK_K / 8);
    ثابت sycl::half * d_پایه     = (ثابت sycl::half *)(scales_پایه + (size_t)nb * 12);

    عائم tmp = 0; // partial sum for thread in warp

#إذا QK_K == 256

    ثابت uint16_t kmask1 = 0x0303;
    ثابت uint16_t kmask2 = 0x0f0f;

    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0,1

    ثابت صحيح n  = K_QUANTS_در_هر_ITERATION;               // iterations in the inner loop
    ثابت صحيح گام = 16/K_QUANTS_در_هر_ITERATION;
    ثابت صحيح درون = tid % گام;                           // 0...15 or 0...7

    ثابت صحيح l0 = n*درون;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ثابت int8_t * s = (ثابت int8_t *)utmp;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {
        ثابت صحيح bi = ib0 + i;

        ثابت uint8_t * h  = hmask_پایه + bi * (QK_K / 8) + l0;

        ثابت عائم d = d_پایه[bi];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی =  32*im + l0;
            ثابت صحيح y_جابه_جایی = 128*im + l0;
            ثابت uint16_t s_لغزش = 4*im;
            ثابت uint8_t m = 1 << (4*im);

            ثابت عائم   * y  = yy + i * QK_K + y_جابه_جایی;
            ثابت uint8_t * q  = qs_پایه + bi * (QK_K / 4) + q_جابه_جایی;

            ثابت uint16_t * a = (ثابت uint16_t *)(scales_پایه + bi * 12);
            utmp[0] = ((a[0] >> s_لغزش) & kmask2) | (((a[4] >> (s_لغزش + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_لغزش) & kmask2) | (((a[5] >> (s_لغزش + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_لغزش) & kmask2) | (((a[4] >> (s_لغزش + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_لغزش) & kmask2) | (((a[5] >> (s_لغزش + 2)) & kmask1) << 4);

            عائم جمع = 0;
            لكل (صحيح l = 0; l < n; ++l) {
                جمع += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                جمع += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * جمع;
        }
    }
#وإلا
    GGML_استفادهنشده(vx);
    GGML_استفادهنشده(yy);
    GGML_استفادهنشده(ncols);
    GGML_استفادهنشده(قلم_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (قلم_ct1.get_local_id(2) == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q4_k(ثابت فراغ *__restrict__ vx,
                                        ثابت عائم *__restrict__ yy,
                                        عائم *__restrict__ dst,
                                        ثابت صحيح ncols, صحيح nrows,
                                        ثابت sycl::nd_قلم<3> &قلم_ct1) {

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;
    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    ثابت بلوک_q4_K * x = (ثابت بلوک_q4_K *)vx + ib0;

#إذا QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0,1

    ثابت صحيح گام = 8/K_QUANTS_در_هر_ITERATION;           // 8 or 4

    ثابت صحيح il_پایه = tid/گام;                        // 0 or 1 (was 0...3)
    ثابت صحيح ir  = tid - گام*il_پایه;                  // 0...7 or 0...3
    ثابت صحيح n   = 2 * K_QUANTS_در_هر_ITERATION;          // 2 or 4

    ثابت صحيح درون = il_پایه%2;

    ثابت صحيح l0 = n*(2*ir + درون);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

#إذا K_QUANTS_در_هر_ITERATION == 2
    uint32_t q32[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q32;
#وإلا
    uint16_t q16[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;
#پایان_اگر

    عائم tmp = 0; // partial sum for thread in warp

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {

        ثابت عائم dall = x[i].dm[0];
        ثابت عائم dmin = x[i].dm[1];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی = 32*im + l0;
            ثابت صحيح y_جابه_جایی = 64*im + l0;

            ثابت عائم   * y1 = yy + i*QK_K + y_جابه_جایی;
            ثابت عائم   * y2 = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#إذا K_QUANTS_در_هر_ITERATION == 2
            ثابت uint32_t * q1 = (ثابت uint32_t *)(x[i].qs + q_جابه_جایی);
            ثابت uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            عائم smin = 0;
            لكل (صحيح l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#وإلا
            ثابت uint16_t * q1 = (ثابت uint16_t *)(x[i].qs + q_جابه_جایی);
            ثابت uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            عائم smin = 0;
            لكل (صحيح l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#پایان_اگر
        }

    }
#وإلا
    ثابت صحيح tid = قلم_ct1.get_local_id(2)/(2*K_QUANTS_در_هر_ITERATION);  // 0...15
    ثابت صحيح ix  = قلم_ct1.get_local_id(2)%(2*K_QUANTS_در_هر_ITERATION);

    ثابت صحيح گام = tid * K_QUANTS_در_هر_ITERATION;

    uint16_t aux16[2];
    ثابت uint8_t * s = (ثابت uint8_t *)aux16;

    عائم tmp = 0;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {
        ثابت uint8_t * q = x[i].qs + گام;
        ثابت عائم   * y = yy + i*QK_K + گام;
        ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ثابت عائم d = (عائم)x[i].dm[0];
        ثابت عائم m = (عائم)x[i].dm[1];
        عائم جمع = 0.f;
        لكل (صحيح j = 0; j < K_QUANTS_در_هر_ITERATION; ++j) {
            جمع += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += جمع;
    }

#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (tid == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q4_k_reorder(ثابت فراغ *__restrict__ vx,
                                                ثابت عائم *__restrict__ yy,
                                                عائم *__restrict__ dst,
                                                ثابت صحيح ncols, صحيح nrows,
                                                ثابت sycl::nd_قلم<3> &قلم_ct1) {

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;
    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ثابت صحيح nb = nrows * رقم_بلوکها_در_هر_ردیف;
    ثابت uint8_t     * qs_پایه     = (ثابت uint8_t *)vx;
    ثابت uint8_t     * scales_پایه = qs_پایه + (size_t)nb * (QK_K / 2);
    ثابت sycl::half2 * dm_پایه     = (ثابت sycl::half2 *)(scales_پایه + (size_t)nb * K_SCALE_اندازه);

#إذا QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0,1

    ثابت صحيح گام = 8/K_QUANTS_در_هر_ITERATION;           // 8 or 4

    ثابت صحيح il_پایه = tid/گام;                        // 0 or 1 (was 0...3)
    ثابت صحيح ir  = tid - گام*il_پایه;                  // 0...7 or 0...3
    ثابت صحيح n   = 2 * K_QUANTS_در_هر_ITERATION;          // 2 or 4

    ثابت صحيح درون = il_پایه%2;

    ثابت صحيح l0 = n*(2*ir + درون);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

#إذا K_QUANTS_در_هر_ITERATION == 2
    uint32_t q32[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q32;
#وإلا
    uint16_t q16[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;
#پایان_اگر

    عائم tmp = 0; // partial sum for thread in warp

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {
        ثابت صحيح bi = ib0 + i;

        ثابت sycl::half2 dm_val = dm_پایه[bi];
        ثابت عائم dall = dm_val[0];
        ثابت عائم dmin = dm_val[1];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی = 32*im + l0;
            ثابت صحيح y_جابه_جایی = 64*im + l0;

            ثابت عائم   * y1 = yy + i*QK_K + y_جابه_جایی;
            ثابت عائم   * y2 = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)(scales_پایه + bi * K_SCALE_اندازه);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#إذا K_QUANTS_در_هر_ITERATION == 2
            ثابت uint32_t * q1 = (ثابت uint32_t *)(qs_پایه + bi * (QK_K / 2) + q_جابه_جایی);
            ثابت uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            عائم smin = 0;
            لكل (صحيح l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#وإلا
            ثابت uint16_t * q1 = (ثابت uint16_t *)(qs_پایه + bi * (QK_K / 2) + q_جابه_جایی);
            ثابت uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            عائم smin = 0;
            لكل (صحيح l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#پایان_اگر
        }

    }
#وإلا
    ثابت صحيح tid = قلم_ct1.get_local_id(2)/(2*K_QUANTS_در_هر_ITERATION);  // 0...15
    ثابت صحيح ix  = قلم_ct1.get_local_id(2)%(2*K_QUANTS_در_هر_ITERATION);

    ثابت صحيح گام = tid * K_QUANTS_در_هر_ITERATION;

    uint16_t aux16[2];
    ثابت uint8_t * s = (ثابت uint8_t *)aux16;

    عائم tmp = 0;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {
        ثابت صحيح bi = ib0 + i;

        ثابت uint8_t * q = qs_پایه + bi * (QK_K / 2) + گام;
        ثابت عائم   * y = yy + i*QK_K + گام;
        ثابت uint16_t * a = (ثابت uint16_t *)(scales_پایه + bi * K_SCALE_اندازه);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ثابت sycl::half2 dm_val = dm_پایه[bi];
        ثابت عائم d = (عائم)dm_val[0];
        ثابت عائم m = (عائم)dm_val[1];
        عائم جمع = 0.f;
        لكل (صحيح j = 0; j < K_QUANTS_در_هر_ITERATION; ++j) {
            جمع += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += جمع;
    }

#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (tid == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q5_k(ثابت فراغ *__restrict__ vx,
                                        ثابت عائم *__restrict__ yy,
                                        عائم *__restrict__ dst,
                                        ثابت صحيح ncols, صحيح nrows,
                                        ثابت sycl::nd_قلم<3> &قلم_ct1) {

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;
    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    ثابت بلوک_q5_K * x = (ثابت بلوک_q5_K *)vx + ib0;

    عائم tmp = 0; // partial sum for thread in warp

#إذا QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحيح tid = قلم_ct1.get_local_id(2) / 2; // 0...7
    ثابت صحيح ix = قلم_ct1.get_local_id(2) % 2;

    ثابت صحيح il_پایه = tid/4;     // 0 or 1 (was 0...3)
    ثابت صحيح ir  = tid - 4*il_پایه;// 0...3
    ثابت صحيح n   = 2;

    ثابت صحيح درون = il_پایه%2;

    ثابت صحيح l0 = n*(2*ir + درون);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

    uint16_t q16[8];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2) {

        ثابت uint8_t * qh  = x[i].qh + l0;
        ثابت عائم dall = x[i].dm[0];
        ثابت عائم dmin = x[i].dm[1];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی = 32*im + l0;
            ثابت صحيح y_جابه_جایی = 64*im + l0;

            ثابت uint8_t hm1  = 1 << (2*im);
            ثابت uint8_t hm2  = hm1 << 4;

            ثابت uint8_t * ql1 = x[i].qs + q_جابه_جایی;
            ثابت عائم   * y1  = yy + i*QK_K + y_جابه_جایی;
            ثابت عائم   * y2  = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 جمع = {0.f, 0.f, 0.f, 0.f};
            عائم smin = 0;
            ثابت uint16_t * q1 = (ثابت uint16_t *)ql1;
            ثابت uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            لكل (صحيح l = 0; l < n; ++l) {
                جمع.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                جمع.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                جمع.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                جمع.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (جمع.x() * sc[0] + جمع.y() * sc[1] + جمع.z() * sc[4] +
                           جمع.w() * sc[5]) -
                   dmin * smin;
        }
    }

#وإلا
    ثابت صحيح tid = قلم_ct1.get_local_id(2)/(2*K_QUANTS_در_هر_ITERATION);  // 0...15
    ثابت صحيح ix  = قلم_ct1.get_local_id(2)%(2*K_QUANTS_در_هر_ITERATION);
    ثابت صحيح گام = tid * K_QUANTS_در_هر_ITERATION;
    ثابت صحيح im = گام/8;
    ثابت صحيح درون = گام%8;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {
        ثابت uint8_t * q = x[i].qs + گام;
        ثابت int8_t  * s = x[i].scales;
        ثابت عائم   * y = yy + i*QK_K + گام;
        ثابت عائم     d = x[i].d;
        عائم جمع = 0.f;
        لكل (صحيح j = 0; j < K_QUANTS_در_هر_ITERATION; ++j) {
            ثابت uint8_t h = x[i].qh[درون+j] >> im;
            جمع += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += جمع;
    }
#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (قلم_ct1.get_local_id(2) == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q5_k_reorder(ثابت فراغ *__restrict__ vx,
                                                ثابت عائم *__restrict__ yy,
                                                عائم *__restrict__ dst,
                                                ثابت صحيح ncols, صحيح nrows,
                                                ثابت sycl::nd_قلم<3> &قلم_ct1) {

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;
    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ثابت صحيح nb = nrows * رقم_بلوکها_در_هر_ردیف;
    ثابت uint8_t     * qs_پایه     = (ثابت uint8_t *)vx;
    ثابت uint8_t     * qh_پایه     = qs_پایه + (size_t)nb * (QK_K / 2);
    ثابت uint8_t     * scales_پایه = qh_پایه + (size_t)nb * (QK_K / 8);
    ثابت sycl::half2 * dm_پایه     = (ثابت sycl::half2 *)(scales_پایه + (size_t)nb * K_SCALE_اندازه);

    عائم tmp = 0; // partial sum for thread in warp

#إذا QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحيح tid = قلم_ct1.get_local_id(2) / 2; // 0...15
    ثابت صحيح ix = قلم_ct1.get_local_id(2) % 2;

    ثابت صحيح il_پایه = tid/4;      // 0...3
    ثابت صحيح ir  = tid - 4*il_پایه;// 0...3
    ثابت صحيح n   = 2;

    ثابت صحيح درون = il_پایه%2;

    ثابت صحيح l0 = n*(2*ir + درون);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

    uint16_t q16[8];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2) {
        ثابت صحيح bi = ib0 + i;

        ثابت uint8_t * qh  = qh_پایه + bi * (QK_K / 8) + l0;
        ثابت sycl::half2 dm_val = dm_پایه[bi];
        ثابت عائم dall = dm_val[0];
        ثابت عائم dmin = dm_val[1];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح q_جابه_جایی = 32*im + l0;
            ثابت صحيح y_جابه_جایی = 64*im + l0;

            ثابت uint8_t hm1  = 1 << (2*im);
            ثابت uint8_t hm2  = hm1 << 4;

            ثابت uint8_t * ql1 = qs_پایه + bi * (QK_K / 2) + q_جابه_جایی;
            ثابت عائم   * y1  = yy + i*QK_K + y_جابه_جایی;
            ثابت عائم   * y2  = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)(scales_پایه + bi * K_SCALE_اندازه);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 جمع = {0.f, 0.f, 0.f, 0.f};
            عائم smin = 0;
            ثابت uint16_t * q1 = (ثابت uint16_t *)ql1;
            ثابت uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            لكل (صحيح l = 0; l < n; ++l) {
                جمع.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                جمع.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                جمع.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                جمع.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (جمع.x() * sc[0] + جمع.y() * sc[1] + جمع.z() * sc[4] +
                           جمع.w() * sc[5]) -
                   dmin * smin;
        }
    }
#وإلا
    // The reordered Q5_K layout is only produced for QK_K == 256.
#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (قلم_ct1.get_local_id(2) == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q6_k(ثابت فراغ * __restrict__ vx, ثابت عائم * __restrict__ yy, عائم * __restrict__ dst, ثابت صحيح ncols, صحيح nrows,
                                        ثابت sycl::nd_قلم<3> &قلم_ct1) {

    static_assert(16%K_QUANTS_در_هر_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;

    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    ثابت بلوک_q6_K * x = (ثابت بلوک_q6_K *)vx + ib0;

#إذا QK_K == 256

    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0, 1

    ثابت صحيح گام = 16/K_QUANTS_در_هر_ITERATION;          // 16 or 8

    ثابت صحيح درون = tid % گام;                           // 0...15 or 0...7

#إذا K_QUANTS_در_هر_ITERATION == 1
    ثابت صحيح l0 = K_QUANTS_در_هر_ITERATION*درون;            // 0...15
    ثابت صحيح is = 0;
#وإلا
    ثابت صحيح l0 = 4 * درون;                               // 0, 4, 8, ..., 28
    ثابت صحيح is = درون / 4;
#پایان_اگر

    عائم tmp = 0; // partial sum for thread in warp

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {

        ثابت عائم d = x[i].d;

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح ql_جابه_جایی = 64*im + l0;
            ثابت صحيح qh_جابه_جایی = 32*im + l0;
            ثابت صحيح s_جابه_جایی  =  8*im + is;
            ثابت صحيح y_جابه_جایی = 128*im + l0;

            ثابت عائم   * y  = yy + i * QK_K + y_جابه_جایی;
            ثابت uint8_t * ql = x[i].ql + ql_جابه_جایی;
            ثابت uint8_t * qh = x[i].qh + qh_جابه_جایی;
            ثابت int8_t  * s  = x[i].scales + s_جابه_جایی;

#إذا K_QUANTS_در_هر_ITERATION == 1
            عائم جمع = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += جمع;
#وإلا
            عائم جمع = 0;
            لكل (صحيح l = 0; l < 4; ++l) {
                جمع += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += جمع;
#پایان_اگر
        }

    }

#وإلا

    ثابت صحيح tid = قلم_ct1.get_local_id(2)/(2*K_QUANTS_در_هر_ITERATION);  // 0...7
    ثابت صحيح ix  = قلم_ct1.get_local_id(2)%(2*K_QUANTS_در_هر_ITERATION);  // 0...3

    ثابت صحيح گام = tid * K_QUANTS_در_هر_ITERATION;

    عائم tmp = 0; // partial sum for thread in warp

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {

        ثابت عائم   * y  = yy + i * QK_K + گام;
        ثابت uint8_t * ql = x[i].ql + گام;
        ثابت uint8_t * qh = x[i].qh + گام;
        ثابت int8_t  * s  = x[i].scales;

        ثابت عائم d = x[i+0].d;

        عائم جمع = 0;
        لكل (صحيح j = 0; j < K_QUANTS_در_هر_ITERATION; ++j) {
            جمع += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += جمع;

    }

#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (tid == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q6_k_reorder(ثابت فراغ * __restrict__ vx, ثابت عائم * __restrict__ yy, عائم * __restrict__ dst, ثابت صحيح ncols, صحيح nrows,
                                                ثابت sycl::nd_قلم<3> &قلم_ct1) {

    static_assert(16%K_QUANTS_در_هر_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
    إذا (ردیف >= nrows) أرجع;

    ثابت صحيح رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت صحيح ib0 = ردیف*رقم_بلوکها_در_هر_ردیف;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    ثابت صحيح nb = nrows * رقم_بلوکها_در_هر_ردیف;
    ثابت uint8_t   * ql_پایه     = (ثابت uint8_t *)vx;
    ثابت uint8_t   * qh_پایه     = ql_پایه + (size_t)nb * (QK_K / 2);
    ثابت int8_t    * scales_پایه = (ثابت int8_t *)(qh_پایه + (size_t)nb * (QK_K / 4));
    ثابت sycl::half * d_پایه     = (ثابت sycl::half *)((ثابت uint8_t *)scales_پایه + (size_t)nb * (QK_K / 16));

#إذا QK_K == 256

    ثابت صحيح tid =
        قلم_ct1.get_local_id(2) / K_QUANTS_در_هر_ITERATION; // 0...7 or 0...15
    ثابت صحيح ix =
        قلم_ct1.get_local_id(2) % K_QUANTS_در_هر_ITERATION; // 0 or 0, 1

    ثابت صحيح گام = 16/K_QUANTS_در_هر_ITERATION;          // 16 or 8

    ثابت صحيح درون = tid % گام;                           // 0...15 or 0...7

#إذا K_QUANTS_در_هر_ITERATION == 1
    ثابت صحيح l0 = K_QUANTS_در_هر_ITERATION*درون;            // 0...15
    ثابت صحيح is = 0;
#وإلا
    ثابت صحيح l0 = 4 * درون;                               // 0, 4, 8, ..., 28
    ثابت صحيح is = درون / 4;
#پایان_اگر

    عائم tmp = 0; // partial sum for thread in warp

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += K_QUANTS_در_هر_ITERATION) {
        ثابت صحيح bi = ib0 + i;

        ثابت عائم d = d_پایه[bi];

        لكل (صحيح im = 0; im < 2; ++im) {
            ثابت صحيح ql_جابه_جایی = 64*im + l0;
            ثابت صحيح qh_جابه_جایی = 32*im + l0;
            ثابت صحيح s_جابه_جایی  =  8*im + is;
            ثابت صحيح y_جابه_جایی = 128*im + l0;

            ثابت عائم   * y  = yy + i * QK_K + y_جابه_جایی;
            ثابت uint8_t * ql = ql_پایه + bi * (QK_K / 2) + ql_جابه_جایی;
            ثابت uint8_t * qh = qh_پایه + bi * (QK_K / 4) + qh_جابه_جایی;
            ثابت int8_t  * s  = scales_پایه + bi * (QK_K / 16) + s_جابه_جایی;

#إذا K_QUANTS_در_هر_ITERATION == 1
            عائم جمع = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += جمع;
#وإلا
            عائم جمع = 0;
            لكل (صحيح l = 0; l < 4; ++l) {
                جمع += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += جمع;
#پایان_اگر
        }

    }

#وإلا

    ثابت صحيح tid = قلم_ct1.get_local_id(2)/(2*K_QUANTS_در_هر_ITERATION);  // 0...7
    ثابت صحيح ix  = قلم_ct1.get_local_id(2)%(2*K_QUANTS_در_هر_ITERATION);  // 0...3

    ثابت صحيح گام = tid * K_QUANTS_در_هر_ITERATION;

    عائم tmp = 0; // partial sum for thread in warp

    لكل (صحيح i = ix; i < رقم_بلوکها_در_هر_ردیف; i += 2*K_QUANTS_در_هر_ITERATION) {
        ثابت صحيح bi = ib0 + i;

        ثابت عائم   * y  = yy + i * QK_K + گام;
        ثابت uint8_t * ql = ql_پایه + bi * (QK_K / 2) + گام;
        ثابت uint8_t * qh = qh_پایه + bi * (QK_K / 4) + گام;
        ثابت int8_t  * s  = scales_پایه + bi * (QK_K / 16);

        ثابت عائم d = d_پایه[bi];

        عائم جمع = 0;
        لكل (صحيح j = 0; j < K_QUANTS_در_هر_ITERATION; ++j) {
            جمع += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += جمع;

    }

#پایان_اگر

    // sum up partial sums and write back result
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        tmp +=
            dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
    }

    إذا (tid == 0) {
        dst[ردیف] = tmp;
    }
}

ساكن فراغ dequantize_mul_mat_vec_q4_0_sycl_reorder(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}


ساكن فراغ dequantize_mul_mat_vec_q4_0_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q1_0_sycl_reorder(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q1_0_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q4_1_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q5_0_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q5_1_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q8_0_sycl_reorder(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                ثابت صحيح ردیف = قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(1) +
                    قلم_ct1.get_local_id(1);
                إذا (ردیف >= nrows) أرجع;

                ثابت صحيح tid = قلم_ct1.get_local_id(2);
                ثابت صحيح iter_stride = 8*2*GGML_SYCL_DMMV_X;
                ثابت صحيح vals_در_هر_iter = iter_stride / WARP_اندازه;
                ثابت صحيح ncols_چپ = ncols % (QK8_0*WARP_اندازه);
                ثابت صحيح ncols_align = ncols - ncols_چپ;

#اگر_تعریف_شده GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#وإلا
                عائم tmp = 0.0f;
#پایان_اگر
                ثابت حرف *d_ptr = (ثابت حرف*)vx + ncols*nrows;  // d after all qs

                صحيح i = 0;
                لكل (i = 0; i < ncols_align; i += iter_stride) {
                    ثابت صحيح ستون = i + vals_در_هر_iter*tid;
                    ثابت صحيح ib = (ردیف*ncols + ستون)/QK8_0;
                    ثابت صحيح iqs = ستون % QK8_0;

#رهنمود unroll
                    لكل (صحيح j = 0; j < vals_در_هر_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ثابت فراغ *)d_ptr, ib, (ثابت فراغ *)vx,
                                                ib * QK8_0 + iqs + j, v);

#اگر_تعریف_شده GGML_SYCL_F16
                        dfloat2 t1{y[ستون + j + 0], y[ستون + j + 1]};
                        tmp += v * t1;
#وإلا
                        tmp += v.x() * y[ستون + j + 0];
                        tmp += v.y() * y[ستون + j + 1];
#پایان_اگر
                    }
                }

                // handle remaining columns
                لكل (; i < ncols; i += iter_stride) {
                    إذا (tid >= ncols_چپ/QK8_0) استمر;
                    ثابت صحيح ستون = i + vals_در_هر_iter*tid;
                    ثابت صحيح ib = (ردیف*ncols + ستون)/QK8_0;
                    ثابت صحيح iqs = ستون % QK8_0;

#رهنمود unroll
                    لكل (صحيح j = 0; j < vals_در_هر_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ثابت فراغ *)d_ptr, ib, (ثابت فراغ *)vx,
                                                ib * QK8_0 + iqs + j, v);

#اگر_تعریف_شده GGML_SYCL_F16
                        dfloat2 t1{y[ستون + j + 0], y[ستون + j + 1]};
                        tmp += v * t1;
#وإلا
                        tmp += v.x() * y[ستون + j + 0];
                        tmp += v.y() * y[ستون + j + 1];
#پایان_اگر
                    }
                }

                // reduce
                ثابت صحيح ماسک_شروع = ncols > GGML_SYCL_DMMV_X ? WARP_اندازه >> 1 : WARP_اندازه >> 2;
                لكل (صحيح ماسک = ماسک_شروع; ماسک > 0; ماسک >>= 1) {
                    tmp += dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), tmp, ماسک);
                }

                إذا (tid == 0) {
#اگر_تعریف_شده GGML_SYCL_F16
                    dst[ردیف] = tmp.x() + tmp.y();
#وإلا
                    dst[ردیف] = tmp;
#پایان_اگر
                }
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q8_0_sycl(ثابت فراغ *vx, ثابت dfloat *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحيح بلوک_رقم_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, GGML_SYCL_MMV_Y, WARP_اندازه);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
            [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, قلم_ct1);
            });
    }
}

ساكن فراغ dequantize_mul_mat_vec_q2_K_sycl(ثابت فراغ *vx, ثابت عائم *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q2_K_sycl_reorder(ثابت فراغ *vx, ثابت عائم *y,
                                                     عائم *dst, ثابت صحيح ncols,
                                                     ثابت صحيح nrows,
                                                     dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q3_K_sycl(ثابت فراغ *vx, ثابت عائم *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q3_K_sycl_reorder(ثابت فراغ *vx, ثابت عائم *y,
                                                     عائم *dst, ثابت صحيح ncols,
                                                     ثابت صحيح nrows,
                                                     dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q4_K_sycl(ثابت فراغ *vx, ثابت عائم *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q5_K_sycl(ثابت فراغ *vx, ثابت عائم *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q6_K_sycl(ثابت فراغ *vx, ثابت عائم *y,
                                             عائم *dst, ثابت صحيح ncols,
                                             ثابت صحيح nrows,
                                             dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
استفاده ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_اندازه;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
بنية esimd_dmmv_ردیف_pair_policy {
    ثابت فراغ * vx;
    عائم *      dst;

    ESIMD_INLINE ثابت فراغ * w_a() ثابت { أرجع vx; }
    ESIMD_INLINE ثابت فراغ * w_b() ثابت { أرجع vx; }

    ESIMD_INLINE صحيح  ردیف_a(صحيح g) ثابت { أرجع g * 2; }
    ESIMD_INLINE صحيح  ردیف_b(صحيح g) ثابت { أرجع g * 2 + 1; }
    ESIMD_INLINE منطقي has_b(صحيح g, صحيح nrows) ثابت { أرجع ردیف_b(g) < nrows; }

    ESIMD_INLINE فراغ epilogue(صحيح g, صحيح nrows, عائم جمع_a, عائم جمع_b) ثابت {
        dst[ردیف_a(g)] = جمع_a;
        إذا (has_b(g, nrows)) {
            dst[ردیف_b(g)] = جمع_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
بنية esimd_dmmv_glu_policy {
    ثابت فراغ * vgate;
    ثابت فراغ * vup;
    عائم *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE ثابت فراغ * w_a() ثابت { أرجع vgate; }
    ESIMD_INLINE ثابت فراغ * w_b() ثابت { أرجع vup; }

    ESIMD_INLINE صحيح  ردیف_a(صحيح g) ثابت { أرجع g; }
    ESIMD_INLINE صحيح  ردیف_b(صحيح g) ثابت { أرجع g; }
    ESIMD_INLINE منطقي has_b(صحيح, صحيح) ثابت { أرجع درست; } // up always contributes

    ESIMD_INLINE فراغ epilogue(صحيح g, صحيح, عائم gate, عائم به_بالا) ثابت {
        استفاده نطاق sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        ثابت عائم GELU_COEF_A    = 0.044715f;
        ثابت عائم SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        ثابت عائم NEG_LOG2E      = -1.44269504088896341f;

        ثابت عائم arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<عائم, 16> t = arg * NEG_LOG2E;
        simd<عائم, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * به_بالا;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
قالب <ggml_type T, نام_نوع Policy>
ESIMD_INLINE فراغ dequantize_mul_mat_vec_reorder_esimd(
        ثابت Policy & policy, ثابت عائم * y,
        ثابت صحيح ncols, ثابت صحيح nrows,
        sycl::local_accessor<عائم, 1> lmem,
        ثابت sycl::nd_قلم<1> & it) {
    استفاده نطاق sycl::ext::intel::esimd;
    استفاده traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    ثابت صحيح    رقم_بلوکها_در_هر_ردیف = ncols / QK_K;
    ثابت size_t nb = (size_t) nrows * رقم_بلوکها_در_هر_ردیف;
    ثابت خودکار   pa = traits::make_ptrs(policy.w_a(), nb);
    ثابت خودکار   pb = traits::make_ptrs(policy.w_b(), nb);

    ثابت صحيح  tid   = it.get_local_id(0);
    ثابت صحيح  g     = it.get_گروه(0);
    ثابت صحيح  ردیف_a = policy.ردیف_a(g);
    ثابت صحيح  ردیف_b = policy.ردیف_b(g);
    ثابت منطقي has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<عائم, 32> acc_a = 0.0f;
    simd<عائم, 32> acc_b = 0.0f;

    لكل (صحيح ib = tid; ib < رقم_بلوکها_در_هر_ردیف; ib += GGML_SYCL_DMMV_ESIMD_WG_اندازه) {
        simd<عائم, 256> y_vec = بلوک_بارگیری<عائم, 256>(y + (size_t) ib * QK_K);

        ثابت size_t bi_a = (size_t) ردیف_a * رقم_بلوکها_در_هر_ردیف + ib;
        ثابت size_t bi_b = (size_t) ردیف_b * رقم_بلوکها_در_هر_ردیف + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<عائم>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<عائم>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    إذا (tid == 0) {
        عائم جمع_a = 0.0f;
        عائم جمع_b = 0.0f;
        لكل (صحيح p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_اندازه; ++p) {
            جمع_a += lmem[p * 2 + 0];
            جمع_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, جمع_a, جمع_b);
    }
}

ساكن فراغ dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(ثابت فراغ *vx, ثابت عائم *y,
                                                           عائم *dst, ثابت صحيح ncols,
                                                           ثابت صحيح nrows,
                                                           dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_اندازه), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_ردیف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساكن فراغ dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(ثابت فراغ *vx, ثابت عائم *y,
                                                           عائم *dst, ثابت صحيح ncols,
                                                           ثابت صحيح nrows,
                                                           dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_اندازه), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_ردیف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساكن فراغ dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(ثابت فراغ *vx, ثابت عائم *y,
                                                           عائم *dst, ثابت صحيح ncols,
                                                           ثابت صحيح nrows,
                                                           dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_اندازه), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_ردیف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساكن فراغ dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(ثابت فراغ *vx, ثابت عائم *y,
                                                           عائم *dst, ثابت صحيح ncols,
                                                           ثابت صحيح nrows,
                                                           dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_اندازه), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_ردیف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساكن فراغ dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(ثابت فراغ *vx, ثابت عائم *y,
                                                           عائم *dst, ثابت صحيح ncols,
                                                           ثابت صحيح nrows,
                                                           dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_اندازه), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_ردیف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
قالب <صحيح NBLK>
ESIMD_INLINE فراغ q8_0_mac_stripe(
        ثابت int8_t * qs_a, ثابت int8_t * qs_b,
        ثابت sycl::half * d_a, ثابت sycl::half * d_b, منطقي has_b,
        sycl::ext::intel::esimd::simd<عائم, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<عائم, 32> & acc_a,
        sycl::ext::intel::esimd::simd<عائم, 32> & acc_b) {
    استفاده نطاق sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = بلوک_بارگیری<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = بلوک_بارگیری<sycl::half, NBLK>(d_a, element_aligned_برچسب{});
    simd<sycl::half, NBLK>  db = 0;
    إذا (has_b) {
        qb = بلوک_بارگیری<int8_t, 32 * NBLK>(qs_b);
        db = بلوک_بارگیری<sycl::half, NBLK>(d_b, element_aligned_برچسب{});
    }

    simd<عائم, NBLK> da_f = convert<عائم>(da);
    simd<عائم, NBLK> db_f = convert<عائم>(db);

#رهنمود unroll
    لكل (صحيح s = 0; s < NBLK; ++s) {
        simd<عائم, 32>  y_s  = y_vec.قالب select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.قالب select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.قالب select<32, 1>(s * 32);
        ثابت عائم sa = da_f[s];
        ثابت عائم sb = db_f[s];
        acc_a += y_s * (convert<عائم>(qa_s) * sa);
        acc_b += y_s * (convert<عائم>(qb_s) * sb);
    }
}

قالب <صحيح WG, نام_نوع Policy>
ESIMD_INLINE فراغ dequantize_mul_mat_vec_q8_0_reorder_esimd(
        ثابت Policy & policy, ثابت عائم * y,
        ثابت صحيح ncols, ثابت صحيح nrows,
        sycl::local_accessor<عائم, 1> lmem,
        ثابت sycl::nd_قلم<1> & it) {
    استفاده نطاق sycl::ext::intel::esimd;

    constexpr صحيح STRIPE = 8;

    ثابت صحيح          nblk_ردیف = ncols / QK8_0;
    ثابت size_t       nb       = (size_t) nrows * nblk_ردیف;
    ثابت int8_t *     qs_a     = (ثابت int8_t *) policy.w_a();
    ثابت int8_t *     qs_b     = (ثابت int8_t *) policy.w_b();
    ثابت sycl::half * d_a      = (ثابت sycl::half *) (qs_a + nb * QK8_0);
    ثابت sycl::half * d_b      = (ثابت sycl::half *) (qs_b + nb * QK8_0);

    ثابت صحيح  tid   = it.get_local_id(0);
    ثابت صحيح  g     = it.get_گروه(0);
    ثابت منطقي has_b = policy.has_b(g, nrows);

    ثابت size_t پایه_a = (size_t) policy.ردیف_a(g) * nblk_ردیف;
    ثابت size_t پایه_b = (size_t) policy.ردیف_b(g) * nblk_ردیف;

    simd<عائم, 32> acc_a = 0.0f;
    simd<عائم, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    صحيح ib = 0;
    لكل (; ib + WG * STRIPE <= nblk_ردیف; ib += WG * STRIPE) {
        ثابت صحيح b = ib + tid * STRIPE;
        simd<عائم, 256> y_vec = بلوک_بارگیری<عائم, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (پایه_a + b) * QK8_0, qs_b + (پایه_b + b) * QK8_0,
                                d_a + پایه_a + b, d_b + پایه_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    لكل (صحيح b = ib + tid; b < nblk_ردیف; b += WG) {
        simd<عائم, 32> y_vec = بلوک_بارگیری<عائم, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (پایه_a + b) * QK8_0, qs_b + (پایه_b + b) * QK8_0,
                           d_a + پایه_a + b, d_b + پایه_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<عائم>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<عائم>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    إذا (tid == 0) {
        عائم جمع_a = 0.0f;
        عائم جمع_b = 0.0f;
        لكل (صحيح p = 0; p < WG; ++p) {
            جمع_a += lmem[p * 2 + 0];
            جمع_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, جمع_a, جمع_b);
    }
}

قالب <صحيح WG, نام_نوع Policy>
ساكن فراغ q8_0_esimd_launch(ثابت Policy & policy, ثابت عائم * y, ثابت صحيح ncols, ثابت صحيح nrows,
                              ثابت صحيح workgroups, dpct::صف_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
قالب <نام_نوع Policy>
ساكن فراغ q8_0_esimd_launch_scaled(ثابت Policy & policy, ثابت عائم * y, ثابت صحيح ncols, ثابت صحيح nrows,
                                     ثابت صحيح workgroups, dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    ثابت صحيح nblk_ردیف = ncols / QK8_0;
    إذا (nblk_ردیف >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } وإلا إذا (nblk_ردیف >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } وإلا إذا (nblk_ردیف >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } وإلا {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

ساكن فراغ dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(ثابت فراغ *vx, ثابت عائم *y,
                                                           عائم *dst, ثابت صحيح ncols,
                                                           ثابت صحيح nrows,
                                                           dpct::صف_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_ردیف_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
قالب <ggml_type T>
ساكن فراغ dequantize_mul_mat_vec_glu_reorder_esimd(ثابت فراغ * vx, ثابت فراغ * vgate,
                                                     ثابت عائم * y, عائم * dst,
                                                     ثابت صحيح ncols, ثابت صحيح nrows,
                                                     ثابت ggml_glu_op glu_op,
                                                     dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<عائم, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_اندازه), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_اندازه)),
            [=](sycl::nd_قلم<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

منطقي ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(تعداد ggml_type src0_type, تعداد ggml_glu_op glu_op,
                                                        ثابت فراغ * vx, ثابت فراغ * vgate, ثابت عائم * y,
                                                        عائم * dst, صحيح ncols, صحيح nrows, dpct::صف_ptr stream) {
    بدل (src0_type) {
        حالة GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            أرجع درست;
        حالة GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            أرجع درست;
        حالة GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            أرجع درست;
        حالة GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            أرجع درست;
        حالة GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            أرجع درست;
        حالة GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            أرجع درست;
        افتراضي:
            أرجع خطأ;
    }
}

#پایان_اگر // GGML_SYCL_DMMV_HAS_ESIMD

ساكن فراغ dequantize_mul_mat_vec_q4_K_sycl_reorder(ثابت فراغ *vx, ثابت عائم *y,
                                                     عائم *dst, ثابت صحيح ncols,
                                                     ثابت صحيح nrows,
                                                     dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q5_K_sycl_reorder(ثابت فراغ *vx, ثابت عائم *y,
                                                     عائم *dst, ثابت صحيح ncols,
                                                     ثابت صحيح nrows,
                                                     dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

ساكن فراغ dequantize_mul_mat_vec_q6_K_sycl_reorder(ثابت فراغ *vx, ثابت عائم *y,
                                                     عائم *dst, ثابت صحيح ncols,
                                                     ثابت صحيح nrows,
                                                     dpct::صف_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحيح ny = 2 / K_QUANTS_در_هر_ITERATION;
    ثابت صحيح بلوک_رقم_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلوک_nums(1, 1, بلوک_رقم_y);
    ثابت sycl::range<3> بلوک_dims(1, ny, WARP_اندازه);
    stream->parallel_for(
        sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) [[sycl::reqd_sub_گروه_اندازه(WARP_اندازه)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, قلم_ct1);
        });
}

فراغ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت عائم *src1_ddf_i, ثابت حرف *src1_ddq_i,
    عائم *dst_dd_i, ثابت int64_t ردیف_کم, ثابت int64_t ردیف_بالا,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_ردیف_اندازه,
    ثابت dpct::صف_ptr &stream) {

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ردیف_diff = ردیف_بالا - ردیف_کم;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#اگر_تعریف_شده GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = اشاره_تهی; // dfloat == half

#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    ثابت منطقي q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#وإلا
    ثابت منطقي q8_0_esimd = خطأ;
#پایان_اگر

    منطقي src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    إذا (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        ثابت مقصد_fp16_sycl_t مقصد_fp16_sycl = ggml_get_مقصد_fp16_sycl(src1->type, dst);
        GGML_ASSERT(مقصد_fp16_sycl != اشاره_تهی);
        مقصد_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#وإلا
    ثابت dfloat * src1_dfloat = (ثابت dfloat *) src1_ddf_i; // dfloat == float, no conversion
#پایان_اگر // GGML_SYCL_F16

    بدل (src0->type) {
        حالة GGML_TYPE_Q1_0:
            إذا ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            } وإلا {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q4_0:
            إذا ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            } وإلا {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            اكسر;
        حالة GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            اكسر;
        حالة GGML_TYPE_Q8_0:
            إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
                إذا (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                } وإلا
#پایان_اگر
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
                }
            } وإلا {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q2_K:
            إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
                إذا (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
                وإلا
#پایان_اگر
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
            } وإلا {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q3_K:
            إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
                إذا (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
                وإلا
#پایان_اگر
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
            } وإلا {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q4_K:
            إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
                إذا (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
                وإلا
#پایان_اگر
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
            } وإلا {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q5_K:
            إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
                إذا (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
                وإلا
#پایان_اگر
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
            } وإلا {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_Q6_K:
            إذا ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
                إذا (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
                وإلا
#پایان_اگر
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
                }
            } وإلا {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ردیف_diff, stream);
            }
            اكسر;
        حالة GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            اكسر;
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_BF16
        حالة GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ردیف_diff, stream);
            اكسر;
#پایان_اگر
        افتراضي:
            چاپ_کن("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_استفادهنشده(src1);
    GGML_استفادهنشده(dst);
    GGML_استفادهنشده(src1_ddq_i);
    GGML_استفادهنشده(src1_ncols);
    GGML_استفادهنشده(src1_padded_ردیف_اندازه);
    GGML_استفادهنشده(ctx);
}
