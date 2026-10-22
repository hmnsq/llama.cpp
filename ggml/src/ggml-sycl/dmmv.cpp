#شامل "convert.hpp"
#شامل "dmmv.hpp"
#شامل "dequantize.hpp"
#شامل "presets.hpp"

#اگر defined(__INTEL_LLVM_COMPILER)
    #اگر __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #شامل <sycl/ext/oneapi/bfloat16.hpp>
        #تعریف_کرو GGML_SYCL_DMMV_HAS_BF16
    #اگر_ختم
    #شامل <sycl/ext/intel/esimd.hpp>
    #شامل "esimd.hpp"
#اگر_ختم

ساکن خالی convert_f16(ثابت خالی * vx, ثابت int64_t ib, ثابت صحیح iqs, dfloat2 & v){
    ثابت sycl::half *x = (ثابت sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_BF16
ساکن خالی convert_bf16(ثابت خالی * vx, ثابت int64_t ib, ثابت صحیح iqs, dfloat2 & v){
    ثابت sycl::ext::oneapi::bfloat16 *x = (ثابت sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#اگر_ختم

ساکن خالی convert_f32(ثابت خالی * vx, ثابت int64_t ib, ثابت صحیح iqs, dfloat2 & v){
    ثابت اعشاری * x = (ثابت اعشاری *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t dequantize_kernel>
ساکن خالی dequantize_mul_mat_vec(ثابت خالی * __restrict__ vx, ثابت dfloat * __restrict__ y, اعشاری * __restrict__ dst, ثابت صحیح ncols, ثابت صحیح nrows,
                                   ثابت sycl::nd_شے<3> &شے_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح tid = شے_ct1.get_local_id(2);

    ثابت صحیح iter_stride = 2*GGML_SYCL_DMMV_X;
    ثابت صحیح vals_فی_iter = iter_stride / WARP_سائز; // num quantized vals per thread and i iter
    ثابت صحیح y_انحراف = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#اگر_تعریف_شدہ GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#ورنہ
    اعشاری tmp = 0.0f;
#اگر_ختم // GGML_SYCL_F16

    برائے (صحیح i = 0; i < ncols; i += iter_stride) {
        ثابت صحیح ستون = i + vals_فی_iter*tid;
        اگر (ستون >= ncols) {
            جاری;
        }
        ثابت صحیح ib = (صف*ncols + ستون)/qk; // x block index
        ثابت صحیح iqs = (ستون%qk)/qr; // x quant index
        ثابت صحیح iybs = ستون - ستون%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ہدایت unroll
        برائے (صحیح j = 0; j < vals_فی_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#اگر_تعریف_شدہ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_انحراف]};

            tmp += v * t1;
#ورنہ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_انحراف];
#اگر_ختم // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ثابت صحیح نقاب_آغاز = ncols > GGML_SYCL_DMMV_X ? WARP_سائز >> 1 : WARP_سائز >> 2;
    برائے (صحیح نقاب = نقاب_آغاز; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (tid == 0) {
#اگر_تعریف_شدہ GGML_SYCL_F16
        dst[صف] = tmp.x() + tmp.y();
#ورنہ
        dst[صف] = tmp;
#اگر_ختم // GGML_SYCL_F16
    }
}

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
ساکن خالی dequantize_mul_mat_vec_reorder(ثابت خالی * __restrict__ vx, ثابت dfloat * __restrict__ y, اعشاری * __restrict__ dst, ثابت صحیح ncols, ثابت صحیح nrows,
                                   ثابت sycl::nd_شے<3> &شے_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);

    اگر (صف >= nrows) {
        واپس;
    }

    ثابت صحیح tid = شے_ct1.get_local_id(2);


    ثابت صحیح ncols_بایاں = ncols % (QK4_0*WARP_سائز);
    ثابت صحیح ncols_align = ncols - ncols_بایاں;
    ثابت صحیح iter_stride = 8*2*GGML_SYCL_DMMV_X;
    ثابت صحیح vals_فی_iter = iter_stride / WARP_سائز; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    ثابت صحیح y_انحراف = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#اگر_تعریف_شدہ GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#ورنہ
    اعشاری tmp = 0.0f;
#اگر_ختم // GGML_SYCL_F16
    ثابت حرف *d_ptr = (ثابت حرف*)vx+ncols*nrows/2;
    صحیح i=0;
    برائے (i = 0; i < ncols_align; i += iter_stride) {
        ثابت صحیح ستون = i + vals_فی_iter*tid;
        ثابت صحیح ib = (صف*ncols + ستون)/qk; // x block index
        ثابت صحیح iqs = (ستون%qk)/qr; // x quant index
        ثابت صحیح iybs = ستون - ستون%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ہدایت unroll
        برائے (صحیح j = 0; j < vals_فی_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ثابت خالی *)d_ptr, ib, (ثابت خالی *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#اگر_تعریف_شدہ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_انحراف]};

            tmp += v * t1;
#ورنہ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_انحراف];
#اگر_ختم // GGML_SYCL_F16
        }
    }

    برائے (; i < ncols; i += iter_stride) {
        اگر (tid>=ncols_بایاں/QK4_0) جاری;
        ثابت صحیح ستون = i + vals_فی_iter*tid;
        ثابت صحیح ib = (صف*ncols + ستون)/qk; // x block index
        ثابت صحیح iqs = (ستون%qk)/qr; // x quant index
        ثابت صحیح iybs = ستون - ستون%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ہدایت unroll
        برائے (صحیح j = 0; j < vals_فی_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ثابت خالی *)d_ptr, ib, (ثابت خالی *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#اگر_تعریف_شدہ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_انحراف]};

            tmp += v * t1;
#ورنہ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_انحراف];
#اگر_ختم // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ثابت صحیح نقاب_آغاز = ncols > GGML_SYCL_DMMV_X ? WARP_سائز >> 1 : WARP_سائز >> 2;
    برائے (صحیح نقاب = نقاب_آغاز; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (tid == 0) {
#اگر_تعریف_شدہ GGML_SYCL_F16
        dst[صف] = tmp.x() + tmp.y();
#ورنہ
        dst[صف] = tmp;
#اگر_ختم // GGML_SYCL_F16
    }
}

ساکن خالی convert_mul_mat_vec_f16_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                         اعشاری *dst, ثابت صحیح ncols,
                                         ثابت صحیح nrows,
                                         dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, شے_ct1);
            });
    }
}

#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_BF16
ساکن خالی convert_mul_mat_vec_bf16_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                          اعشاری *dst, ثابت صحیح ncols,
                                          ثابت صحیح nrows,
                                          dpct::قطار_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, شے_ct1);
            });
    }
}
#اگر_ختم

ساکن خالی dequantize_mul_mat_vec_q2_k(ثابت خالی *__restrict__ vx,
                                        ثابت اعشاری *__restrict__ yy,
                                        اعشاری *__restrict__ dst,
                                        ثابت صحیح ncols, صحیح nrows,
                                        ثابت sycl::nd_شے<3> &شے_ct1) {

    static_assert(16%K_QUANTS_فی_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;

    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    ثابت بلاک_q2_K * x = (ثابت بلاک_q2_K *)vx + ib0;

    اعشاری tmp = 0; // partial sum for thread in warp

#اگر QK_K == 256
    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0,1

    ثابت صحیح قدم = 16/K_QUANTS_فی_ITERATION;

    ثابت صحیح اندر = tid % قدم;                           // 0...15 or 0...7

    ثابت صحیح l0 = K_QUANTS_فی_ITERATION*اندر;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ثابت uint8_t * d = (ثابت uint8_t *)aux;
    ثابت uint8_t * m = (ثابت uint8_t *)(aux + 2);

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {

        ثابت اعشاری dall = x[i].dm[0];
        ثابت اعشاری dmin = x[i].dm[1];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف = 32*im + l0;
            ثابت صحیح s_انحراف = 8*im;
            ثابت صحیح y_انحراف = 128*im + l0;

            ثابت اعشاری   * y = yy + i * QK_K + y_انحراف;
            ثابت uint8_t * q = x[i].qs + q_انحراف;

            ثابت uint32_t * a = (ثابت uint32_t *)(x[i].scales + s_انحراف);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            اعشاری sum1 = 0, sum2 = 0;
            برائے (صحیح l = 0; l < K_QUANTS_فی_ITERATION; ++l) {
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
#ورنہ
    ثابت صحیح tid = شے_ct1.get_local_id(2) /
                    (2 * K_QUANTS_فی_ITERATION); // 0...15 or 0...7
    ثابت صحیح ix = شے_ct1.get_local_id(2) %
                   (2 * K_QUANTS_فی_ITERATION); // 0....1 or 0...3
    ثابت صحیح انحراف = tid * K_QUANTS_فی_ITERATION;

    uint32_t uaux[2];
    ثابت uint8_t * d = (ثابت uint8_t *)uaux;


    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {

        ثابت اعشاری   * y = yy + i * QK_K + انحراف;
        ثابت uint8_t * q = x[i].qs + انحراف;
        ثابت uint32_t * s = (ثابت uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        ثابت sycl::float2 dall =
            x[i].dm.convert<اعشاری, sycl::rounding_منوال::automatic>();

        اعشاری sum1 = 0, sum2 = 0;
        برائے (صحیح l = 0; l < K_QUANTS_فی_ITERATION; ++l) {
            ثابت uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q2_k_reorder(ثابت خالی *__restrict__ vx,
                                                ثابت اعشاری *__restrict__ yy,
                                                اعشاری *__restrict__ dst,
                                                ثابت صحیح ncols, صحیح nrows,
                                                ثابت sycl::nd_شے<3> &شے_ct1) {

    static_assert(16%K_QUANTS_فی_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;

    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    ثابت صحیح nb = nrows * عدد_بلاکیں_فی_صف;
    ثابت uint8_t     * qs_بنیاد     = (ثابت uint8_t *)vx;
    ثابت uint8_t     * scales_بنیاد = qs_بنیاد + (size_t)nb * (QK_K / 4);
    ثابت sycl::half2 * dm_بنیاد     = (ثابت sycl::half2 *)(scales_بنیاد + (size_t)nb * (QK_K / 16));

    اعشاری tmp = 0; // partial sum for thread in warp

#اگر QK_K == 256
    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0,1

    ثابت صحیح قدم = 16/K_QUANTS_فی_ITERATION;

    ثابت صحیح اندر = tid % قدم;                           // 0...15 or 0...7

    ثابت صحیح l0 = K_QUANTS_فی_ITERATION*اندر;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ثابت uint8_t * d = (ثابت uint8_t *)aux;
    ثابت uint8_t * m = (ثابت uint8_t *)(aux + 2);

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {
        ثابت صحیح bi = ib0 + i;

        ثابت sycl::half2 dm_val = dm_بنیاد[bi];
        ثابت اعشاری dall = dm_val[0];
        ثابت اعشاری dmin = dm_val[1];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف = 32*im + l0;
            ثابت صحیح s_انحراف = 8*im;
            ثابت صحیح y_انحراف = 128*im + l0;

            ثابت اعشاری   * y = yy + i * QK_K + y_انحراف;
            ثابت uint8_t * q = qs_بنیاد + bi * (QK_K / 4) + q_انحراف;

            ثابت uint32_t * a = (ثابت uint32_t *)(scales_بنیاد + bi * (QK_K / 16) + s_انحراف);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            اعشاری sum1 = 0, sum2 = 0;
            برائے (صحیح l = 0; l < K_QUANTS_فی_ITERATION; ++l) {
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
#ورنہ
    GGML_غیر_مستعمل(vx);
    GGML_غیر_مستعمل(yy);
    GGML_غیر_مستعمل(ncols);
    GGML_غیر_مستعمل(شے_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q3_k(ثابت خالی *__restrict__ vx,
                                        ثابت اعشاری *__restrict__ yy,
                                        اعشاری *__restrict__ dst,
                                        ثابت صحیح ncols, صحیح nrows,
                                        ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;

    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    ثابت بلاک_q3_K * x = (ثابت بلاک_q3_K *)vx + ib0;

    اعشاری tmp = 0; // partial sum for thread in warp

#اگر QK_K == 256

    ثابت uint16_t kmask1 = 0x0303;
    ثابت uint16_t kmask2 = 0x0f0f;

    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0,1

    ثابت صحیح n  = K_QUANTS_فی_ITERATION;               // iterations in the inner loop
    ثابت صحیح قدم = 16/K_QUANTS_فی_ITERATION;
    ثابت صحیح اندر = tid % قدم;                           // 0...15 or 0...7

    ثابت صحیح l0 = n*اندر;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ثابت int8_t * s = (ثابت int8_t *)utmp;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {

        ثابت uint8_t * h = x[i].hmask + l0;
        ثابت اعشاری d = x[i].d;

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف =  32*im + l0;
            ثابت صحیح y_انحراف = 128*im + l0;
            ثابت uint16_t s_سرکاؤ = 4*im;
            ثابت uint8_t m = 1 << (4*im);

            ثابت اعشاری   * y  = yy + i * QK_K + y_انحراف;
            ثابت uint8_t * q = x[i].qs + q_انحراف;

            ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_سرکاؤ) & kmask2) | (((a[4] >> (s_سرکاؤ + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_سرکاؤ) & kmask2) | (((a[5] >> (s_سرکاؤ + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_سرکاؤ) & kmask2) | (((a[4] >> (s_سرکاؤ + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_سرکاؤ) & kmask2) | (((a[5] >> (s_سرکاؤ + 2)) & kmask1) << 4);

            اعشاری مجموع = 0;
            برائے (صحیح l = 0; l < n; ++l) {
                مجموع += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                مجموع += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * مجموع;
        }

    }
#ورنہ

    ثابت صحیح tid = شے_ct1.get_local_id(2)/(2*K_QUANTS_فی_ITERATION);  // 0...15 or 0...7
    ثابت صحیح ix  = شے_ct1.get_local_id(2)%(2*K_QUANTS_فی_ITERATION);  // 0....1 or 0...3
    ثابت صحیح انحراف = tid * K_QUANTS_فی_ITERATION;         // 0...15 or 0...14
    ثابت صحیح اندر = انحراف/8;                                 // 0 or 1
    ثابت صحیح im = انحراف%8;                                 // 0...7

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {

        ثابت اعشاری   * y = yy + i * QK_K + انحراف;
        ثابت uint8_t * q = x[i].qs + انحراف;
        ثابت uint8_t * s = x[i].scales;

        ثابت اعشاری dall = (اعشاری)x[i].d;

        اعشاری مجموع = 0;
        برائے (صحیح l = 0; l < K_QUANTS_فی_ITERATION; ++l) {
            ثابت uint8_t hl = x[i].hmask[im+l] >> اندر;
            ثابت uint8_t ql = q[l];
            مجموع += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += مجموع;
    }
#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q3_k_reorder(ثابت خالی *__restrict__ vx,
                                                ثابت اعشاری *__restrict__ yy,
                                                اعشاری *__restrict__ dst,
                                                ثابت صحیح ncols, صحیح nrows,
                                                ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;

    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    ثابت صحیح nb = nrows * عدد_بلاکیں_فی_صف;
    ثابت uint8_t   * qs_بنیاد     = (ثابت uint8_t *)vx;
    ثابت uint8_t   * hmask_بنیاد  = qs_بنیاد + (size_t)nb * (QK_K / 4);
    ثابت uint8_t   * scales_بنیاد = hmask_بنیاد + (size_t)nb * (QK_K / 8);
    ثابت sycl::half * d_بنیاد     = (ثابت sycl::half *)(scales_بنیاد + (size_t)nb * 12);

    اعشاری tmp = 0; // partial sum for thread in warp

#اگر QK_K == 256

    ثابت uint16_t kmask1 = 0x0303;
    ثابت uint16_t kmask2 = 0x0f0f;

    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0,1

    ثابت صحیح n  = K_QUANTS_فی_ITERATION;               // iterations in the inner loop
    ثابت صحیح قدم = 16/K_QUANTS_فی_ITERATION;
    ثابت صحیح اندر = tid % قدم;                           // 0...15 or 0...7

    ثابت صحیح l0 = n*اندر;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ثابت int8_t * s = (ثابت int8_t *)utmp;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {
        ثابت صحیح bi = ib0 + i;

        ثابت uint8_t * h  = hmask_بنیاد + bi * (QK_K / 8) + l0;

        ثابت اعشاری d = d_بنیاد[bi];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف =  32*im + l0;
            ثابت صحیح y_انحراف = 128*im + l0;
            ثابت uint16_t s_سرکاؤ = 4*im;
            ثابت uint8_t m = 1 << (4*im);

            ثابت اعشاری   * y  = yy + i * QK_K + y_انحراف;
            ثابت uint8_t * q  = qs_بنیاد + bi * (QK_K / 4) + q_انحراف;

            ثابت uint16_t * a = (ثابت uint16_t *)(scales_بنیاد + bi * 12);
            utmp[0] = ((a[0] >> s_سرکاؤ) & kmask2) | (((a[4] >> (s_سرکاؤ + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_سرکاؤ) & kmask2) | (((a[5] >> (s_سرکاؤ + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_سرکاؤ) & kmask2) | (((a[4] >> (s_سرکاؤ + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_سرکاؤ) & kmask2) | (((a[5] >> (s_سرکاؤ + 2)) & kmask1) << 4);

            اعشاری مجموع = 0;
            برائے (صحیح l = 0; l < n; ++l) {
                مجموع += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                مجموع += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * مجموع;
        }
    }
#ورنہ
    GGML_غیر_مستعمل(vx);
    GGML_غیر_مستعمل(yy);
    GGML_غیر_مستعمل(ncols);
    GGML_غیر_مستعمل(شے_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q4_k(ثابت خالی *__restrict__ vx,
                                        ثابت اعشاری *__restrict__ yy,
                                        اعشاری *__restrict__ dst,
                                        ثابت صحیح ncols, صحیح nrows,
                                        ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;
    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    ثابت بلاک_q4_K * x = (ثابت بلاک_q4_K *)vx + ib0;

#اگر QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0,1

    ثابت صحیح قدم = 8/K_QUANTS_فی_ITERATION;           // 8 or 4

    ثابت صحیح il_بنیاد = tid/قدم;                        // 0 or 1 (was 0...3)
    ثابت صحیح ir  = tid - قدم*il_بنیاد;                  // 0...7 or 0...3
    ثابت صحیح n   = 2 * K_QUANTS_فی_ITERATION;          // 2 or 4

    ثابت صحیح اندر = il_بنیاد%2;

    ثابت صحیح l0 = n*(2*ir + اندر);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

#اگر K_QUANTS_فی_ITERATION == 2
    uint32_t q32[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q32;
#ورنہ
    uint16_t q16[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;
#اگر_ختم

    اعشاری tmp = 0; // partial sum for thread in warp

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {

        ثابت اعشاری dall = x[i].dm[0];
        ثابت اعشاری dmin = x[i].dm[1];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف = 32*im + l0;
            ثابت صحیح y_انحراف = 64*im + l0;

            ثابت اعشاری   * y1 = yy + i*QK_K + y_انحراف;
            ثابت اعشاری   * y2 = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#اگر K_QUANTS_فی_ITERATION == 2
            ثابت uint32_t * q1 = (ثابت uint32_t *)(x[i].qs + q_انحراف);
            ثابت uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            اعشاری smin = 0;
            برائے (صحیح l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#ورنہ
            ثابت uint16_t * q1 = (ثابت uint16_t *)(x[i].qs + q_انحراف);
            ثابت uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            اعشاری smin = 0;
            برائے (صحیح l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#اگر_ختم
        }

    }
#ورنہ
    ثابت صحیح tid = شے_ct1.get_local_id(2)/(2*K_QUANTS_فی_ITERATION);  // 0...15
    ثابت صحیح ix  = شے_ct1.get_local_id(2)%(2*K_QUANTS_فی_ITERATION);

    ثابت صحیح قدم = tid * K_QUANTS_فی_ITERATION;

    uint16_t aux16[2];
    ثابت uint8_t * s = (ثابت uint8_t *)aux16;

    اعشاری tmp = 0;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {
        ثابت uint8_t * q = x[i].qs + قدم;
        ثابت اعشاری   * y = yy + i*QK_K + قدم;
        ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ثابت اعشاری d = (اعشاری)x[i].dm[0];
        ثابت اعشاری m = (اعشاری)x[i].dm[1];
        اعشاری مجموع = 0.f;
        برائے (صحیح j = 0; j < K_QUANTS_فی_ITERATION; ++j) {
            مجموع += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += مجموع;
    }

#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (tid == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q4_k_reorder(ثابت خالی *__restrict__ vx,
                                                ثابت اعشاری *__restrict__ yy,
                                                اعشاری *__restrict__ dst,
                                                ثابت صحیح ncols, صحیح nrows,
                                                ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;
    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ثابت صحیح nb = nrows * عدد_بلاکیں_فی_صف;
    ثابت uint8_t     * qs_بنیاد     = (ثابت uint8_t *)vx;
    ثابت uint8_t     * scales_بنیاد = qs_بنیاد + (size_t)nb * (QK_K / 2);
    ثابت sycl::half2 * dm_بنیاد     = (ثابت sycl::half2 *)(scales_بنیاد + (size_t)nb * K_SCALE_سائز);

#اگر QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0,1

    ثابت صحیح قدم = 8/K_QUANTS_فی_ITERATION;           // 8 or 4

    ثابت صحیح il_بنیاد = tid/قدم;                        // 0 or 1 (was 0...3)
    ثابت صحیح ir  = tid - قدم*il_بنیاد;                  // 0...7 or 0...3
    ثابت صحیح n   = 2 * K_QUANTS_فی_ITERATION;          // 2 or 4

    ثابت صحیح اندر = il_بنیاد%2;

    ثابت صحیح l0 = n*(2*ir + اندر);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

#اگر K_QUANTS_فی_ITERATION == 2
    uint32_t q32[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q32;
#ورنہ
    uint16_t q16[4];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;
#اگر_ختم

    اعشاری tmp = 0; // partial sum for thread in warp

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {
        ثابت صحیح bi = ib0 + i;

        ثابت sycl::half2 dm_val = dm_بنیاد[bi];
        ثابت اعشاری dall = dm_val[0];
        ثابت اعشاری dmin = dm_val[1];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف = 32*im + l0;
            ثابت صحیح y_انحراف = 64*im + l0;

            ثابت اعشاری   * y1 = yy + i*QK_K + y_انحراف;
            ثابت اعشاری   * y2 = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)(scales_بنیاد + bi * K_SCALE_سائز);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#اگر K_QUANTS_فی_ITERATION == 2
            ثابت uint32_t * q1 = (ثابت uint32_t *)(qs_بنیاد + bi * (QK_K / 2) + q_انحراف);
            ثابت uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            اعشاری smin = 0;
            برائے (صحیح l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#ورنہ
            ثابت uint16_t * q1 = (ثابت uint16_t *)(qs_بنیاد + bi * (QK_K / 2) + q_انحراف);
            ثابت uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            اعشاری smin = 0;
            برائے (صحیح l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#اگر_ختم
        }

    }
#ورنہ
    ثابت صحیح tid = شے_ct1.get_local_id(2)/(2*K_QUANTS_فی_ITERATION);  // 0...15
    ثابت صحیح ix  = شے_ct1.get_local_id(2)%(2*K_QUANTS_فی_ITERATION);

    ثابت صحیح قدم = tid * K_QUANTS_فی_ITERATION;

    uint16_t aux16[2];
    ثابت uint8_t * s = (ثابت uint8_t *)aux16;

    اعشاری tmp = 0;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {
        ثابت صحیح bi = ib0 + i;

        ثابت uint8_t * q = qs_بنیاد + bi * (QK_K / 2) + قدم;
        ثابت اعشاری   * y = yy + i*QK_K + قدم;
        ثابت uint16_t * a = (ثابت uint16_t *)(scales_بنیاد + bi * K_SCALE_سائز);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ثابت sycl::half2 dm_val = dm_بنیاد[bi];
        ثابت اعشاری d = (اعشاری)dm_val[0];
        ثابت اعشاری m = (اعشاری)dm_val[1];
        اعشاری مجموع = 0.f;
        برائے (صحیح j = 0; j < K_QUANTS_فی_ITERATION; ++j) {
            مجموع += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += مجموع;
    }

#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (tid == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q5_k(ثابت خالی *__restrict__ vx,
                                        ثابت اعشاری *__restrict__ yy,
                                        اعشاری *__restrict__ dst,
                                        ثابت صحیح ncols, صحیح nrows,
                                        ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;
    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    ثابت بلاک_q5_K * x = (ثابت بلاک_q5_K *)vx + ib0;

    اعشاری tmp = 0; // partial sum for thread in warp

#اگر QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحیح tid = شے_ct1.get_local_id(2) / 2; // 0...7
    ثابت صحیح ix = شے_ct1.get_local_id(2) % 2;

    ثابت صحیح il_بنیاد = tid/4;     // 0 or 1 (was 0...3)
    ثابت صحیح ir  = tid - 4*il_بنیاد;// 0...3
    ثابت صحیح n   = 2;

    ثابت صحیح اندر = il_بنیاد%2;

    ثابت صحیح l0 = n*(2*ir + اندر);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

    uint16_t q16[8];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2) {

        ثابت uint8_t * qh  = x[i].qh + l0;
        ثابت اعشاری dall = x[i].dm[0];
        ثابت اعشاری dmin = x[i].dm[1];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف = 32*im + l0;
            ثابت صحیح y_انحراف = 64*im + l0;

            ثابت uint8_t hm1  = 1 << (2*im);
            ثابت uint8_t hm2  = hm1 << 4;

            ثابت uint8_t * ql1 = x[i].qs + q_انحراف;
            ثابت اعشاری   * y1  = yy + i*QK_K + y_انحراف;
            ثابت اعشاری   * y2  = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 مجموع = {0.f, 0.f, 0.f, 0.f};
            اعشاری smin = 0;
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
            برائے (صحیح l = 0; l < n; ++l) {
                مجموع.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                مجموع.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                مجموع.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                مجموع.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (مجموع.x() * sc[0] + مجموع.y() * sc[1] + مجموع.z() * sc[4] +
                           مجموع.w() * sc[5]) -
                   dmin * smin;
        }
    }

#ورنہ
    ثابت صحیح tid = شے_ct1.get_local_id(2)/(2*K_QUANTS_فی_ITERATION);  // 0...15
    ثابت صحیح ix  = شے_ct1.get_local_id(2)%(2*K_QUANTS_فی_ITERATION);
    ثابت صحیح قدم = tid * K_QUANTS_فی_ITERATION;
    ثابت صحیح im = قدم/8;
    ثابت صحیح اندر = قدم%8;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {
        ثابت uint8_t * q = x[i].qs + قدم;
        ثابت int8_t  * s = x[i].scales;
        ثابت اعشاری   * y = yy + i*QK_K + قدم;
        ثابت اعشاری     d = x[i].d;
        اعشاری مجموع = 0.f;
        برائے (صحیح j = 0; j < K_QUANTS_فی_ITERATION; ++j) {
            ثابت uint8_t h = x[i].qh[اندر+j] >> im;
            مجموع += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += مجموع;
    }
#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q5_k_reorder(ثابت خالی *__restrict__ vx,
                                                ثابت اعشاری *__restrict__ yy,
                                                اعشاری *__restrict__ dst,
                                                ثابت صحیح ncols, صحیح nrows,
                                                ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;
    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ثابت صحیح nb = nrows * عدد_بلاکیں_فی_صف;
    ثابت uint8_t     * qs_بنیاد     = (ثابت uint8_t *)vx;
    ثابت uint8_t     * qh_بنیاد     = qs_بنیاد + (size_t)nb * (QK_K / 2);
    ثابت uint8_t     * scales_بنیاد = qh_بنیاد + (size_t)nb * (QK_K / 8);
    ثابت sycl::half2 * dm_بنیاد     = (ثابت sycl::half2 *)(scales_بنیاد + (size_t)nb * K_SCALE_سائز);

    اعشاری tmp = 0; // partial sum for thread in warp

#اگر QK_K == 256
    ثابت uint16_t kmask1 = 0x3f3f;
    ثابت uint16_t kmask2 = 0x0f0f;
    ثابت uint16_t kmask3 = 0xc0c0;

    ثابت صحیح tid = شے_ct1.get_local_id(2) / 2; // 0...15
    ثابت صحیح ix = شے_ct1.get_local_id(2) % 2;

    ثابت صحیح il_بنیاد = tid/4;      // 0...3
    ثابت صحیح ir  = tid - 4*il_بنیاد;// 0...3
    ثابت صحیح n   = 2;

    ثابت صحیح اندر = il_بنیاد%2;

    ثابت صحیح l0 = n*(2*ir + اندر);

    uint16_t aux[4];
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;

    uint16_t q16[8];
    ثابت uint8_t * q4 = (ثابت uint8_t *)q16;

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2) {
        ثابت صحیح bi = ib0 + i;

        ثابت uint8_t * qh  = qh_بنیاد + bi * (QK_K / 8) + l0;
        ثابت sycl::half2 dm_val = dm_بنیاد[bi];
        ثابت اعشاری dall = dm_val[0];
        ثابت اعشاری dmin = dm_val[1];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح q_انحراف = 32*im + l0;
            ثابت صحیح y_انحراف = 64*im + l0;

            ثابت uint8_t hm1  = 1 << (2*im);
            ثابت uint8_t hm2  = hm1 << 4;

            ثابت uint8_t * ql1 = qs_بنیاد + bi * (QK_K / 2) + q_انحراف;
            ثابت اعشاری   * y1  = yy + i*QK_K + y_انحراف;
            ثابت اعشاری   * y2  = y1 + 128;

            ثابت uint16_t * a = (ثابت uint16_t *)(scales_بنیاد + bi * K_SCALE_سائز);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 مجموع = {0.f, 0.f, 0.f, 0.f};
            اعشاری smin = 0;
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
            برائے (صحیح l = 0; l < n; ++l) {
                مجموع.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                مجموع.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                مجموع.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                مجموع.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (مجموع.x() * sc[0] + مجموع.y() * sc[1] + مجموع.z() * sc[4] +
                           مجموع.w() * sc[5]) -
                   dmin * smin;
        }
    }
#ورنہ
    // The reordered Q5_K layout is only produced for QK_K == 256.
#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q6_k(ثابت خالی * __restrict__ vx, ثابت اعشاری * __restrict__ yy, اعشاری * __restrict__ dst, ثابت صحیح ncols, صحیح nrows,
                                        ثابت sycl::nd_شے<3> &شے_ct1) {

    static_assert(16%K_QUANTS_فی_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;

    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    ثابت بلاک_q6_K * x = (ثابت بلاک_q6_K *)vx + ib0;

#اگر QK_K == 256

    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0, 1

    ثابت صحیح قدم = 16/K_QUANTS_فی_ITERATION;          // 16 or 8

    ثابت صحیح اندر = tid % قدم;                           // 0...15 or 0...7

#اگر K_QUANTS_فی_ITERATION == 1
    ثابت صحیح l0 = K_QUANTS_فی_ITERATION*اندر;            // 0...15
    ثابت صحیح is = 0;
#ورنہ
    ثابت صحیح l0 = 4 * اندر;                               // 0, 4, 8, ..., 28
    ثابت صحیح is = اندر / 4;
#اگر_ختم

    اعشاری tmp = 0; // partial sum for thread in warp

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {

        ثابت اعشاری d = x[i].d;

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح ql_انحراف = 64*im + l0;
            ثابت صحیح qh_انحراف = 32*im + l0;
            ثابت صحیح s_انحراف  =  8*im + is;
            ثابت صحیح y_انحراف = 128*im + l0;

            ثابت اعشاری   * y  = yy + i * QK_K + y_انحراف;
            ثابت uint8_t * ql = x[i].ql + ql_انحراف;
            ثابت uint8_t * qh = x[i].qh + qh_انحراف;
            ثابت int8_t  * s  = x[i].scales + s_انحراف;

#اگر K_QUANTS_فی_ITERATION == 1
            اعشاری مجموع = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += مجموع;
#ورنہ
            اعشاری مجموع = 0;
            برائے (صحیح l = 0; l < 4; ++l) {
                مجموع += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += مجموع;
#اگر_ختم
        }

    }

#ورنہ

    ثابت صحیح tid = شے_ct1.get_local_id(2)/(2*K_QUANTS_فی_ITERATION);  // 0...7
    ثابت صحیح ix  = شے_ct1.get_local_id(2)%(2*K_QUANTS_فی_ITERATION);  // 0...3

    ثابت صحیح قدم = tid * K_QUANTS_فی_ITERATION;

    اعشاری tmp = 0; // partial sum for thread in warp

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {

        ثابت اعشاری   * y  = yy + i * QK_K + قدم;
        ثابت uint8_t * ql = x[i].ql + قدم;
        ثابت uint8_t * qh = x[i].qh + قدم;
        ثابت int8_t  * s  = x[i].scales;

        ثابت اعشاری d = x[i+0].d;

        اعشاری مجموع = 0;
        برائے (صحیح j = 0; j < K_QUANTS_فی_ITERATION; ++j) {
            مجموع += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += مجموع;

    }

#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (tid == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q6_k_reorder(ثابت خالی * __restrict__ vx, ثابت اعشاری * __restrict__ yy, اعشاری * __restrict__ dst, ثابت صحیح ncols, صحیح nrows,
                                                ثابت sycl::nd_شے<3> &شے_ct1) {

    static_assert(16%K_QUANTS_فی_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
    اگر (صف >= nrows) واپس;

    ثابت صحیح عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت صحیح ib0 = صف*عدد_بلاکیں_فی_صف;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    ثابت صحیح nb = nrows * عدد_بلاکیں_فی_صف;
    ثابت uint8_t   * ql_بنیاد     = (ثابت uint8_t *)vx;
    ثابت uint8_t   * qh_بنیاد     = ql_بنیاد + (size_t)nb * (QK_K / 2);
    ثابت int8_t    * scales_بنیاد = (ثابت int8_t *)(qh_بنیاد + (size_t)nb * (QK_K / 4));
    ثابت sycl::half * d_بنیاد     = (ثابت sycl::half *)((ثابت uint8_t *)scales_بنیاد + (size_t)nb * (QK_K / 16));

#اگر QK_K == 256

    ثابت صحیح tid =
        شے_ct1.get_local_id(2) / K_QUANTS_فی_ITERATION; // 0...7 or 0...15
    ثابت صحیح ix =
        شے_ct1.get_local_id(2) % K_QUANTS_فی_ITERATION; // 0 or 0, 1

    ثابت صحیح قدم = 16/K_QUANTS_فی_ITERATION;          // 16 or 8

    ثابت صحیح اندر = tid % قدم;                           // 0...15 or 0...7

#اگر K_QUANTS_فی_ITERATION == 1
    ثابت صحیح l0 = K_QUANTS_فی_ITERATION*اندر;            // 0...15
    ثابت صحیح is = 0;
#ورنہ
    ثابت صحیح l0 = 4 * اندر;                               // 0, 4, 8, ..., 28
    ثابت صحیح is = اندر / 4;
#اگر_ختم

    اعشاری tmp = 0; // partial sum for thread in warp

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += K_QUANTS_فی_ITERATION) {
        ثابت صحیح bi = ib0 + i;

        ثابت اعشاری d = d_بنیاد[bi];

        برائے (صحیح im = 0; im < 2; ++im) {
            ثابت صحیح ql_انحراف = 64*im + l0;
            ثابت صحیح qh_انحراف = 32*im + l0;
            ثابت صحیح s_انحراف  =  8*im + is;
            ثابت صحیح y_انحراف = 128*im + l0;

            ثابت اعشاری   * y  = yy + i * QK_K + y_انحراف;
            ثابت uint8_t * ql = ql_بنیاد + bi * (QK_K / 2) + ql_انحراف;
            ثابت uint8_t * qh = qh_بنیاد + bi * (QK_K / 4) + qh_انحراف;
            ثابت int8_t  * s  = scales_بنیاد + bi * (QK_K / 16) + s_انحراف;

#اگر K_QUANTS_فی_ITERATION == 1
            اعشاری مجموع = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += مجموع;
#ورنہ
            اعشاری مجموع = 0;
            برائے (صحیح l = 0; l < 4; ++l) {
                مجموع += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += مجموع;
#اگر_ختم
        }

    }

#ورنہ

    ثابت صحیح tid = شے_ct1.get_local_id(2)/(2*K_QUANTS_فی_ITERATION);  // 0...7
    ثابت صحیح ix  = شے_ct1.get_local_id(2)%(2*K_QUANTS_فی_ITERATION);  // 0...3

    ثابت صحیح قدم = tid * K_QUANTS_فی_ITERATION;

    اعشاری tmp = 0; // partial sum for thread in warp

    برائے (صحیح i = ix; i < عدد_بلاکیں_فی_صف; i += 2*K_QUANTS_فی_ITERATION) {
        ثابت صحیح bi = ib0 + i;

        ثابت اعشاری   * y  = yy + i * QK_K + قدم;
        ثابت uint8_t * ql = ql_بنیاد + bi * (QK_K / 2) + قدم;
        ثابت uint8_t * qh = qh_بنیاد + bi * (QK_K / 4) + قدم;
        ثابت int8_t  * s  = scales_بنیاد + bi * (QK_K / 16);

        ثابت اعشاری d = d_بنیاد[bi];

        اعشاری مجموع = 0;
        برائے (صحیح j = 0; j < K_QUANTS_فی_ITERATION; ++j) {
            مجموع += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += مجموع;

    }

#اگر_ختم

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (tid == 0) {
        dst[صف] = tmp;
    }
}

ساکن خالی dequantize_mul_mat_vec_q4_0_sycl_reorder(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}


ساکن خالی dequantize_mul_mat_vec_q4_0_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q1_0_sycl_reorder(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q1_0_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q4_1_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q5_0_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q5_1_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q8_0_sycl_reorder(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                ثابت صحیح صف = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(1) +
                    شے_ct1.get_local_id(1);
                اگر (صف >= nrows) واپس;

                ثابت صحیح tid = شے_ct1.get_local_id(2);
                ثابت صحیح iter_stride = 8*2*GGML_SYCL_DMMV_X;
                ثابت صحیح vals_فی_iter = iter_stride / WARP_سائز;
                ثابت صحیح ncols_بایاں = ncols % (QK8_0*WARP_سائز);
                ثابت صحیح ncols_align = ncols - ncols_بایاں;

#اگر_تعریف_شدہ GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#ورنہ
                اعشاری tmp = 0.0f;
#اگر_ختم
                ثابت حرف *d_ptr = (ثابت حرف*)vx + ncols*nrows;  // d after all qs

                صحیح i = 0;
                برائے (i = 0; i < ncols_align; i += iter_stride) {
                    ثابت صحیح ستون = i + vals_فی_iter*tid;
                    ثابت صحیح ib = (صف*ncols + ستون)/QK8_0;
                    ثابت صحیح iqs = ستون % QK8_0;

#ہدایت unroll
                    برائے (صحیح j = 0; j < vals_فی_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ثابت خالی *)d_ptr, ib, (ثابت خالی *)vx,
                                                ib * QK8_0 + iqs + j, v);

#اگر_تعریف_شدہ GGML_SYCL_F16
                        dfloat2 t1{y[ستون + j + 0], y[ستون + j + 1]};
                        tmp += v * t1;
#ورنہ
                        tmp += v.x() * y[ستون + j + 0];
                        tmp += v.y() * y[ستون + j + 1];
#اگر_ختم
                    }
                }

                // handle remaining columns
                برائے (; i < ncols; i += iter_stride) {
                    اگر (tid >= ncols_بایاں/QK8_0) جاری;
                    ثابت صحیح ستون = i + vals_فی_iter*tid;
                    ثابت صحیح ib = (صف*ncols + ستون)/QK8_0;
                    ثابت صحیح iqs = ستون % QK8_0;

#ہدایت unroll
                    برائے (صحیح j = 0; j < vals_فی_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ثابت خالی *)d_ptr, ib, (ثابت خالی *)vx,
                                                ib * QK8_0 + iqs + j, v);

#اگر_تعریف_شدہ GGML_SYCL_F16
                        dfloat2 t1{y[ستون + j + 0], y[ستون + j + 1]};
                        tmp += v * t1;
#ورنہ
                        tmp += v.x() * y[ستون + j + 0];
                        tmp += v.y() * y[ستون + j + 1];
#اگر_ختم
                    }
                }

                // reduce
                ثابت صحیح نقاب_آغاز = ncols > GGML_SYCL_DMMV_X ? WARP_سائز >> 1 : WARP_سائز >> 2;
                برائے (صحیح نقاب = نقاب_آغاز; نقاب > 0; نقاب >>= 1) {
                    tmp += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
                }

                اگر (tid == 0) {
#اگر_تعریف_شدہ GGML_SYCL_F16
                    dst[صف] = tmp.x() + tmp.y();
#ورنہ
                    dst[صف] = tmp;
#اگر_ختم
                }
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q8_0_sycl(ثابت خالی *vx, ثابت dfloat *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ثابت صحیح بلاک_عدد_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, GGML_SYCL_MMV_Y, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, شے_ct1);
            });
    }
}

ساکن خالی dequantize_mul_mat_vec_q2_K_sycl(ثابت خالی *vx, ثابت اعشاری *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q2_K_sycl_reorder(ثابت خالی *vx, ثابت اعشاری *y,
                                                     اعشاری *dst, ثابت صحیح ncols,
                                                     ثابت صحیح nrows,
                                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q3_K_sycl(ثابت خالی *vx, ثابت اعشاری *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q3_K_sycl_reorder(ثابت خالی *vx, ثابت اعشاری *y,
                                                     اعشاری *dst, ثابت صحیح ncols,
                                                     ثابت صحیح nrows,
                                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q4_K_sycl(ثابت خالی *vx, ثابت اعشاری *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q5_K_sycl(ثابت خالی *vx, ثابت اعشاری *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q6_K_sycl(ثابت خالی *vx, ثابت اعشاری *y,
                                             اعشاری *dst, ثابت صحیح ncols,
                                             ثابت صحیح nrows,
                                             dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
استعمال ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_سائز;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
ساخت esimd_dmmv_صف_pair_policy {
    ثابت خالی * vx;
    اعشاری *      dst;

    ESIMD_INLINE ثابت خالی * w_a() ثابت { واپس vx; }
    ESIMD_INLINE ثابت خالی * w_b() ثابت { واپس vx; }

    ESIMD_INLINE صحیح  صف_a(صحیح g) ثابت { واپس g * 2; }
    ESIMD_INLINE صحیح  صف_b(صحیح g) ثابت { واپس g * 2 + 1; }
    ESIMD_INLINE منطقی has_b(صحیح g, صحیح nrows) ثابت { واپس صف_b(g) < nrows; }

    ESIMD_INLINE خالی epilogue(صحیح g, صحیح nrows, اعشاری مجموع_a, اعشاری مجموع_b) ثابت {
        dst[صف_a(g)] = مجموع_a;
        اگر (has_b(g, nrows)) {
            dst[صف_b(g)] = مجموع_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
ساخت esimd_dmmv_glu_policy {
    ثابت خالی * vgate;
    ثابت خالی * vup;
    اعشاری *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE ثابت خالی * w_a() ثابت { واپس vgate; }
    ESIMD_INLINE ثابت خالی * w_b() ثابت { واپس vup; }

    ESIMD_INLINE صحیح  صف_a(صحیح g) ثابت { واپس g; }
    ESIMD_INLINE صحیح  صف_b(صحیح g) ثابت { واپس g; }
    ESIMD_INLINE منطقی has_b(صحیح, صحیح) ثابت { واپس سچ; } // up always contributes

    ESIMD_INLINE خالی epilogue(صحیح g, صحیح, اعشاری gate, اعشاری اوپر) ثابت {
        استعمال نامخلا sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        ثابت اعشاری GELU_COEF_A    = 0.044715f;
        ثابت اعشاری SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        ثابت اعشاری NEG_LOG2E      = -1.44269504088896341f;

        ثابت اعشاری arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<اعشاری, 16> t = arg * NEG_LOG2E;
        simd<اعشاری, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * اوپر;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
سانچہ <ggml_type T, قسم_نام Policy>
ESIMD_INLINE خالی dequantize_mul_mat_vec_reorder_esimd(
        ثابت Policy & policy, ثابت اعشاری * y,
        ثابت صحیح ncols, ثابت صحیح nrows,
        sycl::local_accessor<اعشاری, 1> lmem,
        ثابت sycl::nd_شے<1> & it) {
    استعمال نامخلا sycl::ext::intel::esimd;
    استعمال traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    ثابت صحیح    عدد_بلاکیں_فی_صف = ncols / QK_K;
    ثابت size_t nb = (size_t) nrows * عدد_بلاکیں_فی_صف;
    ثابت خودکار   pa = traits::make_ptrs(policy.w_a(), nb);
    ثابت خودکار   pb = traits::make_ptrs(policy.w_b(), nb);

    ثابت صحیح  tid   = it.get_local_id(0);
    ثابت صحیح  g     = it.get_گروہ(0);
    ثابت صحیح  صف_a = policy.صف_a(g);
    ثابت صحیح  صف_b = policy.صف_b(g);
    ثابت منطقی has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<اعشاری, 32> acc_a = 0.0f;
    simd<اعشاری, 32> acc_b = 0.0f;

    برائے (صحیح ib = tid; ib < عدد_بلاکیں_فی_صف; ib += GGML_SYCL_DMMV_ESIMD_WG_سائز) {
        simd<اعشاری, 256> y_vec = بلاک_بھرو<اعشاری, 256>(y + (size_t) ib * QK_K);

        ثابت size_t bi_a = (size_t) صف_a * عدد_بلاکیں_فی_صف + ib;
        ثابت size_t bi_b = (size_t) صف_b * عدد_بلاکیں_فی_صف + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<اعشاری>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<اعشاری>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    اگر (tid == 0) {
        اعشاری مجموع_a = 0.0f;
        اعشاری مجموع_b = 0.0f;
        برائے (صحیح p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_سائز; ++p) {
            مجموع_a += lmem[p * 2 + 0];
            مجموع_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, مجموع_a, مجموع_b);
    }
}

ساکن خالی dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(ثابت خالی *vx, ثابت اعشاری *y,
                                                           اعشاری *dst, ثابت صحیح ncols,
                                                           ثابت صحیح nrows,
                                                           dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_سائز), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_صف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساکن خالی dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(ثابت خالی *vx, ثابت اعشاری *y,
                                                           اعشاری *dst, ثابت صحیح ncols,
                                                           ثابت صحیح nrows,
                                                           dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_سائز), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_صف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساکن خالی dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(ثابت خالی *vx, ثابت اعشاری *y,
                                                           اعشاری *dst, ثابت صحیح ncols,
                                                           ثابت صحیح nrows,
                                                           dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_سائز), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_صف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساکن خالی dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(ثابت خالی *vx, ثابت اعشاری *y,
                                                           اعشاری *dst, ثابت صحیح ncols,
                                                           ثابت صحیح nrows,
                                                           dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_سائز), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_صف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ساکن خالی dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(ثابت خالی *vx, ثابت اعشاری *y,
                                                           اعشاری *dst, ثابت صحیح ncols,
                                                           ثابت صحیح nrows,
                                                           dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_سائز), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_صف_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
سانچہ <صحیح NBLK>
ESIMD_INLINE خالی q8_0_mac_stripe(
        ثابت int8_t * qs_a, ثابت int8_t * qs_b,
        ثابت sycl::half * d_a, ثابت sycl::half * d_b, منطقی has_b,
        sycl::ext::intel::esimd::simd<اعشاری, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<اعشاری, 32> & acc_a,
        sycl::ext::intel::esimd::simd<اعشاری, 32> & acc_b) {
    استعمال نامخلا sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = بلاک_بھرو<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = بلاک_بھرو<sycl::half, NBLK>(d_a, element_aligned_لیبل{});
    simd<sycl::half, NBLK>  db = 0;
    اگر (has_b) {
        qb = بلاک_بھرو<int8_t, 32 * NBLK>(qs_b);
        db = بلاک_بھرو<sycl::half, NBLK>(d_b, element_aligned_لیبل{});
    }

    simd<اعشاری, NBLK> da_f = convert<اعشاری>(da);
    simd<اعشاری, NBLK> db_f = convert<اعشاری>(db);

#ہدایت unroll
    برائے (صحیح s = 0; s < NBLK; ++s) {
        simd<اعشاری, 32>  y_s  = y_vec.سانچہ select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.سانچہ select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.سانچہ select<32, 1>(s * 32);
        ثابت اعشاری sa = da_f[s];
        ثابت اعشاری sb = db_f[s];
        acc_a += y_s * (convert<اعشاری>(qa_s) * sa);
        acc_b += y_s * (convert<اعشاری>(qb_s) * sb);
    }
}

سانچہ <صحیح WG, قسم_نام Policy>
ESIMD_INLINE خالی dequantize_mul_mat_vec_q8_0_reorder_esimd(
        ثابت Policy & policy, ثابت اعشاری * y,
        ثابت صحیح ncols, ثابت صحیح nrows,
        sycl::local_accessor<اعشاری, 1> lmem,
        ثابت sycl::nd_شے<1> & it) {
    استعمال نامخلا sycl::ext::intel::esimd;

    constexpr صحیح STRIPE = 8;

    ثابت صحیح          nblk_صف = ncols / QK8_0;
    ثابت size_t       nb       = (size_t) nrows * nblk_صف;
    ثابت int8_t *     qs_a     = (ثابت int8_t *) policy.w_a();
    ثابت int8_t *     qs_b     = (ثابت int8_t *) policy.w_b();
    ثابت sycl::half * d_a      = (ثابت sycl::half *) (qs_a + nb * QK8_0);
    ثابت sycl::half * d_b      = (ثابت sycl::half *) (qs_b + nb * QK8_0);

    ثابت صحیح  tid   = it.get_local_id(0);
    ثابت صحیح  g     = it.get_گروہ(0);
    ثابت منطقی has_b = policy.has_b(g, nrows);

    ثابت size_t بنیاد_a = (size_t) policy.صف_a(g) * nblk_صف;
    ثابت size_t بنیاد_b = (size_t) policy.صف_b(g) * nblk_صف;

    simd<اعشاری, 32> acc_a = 0.0f;
    simd<اعشاری, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    صحیح ib = 0;
    برائے (; ib + WG * STRIPE <= nblk_صف; ib += WG * STRIPE) {
        ثابت صحیح b = ib + tid * STRIPE;
        simd<اعشاری, 256> y_vec = بلاک_بھرو<اعشاری, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (بنیاد_a + b) * QK8_0, qs_b + (بنیاد_b + b) * QK8_0,
                                d_a + بنیاد_a + b, d_b + بنیاد_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    برائے (صحیح b = ib + tid; b < nblk_صف; b += WG) {
        simd<اعشاری, 32> y_vec = بلاک_بھرو<اعشاری, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (بنیاد_a + b) * QK8_0, qs_b + (بنیاد_b + b) * QK8_0,
                           d_a + بنیاد_a + b, d_b + بنیاد_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<اعشاری>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<اعشاری>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    اگر (tid == 0) {
        اعشاری مجموع_a = 0.0f;
        اعشاری مجموع_b = 0.0f;
        برائے (صحیح p = 0; p < WG; ++p) {
            مجموع_a += lmem[p * 2 + 0];
            مجموع_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, مجموع_a, مجموع_b);
    }
}

سانچہ <صحیح WG, قسم_نام Policy>
ساکن خالی q8_0_esimd_launch(ثابت Policy & policy, ثابت اعشاری * y, ثابت صحیح ncols, ثابت صحیح nrows,
                              ثابت صحیح workgroups, dpct::قطار_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
سانچہ <قسم_نام Policy>
ساکن خالی q8_0_esimd_launch_scaled(ثابت Policy & policy, ثابت اعشاری * y, ثابت صحیح ncols, ثابت صحیح nrows,
                                     ثابت صحیح workgroups, dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    ثابت صحیح nblk_صف = ncols / QK8_0;
    اگر (nblk_صف >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } ورنہ اگر (nblk_صف >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } ورنہ اگر (nblk_صف >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } ورنہ {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

ساکن خالی dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(ثابت خالی *vx, ثابت اعشاری *y,
                                                           اعشاری *dst, ثابت صحیح ncols,
                                                           ثابت صحیح nrows,
                                                           dpct::قطار_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_صف_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
سانچہ <ggml_type T>
ساکن خالی dequantize_mul_mat_vec_glu_reorder_esimd(ثابت خالی * vx, ثابت خالی * vgate,
                                                     ثابت اعشاری * y, اعشاری * dst,
                                                     ثابت صحیح ncols, ثابت صحیح nrows,
                                                     ثابت ggml_glu_op glu_op,
                                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<اعشاری, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_سائز), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_سائز)),
            [=](sycl::nd_شے<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

منطقی ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(شمار ggml_type src0_type, شمار ggml_glu_op glu_op,
                                                        ثابت خالی * vx, ثابت خالی * vgate, ثابت اعشاری * y,
                                                        اعشاری * dst, صحیح ncols, صحیح nrows, dpct::قطار_ptr stream) {
    انتخاب (src0_type) {
        حالت GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            واپس سچ;
        حالت GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            واپس سچ;
        حالت GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            واپس سچ;
        حالت GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            واپس سچ;
        حالت GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            واپس سچ;
        حالت GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

#اگر_ختم // GGML_SYCL_DMMV_HAS_ESIMD

ساکن خالی dequantize_mul_mat_vec_q4_K_sycl_reorder(ثابت خالی *vx, ثابت اعشاری *y,
                                                     اعشاری *dst, ثابت صحیح ncols,
                                                     ثابت صحیح nrows,
                                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q5_K_sycl_reorder(ثابت خالی *vx, ثابت اعشاری *y,
                                                     اعشاری *dst, ثابت صحیح ncols,
                                                     ثابت صحیح nrows,
                                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

ساکن خالی dequantize_mul_mat_vec_q6_K_sycl_reorder(ثابت خالی *vx, ثابت اعشاری *y,
                                                     اعشاری *dst, ثابت صحیح ncols,
                                                     ثابت صحیح nrows,
                                                     dpct::قطار_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ثابت صحیح ny = 2 / K_QUANTS_فی_ITERATION;
    ثابت صحیح بلاک_عدد_y = (nrows + ny - 1) / ny;
    ثابت sycl::range<3> بلاک_nums(1, 1, بلاک_عدد_y);
    ثابت sycl::range<3> بلاک_dims(1, ny, WARP_سائز);
    stream->parallel_for(
        sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, شے_ct1);
        });
}

خالی ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت اعشاری *src1_ddf_i, ثابت حرف *src1_ddq_i,
    اعشاری *dst_dd_i, ثابت int64_t صف_کم, ثابت int64_t صف_بلند,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائز,
    ثابت dpct::قطار_ptr &stream) {

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t صف_diff = صف_بلند - صف_کم;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#اگر_تعریف_شدہ GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = خالی_اشاریہ; // dfloat == half

#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    ثابت منطقی q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#ورنہ
    ثابت منطقی q8_0_esimd = جھوٹ;
#اگر_ختم

    منطقی src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    اگر (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        ثابت مقصد_fp16_sycl_t مقصد_fp16_sycl = ggml_get_مقصد_fp16_sycl(src1->type, dst);
        GGML_ASSERT(مقصد_fp16_sycl != خالی_اشاریہ);
        مقصد_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#ورنہ
    ثابت dfloat * src1_dfloat = (ثابت dfloat *) src1_ddf_i; // dfloat == float, no conversion
#اگر_ختم // GGML_SYCL_F16

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q1_0:
            اگر ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            } ورنہ {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q4_0:
            اگر ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            } ورنہ {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            توڑو;
        حالت GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            توڑو;
        حالت GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            توڑو;
        حالت GGML_TYPE_Q8_0:
            اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
                اگر (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                } ورنہ
#اگر_ختم
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
                }
            } ورنہ {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q2_K:
            اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
                اگر (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
                ورنہ
#اگر_ختم
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
            } ورنہ {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q3_K:
            اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
                اگر (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
                ورنہ
#اگر_ختم
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
            } ورنہ {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q4_K:
            اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
                اگر (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
                ورنہ
#اگر_ختم
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
            } ورنہ {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q5_K:
            اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
                اگر (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
                ورنہ
#اگر_ختم
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
            } ورنہ {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_Q6_K:
            اگر ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
                اگر (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
                ورنہ
#اگر_ختم
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
                }
            } ورنہ {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, صف_diff, stream);
            }
            توڑو;
        حالت GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            توڑو;
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_BF16
        حالت GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, صف_diff, stream);
            توڑو;
#اگر_ختم
        طےشدہ:
            لکھو("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_غیر_مستعمل(src1);
    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(src1_ddq_i);
    GGML_غیر_مستعمل(src1_ncols);
    GGML_غیر_مستعمل(src1_padded_صف_سائز);
    GGML_غیر_مستعمل(ctx);
}
