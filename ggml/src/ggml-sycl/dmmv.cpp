#అన్తర్భావ "convert.hpp"
#అన్తర్భావ "dmmv.hpp"
#అన్తర్భావ "dequantize.hpp"
#అన్తర్భావ "presets.hpp"

#ఒకవేళ defined(__INTEL_LLVM_COMPILER)
    #ఒకవేళ __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #అన్తర్భావ <sycl/ext/oneapi/bfloat16.hpp>
        #నిర్వచించు GGML_SYCL_DMMV_HAS_BF16
    #అయితే_ముగింపు
    #అన్తర్భావ <sycl/ext/intel/esimd.hpp>
    #అన్తర్భావ "esimd.hpp"
#అయితే_ముగింపు

స్థిర శూన్య convert_f16(స్థిరరాశి శూన్య * vx, స్థిరరాశి int64_t ib, స్థిరరాశి పూర్ణాంకం iqs, dfloat2 & v){
    స్థిరరాశి sycl::half *x = (స్థిరరాశి sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_BF16
స్థిర శూన్య convert_bf16(స్థిరరాశి శూన్య * vx, స్థిరరాశి int64_t ib, స్థిరరాశి పూర్ణాంకం iqs, dfloat2 & v){
    స్థిరరాశి sycl::ext::oneapi::bfloat16 *x = (స్థిరరాశి sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#అయితే_ముగింపు

స్థిర శూన్య convert_f32(స్థిరరాశి శూన్య * vx, స్థిరరాశి int64_t ib, స్థిరరాశి పూర్ణాంకం iqs, dfloat2 & v){
    స్థిరరాశి దశాంశం * x = (స్థిరరాశి దశాంశం *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dequantize_kernel>
స్థిర శూన్య dequantize_mul_mat_vec(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి dfloat * __restrict__ y, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                   స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);

    స్థిరరాశి పూర్ణాంకం iter_stride = 2*GGML_SYCL_DMMV_X;
    స్థిరరాశి పూర్ణాంకం vals_చొప్పున_iter = iter_stride / WARP_కొలత; // num quantized vals per thread and i iter
    స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#నిర్వచితమైతే GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#లేకపోతే
    దశాంశం tmp = 0.0f;
#అయితే_ముగింపు // GGML_SYCL_F16

    కొరకు (పూర్ణాంకం i = 0; i < ncols; i += iter_stride) {
        స్థిరరాశి పూర్ణాంకం నిలువు = i + vals_చొప్పున_iter*tid;
        ఒకవేళ (నిలువు >= ncols) {
            కొనసాగు;
        }
        స్థిరరాశి పూర్ణాంకం ib = (అడ్డువరుస*ncols + నిలువు)/qk; // x block index
        స్థిరరాశి పూర్ణాంకం iqs = (నిలువు%qk)/qr; // x quant index
        స్థిరరాశి పూర్ణాంకం iybs = నిలువు - నిలువు%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < vals_చొప్పున_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#నిర్వచితమైతే GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_స్థానభ్రంశం]};

            tmp += v * t1;
#లేకపోతే
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_స్థానభ్రంశం];
#అయితే_ముగింపు // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    స్థిరరాశి పూర్ణాంకం ముసుగు_ప్రారంభం = ncols > GGML_SYCL_DMMV_X ? WARP_కొలత >> 1 : WARP_కొలత >> 2;
    కొరకు (పూర్ణాంకం ముసుగు = ముసుగు_ప్రారంభం; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (tid == 0) {
#నిర్వచితమైతే GGML_SYCL_F16
        dst[అడ్డువరుస] = tmp.x() + tmp.y();
#లేకపోతే
        dst[అడ్డువరుస] = tmp;
#అయితే_ముగింపు // GGML_SYCL_F16
    }
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
స్థిర శూన్య dequantize_mul_mat_vec_reorder(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి dfloat * __restrict__ y, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                   స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);

    ఒకవేళ (అడ్డువరుస >= nrows) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);


    స్థిరరాశి పూర్ణాంకం ncols_ఎడమ = ncols % (QK4_0*WARP_కొలత);
    స్థిరరాశి పూర్ణాంకం ncols_align = ncols - ncols_ఎడమ;
    స్థిరరాశి పూర్ణాంకం iter_stride = 8*2*GGML_SYCL_DMMV_X;
    స్థిరరాశి పూర్ణాంకం vals_చొప్పున_iter = iter_stride / WARP_కొలత; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#నిర్వచితమైతే GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#లేకపోతే
    దశాంశం tmp = 0.0f;
#అయితే_ముగింపు // GGML_SYCL_F16
    స్థిరరాశి అక్షరం *d_ptr = (స్థిరరాశి అక్షరం*)vx+ncols*nrows/2;
    పూర్ణాంకం i=0;
    కొరకు (i = 0; i < ncols_align; i += iter_stride) {
        స్థిరరాశి పూర్ణాంకం నిలువు = i + vals_చొప్పున_iter*tid;
        స్థిరరాశి పూర్ణాంకం ib = (అడ్డువరుస*ncols + నిలువు)/qk; // x block index
        స్థిరరాశి పూర్ణాంకం iqs = (నిలువు%qk)/qr; // x quant index
        స్థిరరాశి పూర్ణాంకం iybs = నిలువు - నిలువు%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < vals_చొప్పున_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((స్థిరరాశి శూన్య *)d_ptr, ib, (స్థిరరాశి శూన్య *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#నిర్వచితమైతే GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_స్థానభ్రంశం]};

            tmp += v * t1;
#లేకపోతే
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_స్థానభ్రంశం];
#అయితే_ముగింపు // GGML_SYCL_F16
        }
    }

    కొరకు (; i < ncols; i += iter_stride) {
        ఒకవేళ (tid>=ncols_ఎడమ/QK4_0) కొనసాగు;
        స్థిరరాశి పూర్ణాంకం నిలువు = i + vals_చొప్పున_iter*tid;
        స్థిరరాశి పూర్ణాంకం ib = (అడ్డువరుస*ncols + నిలువు)/qk; // x block index
        స్థిరరాశి పూర్ణాంకం iqs = (నిలువు%qk)/qr; // x quant index
        స్థిరరాశి పూర్ణాంకం iybs = నిలువు - నిలువు%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < vals_చొప్పున_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((స్థిరరాశి శూన్య *)d_ptr, ib, (స్థిరరాశి శూన్య *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#నిర్వచితమైతే GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_స్థానభ్రంశం]};

            tmp += v * t1;
#లేకపోతే
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_స్థానభ్రంశం];
#అయితే_ముగింపు // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    స్థిరరాశి పూర్ణాంకం ముసుగు_ప్రారంభం = ncols > GGML_SYCL_DMMV_X ? WARP_కొలత >> 1 : WARP_కొలత >> 2;
    కొరకు (పూర్ణాంకం ముసుగు = ముసుగు_ప్రారంభం; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (tid == 0) {
#నిర్వచితమైతే GGML_SYCL_F16
        dst[అడ్డువరుస] = tmp.x() + tmp.y();
#లేకపోతే
        dst[అడ్డువరుస] = tmp;
#అయితే_ముగింపు // GGML_SYCL_F16
    }
}

స్థిర శూన్య convert_mul_mat_vec_f16_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                         దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                         స్థిరరాశి పూర్ణాంకం nrows,
                                         dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, అంశం_ct1);
            });
    }
}

#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_BF16
స్థిర శూన్య convert_mul_mat_vec_bf16_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                          దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                          స్థిరరాశి పూర్ణాంకం nrows,
                                          dpct::వరుస_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, అంశం_ct1);
            });
    }
}
#అయితే_ముగింపు

స్థిర శూన్య dequantize_mul_mat_vec_q2_k(స్థిరరాశి శూన్య *__restrict__ vx,
                                        స్థిరరాశి దశాంశం *__restrict__ yy,
                                        దశాంశం *__restrict__ dst,
                                        స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    static_assert(16%K_QUANTS_చొప్పున_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    స్థిరరాశి ఖండం_q2_K * x = (స్థిరరాశి ఖండం_q2_K *)vx + ib0;

    దశాంశం tmp = 0; // partial sum for thread in warp

#ఒకవేళ QK_K == 256
    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0,1

    స్థిరరాశి పూర్ణాంకం అడుగు = 16/K_QUANTS_చొప్పున_ITERATION;

    స్థిరరాశి పూర్ణాంకం లోపల = tid % అడుగు;                           // 0...15 or 0...7

    స్థిరరాశి పూర్ణాంకం l0 = K_QUANTS_చొప్పున_ITERATION*లోపల;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    స్థిరరాశి uint8_t * d = (స్థిరరాశి uint8_t *)aux;
    స్థిరరాశి uint8_t * m = (స్థిరరాశి uint8_t *)(aux + 2);

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి దశాంశం dall = x[i].dm[0];
        స్థిరరాశి దశాంశం dmin = x[i].dm[1];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం s_స్థానభ్రంశం = 8*im;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 128*im + l0;

            స్థిరరాశి దశాంశం   * y = yy + i * QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * q = x[i].qs + q_స్థానభ్రంశం;

            స్థిరరాశి uint32_t * a = (స్థిరరాశి uint32_t *)(x[i].scales + s_స్థానభ్రంశం);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            దశాంశం sum1 = 0, sum2 = 0;
            కొరకు (పూర్ణాంకం l = 0; l < K_QUANTS_చొప్పున_ITERATION; ++l) {
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
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2) /
                    (2 * K_QUANTS_చొప్పున_ITERATION); // 0...15 or 0...7
    స్థిరరాశి పూర్ణాంకం ix = అంశం_ct1.get_local_id(2) %
                   (2 * K_QUANTS_చొప్పున_ITERATION); // 0....1 or 0...3
    స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = tid * K_QUANTS_చొప్పున_ITERATION;

    uint32_t uaux[2];
    స్థిరరాశి uint8_t * d = (స్థిరరాశి uint8_t *)uaux;


    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి దశాంశం   * y = yy + i * QK_K + స్థానభ్రంశం;
        స్థిరరాశి uint8_t * q = x[i].qs + స్థానభ్రంశం;
        స్థిరరాశి uint32_t * s = (స్థిరరాశి uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        స్థిరరాశి sycl::float2 dall =
            x[i].dm.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

        దశాంశం sum1 = 0, sum2 = 0;
        కొరకు (పూర్ణాంకం l = 0; l < K_QUANTS_చొప్పున_ITERATION; ++l) {
            స్థిరరాశి uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q2_k_reorder(స్థిరరాశి శూన్య *__restrict__ vx,
                                                స్థిరరాశి దశాంశం *__restrict__ yy,
                                                దశాంశం *__restrict__ dst,
                                                స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                                స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    static_assert(16%K_QUANTS_చొప్పున_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    స్థిరరాశి పూర్ణాంకం nb = nrows * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి uint8_t     * qs_పునాది     = (స్థిరరాశి uint8_t *)vx;
    స్థిరరాశి uint8_t     * scales_పునాది = qs_పునాది + (size_t)nb * (QK_K / 4);
    స్థిరరాశి sycl::half2 * dm_పునాది     = (స్థిరరాశి sycl::half2 *)(scales_పునాది + (size_t)nb * (QK_K / 16));

    దశాంశం tmp = 0; // partial sum for thread in warp

#ఒకవేళ QK_K == 256
    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0,1

    స్థిరరాశి పూర్ణాంకం అడుగు = 16/K_QUANTS_చొప్పున_ITERATION;

    స్థిరరాశి పూర్ణాంకం లోపల = tid % అడుగు;                           // 0...15 or 0...7

    స్థిరరాశి పూర్ణాంకం l0 = K_QUANTS_చొప్పున_ITERATION*లోపల;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    స్థిరరాశి uint8_t * d = (స్థిరరాశి uint8_t *)aux;
    స్థిరరాశి uint8_t * m = (స్థిరరాశి uint8_t *)(aux + 2);

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి sycl::half2 dm_val = dm_పునాది[bi];
        స్థిరరాశి దశాంశం dall = dm_val[0];
        స్థిరరాశి దశాంశం dmin = dm_val[1];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం s_స్థానభ్రంశం = 8*im;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 128*im + l0;

            స్థిరరాశి దశాంశం   * y = yy + i * QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * q = qs_పునాది + bi * (QK_K / 4) + q_స్థానభ్రంశం;

            స్థిరరాశి uint32_t * a = (స్థిరరాశి uint32_t *)(scales_పునాది + bi * (QK_K / 16) + s_స్థానభ్రంశం);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            దశాంశం sum1 = 0, sum2 = 0;
            కొరకు (పూర్ణాంకం l = 0; l < K_QUANTS_చొప్పున_ITERATION; ++l) {
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
#లేకపోతే
    GGML_వాడని(vx);
    GGML_వాడని(yy);
    GGML_వాడని(ncols);
    GGML_వాడని(అంశం_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q3_k(స్థిరరాశి శూన్య *__restrict__ vx,
                                        స్థిరరాశి దశాంశం *__restrict__ yy,
                                        దశాంశం *__restrict__ dst,
                                        స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    స్థిరరాశి ఖండం_q3_K * x = (స్థిరరాశి ఖండం_q3_K *)vx + ib0;

    దశాంశం tmp = 0; // partial sum for thread in warp

#ఒకవేళ QK_K == 256

    స్థిరరాశి uint16_t kmask1 = 0x0303;
    స్థిరరాశి uint16_t kmask2 = 0x0f0f;

    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0,1

    స్థిరరాశి పూర్ణాంకం n  = K_QUANTS_చొప్పున_ITERATION;               // iterations in the inner loop
    స్థిరరాశి పూర్ణాంకం అడుగు = 16/K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం లోపల = tid % అడుగు;                           // 0...15 or 0...7

    స్థిరరాశి పూర్ణాంకం l0 = n*లోపల;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    స్థిరరాశి int8_t * s = (స్థిరరాశి int8_t *)utmp;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి uint8_t * h = x[i].hmask + l0;
        స్థిరరాశి దశాంశం d = x[i].d;

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం =  32*im + l0;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 128*im + l0;
            స్థిరరాశి uint16_t s_జరుపు = 4*im;
            స్థిరరాశి uint8_t m = 1 << (4*im);

            స్థిరరాశి దశాంశం   * y  = yy + i * QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * q = x[i].qs + q_స్థానభ్రంశం;

            స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_జరుపు) & kmask2) | (((a[4] >> (s_జరుపు + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_జరుపు) & kmask2) | (((a[5] >> (s_జరుపు + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_జరుపు) & kmask2) | (((a[4] >> (s_జరుపు + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_జరుపు) & kmask2) | (((a[5] >> (s_జరుపు + 2)) & kmask1) << 4);

            దశాంశం కూడిక = 0;
            కొరకు (పూర్ణాంకం l = 0; l < n; ++l) {
                కూడిక += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                కూడిక += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * కూడిక;
        }

    }
#లేకపోతే

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2)/(2*K_QUANTS_చొప్పున_ITERATION);  // 0...15 or 0...7
    స్థిరరాశి పూర్ణాంకం ix  = అంశం_ct1.get_local_id(2)%(2*K_QUANTS_చొప్పున_ITERATION);  // 0....1 or 0...3
    స్థిరరాశి పూర్ణాంకం స్థానభ్రంశం = tid * K_QUANTS_చొప్పున_ITERATION;         // 0...15 or 0...14
    స్థిరరాశి పూర్ణాంకం లోపల = స్థానభ్రంశం/8;                                 // 0 or 1
    స్థిరరాశి పూర్ణాంకం im = స్థానభ్రంశం%8;                                 // 0...7

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి దశాంశం   * y = yy + i * QK_K + స్థానభ్రంశం;
        స్థిరరాశి uint8_t * q = x[i].qs + స్థానభ్రంశం;
        స్థిరరాశి uint8_t * s = x[i].scales;

        స్థిరరాశి దశాంశం dall = (దశాంశం)x[i].d;

        దశాంశం కూడిక = 0;
        కొరకు (పూర్ణాంకం l = 0; l < K_QUANTS_చొప్పున_ITERATION; ++l) {
            స్థిరరాశి uint8_t hl = x[i].hmask[im+l] >> లోపల;
            స్థిరరాశి uint8_t ql = q[l];
            కూడిక += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += కూడిక;
    }
#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q3_k_reorder(స్థిరరాశి శూన్య *__restrict__ vx,
                                                స్థిరరాశి దశాంశం *__restrict__ yy,
                                                దశాంశం *__restrict__ dst,
                                                స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                                స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    స్థిరరాశి పూర్ణాంకం nb = nrows * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి uint8_t   * qs_పునాది     = (స్థిరరాశి uint8_t *)vx;
    స్థిరరాశి uint8_t   * hmask_పునాది  = qs_పునాది + (size_t)nb * (QK_K / 4);
    స్థిరరాశి uint8_t   * scales_పునాది = hmask_పునాది + (size_t)nb * (QK_K / 8);
    స్థిరరాశి sycl::half * d_పునాది     = (స్థిరరాశి sycl::half *)(scales_పునాది + (size_t)nb * 12);

    దశాంశం tmp = 0; // partial sum for thread in warp

#ఒకవేళ QK_K == 256

    స్థిరరాశి uint16_t kmask1 = 0x0303;
    స్థిరరాశి uint16_t kmask2 = 0x0f0f;

    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0,1

    స్థిరరాశి పూర్ణాంకం n  = K_QUANTS_చొప్పున_ITERATION;               // iterations in the inner loop
    స్థిరరాశి పూర్ణాంకం అడుగు = 16/K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం లోపల = tid % అడుగు;                           // 0...15 or 0...7

    స్థిరరాశి పూర్ణాంకం l0 = n*లోపల;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    స్థిరరాశి int8_t * s = (స్థిరరాశి int8_t *)utmp;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి uint8_t * h  = hmask_పునాది + bi * (QK_K / 8) + l0;

        స్థిరరాశి దశాంశం d = d_పునాది[bi];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం =  32*im + l0;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 128*im + l0;
            స్థిరరాశి uint16_t s_జరుపు = 4*im;
            స్థిరరాశి uint8_t m = 1 << (4*im);

            స్థిరరాశి దశాంశం   * y  = yy + i * QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * q  = qs_పునాది + bi * (QK_K / 4) + q_స్థానభ్రంశం;

            స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)(scales_పునాది + bi * 12);
            utmp[0] = ((a[0] >> s_జరుపు) & kmask2) | (((a[4] >> (s_జరుపు + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_జరుపు) & kmask2) | (((a[5] >> (s_జరుపు + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_జరుపు) & kmask2) | (((a[4] >> (s_జరుపు + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_జరుపు) & kmask2) | (((a[5] >> (s_జరుపు + 2)) & kmask1) << 4);

            దశాంశం కూడిక = 0;
            కొరకు (పూర్ణాంకం l = 0; l < n; ++l) {
                కూడిక += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                కూడిక += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * కూడిక;
        }
    }
#లేకపోతే
    GGML_వాడని(vx);
    GGML_వాడని(yy);
    GGML_వాడని(ncols);
    GGML_వాడని(అంశం_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q4_k(స్థిరరాశి శూన్య *__restrict__ vx,
                                        స్థిరరాశి దశాంశం *__restrict__ yy,
                                        దశాంశం *__restrict__ dst,
                                        స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    స్థిరరాశి ఖండం_q4_K * x = (స్థిరరాశి ఖండం_q4_K *)vx + ib0;

#ఒకవేళ QK_K == 256
    స్థిరరాశి uint16_t kmask1 = 0x3f3f;
    స్థిరరాశి uint16_t kmask2 = 0x0f0f;
    స్థిరరాశి uint16_t kmask3 = 0xc0c0;

    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0,1

    స్థిరరాశి పూర్ణాంకం అడుగు = 8/K_QUANTS_చొప్పున_ITERATION;           // 8 or 4

    స్థిరరాశి పూర్ణాంకం il_పునాది = tid/అడుగు;                        // 0 or 1 (was 0...3)
    స్థిరరాశి పూర్ణాంకం ir  = tid - అడుగు*il_పునాది;                  // 0...7 or 0...3
    స్థిరరాశి పూర్ణాంకం n   = 2 * K_QUANTS_చొప్పున_ITERATION;          // 2 or 4

    స్థిరరాశి పూర్ణాంకం లోపల = il_పునాది%2;

    స్థిరరాశి పూర్ణాంకం l0 = n*(2*ir + లోపల);

    uint16_t aux[4];
    స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *)aux;

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 2
    uint32_t q32[4];
    స్థిరరాశి uint8_t * q4 = (స్థిరరాశి uint8_t *)q32;
#లేకపోతే
    uint16_t q16[4];
    స్థిరరాశి uint8_t * q4 = (స్థిరరాశి uint8_t *)q16;
#అయితే_ముగింపు

    దశాంశం tmp = 0; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి దశాంశం dall = x[i].dm[0];
        స్థిరరాశి దశాంశం dmin = x[i].dm[1];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 64*im + l0;

            స్థిరరాశి దశాంశం   * y1 = yy + i*QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం   * y2 = y1 + 128;

            స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 2
            స్థిరరాశి uint32_t * q1 = (స్థిరరాశి uint32_t *)(x[i].qs + q_స్థానభ్రంశం);
            స్థిరరాశి uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            దశాంశం smin = 0;
            కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#లేకపోతే
            స్థిరరాశి uint16_t * q1 = (స్థిరరాశి uint16_t *)(x[i].qs + q_స్థానభ్రంశం);
            స్థిరరాశి uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            దశాంశం smin = 0;
            కొరకు (పూర్ణాంకం l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#అయితే_ముగింపు
        }

    }
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2)/(2*K_QUANTS_చొప్పున_ITERATION);  // 0...15
    స్థిరరాశి పూర్ణాంకం ix  = అంశం_ct1.get_local_id(2)%(2*K_QUANTS_చొప్పున_ITERATION);

    స్థిరరాశి పూర్ణాంకం అడుగు = tid * K_QUANTS_చొప్పున_ITERATION;

    uint16_t aux16[2];
    స్థిరరాశి uint8_t * s = (స్థిరరాశి uint8_t *)aux16;

    దశాంశం tmp = 0;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి uint8_t * q = x[i].qs + అడుగు;
        స్థిరరాశి దశాంశం   * y = yy + i*QK_K + అడుగు;
        స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        స్థిరరాశి దశాంశం d = (దశాంశం)x[i].dm[0];
        స్థిరరాశి దశాంశం m = (దశాంశం)x[i].dm[1];
        దశాంశం కూడిక = 0.f;
        కొరకు (పూర్ణాంకం j = 0; j < K_QUANTS_చొప్పున_ITERATION; ++j) {
            కూడిక += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += కూడిక;
    }

#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (tid == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q4_k_reorder(స్థిరరాశి శూన్య *__restrict__ vx,
                                                స్థిరరాశి దశాంశం *__restrict__ yy,
                                                దశాంశం *__restrict__ dst,
                                                స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                                స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    స్థిరరాశి పూర్ణాంకం nb = nrows * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి uint8_t     * qs_పునాది     = (స్థిరరాశి uint8_t *)vx;
    స్థిరరాశి uint8_t     * scales_పునాది = qs_పునాది + (size_t)nb * (QK_K / 2);
    స్థిరరాశి sycl::half2 * dm_పునాది     = (స్థిరరాశి sycl::half2 *)(scales_పునాది + (size_t)nb * K_SCALE_కొలత);

#ఒకవేళ QK_K == 256
    స్థిరరాశి uint16_t kmask1 = 0x3f3f;
    స్థిరరాశి uint16_t kmask2 = 0x0f0f;
    స్థిరరాశి uint16_t kmask3 = 0xc0c0;

    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0,1

    స్థిరరాశి పూర్ణాంకం అడుగు = 8/K_QUANTS_చొప్పున_ITERATION;           // 8 or 4

    స్థిరరాశి పూర్ణాంకం il_పునాది = tid/అడుగు;                        // 0 or 1 (was 0...3)
    స్థిరరాశి పూర్ణాంకం ir  = tid - అడుగు*il_పునాది;                  // 0...7 or 0...3
    స్థిరరాశి పూర్ణాంకం n   = 2 * K_QUANTS_చొప్పున_ITERATION;          // 2 or 4

    స్థిరరాశి పూర్ణాంకం లోపల = il_పునాది%2;

    స్థిరరాశి పూర్ణాంకం l0 = n*(2*ir + లోపల);

    uint16_t aux[4];
    స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *)aux;

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 2
    uint32_t q32[4];
    స్థిరరాశి uint8_t * q4 = (స్థిరరాశి uint8_t *)q32;
#లేకపోతే
    uint16_t q16[4];
    స్థిరరాశి uint8_t * q4 = (స్థిరరాశి uint8_t *)q16;
#అయితే_ముగింపు

    దశాంశం tmp = 0; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి sycl::half2 dm_val = dm_పునాది[bi];
        స్థిరరాశి దశాంశం dall = dm_val[0];
        స్థిరరాశి దశాంశం dmin = dm_val[1];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 64*im + l0;

            స్థిరరాశి దశాంశం   * y1 = yy + i*QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం   * y2 = y1 + 128;

            స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)(scales_పునాది + bi * K_SCALE_కొలత);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 2
            స్థిరరాశి uint32_t * q1 = (స్థిరరాశి uint32_t *)(qs_పునాది + bi * (QK_K / 2) + q_స్థానభ్రంశం);
            స్థిరరాశి uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            దశాంశం smin = 0;
            కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#లేకపోతే
            స్థిరరాశి uint16_t * q1 = (స్థిరరాశి uint16_t *)(qs_పునాది + bi * (QK_K / 2) + q_స్థానభ్రంశం);
            స్థిరరాశి uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            దశాంశం smin = 0;
            కొరకు (పూర్ణాంకం l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#అయితే_ముగింపు
        }

    }
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2)/(2*K_QUANTS_చొప్పున_ITERATION);  // 0...15
    స్థిరరాశి పూర్ణాంకం ix  = అంశం_ct1.get_local_id(2)%(2*K_QUANTS_చొప్పున_ITERATION);

    స్థిరరాశి పూర్ణాంకం అడుగు = tid * K_QUANTS_చొప్పున_ITERATION;

    uint16_t aux16[2];
    స్థిరరాశి uint8_t * s = (స్థిరరాశి uint8_t *)aux16;

    దశాంశం tmp = 0;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి uint8_t * q = qs_పునాది + bi * (QK_K / 2) + అడుగు;
        స్థిరరాశి దశాంశం   * y = yy + i*QK_K + అడుగు;
        స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)(scales_పునాది + bi * K_SCALE_కొలత);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        స్థిరరాశి sycl::half2 dm_val = dm_పునాది[bi];
        స్థిరరాశి దశాంశం d = (దశాంశం)dm_val[0];
        స్థిరరాశి దశాంశం m = (దశాంశం)dm_val[1];
        దశాంశం కూడిక = 0.f;
        కొరకు (పూర్ణాంకం j = 0; j < K_QUANTS_చొప్పున_ITERATION; ++j) {
            కూడిక += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += కూడిక;
    }

#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (tid == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_k(స్థిరరాశి శూన్య *__restrict__ vx,
                                        స్థిరరాశి దశాంశం *__restrict__ yy,
                                        దశాంశం *__restrict__ dst,
                                        స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    స్థిరరాశి ఖండం_q5_K * x = (స్థిరరాశి ఖండం_q5_K *)vx + ib0;

    దశాంశం tmp = 0; // partial sum for thread in warp

#ఒకవేళ QK_K == 256
    స్థిరరాశి uint16_t kmask1 = 0x3f3f;
    స్థిరరాశి uint16_t kmask2 = 0x0f0f;
    స్థిరరాశి uint16_t kmask3 = 0xc0c0;

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2) / 2; // 0...7
    స్థిరరాశి పూర్ణాంకం ix = అంశం_ct1.get_local_id(2) % 2;

    స్థిరరాశి పూర్ణాంకం il_పునాది = tid/4;     // 0 or 1 (was 0...3)
    స్థిరరాశి పూర్ణాంకం ir  = tid - 4*il_పునాది;// 0...3
    స్థిరరాశి పూర్ణాంకం n   = 2;

    స్థిరరాశి పూర్ణాంకం లోపల = il_పునాది%2;

    స్థిరరాశి పూర్ణాంకం l0 = n*(2*ir + లోపల);

    uint16_t aux[4];
    స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *)aux;

    uint16_t q16[8];
    స్థిరరాశి uint8_t * q4 = (స్థిరరాశి uint8_t *)q16;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2) {

        స్థిరరాశి uint8_t * qh  = x[i].qh + l0;
        స్థిరరాశి దశాంశం dall = x[i].dm[0];
        స్థిరరాశి దశాంశం dmin = x[i].dm[1];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 64*im + l0;

            స్థిరరాశి uint8_t hm1  = 1 << (2*im);
            స్థిరరాశి uint8_t hm2  = hm1 << 4;

            స్థిరరాశి uint8_t * ql1 = x[i].qs + q_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం   * y1  = yy + i*QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం   * y2  = y1 + 128;

            స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 కూడిక = {0.f, 0.f, 0.f, 0.f};
            దశాంశం smin = 0;
            స్థిరరాశి uint16_t * q1 = (స్థిరరాశి uint16_t *)ql1;
            స్థిరరాశి uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            కొరకు (పూర్ణాంకం l = 0; l < n; ++l) {
                కూడిక.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                కూడిక.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                కూడిక.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                కూడిక.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (కూడిక.x() * sc[0] + కూడిక.y() * sc[1] + కూడిక.z() * sc[4] +
                           కూడిక.w() * sc[5]) -
                   dmin * smin;
        }
    }

#లేకపోతే
    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2)/(2*K_QUANTS_చొప్పున_ITERATION);  // 0...15
    స్థిరరాశి పూర్ణాంకం ix  = అంశం_ct1.get_local_id(2)%(2*K_QUANTS_చొప్పున_ITERATION);
    స్థిరరాశి పూర్ణాంకం అడుగు = tid * K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం im = అడుగు/8;
    స్థిరరాశి పూర్ణాంకం లోపల = అడుగు%8;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి uint8_t * q = x[i].qs + అడుగు;
        స్థిరరాశి int8_t  * s = x[i].scales;
        స్థిరరాశి దశాంశం   * y = yy + i*QK_K + అడుగు;
        స్థిరరాశి దశాంశం     d = x[i].d;
        దశాంశం కూడిక = 0.f;
        కొరకు (పూర్ణాంకం j = 0; j < K_QUANTS_చొప్పున_ITERATION; ++j) {
            స్థిరరాశి uint8_t h = x[i].qh[లోపల+j] >> im;
            కూడిక += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += కూడిక;
    }
#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_k_reorder(స్థిరరాశి శూన్య *__restrict__ vx,
                                                స్థిరరాశి దశాంశం *__restrict__ yy,
                                                దశాంశం *__restrict__ dst,
                                                స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                                స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    స్థిరరాశి పూర్ణాంకం nb = nrows * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి uint8_t     * qs_పునాది     = (స్థిరరాశి uint8_t *)vx;
    స్థిరరాశి uint8_t     * qh_పునాది     = qs_పునాది + (size_t)nb * (QK_K / 2);
    స్థిరరాశి uint8_t     * scales_పునాది = qh_పునాది + (size_t)nb * (QK_K / 8);
    స్థిరరాశి sycl::half2 * dm_పునాది     = (స్థిరరాశి sycl::half2 *)(scales_పునాది + (size_t)nb * K_SCALE_కొలత);

    దశాంశం tmp = 0; // partial sum for thread in warp

#ఒకవేళ QK_K == 256
    స్థిరరాశి uint16_t kmask1 = 0x3f3f;
    స్థిరరాశి uint16_t kmask2 = 0x0f0f;
    స్థిరరాశి uint16_t kmask3 = 0xc0c0;

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2) / 2; // 0...15
    స్థిరరాశి పూర్ణాంకం ix = అంశం_ct1.get_local_id(2) % 2;

    స్థిరరాశి పూర్ణాంకం il_పునాది = tid/4;      // 0...3
    స్థిరరాశి పూర్ణాంకం ir  = tid - 4*il_పునాది;// 0...3
    స్థిరరాశి పూర్ణాంకం n   = 2;

    స్థిరరాశి పూర్ణాంకం లోపల = il_పునాది%2;

    స్థిరరాశి పూర్ణాంకం l0 = n*(2*ir + లోపల);

    uint16_t aux[4];
    స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *)aux;

    uint16_t q16[8];
    స్థిరరాశి uint8_t * q4 = (స్థిరరాశి uint8_t *)q16;

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి uint8_t * qh  = qh_పునాది + bi * (QK_K / 8) + l0;
        స్థిరరాశి sycl::half2 dm_val = dm_పునాది[bi];
        స్థిరరాశి దశాంశం dall = dm_val[0];
        స్థిరరాశి దశాంశం dmin = dm_val[1];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 64*im + l0;

            స్థిరరాశి uint8_t hm1  = 1 << (2*im);
            స్థిరరాశి uint8_t hm2  = hm1 << 4;

            స్థిరరాశి uint8_t * ql1 = qs_పునాది + bi * (QK_K / 2) + q_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం   * y1  = yy + i*QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం   * y2  = y1 + 128;

            స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)(scales_పునాది + bi * K_SCALE_కొలత);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 కూడిక = {0.f, 0.f, 0.f, 0.f};
            దశాంశం smin = 0;
            స్థిరరాశి uint16_t * q1 = (స్థిరరాశి uint16_t *)ql1;
            స్థిరరాశి uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            కొరకు (పూర్ణాంకం l = 0; l < n; ++l) {
                కూడిక.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                కూడిక.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                కూడిక.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                కూడిక.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (కూడిక.x() * sc[0] + కూడిక.y() * sc[1] + కూడిక.z() * sc[4] +
                           కూడిక.w() * sc[5]) -
                   dmin * smin;
        }
    }
#లేకపోతే
    // The reordered Q5_K layout is only produced for QK_K == 256.
#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q6_k(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి దశాంశం * __restrict__ yy, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    static_assert(16%K_QUANTS_చొప్పున_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    స్థిరరాశి ఖండం_q6_K * x = (స్థిరరాశి ఖండం_q6_K *)vx + ib0;

#ఒకవేళ QK_K == 256

    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0, 1

    స్థిరరాశి పూర్ణాంకం అడుగు = 16/K_QUANTS_చొప్పున_ITERATION;          // 16 or 8

    స్థిరరాశి పూర్ణాంకం లోపల = tid % అడుగు;                           // 0...15 or 0...7

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 1
    స్థిరరాశి పూర్ణాంకం l0 = K_QUANTS_చొప్పున_ITERATION*లోపల;            // 0...15
    స్థిరరాశి పూర్ణాంకం is = 0;
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం l0 = 4 * లోపల;                               // 0, 4, 8, ..., 28
    స్థిరరాశి పూర్ణాంకం is = లోపల / 4;
#అయితే_ముగింపు

    దశాంశం tmp = 0; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి దశాంశం d = x[i].d;

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం ql_స్థానభ్రంశం = 64*im + l0;
            స్థిరరాశి పూర్ణాంకం qh_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం s_స్థానభ్రంశం  =  8*im + is;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 128*im + l0;

            స్థిరరాశి దశాంశం   * y  = yy + i * QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * ql = x[i].ql + ql_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * qh = x[i].qh + qh_స్థానభ్రంశం;
            స్థిరరాశి int8_t  * s  = x[i].scales + s_స్థానభ్రంశం;

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 1
            దశాంశం కూడిక = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += కూడిక;
#లేకపోతే
            దశాంశం కూడిక = 0;
            కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
                కూడిక += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += కూడిక;
#అయితే_ముగింపు
        }

    }

#లేకపోతే

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2)/(2*K_QUANTS_చొప్పున_ITERATION);  // 0...7
    స్థిరరాశి పూర్ణాంకం ix  = అంశం_ct1.get_local_id(2)%(2*K_QUANTS_చొప్పున_ITERATION);  // 0...3

    స్థిరరాశి పూర్ణాంకం అడుగు = tid * K_QUANTS_చొప్పున_ITERATION;

    దశాంశం tmp = 0; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {

        స్థిరరాశి దశాంశం   * y  = yy + i * QK_K + అడుగు;
        స్థిరరాశి uint8_t * ql = x[i].ql + అడుగు;
        స్థిరరాశి uint8_t * qh = x[i].qh + అడుగు;
        స్థిరరాశి int8_t  * s  = x[i].scales;

        స్థిరరాశి దశాంశం d = x[i+0].d;

        దశాంశం కూడిక = 0;
        కొరకు (పూర్ణాంకం j = 0; j < K_QUANTS_చొప్పున_ITERATION; ++j) {
            కూడిక += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += కూడిక;

    }

#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (tid == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q6_k_reorder(స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి దశాంశం * __restrict__ yy, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం nrows,
                                                స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    static_assert(16%K_QUANTS_చొప్పున_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
    ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి పూర్ణాంకం ib0 = అడ్డువరుస*అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    స్థిరరాశి పూర్ణాంకం nb = nrows * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి uint8_t   * ql_పునాది     = (స్థిరరాశి uint8_t *)vx;
    స్థిరరాశి uint8_t   * qh_పునాది     = ql_పునాది + (size_t)nb * (QK_K / 2);
    స్థిరరాశి int8_t    * scales_పునాది = (స్థిరరాశి int8_t *)(qh_పునాది + (size_t)nb * (QK_K / 4));
    స్థిరరాశి sycl::half * d_పునాది     = (స్థిరరాశి sycl::half *)((స్థిరరాశి uint8_t *)scales_పునాది + (size_t)nb * (QK_K / 16));

#ఒకవేళ QK_K == 256

    స్థిరరాశి పూర్ణాంకం tid =
        అంశం_ct1.get_local_id(2) / K_QUANTS_చొప్పున_ITERATION; // 0...7 or 0...15
    స్థిరరాశి పూర్ణాంకం ix =
        అంశం_ct1.get_local_id(2) % K_QUANTS_చొప్పున_ITERATION; // 0 or 0, 1

    స్థిరరాశి పూర్ణాంకం అడుగు = 16/K_QUANTS_చొప్పున_ITERATION;          // 16 or 8

    స్థిరరాశి పూర్ణాంకం లోపల = tid % అడుగు;                           // 0...15 or 0...7

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 1
    స్థిరరాశి పూర్ణాంకం l0 = K_QUANTS_చొప్పున_ITERATION*లోపల;            // 0...15
    స్థిరరాశి పూర్ణాంకం is = 0;
#లేకపోతే
    స్థిరరాశి పూర్ణాంకం l0 = 4 * లోపల;                               // 0, 4, 8, ..., 28
    స్థిరరాశి పూర్ణాంకం is = లోపల / 4;
#అయితే_ముగింపు

    దశాంశం tmp = 0; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి దశాంశం d = d_పునాది[bi];

        కొరకు (పూర్ణాంకం im = 0; im < 2; ++im) {
            స్థిరరాశి పూర్ణాంకం ql_స్థానభ్రంశం = 64*im + l0;
            స్థిరరాశి పూర్ణాంకం qh_స్థానభ్రంశం = 32*im + l0;
            స్థిరరాశి పూర్ణాంకం s_స్థానభ్రంశం  =  8*im + is;
            స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = 128*im + l0;

            స్థిరరాశి దశాంశం   * y  = yy + i * QK_K + y_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * ql = ql_పునాది + bi * (QK_K / 2) + ql_స్థానభ్రంశం;
            స్థిరరాశి uint8_t * qh = qh_పునాది + bi * (QK_K / 4) + qh_స్థానభ్రంశం;
            స్థిరరాశి int8_t  * s  = scales_పునాది + bi * (QK_K / 16) + s_స్థానభ్రంశం;

#ఒకవేళ K_QUANTS_చొప్పున_ITERATION == 1
            దశాంశం కూడిక = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += కూడిక;
#లేకపోతే
            దశాంశం కూడిక = 0;
            కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
                కూడిక += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += కూడిక;
#అయితే_ముగింపు
        }

    }

#లేకపోతే

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2)/(2*K_QUANTS_చొప్పున_ITERATION);  // 0...7
    స్థిరరాశి పూర్ణాంకం ix  = అంశం_ct1.get_local_id(2)%(2*K_QUANTS_చొప్పున_ITERATION);  // 0...3

    స్థిరరాశి పూర్ణాంకం అడుగు = tid * K_QUANTS_చొప్పున_ITERATION;

    దశాంశం tmp = 0; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం i = ix; i < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; i += 2*K_QUANTS_చొప్పున_ITERATION) {
        స్థిరరాశి పూర్ణాంకం bi = ib0 + i;

        స్థిరరాశి దశాంశం   * y  = yy + i * QK_K + అడుగు;
        స్థిరరాశి uint8_t * ql = ql_పునాది + bi * (QK_K / 2) + అడుగు;
        స్థిరరాశి uint8_t * qh = qh_పునాది + bi * (QK_K / 4) + అడుగు;
        స్థిరరాశి int8_t  * s  = scales_పునాది + bi * (QK_K / 16);

        స్థిరరాశి దశాంశం d = d_పునాది[bi];

        దశాంశం కూడిక = 0;
        కొరకు (పూర్ణాంకం j = 0; j < K_QUANTS_చొప్పున_ITERATION; ++j) {
            కూడిక += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += కూడిక;

    }

#అయితే_ముగింపు

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (tid == 0) {
        dst[అడ్డువరుస] = tmp;
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q4_0_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}


స్థిర శూన్య dequantize_mul_mat_vec_q4_0_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q1_0_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q1_0_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q4_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_0_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_1_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q8_0_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(1) +
                    అంశం_ct1.get_local_id(1);
                ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

                స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
                స్థిరరాశి పూర్ణాంకం iter_stride = 8*2*GGML_SYCL_DMMV_X;
                స్థిరరాశి పూర్ణాంకం vals_చొప్పున_iter = iter_stride / WARP_కొలత;
                స్థిరరాశి పూర్ణాంకం ncols_ఎడమ = ncols % (QK8_0*WARP_కొలత);
                స్థిరరాశి పూర్ణాంకం ncols_align = ncols - ncols_ఎడమ;

#నిర్వచితమైతే GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#లేకపోతే
                దశాంశం tmp = 0.0f;
#అయితే_ముగింపు
                స్థిరరాశి అక్షరం *d_ptr = (స్థిరరాశి అక్షరం*)vx + ncols*nrows;  // d after all qs

                పూర్ణాంకం i = 0;
                కొరకు (i = 0; i < ncols_align; i += iter_stride) {
                    స్థిరరాశి పూర్ణాంకం నిలువు = i + vals_చొప్పున_iter*tid;
                    స్థిరరాశి పూర్ణాంకం ib = (అడ్డువరుస*ncols + నిలువు)/QK8_0;
                    స్థిరరాశి పూర్ణాంకం iqs = నిలువు % QK8_0;

#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < vals_చొప్పున_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((స్థిరరాశి శూన్య *)d_ptr, ib, (స్థిరరాశి శూన్య *)vx,
                                                ib * QK8_0 + iqs + j, v);

#నిర్వచితమైతే GGML_SYCL_F16
                        dfloat2 t1{y[నిలువు + j + 0], y[నిలువు + j + 1]};
                        tmp += v * t1;
#లేకపోతే
                        tmp += v.x() * y[నిలువు + j + 0];
                        tmp += v.y() * y[నిలువు + j + 1];
#అయితే_ముగింపు
                    }
                }

                // handle remaining columns
                కొరకు (; i < ncols; i += iter_stride) {
                    ఒకవేళ (tid >= ncols_ఎడమ/QK8_0) కొనసాగు;
                    స్థిరరాశి పూర్ణాంకం నిలువు = i + vals_చొప్పున_iter*tid;
                    స్థిరరాశి పూర్ణాంకం ib = (అడ్డువరుస*ncols + నిలువు)/QK8_0;
                    స్థిరరాశి పూర్ణాంకం iqs = నిలువు % QK8_0;

#సూచన unroll
                    కొరకు (పూర్ణాంకం j = 0; j < vals_చొప్పున_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((స్థిరరాశి శూన్య *)d_ptr, ib, (స్థిరరాశి శూన్య *)vx,
                                                ib * QK8_0 + iqs + j, v);

#నిర్వచితమైతే GGML_SYCL_F16
                        dfloat2 t1{y[నిలువు + j + 0], y[నిలువు + j + 1]};
                        tmp += v * t1;
#లేకపోతే
                        tmp += v.x() * y[నిలువు + j + 0];
                        tmp += v.y() * y[నిలువు + j + 1];
#అయితే_ముగింపు
                    }
                }

                // reduce
                స్థిరరాశి పూర్ణాంకం ముసుగు_ప్రారంభం = ncols > GGML_SYCL_DMMV_X ? WARP_కొలత >> 1 : WARP_కొలత >> 2;
                కొరకు (పూర్ణాంకం ముసుగు = ముసుగు_ప్రారంభం; ముసుగు > 0; ముసుగు >>= 1) {
                    tmp += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
                }

                ఒకవేళ (tid == 0) {
#నిర్వచితమైతే GGML_SYCL_F16
                    dst[అడ్డువరుస] = tmp.x() + tmp.y();
#లేకపోతే
                    dst[అడ్డువరుస] = tmp;
#అయితే_ముగింపు
                }
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q8_0_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి dfloat *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, GGML_SYCL_MMV_Y, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, అంశం_ct1);
            });
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q2_K_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q2_K_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                     దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                     స్థిరరాశి పూర్ణాంకం nrows,
                                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q3_K_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q3_K_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                     దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                     స్థిరరాశి పూర్ణాంకం nrows,
                                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q4_K_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_K_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q6_K_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                             దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                             స్థిరరాశి పూర్ణాంకం nrows,
                                             dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
ఉపయోగించి ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_కొలత;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
నిర్మాణం esimd_dmmv_అడ్డువరుస_pair_policy {
    స్థిరరాశి శూన్య * vx;
    దశాంశం *      dst;

    ESIMD_INLINE స్థిరరాశి శూన్య * w_a() స్థిరరాశి { తిరిగి vx; }
    ESIMD_INLINE స్థిరరాశి శూన్య * w_b() స్థిరరాశి { తిరిగి vx; }

    ESIMD_INLINE పూర్ణాంకం  అడ్డువరుస_a(పూర్ణాంకం g) స్థిరరాశి { తిరిగి g * 2; }
    ESIMD_INLINE పూర్ణాంకం  అడ్డువరుస_b(పూర్ణాంకం g) స్థిరరాశి { తిరిగి g * 2 + 1; }
    ESIMD_INLINE తార్కిక has_b(పూర్ణాంకం g, పూర్ణాంకం nrows) స్థిరరాశి { తిరిగి అడ్డువరుస_b(g) < nrows; }

    ESIMD_INLINE శూన్య epilogue(పూర్ణాంకం g, పూర్ణాంకం nrows, దశాంశం కూడిక_a, దశాంశం కూడిక_b) స్థిరరాశి {
        dst[అడ్డువరుస_a(g)] = కూడిక_a;
        ఒకవేళ (has_b(g, nrows)) {
            dst[అడ్డువరుస_b(g)] = కూడిక_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
నిర్మాణం esimd_dmmv_glu_policy {
    స్థిరరాశి శూన్య * vgate;
    స్థిరరాశి శూన్య * vup;
    దశాంశం *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE స్థిరరాశి శూన్య * w_a() స్థిరరాశి { తిరిగి vgate; }
    ESIMD_INLINE స్థిరరాశి శూన్య * w_b() స్థిరరాశి { తిరిగి vup; }

    ESIMD_INLINE పూర్ణాంకం  అడ్డువరుస_a(పూర్ణాంకం g) స్థిరరాశి { తిరిగి g; }
    ESIMD_INLINE పూర్ణాంకం  అడ్డువరుస_b(పూర్ణాంకం g) స్థిరరాశి { తిరిగి g; }
    ESIMD_INLINE తార్కిక has_b(పూర్ణాంకం, పూర్ణాంకం) స్థిరరాశి { తిరిగి నిజం; } // up always contributes

    ESIMD_INLINE శూన్య epilogue(పూర్ణాంకం g, పూర్ణాంకం, దశాంశం gate, దశాంశం పైకి) స్థిరరాశి {
        ఉపయోగించి నామాకాశ sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        స్థిరరాశి దశాంశం GELU_COEF_A    = 0.044715f;
        స్థిరరాశి దశాంశం SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        స్థిరరాశి దశాంశం NEG_LOG2E      = -1.44269504088896341f;

        స్థిరరాశి దశాంశం arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<దశాంశం, 16> t = arg * NEG_LOG2E;
        simd<దశాంశం, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * పైకి;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
మూస <ggml_type T, రకం_పేరు Policy>
ESIMD_INLINE శూన్య dequantize_mul_mat_vec_reorder_esimd(
        స్థిరరాశి Policy & policy, స్థిరరాశి దశాంశం * y,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        sycl::local_accessor<దశాంశం, 1> lmem,
        స్థిరరాశి sycl::nd_అంశం<1> & it) {
    ఉపయోగించి నామాకాశ sycl::ext::intel::esimd;
    ఉపయోగించి traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    స్థిరరాశి పూర్ణాంకం    అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = ncols / QK_K;
    స్థిరరాశి size_t nb = (size_t) nrows * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
    స్థిరరాశి స్వయంచాలక   pa = traits::make_ptrs(policy.w_a(), nb);
    స్థిరరాశి స్వయంచాలక   pb = traits::make_ptrs(policy.w_b(), nb);

    స్థిరరాశి పూర్ణాంకం  tid   = it.get_local_id(0);
    స్థిరరాశి పూర్ణాంకం  g     = it.get_సమూహం(0);
    స్థిరరాశి పూర్ణాంకం  అడ్డువరుస_a = policy.అడ్డువరుస_a(g);
    స్థిరరాశి పూర్ణాంకం  అడ్డువరుస_b = policy.అడ్డువరుస_b(g);
    స్థిరరాశి తార్కిక has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<దశాంశం, 32> acc_a = 0.0f;
    simd<దశాంశం, 32> acc_b = 0.0f;

    కొరకు (పూర్ణాంకం ib = tid; ib < అంకె_ఖండాలు_చొప్పున_అడ్డువరుస; ib += GGML_SYCL_DMMV_ESIMD_WG_కొలత) {
        simd<దశాంశం, 256> y_vec = ఖండం_ఎక్కించు<దశాంశం, 256>(y + (size_t) ib * QK_K);

        స్థిరరాశి size_t bi_a = (size_t) అడ్డువరుస_a * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస + ib;
        స్థిరరాశి size_t bi_b = (size_t) అడ్డువరుస_b * అంకె_ఖండాలు_చొప్పున_అడ్డువరుస + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<దశాంశం>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<దశాంశం>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    ఒకవేళ (tid == 0) {
        దశాంశం కూడిక_a = 0.0f;
        దశాంశం కూడిక_b = 0.0f;
        కొరకు (పూర్ణాంకం p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_కొలత; ++p) {
            కూడిక_a += lmem[p * 2 + 0];
            కూడిక_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, కూడిక_a, కూడిక_b);
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                           స్థిరరాశి పూర్ణాంకం nrows,
                                                           dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_కొలత), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_అడ్డువరుస_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

స్థిర శూన్య dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                           స్థిరరాశి పూర్ణాంకం nrows,
                                                           dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_కొలత), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_అడ్డువరుస_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

స్థిర శూన్య dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                           స్థిరరాశి పూర్ణాంకం nrows,
                                                           dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_కొలత), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_అడ్డువరుస_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                           స్థిరరాశి పూర్ణాంకం nrows,
                                                           dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_కొలత), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_అడ్డువరుస_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

స్థిర శూన్య dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                           స్థిరరాశి పూర్ణాంకం nrows,
                                                           dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_కొలత), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_అడ్డువరుస_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
మూస <పూర్ణాంకం NBLK>
ESIMD_INLINE శూన్య q8_0_mac_stripe(
        స్థిరరాశి int8_t * qs_a, స్థిరరాశి int8_t * qs_b,
        స్థిరరాశి sycl::half * d_a, స్థిరరాశి sycl::half * d_b, తార్కిక has_b,
        sycl::ext::intel::esimd::simd<దశాంశం, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<దశాంశం, 32> & acc_a,
        sycl::ext::intel::esimd::simd<దశాంశం, 32> & acc_b) {
    ఉపయోగించి నామాకాశ sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = ఖండం_ఎక్కించు<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = ఖండం_ఎక్కించు<sycl::half, NBLK>(d_a, element_aligned_ట్యాగ్{});
    simd<sycl::half, NBLK>  db = 0;
    ఒకవేళ (has_b) {
        qb = ఖండం_ఎక్కించు<int8_t, 32 * NBLK>(qs_b);
        db = ఖండం_ఎక్కించు<sycl::half, NBLK>(d_b, element_aligned_ట్యాగ్{});
    }

    simd<దశాంశం, NBLK> da_f = convert<దశాంశం>(da);
    simd<దశాంశం, NBLK> db_f = convert<దశాంశం>(db);

#సూచన unroll
    కొరకు (పూర్ణాంకం s = 0; s < NBLK; ++s) {
        simd<దశాంశం, 32>  y_s  = y_vec.మూస select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.మూస select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.మూస select<32, 1>(s * 32);
        స్థిరరాశి దశాంశం sa = da_f[s];
        స్థిరరాశి దశాంశం sb = db_f[s];
        acc_a += y_s * (convert<దశాంశం>(qa_s) * sa);
        acc_b += y_s * (convert<దశాంశం>(qb_s) * sb);
    }
}

మూస <పూర్ణాంకం WG, రకం_పేరు Policy>
ESIMD_INLINE శూన్య dequantize_mul_mat_vec_q8_0_reorder_esimd(
        స్థిరరాశి Policy & policy, స్థిరరాశి దశాంశం * y,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        sycl::local_accessor<దశాంశం, 1> lmem,
        స్థిరరాశి sycl::nd_అంశం<1> & it) {
    ఉపయోగించి నామాకాశ sycl::ext::intel::esimd;

    constexpr పూర్ణాంకం STRIPE = 8;

    స్థిరరాశి పూర్ణాంకం          nblk_అడ్డువరుస = ncols / QK8_0;
    స్థిరరాశి size_t       nb       = (size_t) nrows * nblk_అడ్డువరుస;
    స్థిరరాశి int8_t *     qs_a     = (స్థిరరాశి int8_t *) policy.w_a();
    స్థిరరాశి int8_t *     qs_b     = (స్థిరరాశి int8_t *) policy.w_b();
    స్థిరరాశి sycl::half * d_a      = (స్థిరరాశి sycl::half *) (qs_a + nb * QK8_0);
    స్థిరరాశి sycl::half * d_b      = (స్థిరరాశి sycl::half *) (qs_b + nb * QK8_0);

    స్థిరరాశి పూర్ణాంకం  tid   = it.get_local_id(0);
    స్థిరరాశి పూర్ణాంకం  g     = it.get_సమూహం(0);
    స్థిరరాశి తార్కిక has_b = policy.has_b(g, nrows);

    స్థిరరాశి size_t పునాది_a = (size_t) policy.అడ్డువరుస_a(g) * nblk_అడ్డువరుస;
    స్థిరరాశి size_t పునాది_b = (size_t) policy.అడ్డువరుస_b(g) * nblk_అడ్డువరుస;

    simd<దశాంశం, 32> acc_a = 0.0f;
    simd<దశాంశం, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    పూర్ణాంకం ib = 0;
    కొరకు (; ib + WG * STRIPE <= nblk_అడ్డువరుస; ib += WG * STRIPE) {
        స్థిరరాశి పూర్ణాంకం b = ib + tid * STRIPE;
        simd<దశాంశం, 256> y_vec = ఖండం_ఎక్కించు<దశాంశం, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (పునాది_a + b) * QK8_0, qs_b + (పునాది_b + b) * QK8_0,
                                d_a + పునాది_a + b, d_b + పునాది_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    కొరకు (పూర్ణాంకం b = ib + tid; b < nblk_అడ్డువరుస; b += WG) {
        simd<దశాంశం, 32> y_vec = ఖండం_ఎక్కించు<దశాంశం, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (పునాది_a + b) * QK8_0, qs_b + (పునాది_b + b) * QK8_0,
                           d_a + పునాది_a + b, d_b + పునాది_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<దశాంశం>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<దశాంశం>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    ఒకవేళ (tid == 0) {
        దశాంశం కూడిక_a = 0.0f;
        దశాంశం కూడిక_b = 0.0f;
        కొరకు (పూర్ణాంకం p = 0; p < WG; ++p) {
            కూడిక_a += lmem[p * 2 + 0];
            కూడిక_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, కూడిక_a, కూడిక_b);
    }
}

మూస <పూర్ణాంకం WG, రకం_పేరు Policy>
స్థిర శూన్య q8_0_esimd_launch(స్థిరరాశి Policy & policy, స్థిరరాశి దశాంశం * y, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                              స్థిరరాశి పూర్ణాంకం workgroups, dpct::వరుస_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
మూస <రకం_పేరు Policy>
స్థిర శూన్య q8_0_esimd_launch_scaled(స్థిరరాశి Policy & policy, స్థిరరాశి దశాంశం * y, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                     స్థిరరాశి పూర్ణాంకం workgroups, dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    స్థిరరాశి పూర్ణాంకం nblk_అడ్డువరుస = ncols / QK8_0;
    ఒకవేళ (nblk_అడ్డువరుస >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } లేకపోతే ఒకవేళ (nblk_అడ్డువరుస >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } లేకపోతే ఒకవేళ (nblk_అడ్డువరుస >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } లేకపోతే {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

స్థిర శూన్య dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                           స్థిరరాశి పూర్ణాంకం nrows,
                                                           dpct::వరుస_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_అడ్డువరుస_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
మూస <ggml_type T>
స్థిర శూన్య dequantize_mul_mat_vec_glu_reorder_esimd(స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vgate,
                                                     స్థిరరాశి దశాంశం * y, దశాంశం * dst,
                                                     స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
                                                     స్థిరరాశి ggml_glu_op glu_op,
                                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<దశాంశం, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_కొలత), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_కొలత)),
            [=](sycl::nd_అంశం<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

తార్కిక ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(గణన ggml_type src0_type, గణన ggml_glu_op glu_op,
                                                        స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vgate, స్థిరరాశి దశాంశం * y,
                                                        దశాంశం * dst, పూర్ణాంకం ncols, పూర్ణాంకం nrows, dpct::వరుస_ptr stream) {
    ఎంపిక (src0_type) {
        స్థితి GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

#అయితే_ముగింపు // GGML_SYCL_DMMV_HAS_ESIMD

స్థిర శూన్య dequantize_mul_mat_vec_q4_K_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                     దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                     స్థిరరాశి పూర్ణాంకం nrows,
                                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q5_K_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                     దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                     స్థిరరాశి పూర్ణాంకం nrows,
                                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

స్థిర శూన్య dequantize_mul_mat_vec_q6_K_sycl_reorder(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                                     దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                                     స్థిరరాశి పూర్ణాంకం nrows,
                                                     dpct::వరుస_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం ny = 2 / K_QUANTS_చొప్పున_ITERATION;
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_y = (nrows + ny - 1) / ny;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, ఖండం_అంకె_y);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, ny, WARP_కొలత);
    stream->parallel_for(
        sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, అంశం_ct1);
        });
}

శూన్య ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
    స్థిరరాశి అక్షరం *src0_dd_i, స్థిరరాశి దశాంశం *src1_ddf_i, స్థిరరాశి అక్షరం *src1_ddq_i,
    దశాంశం *dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ, స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి dpct::వరుస_ptr &stream) {

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t అడ్డువరుస_diff = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#నిర్వచితమైతే GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = శూన్యసూచిక; // dfloat == half

#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    స్థిరరాశి తార్కిక q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#లేకపోతే
    స్థిరరాశి తార్కిక q8_0_esimd = అబద్ధం;
#అయితే_ముగింపు

    తార్కిక src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    ఒకవేళ (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        స్థిరరాశి గమ్యం_fp16_sycl_t గమ్యం_fp16_sycl = ggml_get_గమ్యం_fp16_sycl(src1->type, dst);
        GGML_ASSERT(గమ్యం_fp16_sycl != శూన్యసూచిక);
        గమ్యం_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#లేకపోతే
    స్థిరరాశి dfloat * src1_dfloat = (స్థిరరాశి dfloat *) src1_ddf_i; // dfloat == float, no conversion
#అయితే_ముగింపు // GGML_SYCL_F16

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q1_0:
            ఒకవేళ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            } లేకపోతే {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            ఒకవేళ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            } లేకపోతే {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
                ఒకవేళ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                } లేకపోతే
#అయితే_ముగింపు
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
            } లేకపోతే {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
                ఒకవేళ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
                లేకపోతే
#అయితే_ముగింపు
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
            } లేకపోతే {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
                ఒకవేళ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
                లేకపోతే
#అయితే_ముగింపు
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
            } లేకపోతే {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
                ఒకవేళ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
                లేకపోతే
#అయితే_ముగింపు
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
            } లేకపోతే {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
                ఒకవేళ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
                లేకపోతే
#అయితే_ముగింపు
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
            } లేకపోతే {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            ఒకవేళ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
                ఒకవేళ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
                లేకపోతే
#అయితే_ముగింపు
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
                }
            } లేకపోతే {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            }
            ఆపు;
        స్థితి GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            ఆపు;
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_BF16
        స్థితి GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, అడ్డువరుస_diff, stream);
            ఆపు;
#అయితే_ముగింపు
        అప్రమేయ:
            రాయి("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_వాడని(src1);
    GGML_వాడని(dst);
    GGML_వాడని(src1_ddq_i);
    GGML_వాడని(src1_ncols);
    GGML_వాడని(src1_padded_అడ్డువరుస_కొలత);
    GGML_వాడని(ctx);
}
