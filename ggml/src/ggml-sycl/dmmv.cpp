#अन्तर्भाव "convert.hpp"
#अन्तर्भाव "dmmv.hpp"
#अन्तर्भाव "dequantize.hpp"
#अन्तर्भाव "presets.hpp"

#यदि defined(__INTEL_LLVM_COMPILER)
    #यदि __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #अन्तर्भाव <sycl/ext/oneapi/bfloat16.hpp>
        #परिभाषय GGML_SYCL_DMMV_HAS_BF16
    #यदि_अन्तः
    #अन्तर्भाव <sycl/ext/intel/esimd.hpp>
    #अन्तर्भाव "esimd.hpp"
#यदि_अन्तः

स्थिर शून्य convert_f16(ध्रुव शून्य * vx, ध्रुव int64_t ib, ध्रुव पूर्णाङ्क iqs, dfloat2 & v){
    ध्रुव sycl::half *x = (ध्रुव sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_BF16
स्थिर शून्य convert_bf16(ध्रुव शून्य * vx, ध्रुव int64_t ib, ध्रुव पूर्णाङ्क iqs, dfloat2 & v){
    ध्रुव sycl::ext::oneapi::bfloat16 *x = (ध्रुव sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#यदि_अन्तः

स्थिर शून्य convert_f32(ध्रुव शून्य * vx, ध्रुव int64_t ib, ध्रुव पूर्णाङ्क iqs, dfloat2 & v){
    ध्रुव भिन्न * x = (ध्रुव भिन्न *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qr, dequantize_kernel_t dequantize_kernel>
स्थिर शून्य dequantize_mul_mat_vec(ध्रुव शून्य * __restrict__ vx, ध्रुव dfloat * __restrict__ y, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                   ध्रुव sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);

    ध्रुव पूर्णाङ्क iter_stride = 2*GGML_SYCL_DMMV_X;
    ध्रुव पूर्णाङ्क vals_प्रत्येकम्_iter = iter_stride / WARP_आकार; // num quantized vals per thread and i iter
    ध्रुव पूर्णाङ्क y_विस्थापनम् = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#यदि_परिभाषितम् GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#अन्यथा
    भिन्न tmp = 0.0f;
#यदि_अन्तः // GGML_SYCL_F16

    प्रति (पूर्णाङ्क i = 0; i < ncols; i += iter_stride) {
        ध्रुव पूर्णाङ्क col = i + vals_प्रत्येकम्_iter*tid;
        यदि (col >= ncols) {
            अनुवर्त;
        }
        ध्रुव पूर्णाङ्क ib = (row*ncols + col)/qk; // x block index
        ध्रुव पूर्णाङ्क iqs = (col%qk)/qr; // x quant index
        ध्रुव पूर्णाङ्क iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < vals_प्रत्येकम्_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#यदि_परिभाषितम् GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_विस्थापनम्]};

            tmp += v * t1;
#अन्यथा
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_विस्थापनम्];
#यदि_अन्तः // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ध्रुव पूर्णाङ्क आवरणम्_start = ncols > GGML_SYCL_DMMV_X ? WARP_आकार >> 1 : WARP_आकार >> 2;
    प्रति (पूर्णाङ्क आवरणम् = आवरणम्_start; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (tid == 0) {
#यदि_परिभाषितम् GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#अन्यथा
        dst[row] = tmp;
#यदि_अन्तः // GGML_SYCL_F16
    }
}

प्रतिरूपम् <पूर्णाङ्क qk, पूर्णाङ्क qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
स्थिर शून्य dequantize_mul_mat_vec_reorder(ध्रुव शून्य * __restrict__ vx, ध्रुव dfloat * __restrict__ y, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                   ध्रुव sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    यदि (row >= nrows) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);


    ध्रुव पूर्णाङ्क ncols_वामः = ncols % (QK4_0*WARP_आकार);
    ध्रुव पूर्णाङ्क ncols_align = ncols - ncols_वामः;
    ध्रुव पूर्णाङ्क iter_stride = 8*2*GGML_SYCL_DMMV_X;
    ध्रुव पूर्णाङ्क vals_प्रत्येकम्_iter = iter_stride / WARP_आकार; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    ध्रुव पूर्णाङ्क y_विस्थापनम् = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#यदि_परिभाषितम् GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#अन्यथा
    भिन्न tmp = 0.0f;
#यदि_अन्तः // GGML_SYCL_F16
    ध्रुव वर्ण *d_ptr = (ध्रुव वर्ण*)vx+ncols*nrows/2;
    पूर्णाङ्क i=0;
    प्रति (i = 0; i < ncols_align; i += iter_stride) {
        ध्रुव पूर्णाङ्क col = i + vals_प्रत्येकम्_iter*tid;
        ध्रुव पूर्णाङ्क ib = (row*ncols + col)/qk; // x block index
        ध्रुव पूर्णाङ्क iqs = (col%qk)/qr; // x quant index
        ध्रुव पूर्णाङ्क iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < vals_प्रत्येकम्_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ध्रुव शून्य *)d_ptr, ib, (ध्रुव शून्य *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#यदि_परिभाषितम् GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_विस्थापनम्]};

            tmp += v * t1;
#अन्यथा
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_विस्थापनम्];
#यदि_अन्तः // GGML_SYCL_F16
        }
    }

    प्रति (; i < ncols; i += iter_stride) {
        यदि (tid>=ncols_वामः/QK4_0) अनुवर्त;
        ध्रुव पूर्णाङ्क col = i + vals_प्रत्येकम्_iter*tid;
        ध्रुव पूर्णाङ्क ib = (row*ncols + col)/qk; // x block index
        ध्रुव पूर्णाङ्क iqs = (col%qk)/qr; // x quant index
        ध्रुव पूर्णाङ्क iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < vals_प्रत्येकम्_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ध्रुव शून्य *)d_ptr, ib, (ध्रुव शून्य *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#यदि_परिभाषितम् GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_विस्थापनम्]};

            tmp += v * t1;
#अन्यथा
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_विस्थापनम्];
#यदि_अन्तः // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ध्रुव पूर्णाङ्क आवरणम्_start = ncols > GGML_SYCL_DMMV_X ? WARP_आकार >> 1 : WARP_आकार >> 2;
    प्रति (पूर्णाङ्क आवरणम् = आवरणम्_start; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (tid == 0) {
#यदि_परिभाषितम् GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#अन्यथा
        dst[row] = tmp;
#यदि_अन्तः // GGML_SYCL_F16
    }
}

स्थिर शून्य convert_mul_mat_vec_f16_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                         भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                         ध्रुव पूर्णाङ्क nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, item_ct1);
            });
    }
}

#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_BF16
स्थिर शून्य convert_mul_mat_vec_bf16_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                          भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                          ध्रुव पूर्णाङ्क nrows,
                                          dpct::queue_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, item_ct1);
            });
    }
}
#यदि_अन्तः

स्थिर शून्य dequantize_mul_mat_vec_q2_k(ध्रुव शून्य *__restrict__ vx,
                                        ध्रुव भिन्न *__restrict__ yy,
                                        भिन्न *__restrict__ dst,
                                        ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                        ध्रुव sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_प्रत्येकम्_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    ध्रुव खण्डम्_q2_K * x = (ध्रुव खण्डम्_q2_K *)vx + ib0;

    भिन्न tmp = 0; // partial sum for thread in warp

#यदि QK_K == 256
    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0,1

    ध्रुव पूर्णाङ्क चरणः = 16/K_QUANTS_प्रत्येकम्_ITERATION;

    ध्रुव पूर्णाङ्क अन्तर् = tid % चरणः;                           // 0...15 or 0...7

    ध्रुव पूर्णाङ्क l0 = K_QUANTS_प्रत्येकम्_ITERATION*अन्तर्;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ध्रुव uint8_t * d = (ध्रुव uint8_t *)aux;
    ध्रुव uint8_t * m = (ध्रुव uint8_t *)(aux + 2);

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव भिन्न dall = x[i].dm[0];
        ध्रुव भिन्न dmin = x[i].dm[1];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क s_विस्थापनम् = 8*im;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 128*im + l0;

            ध्रुव भिन्न   * y = yy + i * QK_K + y_विस्थापनम्;
            ध्रुव uint8_t * q = x[i].qs + q_विस्थापनम्;

            ध्रुव uint32_t * a = (ध्रुव uint32_t *)(x[i].scales + s_विस्थापनम्);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            भिन्न sum1 = 0, sum2 = 0;
            प्रति (पूर्णाङ्क l = 0; l < K_QUANTS_प्रत्येकम्_ITERATION; ++l) {
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
#अन्यथा
    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2) /
                    (2 * K_QUANTS_प्रत्येकम्_ITERATION); // 0...15 or 0...7
    ध्रुव पूर्णाङ्क ix = item_ct1.get_local_id(2) %
                   (2 * K_QUANTS_प्रत्येकम्_ITERATION); // 0....1 or 0...3
    ध्रुव पूर्णाङ्क विस्थापनम् = tid * K_QUANTS_प्रत्येकम्_ITERATION;

    uint32_t uaux[2];
    ध्रुव uint8_t * d = (ध्रुव uint8_t *)uaux;


    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव भिन्न   * y = yy + i * QK_K + विस्थापनम्;
        ध्रुव uint8_t * q = x[i].qs + विस्थापनम्;
        ध्रुव uint32_t * s = (ध्रुव uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        ध्रुव sycl::float2 dall =
            x[i].dm.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

        भिन्न sum1 = 0, sum2 = 0;
        प्रति (पूर्णाङ्क l = 0; l < K_QUANTS_प्रत्येकम्_ITERATION; ++l) {
            ध्रुव uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q2_k_reorder(ध्रुव शून्य *__restrict__ vx,
                                                ध्रुव भिन्न *__restrict__ yy,
                                                भिन्न *__restrict__ dst,
                                                ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                                ध्रुव sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_प्रत्येकम्_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    ध्रुव पूर्णाङ्क nb = nrows * अङ्कः_खण्डानि_प्रत्येकम्_row;
    ध्रुव uint8_t     * qs_आधारः     = (ध्रुव uint8_t *)vx;
    ध्रुव uint8_t     * scales_आधारः = qs_आधारः + (size_t)nb * (QK_K / 4);
    ध्रुव sycl::half2 * dm_आधारः     = (ध्रुव sycl::half2 *)(scales_आधारः + (size_t)nb * (QK_K / 16));

    भिन्न tmp = 0; // partial sum for thread in warp

#यदि QK_K == 256
    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0,1

    ध्रुव पूर्णाङ्क चरणः = 16/K_QUANTS_प्रत्येकम्_ITERATION;

    ध्रुव पूर्णाङ्क अन्तर् = tid % चरणः;                           // 0...15 or 0...7

    ध्रुव पूर्णाङ्क l0 = K_QUANTS_प्रत्येकम्_ITERATION*अन्तर्;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ध्रुव uint8_t * d = (ध्रुव uint8_t *)aux;
    ध्रुव uint8_t * m = (ध्रुव uint8_t *)(aux + 2);

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव sycl::half2 dm_val = dm_आधारः[bi];
        ध्रुव भिन्न dall = dm_val[0];
        ध्रुव भिन्न dmin = dm_val[1];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क s_विस्थापनम् = 8*im;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 128*im + l0;

            ध्रुव भिन्न   * y = yy + i * QK_K + y_विस्थापनम्;
            ध्रुव uint8_t * q = qs_आधारः + bi * (QK_K / 4) + q_विस्थापनम्;

            ध्रुव uint32_t * a = (ध्रुव uint32_t *)(scales_आधारः + bi * (QK_K / 16) + s_विस्थापनम्);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            भिन्न sum1 = 0, sum2 = 0;
            प्रति (पूर्णाङ्क l = 0; l < K_QUANTS_प्रत्येकम्_ITERATION; ++l) {
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
#अन्यथा
    GGML_अप्रयुक्तम्(vx);
    GGML_अप्रयुक्तम्(yy);
    GGML_अप्रयुक्तम्(ncols);
    GGML_अप्रयुक्तम्(item_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q3_k(ध्रुव शून्य *__restrict__ vx,
                                        ध्रुव भिन्न *__restrict__ yy,
                                        भिन्न *__restrict__ dst,
                                        ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                        ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    ध्रुव खण्डम्_q3_K * x = (ध्रुव खण्डम्_q3_K *)vx + ib0;

    भिन्न tmp = 0; // partial sum for thread in warp

#यदि QK_K == 256

    ध्रुव uint16_t kmask1 = 0x0303;
    ध्रुव uint16_t kmask2 = 0x0f0f;

    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0,1

    ध्रुव पूर्णाङ्क n  = K_QUANTS_प्रत्येकम्_ITERATION;               // iterations in the inner loop
    ध्रुव पूर्णाङ्क चरणः = 16/K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क अन्तर् = tid % चरणः;                           // 0...15 or 0...7

    ध्रुव पूर्णाङ्क l0 = n*अन्तर्;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ध्रुव int8_t * s = (ध्रुव int8_t *)utmp;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव uint8_t * h = x[i].hmask + l0;
        ध्रुव भिन्न d = x[i].d;

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् =  32*im + l0;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 128*im + l0;
            ध्रुव uint16_t s_सरणम् = 4*im;
            ध्रुव uint8_t m = 1 << (4*im);

            ध्रुव भिन्न   * y  = yy + i * QK_K + y_विस्थापनम्;
            ध्रुव uint8_t * q = x[i].qs + q_विस्थापनम्;

            ध्रुव uint16_t * a = (ध्रुव uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_सरणम्) & kmask2) | (((a[4] >> (s_सरणम् + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_सरणम्) & kmask2) | (((a[5] >> (s_सरणम् + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_सरणम्) & kmask2) | (((a[4] >> (s_सरणम् + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_सरणम्) & kmask2) | (((a[5] >> (s_सरणम् + 2)) & kmask1) << 4);

            भिन्न योग_2 = 0;
            प्रति (पूर्णाङ्क l = 0; l < n; ++l) {
                योग_2 += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                योग_2 += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * योग_2;
        }

    }
#अन्यथा

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2)/(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...15 or 0...7
    ध्रुव पूर्णाङ्क ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0....1 or 0...3
    ध्रुव पूर्णाङ्क विस्थापनम् = tid * K_QUANTS_प्रत्येकम्_ITERATION;         // 0...15 or 0...14
    ध्रुव पूर्णाङ्क अन्तर् = विस्थापनम्/8;                                 // 0 or 1
    ध्रुव पूर्णाङ्क im = विस्थापनम्%8;                                 // 0...7

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव भिन्न   * y = yy + i * QK_K + विस्थापनम्;
        ध्रुव uint8_t * q = x[i].qs + विस्थापनम्;
        ध्रुव uint8_t * s = x[i].scales;

        ध्रुव भिन्न dall = (भिन्न)x[i].d;

        भिन्न योग_2 = 0;
        प्रति (पूर्णाङ्क l = 0; l < K_QUANTS_प्रत्येकम्_ITERATION; ++l) {
            ध्रुव uint8_t hl = x[i].hmask[im+l] >> अन्तर्;
            ध्रुव uint8_t ql = q[l];
            योग_2 += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += योग_2;
    }
#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q3_k_reorder(ध्रुव शून्य *__restrict__ vx,
                                                ध्रुव भिन्न *__restrict__ yy,
                                                भिन्न *__restrict__ dst,
                                                ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                                ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    ध्रुव पूर्णाङ्क nb = nrows * अङ्कः_खण्डानि_प्रत्येकम्_row;
    ध्रुव uint8_t   * qs_आधारः     = (ध्रुव uint8_t *)vx;
    ध्रुव uint8_t   * hmask_आधारः  = qs_आधारः + (size_t)nb * (QK_K / 4);
    ध्रुव uint8_t   * scales_आधारः = hmask_आधारः + (size_t)nb * (QK_K / 8);
    ध्रुव sycl::half * d_आधारः     = (ध्रुव sycl::half *)(scales_आधारः + (size_t)nb * 12);

    भिन्न tmp = 0; // partial sum for thread in warp

#यदि QK_K == 256

    ध्रुव uint16_t kmask1 = 0x0303;
    ध्रुव uint16_t kmask2 = 0x0f0f;

    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0,1

    ध्रुव पूर्णाङ्क n  = K_QUANTS_प्रत्येकम्_ITERATION;               // iterations in the inner loop
    ध्रुव पूर्णाङ्क चरणः = 16/K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क अन्तर् = tid % चरणः;                           // 0...15 or 0...7

    ध्रुव पूर्णाङ्क l0 = n*अन्तर्;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ध्रुव int8_t * s = (ध्रुव int8_t *)utmp;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव uint8_t * h  = hmask_आधारः + bi * (QK_K / 8) + l0;

        ध्रुव भिन्न d = d_आधारः[bi];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् =  32*im + l0;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 128*im + l0;
            ध्रुव uint16_t s_सरणम् = 4*im;
            ध्रुव uint8_t m = 1 << (4*im);

            ध्रुव भिन्न   * y  = yy + i * QK_K + y_विस्थापनम्;
            ध्रुव uint8_t * q  = qs_आधारः + bi * (QK_K / 4) + q_विस्थापनम्;

            ध्रुव uint16_t * a = (ध्रुव uint16_t *)(scales_आधारः + bi * 12);
            utmp[0] = ((a[0] >> s_सरणम्) & kmask2) | (((a[4] >> (s_सरणम् + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_सरणम्) & kmask2) | (((a[5] >> (s_सरणम् + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_सरणम्) & kmask2) | (((a[4] >> (s_सरणम् + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_सरणम्) & kmask2) | (((a[5] >> (s_सरणम् + 2)) & kmask1) << 4);

            भिन्न योग_2 = 0;
            प्रति (पूर्णाङ्क l = 0; l < n; ++l) {
                योग_2 += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                योग_2 += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * योग_2;
        }
    }
#अन्यथा
    GGML_अप्रयुक्तम्(vx);
    GGML_अप्रयुक्तम्(yy);
    GGML_अप्रयुक्तम्(ncols);
    GGML_अप्रयुक्तम्(item_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q4_k(ध्रुव शून्य *__restrict__ vx,
                                        ध्रुव भिन्न *__restrict__ yy,
                                        भिन्न *__restrict__ dst,
                                        ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                        ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    ध्रुव खण्डम्_q4_K * x = (ध्रुव खण्डम्_q4_K *)vx + ib0;

#यदि QK_K == 256
    ध्रुव uint16_t kmask1 = 0x3f3f;
    ध्रुव uint16_t kmask2 = 0x0f0f;
    ध्रुव uint16_t kmask3 = 0xc0c0;

    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0,1

    ध्रुव पूर्णाङ्क चरणः = 8/K_QUANTS_प्रत्येकम्_ITERATION;           // 8 or 4

    ध्रुव पूर्णाङ्क il_आधारः = tid/चरणः;                        // 0 or 1 (was 0...3)
    ध्रुव पूर्णाङ्क ir  = tid - चरणः*il_आधारः;                  // 0...7 or 0...3
    ध्रुव पूर्णाङ्क n   = 2 * K_QUANTS_प्रत्येकम्_ITERATION;          // 2 or 4

    ध्रुव पूर्णाङ्क अन्तर् = il_आधारः%2;

    ध्रुव पूर्णाङ्क l0 = n*(2*ir + अन्तर्);

    uint16_t aux[4];
    ध्रुव uint8_t * sc = (ध्रुव uint8_t *)aux;

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 2
    uint32_t q32[4];
    ध्रुव uint8_t * q4 = (ध्रुव uint8_t *)q32;
#अन्यथा
    uint16_t q16[4];
    ध्रुव uint8_t * q4 = (ध्रुव uint8_t *)q16;
#यदि_अन्तः

    भिन्न tmp = 0; // partial sum for thread in warp

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव भिन्न dall = x[i].dm[0];
        ध्रुव भिन्न dmin = x[i].dm[1];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 64*im + l0;

            ध्रुव भिन्न   * y1 = yy + i*QK_K + y_विस्थापनम्;
            ध्रुव भिन्न   * y2 = y1 + 128;

            ध्रुव uint16_t * a = (ध्रुव uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 2
            ध्रुव uint32_t * q1 = (ध्रुव uint32_t *)(x[i].qs + q_विस्थापनम्);
            ध्रुव uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            भिन्न smin = 0;
            प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#अन्यथा
            ध्रुव uint16_t * q1 = (ध्रुव uint16_t *)(x[i].qs + q_विस्थापनम्);
            ध्रुव uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            भिन्न smin = 0;
            प्रति (पूर्णाङ्क l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#यदि_अन्तः
        }

    }
#अन्यथा
    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2)/(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...15
    ध्रुव पूर्णाङ्क ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_प्रत्येकम्_ITERATION);

    ध्रुव पूर्णाङ्क चरणः = tid * K_QUANTS_प्रत्येकम्_ITERATION;

    uint16_t aux16[2];
    ध्रुव uint8_t * s = (ध्रुव uint8_t *)aux16;

    भिन्न tmp = 0;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव uint8_t * q = x[i].qs + चरणः;
        ध्रुव भिन्न   * y = yy + i*QK_K + चरणः;
        ध्रुव uint16_t * a = (ध्रुव uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ध्रुव भिन्न d = (भिन्न)x[i].dm[0];
        ध्रुव भिन्न m = (भिन्न)x[i].dm[1];
        भिन्न योग_2 = 0.f;
        प्रति (पूर्णाङ्क j = 0; j < K_QUANTS_प्रत्येकम्_ITERATION; ++j) {
            योग_2 += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += योग_2;
    }

#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (tid == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q4_k_reorder(ध्रुव शून्य *__restrict__ vx,
                                                ध्रुव भिन्न *__restrict__ yy,
                                                भिन्न *__restrict__ dst,
                                                ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                                ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ध्रुव पूर्णाङ्क nb = nrows * अङ्कः_खण्डानि_प्रत्येकम्_row;
    ध्रुव uint8_t     * qs_आधारः     = (ध्रुव uint8_t *)vx;
    ध्रुव uint8_t     * scales_आधारः = qs_आधारः + (size_t)nb * (QK_K / 2);
    ध्रुव sycl::half2 * dm_आधारः     = (ध्रुव sycl::half2 *)(scales_आधारः + (size_t)nb * K_SCALE_आकार);

#यदि QK_K == 256
    ध्रुव uint16_t kmask1 = 0x3f3f;
    ध्रुव uint16_t kmask2 = 0x0f0f;
    ध्रुव uint16_t kmask3 = 0xc0c0;

    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0,1

    ध्रुव पूर्णाङ्क चरणः = 8/K_QUANTS_प्रत्येकम्_ITERATION;           // 8 or 4

    ध्रुव पूर्णाङ्क il_आधारः = tid/चरणः;                        // 0 or 1 (was 0...3)
    ध्रुव पूर्णाङ्क ir  = tid - चरणः*il_आधारः;                  // 0...7 or 0...3
    ध्रुव पूर्णाङ्क n   = 2 * K_QUANTS_प्रत्येकम्_ITERATION;          // 2 or 4

    ध्रुव पूर्णाङ्क अन्तर् = il_आधारः%2;

    ध्रुव पूर्णाङ्क l0 = n*(2*ir + अन्तर्);

    uint16_t aux[4];
    ध्रुव uint8_t * sc = (ध्रुव uint8_t *)aux;

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 2
    uint32_t q32[4];
    ध्रुव uint8_t * q4 = (ध्रुव uint8_t *)q32;
#अन्यथा
    uint16_t q16[4];
    ध्रुव uint8_t * q4 = (ध्रुव uint8_t *)q16;
#यदि_अन्तः

    भिन्न tmp = 0; // partial sum for thread in warp

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव sycl::half2 dm_val = dm_आधारः[bi];
        ध्रुव भिन्न dall = dm_val[0];
        ध्रुव भिन्न dmin = dm_val[1];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 64*im + l0;

            ध्रुव भिन्न   * y1 = yy + i*QK_K + y_विस्थापनम्;
            ध्रुव भिन्न   * y2 = y1 + 128;

            ध्रुव uint16_t * a = (ध्रुव uint16_t *)(scales_आधारः + bi * K_SCALE_आकार);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 2
            ध्रुव uint32_t * q1 = (ध्रुव uint32_t *)(qs_आधारः + bi * (QK_K / 2) + q_विस्थापनम्);
            ध्रुव uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            भिन्न smin = 0;
            प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#अन्यथा
            ध्रुव uint16_t * q1 = (ध्रुव uint16_t *)(qs_आधारः + bi * (QK_K / 2) + q_विस्थापनम्);
            ध्रुव uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            भिन्न smin = 0;
            प्रति (पूर्णाङ्क l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#यदि_अन्तः
        }

    }
#अन्यथा
    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2)/(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...15
    ध्रुव पूर्णाङ्क ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_प्रत्येकम्_ITERATION);

    ध्रुव पूर्णाङ्क चरणः = tid * K_QUANTS_प्रत्येकम्_ITERATION;

    uint16_t aux16[2];
    ध्रुव uint8_t * s = (ध्रुव uint8_t *)aux16;

    भिन्न tmp = 0;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव uint8_t * q = qs_आधारः + bi * (QK_K / 2) + चरणः;
        ध्रुव भिन्न   * y = yy + i*QK_K + चरणः;
        ध्रुव uint16_t * a = (ध्रुव uint16_t *)(scales_आधारः + bi * K_SCALE_आकार);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ध्रुव sycl::half2 dm_val = dm_आधारः[bi];
        ध्रुव भिन्न d = (भिन्न)dm_val[0];
        ध्रुव भिन्न m = (भिन्न)dm_val[1];
        भिन्न योग_2 = 0.f;
        प्रति (पूर्णाङ्क j = 0; j < K_QUANTS_प्रत्येकम्_ITERATION; ++j) {
            योग_2 += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += योग_2;
    }

#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (tid == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q5_k(ध्रुव शून्य *__restrict__ vx,
                                        ध्रुव भिन्न *__restrict__ yy,
                                        भिन्न *__restrict__ dst,
                                        ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                        ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    ध्रुव खण्डम्_q5_K * x = (ध्रुव खण्डम्_q5_K *)vx + ib0;

    भिन्न tmp = 0; // partial sum for thread in warp

#यदि QK_K == 256
    ध्रुव uint16_t kmask1 = 0x3f3f;
    ध्रुव uint16_t kmask2 = 0x0f0f;
    ध्रुव uint16_t kmask3 = 0xc0c0;

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2) / 2; // 0...7
    ध्रुव पूर्णाङ्क ix = item_ct1.get_local_id(2) % 2;

    ध्रुव पूर्णाङ्क il_आधारः = tid/4;     // 0 or 1 (was 0...3)
    ध्रुव पूर्णाङ्क ir  = tid - 4*il_आधारः;// 0...3
    ध्रुव पूर्णाङ्क n   = 2;

    ध्रुव पूर्णाङ्क अन्तर् = il_आधारः%2;

    ध्रुव पूर्णाङ्क l0 = n*(2*ir + अन्तर्);

    uint16_t aux[4];
    ध्रुव uint8_t * sc = (ध्रुव uint8_t *)aux;

    uint16_t q16[8];
    ध्रुव uint8_t * q4 = (ध्रुव uint8_t *)q16;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2) {

        ध्रुव uint8_t * qh  = x[i].qh + l0;
        ध्रुव भिन्न dall = x[i].dm[0];
        ध्रुव भिन्न dmin = x[i].dm[1];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 64*im + l0;

            ध्रुव uint8_t hm1  = 1 << (2*im);
            ध्रुव uint8_t hm2  = hm1 << 4;

            ध्रुव uint8_t * ql1 = x[i].qs + q_विस्थापनम्;
            ध्रुव भिन्न   * y1  = yy + i*QK_K + y_विस्थापनम्;
            ध्रुव भिन्न   * y2  = y1 + 128;

            ध्रुव uint16_t * a = (ध्रुव uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 योग_2 = {0.f, 0.f, 0.f, 0.f};
            भिन्न smin = 0;
            ध्रुव uint16_t * q1 = (ध्रुव uint16_t *)ql1;
            ध्रुव uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            प्रति (पूर्णाङ्क l = 0; l < n; ++l) {
                योग_2.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                योग_2.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                योग_2.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                योग_2.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (योग_2.x() * sc[0] + योग_2.y() * sc[1] + योग_2.z() * sc[4] +
                           योग_2.w() * sc[5]) -
                   dmin * smin;
        }
    }

#अन्यथा
    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2)/(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...15
    ध्रुव पूर्णाङ्क ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_प्रत्येकम्_ITERATION);
    ध्रुव पूर्णाङ्क चरणः = tid * K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क im = चरणः/8;
    ध्रुव पूर्णाङ्क अन्तर् = चरणः%8;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव uint8_t * q = x[i].qs + चरणः;
        ध्रुव int8_t  * s = x[i].scales;
        ध्रुव भिन्न   * y = yy + i*QK_K + चरणः;
        ध्रुव भिन्न     d = x[i].d;
        भिन्न योग_2 = 0.f;
        प्रति (पूर्णाङ्क j = 0; j < K_QUANTS_प्रत्येकम्_ITERATION; ++j) {
            ध्रुव uint8_t h = x[i].qh[अन्तर्+j] >> im;
            योग_2 += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += योग_2;
    }
#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q5_k_reorder(ध्रुव शून्य *__restrict__ vx,
                                                ध्रुव भिन्न *__restrict__ yy,
                                                भिन्न *__restrict__ dst,
                                                ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                                ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ध्रुव पूर्णाङ्क nb = nrows * अङ्कः_खण्डानि_प्रत्येकम्_row;
    ध्रुव uint8_t     * qs_आधारः     = (ध्रुव uint8_t *)vx;
    ध्रुव uint8_t     * qh_आधारः     = qs_आधारः + (size_t)nb * (QK_K / 2);
    ध्रुव uint8_t     * scales_आधारः = qh_आधारः + (size_t)nb * (QK_K / 8);
    ध्रुव sycl::half2 * dm_आधारः     = (ध्रुव sycl::half2 *)(scales_आधारः + (size_t)nb * K_SCALE_आकार);

    भिन्न tmp = 0; // partial sum for thread in warp

#यदि QK_K == 256
    ध्रुव uint16_t kmask1 = 0x3f3f;
    ध्रुव uint16_t kmask2 = 0x0f0f;
    ध्रुव uint16_t kmask3 = 0xc0c0;

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2) / 2; // 0...15
    ध्रुव पूर्णाङ्क ix = item_ct1.get_local_id(2) % 2;

    ध्रुव पूर्णाङ्क il_आधारः = tid/4;      // 0...3
    ध्रुव पूर्णाङ्क ir  = tid - 4*il_आधारः;// 0...3
    ध्रुव पूर्णाङ्क n   = 2;

    ध्रुव पूर्णाङ्क अन्तर् = il_आधारः%2;

    ध्रुव पूर्णाङ्क l0 = n*(2*ir + अन्तर्);

    uint16_t aux[4];
    ध्रुव uint8_t * sc = (ध्रुव uint8_t *)aux;

    uint16_t q16[8];
    ध्रुव uint8_t * q4 = (ध्रुव uint8_t *)q16;

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव uint8_t * qh  = qh_आधारः + bi * (QK_K / 8) + l0;
        ध्रुव sycl::half2 dm_val = dm_आधारः[bi];
        ध्रुव भिन्न dall = dm_val[0];
        ध्रुव भिन्न dmin = dm_val[1];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क q_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 64*im + l0;

            ध्रुव uint8_t hm1  = 1 << (2*im);
            ध्रुव uint8_t hm2  = hm1 << 4;

            ध्रुव uint8_t * ql1 = qs_आधारः + bi * (QK_K / 2) + q_विस्थापनम्;
            ध्रुव भिन्न   * y1  = yy + i*QK_K + y_विस्थापनम्;
            ध्रुव भिन्न   * y2  = y1 + 128;

            ध्रुव uint16_t * a = (ध्रुव uint16_t *)(scales_आधारः + bi * K_SCALE_आकार);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 योग_2 = {0.f, 0.f, 0.f, 0.f};
            भिन्न smin = 0;
            ध्रुव uint16_t * q1 = (ध्रुव uint16_t *)ql1;
            ध्रुव uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            प्रति (पूर्णाङ्क l = 0; l < n; ++l) {
                योग_2.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                योग_2.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                योग_2.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                योग_2.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (योग_2.x() * sc[0] + योग_2.y() * sc[1] + योग_2.z() * sc[4] +
                           योग_2.w() * sc[5]) -
                   dmin * smin;
        }
    }
#अन्यथा
    // The reordered Q5_K layout is only produced for QK_K == 256.
#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q6_k(ध्रुव शून्य * __restrict__ vx, ध्रुव भिन्न * __restrict__ yy, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                        ध्रुव sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_प्रत्येकम्_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    ध्रुव खण्डम्_q6_K * x = (ध्रुव खण्डम्_q6_K *)vx + ib0;

#यदि QK_K == 256

    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0, 1

    ध्रुव पूर्णाङ्क चरणः = 16/K_QUANTS_प्रत्येकम्_ITERATION;          // 16 or 8

    ध्रुव पूर्णाङ्क अन्तर् = tid % चरणः;                           // 0...15 or 0...7

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 1
    ध्रुव पूर्णाङ्क l0 = K_QUANTS_प्रत्येकम्_ITERATION*अन्तर्;            // 0...15
    ध्रुव पूर्णाङ्क is = 0;
#अन्यथा
    ध्रुव पूर्णाङ्क l0 = 4 * अन्तर्;                               // 0, 4, 8, ..., 28
    ध्रुव पूर्णाङ्क is = अन्तर् / 4;
#यदि_अन्तः

    भिन्न tmp = 0; // partial sum for thread in warp

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव भिन्न d = x[i].d;

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क ql_विस्थापनम् = 64*im + l0;
            ध्रुव पूर्णाङ्क qh_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क s_विस्थापनम्  =  8*im + is;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 128*im + l0;

            ध्रुव भिन्न   * y  = yy + i * QK_K + y_विस्थापनम्;
            ध्रुव uint8_t * ql = x[i].ql + ql_विस्थापनम्;
            ध्रुव uint8_t * qh = x[i].qh + qh_विस्थापनम्;
            ध्रुव int8_t  * s  = x[i].scales + s_विस्थापनम्;

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 1
            भिन्न योग_2 = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += योग_2;
#अन्यथा
            भिन्न योग_2 = 0;
            प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
                योग_2 += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += योग_2;
#यदि_अन्तः
        }

    }

#अन्यथा

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2)/(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...7
    ध्रुव पूर्णाङ्क ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...3

    ध्रुव पूर्णाङ्क चरणः = tid * K_QUANTS_प्रत्येकम्_ITERATION;

    भिन्न tmp = 0; // partial sum for thread in warp

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {

        ध्रुव भिन्न   * y  = yy + i * QK_K + चरणः;
        ध्रुव uint8_t * ql = x[i].ql + चरणः;
        ध्रुव uint8_t * qh = x[i].qh + चरणः;
        ध्रुव int8_t  * s  = x[i].scales;

        ध्रुव भिन्न d = x[i+0].d;

        भिन्न योग_2 = 0;
        प्रति (पूर्णाङ्क j = 0; j < K_QUANTS_प्रत्येकम्_ITERATION; ++j) {
            योग_2 += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += योग_2;

    }

#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (tid == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q6_k_reorder(ध्रुव शून्य * __restrict__ vx, ध्रुव भिन्न * __restrict__ yy, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क nrows,
                                                ध्रुव sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_प्रत्येकम्_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    यदि (row >= nrows) प्रत्यागम;

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव पूर्णाङ्क ib0 = row*अङ्कः_खण्डानि_प्रत्येकम्_row;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    ध्रुव पूर्णाङ्क nb = nrows * अङ्कः_खण्डानि_प्रत्येकम्_row;
    ध्रुव uint8_t   * ql_आधारः     = (ध्रुव uint8_t *)vx;
    ध्रुव uint8_t   * qh_आधारः     = ql_आधारः + (size_t)nb * (QK_K / 2);
    ध्रुव int8_t    * scales_आधारः = (ध्रुव int8_t *)(qh_आधारः + (size_t)nb * (QK_K / 4));
    ध्रुव sycl::half * d_आधारः     = (ध्रुव sycl::half *)((ध्रुव uint8_t *)scales_आधारः + (size_t)nb * (QK_K / 16));

#यदि QK_K == 256

    ध्रुव पूर्णाङ्क tid =
        item_ct1.get_local_id(2) / K_QUANTS_प्रत्येकम्_ITERATION; // 0...7 or 0...15
    ध्रुव पूर्णाङ्क ix =
        item_ct1.get_local_id(2) % K_QUANTS_प्रत्येकम्_ITERATION; // 0 or 0, 1

    ध्रुव पूर्णाङ्क चरणः = 16/K_QUANTS_प्रत्येकम्_ITERATION;          // 16 or 8

    ध्रुव पूर्णाङ्क अन्तर् = tid % चरणः;                           // 0...15 or 0...7

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 1
    ध्रुव पूर्णाङ्क l0 = K_QUANTS_प्रत्येकम्_ITERATION*अन्तर्;            // 0...15
    ध्रुव पूर्णाङ्क is = 0;
#अन्यथा
    ध्रुव पूर्णाङ्क l0 = 4 * अन्तर्;                               // 0, 4, 8, ..., 28
    ध्रुव पूर्णाङ्क is = अन्तर् / 4;
#यदि_अन्तः

    भिन्न tmp = 0; // partial sum for thread in warp

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव भिन्न d = d_आधारः[bi];

        प्रति (पूर्णाङ्क im = 0; im < 2; ++im) {
            ध्रुव पूर्णाङ्क ql_विस्थापनम् = 64*im + l0;
            ध्रुव पूर्णाङ्क qh_विस्थापनम् = 32*im + l0;
            ध्रुव पूर्णाङ्क s_विस्थापनम्  =  8*im + is;
            ध्रुव पूर्णाङ्क y_विस्थापनम् = 128*im + l0;

            ध्रुव भिन्न   * y  = yy + i * QK_K + y_विस्थापनम्;
            ध्रुव uint8_t * ql = ql_आधारः + bi * (QK_K / 2) + ql_विस्थापनम्;
            ध्रुव uint8_t * qh = qh_आधारः + bi * (QK_K / 4) + qh_विस्थापनम्;
            ध्रुव int8_t  * s  = scales_आधारः + bi * (QK_K / 16) + s_विस्थापनम्;

#यदि K_QUANTS_प्रत्येकम्_ITERATION == 1
            भिन्न योग_2 = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += योग_2;
#अन्यथा
            भिन्न योग_2 = 0;
            प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
                योग_2 += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += योग_2;
#यदि_अन्तः
        }

    }

#अन्यथा

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2)/(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...7
    ध्रुव पूर्णाङ्क ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_प्रत्येकम्_ITERATION);  // 0...3

    ध्रुव पूर्णाङ्क चरणः = tid * K_QUANTS_प्रत्येकम्_ITERATION;

    भिन्न tmp = 0; // partial sum for thread in warp

    प्रति (पूर्णाङ्क i = ix; i < अङ्कः_खण्डानि_प्रत्येकम्_row; i += 2*K_QUANTS_प्रत्येकम्_ITERATION) {
        ध्रुव पूर्णाङ्क bi = ib0 + i;

        ध्रुव भिन्न   * y  = yy + i * QK_K + चरणः;
        ध्रुव uint8_t * ql = ql_आधारः + bi * (QK_K / 2) + चरणः;
        ध्रुव uint8_t * qh = qh_आधारः + bi * (QK_K / 4) + चरणः;
        ध्रुव int8_t  * s  = scales_आधारः + bi * (QK_K / 16);

        ध्रुव भिन्न d = d_आधारः[bi];

        भिन्न योग_2 = 0;
        प्रति (पूर्णाङ्क j = 0; j < K_QUANTS_प्रत्येकम्_ITERATION; ++j) {
            योग_2 += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += योग_2;

    }

#यदि_अन्तः

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (tid == 0) {
        dst[row] = tmp;
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q4_0_sycl_reorder(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}


स्थिर शून्य dequantize_mul_mat_vec_q4_0_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q1_0_sycl_reorder(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q1_0_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q4_1_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q5_0_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q5_1_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q8_0_sycl_reorder(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
                यदि (row >= nrows) प्रत्यागम;

                ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
                ध्रुव पूर्णाङ्क iter_stride = 8*2*GGML_SYCL_DMMV_X;
                ध्रुव पूर्णाङ्क vals_प्रत्येकम्_iter = iter_stride / WARP_आकार;
                ध्रुव पूर्णाङ्क ncols_वामः = ncols % (QK8_0*WARP_आकार);
                ध्रुव पूर्णाङ्क ncols_align = ncols - ncols_वामः;

#यदि_परिभाषितम् GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#अन्यथा
                भिन्न tmp = 0.0f;
#यदि_अन्तः
                ध्रुव वर्ण *d_ptr = (ध्रुव वर्ण*)vx + ncols*nrows;  // d after all qs

                पूर्णाङ्क i = 0;
                प्रति (i = 0; i < ncols_align; i += iter_stride) {
                    ध्रुव पूर्णाङ्क col = i + vals_प्रत्येकम्_iter*tid;
                    ध्रुव पूर्णाङ्क ib = (row*ncols + col)/QK8_0;
                    ध्रुव पूर्णाङ्क iqs = col % QK8_0;

#निर्देशः unroll
                    प्रति (पूर्णाङ्क j = 0; j < vals_प्रत्येकम्_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ध्रुव शून्य *)d_ptr, ib, (ध्रुव शून्य *)vx,
                                                ib * QK8_0 + iqs + j, v);

#यदि_परिभाषितम् GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#अन्यथा
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#यदि_अन्तः
                    }
                }

                // handle remaining columns
                प्रति (; i < ncols; i += iter_stride) {
                    यदि (tid >= ncols_वामः/QK8_0) अनुवर्त;
                    ध्रुव पूर्णाङ्क col = i + vals_प्रत्येकम्_iter*tid;
                    ध्रुव पूर्णाङ्क ib = (row*ncols + col)/QK8_0;
                    ध्रुव पूर्णाङ्क iqs = col % QK8_0;

#निर्देशः unroll
                    प्रति (पूर्णाङ्क j = 0; j < vals_प्रत्येकम्_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ध्रुव शून्य *)d_ptr, ib, (ध्रुव शून्य *)vx,
                                                ib * QK8_0 + iqs + j, v);

#यदि_परिभाषितम् GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#अन्यथा
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#यदि_अन्तः
                    }
                }

                // reduce
                ध्रुव पूर्णाङ्क आवरणम्_start = ncols > GGML_SYCL_DMMV_X ? WARP_आकार >> 1 : WARP_आकार >> 2;
                प्रति (पूर्णाङ्क आवरणम् = आवरणम्_start; आवरणम् > 0; आवरणम् >>= 1) {
                    tmp += dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
                }

                यदि (tid == 0) {
#यदि_परिभाषितम् GGML_SYCL_F16
                    dst[row] = tmp.x() + tmp.y();
#अन्यथा
                    dst[row] = tmp;
#यदि_अन्तः
                }
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q8_0_sycl(ध्रुव शून्य *vx, ध्रुव dfloat *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, GGML_SYCL_MMV_Y, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q2_K_sycl(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q2_K_sycl_reorder(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                     भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                     ध्रुव पूर्णाङ्क nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q3_K_sycl(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q3_K_sycl_reorder(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                     भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                     ध्रुव पूर्णाङ्क nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q4_K_sycl(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q5_K_sycl(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q6_K_sycl(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                             भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                             ध्रुव पूर्णाङ्क nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
उपयुज्य ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_आकार;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
संरचना esimd_dmmv_row_pair_policy {
    ध्रुव शून्य * vx;
    भिन्न *      dst;

    ESIMD_INLINE ध्रुव शून्य * w_a() ध्रुव { प्रत्यागम vx; }
    ESIMD_INLINE ध्रुव शून्य * w_b() ध्रुव { प्रत्यागम vx; }

    ESIMD_INLINE पूर्णाङ्क  row_a(पूर्णाङ्क g) ध्रुव { प्रत्यागम g * 2; }
    ESIMD_INLINE पूर्णाङ्क  row_b(पूर्णाङ्क g) ध्रुव { प्रत्यागम g * 2 + 1; }
    ESIMD_INLINE तार्किक has_b(पूर्णाङ्क g, पूर्णाङ्क nrows) ध्रुव { प्रत्यागम row_b(g) < nrows; }

    ESIMD_INLINE शून्य epilogue(पूर्णाङ्क g, पूर्णाङ्क nrows, भिन्न योग_a, भिन्न योग_b) ध्रुव {
        dst[row_a(g)] = योग_a;
        यदि (has_b(g, nrows)) {
            dst[row_b(g)] = योग_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
संरचना esimd_dmmv_glu_policy {
    ध्रुव शून्य * vgate;
    ध्रुव शून्य * vup;
    भिन्न *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE ध्रुव शून्य * w_a() ध्रुव { प्रत्यागम vgate; }
    ESIMD_INLINE ध्रुव शून्य * w_b() ध्रुव { प्रत्यागम vup; }

    ESIMD_INLINE पूर्णाङ्क  row_a(पूर्णाङ्क g) ध्रुव { प्रत्यागम g; }
    ESIMD_INLINE पूर्णाङ्क  row_b(पूर्णाङ्क g) ध्रुव { प्रत्यागम g; }
    ESIMD_INLINE तार्किक has_b(पूर्णाङ्क, पूर्णाङ्क) ध्रुव { प्रत्यागम सत्य; } // up always contributes

    ESIMD_INLINE शून्य epilogue(पूर्णाङ्क g, पूर्णाङ्क, भिन्न gate, भिन्न ऊर्ध्वम्) ध्रुव {
        उपयुज्य नामाकाश sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        ध्रुव भिन्न GELU_COEF_A    = 0.044715f;
        ध्रुव भिन्न SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        ध्रुव भिन्न NEG_LOG2E      = -1.44269504088896341f;

        ध्रुव भिन्न arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<भिन्न, 16> t = arg * NEG_LOG2E;
        simd<भिन्न, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * ऊर्ध्वम्;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
प्रतिरूपम् <ggml_type T, प्रकारनाम Policy>
ESIMD_INLINE शून्य dequantize_mul_mat_vec_reorder_esimd(
        ध्रुव Policy & policy, ध्रुव भिन्न * y,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        sycl::local_accessor<भिन्न, 1> lmem,
        ध्रुव sycl::nd_item<1> & it) {
    उपयुज्य नामाकाश sycl::ext::intel::esimd;
    उपयुज्य traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    ध्रुव पूर्णाङ्क    अङ्कः_खण्डानि_प्रत्येकम्_row = ncols / QK_K;
    ध्रुव size_t nb = (size_t) nrows * अङ्कः_खण्डानि_प्रत्येकम्_row;
    ध्रुव स्वचालित   pa = traits::make_ptrs(policy.w_a(), nb);
    ध्रुव स्वचालित   pb = traits::make_ptrs(policy.w_b(), nb);

    ध्रुव पूर्णाङ्क  tid   = it.get_local_id(0);
    ध्रुव पूर्णाङ्क  g     = it.get_गणः(0);
    ध्रुव पूर्णाङ्क  row_a = policy.row_a(g);
    ध्रुव पूर्णाङ्क  row_b = policy.row_b(g);
    ध्रुव तार्किक has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<भिन्न, 32> acc_a = 0.0f;
    simd<भिन्न, 32> acc_b = 0.0f;

    प्रति (पूर्णाङ्क ib = tid; ib < अङ्कः_खण्डानि_प्रत्येकम्_row; ib += GGML_SYCL_DMMV_ESIMD_WG_आकार) {
        simd<भिन्न, 256> y_vec = खण्डम्_आरोपय<भिन्न, 256>(y + (size_t) ib * QK_K);

        ध्रुव size_t bi_a = (size_t) row_a * अङ्कः_खण्डानि_प्रत्येकम्_row + ib;
        ध्रुव size_t bi_b = (size_t) row_b * अङ्कः_खण्डानि_प्रत्येकम्_row + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<भिन्न>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<भिन्न>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    यदि (tid == 0) {
        भिन्न योग_a = 0.0f;
        भिन्न योग_b = 0.0f;
        प्रति (पूर्णाङ्क p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_आकार; ++p) {
            योग_a += lmem[p * 2 + 0];
            योग_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, योग_a, योग_b);
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                           ध्रुव पूर्णाङ्क nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_आकार), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

स्थिर शून्य dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                           ध्रुव पूर्णाङ्क nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_आकार), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

स्थिर शून्य dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                           ध्रुव पूर्णाङ्क nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_आकार), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

स्थिर शून्य dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                           ध्रुव पूर्णाङ्क nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_आकार), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

स्थिर शून्य dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                           ध्रुव पूर्णाङ्क nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_आकार), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
प्रतिरूपम् <पूर्णाङ्क NBLK>
ESIMD_INLINE शून्य q8_0_mac_stripe(
        ध्रुव int8_t * qs_a, ध्रुव int8_t * qs_b,
        ध्रुव sycl::half * d_a, ध्रुव sycl::half * d_b, तार्किक has_b,
        sycl::ext::intel::esimd::simd<भिन्न, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<भिन्न, 32> & acc_a,
        sycl::ext::intel::esimd::simd<भिन्न, 32> & acc_b) {
    उपयुज्य नामाकाश sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = खण्डम्_आरोपय<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = खण्डम्_आरोपय<sycl::half, NBLK>(d_a, element_aligned_चिह्नकम्{});
    simd<sycl::half, NBLK>  db = 0;
    यदि (has_b) {
        qb = खण्डम्_आरोपय<int8_t, 32 * NBLK>(qs_b);
        db = खण्डम्_आरोपय<sycl::half, NBLK>(d_b, element_aligned_चिह्नकम्{});
    }

    simd<भिन्न, NBLK> da_f = convert<भिन्न>(da);
    simd<भिन्न, NBLK> db_f = convert<भिन्न>(db);

#निर्देशः unroll
    प्रति (पूर्णाङ्क s = 0; s < NBLK; ++s) {
        simd<भिन्न, 32>  y_s  = y_vec.प्रतिरूपम् select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.प्रतिरूपम् select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.प्रतिरूपम् select<32, 1>(s * 32);
        ध्रुव भिन्न sa = da_f[s];
        ध्रुव भिन्न sb = db_f[s];
        acc_a += y_s * (convert<भिन्न>(qa_s) * sa);
        acc_b += y_s * (convert<भिन्न>(qb_s) * sb);
    }
}

प्रतिरूपम् <पूर्णाङ्क WG, प्रकारनाम Policy>
ESIMD_INLINE शून्य dequantize_mul_mat_vec_q8_0_reorder_esimd(
        ध्रुव Policy & policy, ध्रुव भिन्न * y,
        ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
        sycl::local_accessor<भिन्न, 1> lmem,
        ध्रुव sycl::nd_item<1> & it) {
    उपयुज्य नामाकाश sycl::ext::intel::esimd;

    constexpr पूर्णाङ्क STRIPE = 8;

    ध्रुव पूर्णाङ्क          nblk_row = ncols / QK8_0;
    ध्रुव size_t       nb       = (size_t) nrows * nblk_row;
    ध्रुव int8_t *     qs_a     = (ध्रुव int8_t *) policy.w_a();
    ध्रुव int8_t *     qs_b     = (ध्रुव int8_t *) policy.w_b();
    ध्रुव sycl::half * d_a      = (ध्रुव sycl::half *) (qs_a + nb * QK8_0);
    ध्रुव sycl::half * d_b      = (ध्रुव sycl::half *) (qs_b + nb * QK8_0);

    ध्रुव पूर्णाङ्क  tid   = it.get_local_id(0);
    ध्रुव पूर्णाङ्क  g     = it.get_गणः(0);
    ध्रुव तार्किक has_b = policy.has_b(g, nrows);

    ध्रुव size_t आधारः_a = (size_t) policy.row_a(g) * nblk_row;
    ध्रुव size_t आधारः_b = (size_t) policy.row_b(g) * nblk_row;

    simd<भिन्न, 32> acc_a = 0.0f;
    simd<भिन्न, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    पूर्णाङ्क ib = 0;
    प्रति (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        ध्रुव पूर्णाङ्क b = ib + tid * STRIPE;
        simd<भिन्न, 256> y_vec = खण्डम्_आरोपय<भिन्न, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (आधारः_a + b) * QK8_0, qs_b + (आधारः_b + b) * QK8_0,
                                d_a + आधारः_a + b, d_b + आधारः_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    प्रति (पूर्णाङ्क b = ib + tid; b < nblk_row; b += WG) {
        simd<भिन्न, 32> y_vec = खण्डम्_आरोपय<भिन्न, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (आधारः_a + b) * QK8_0, qs_b + (आधारः_b + b) * QK8_0,
                           d_a + आधारः_a + b, d_b + आधारः_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<भिन्न>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<भिन्न>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    यदि (tid == 0) {
        भिन्न योग_a = 0.0f;
        भिन्न योग_b = 0.0f;
        प्रति (पूर्णाङ्क p = 0; p < WG; ++p) {
            योग_a += lmem[p * 2 + 0];
            योग_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, योग_a, योग_b);
    }
}

प्रतिरूपम् <पूर्णाङ्क WG, प्रकारनाम Policy>
स्थिर शून्य q8_0_esimd_launch(ध्रुव Policy & policy, ध्रुव भिन्न * y, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                              ध्रुव पूर्णाङ्क workgroups, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
प्रतिरूपम् <प्रकारनाम Policy>
स्थिर शून्य q8_0_esimd_launch_scaled(ध्रुव Policy & policy, ध्रुव भिन्न * y, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                     ध्रुव पूर्णाङ्क workgroups, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    ध्रुव पूर्णाङ्क nblk_row = ncols / QK8_0;
    यदि (nblk_row >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } अन्यथा यदि (nblk_row >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } अन्यथा यदि (nblk_row >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } अन्यथा {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

स्थिर शून्य dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                           ध्रुव पूर्णाङ्क nrows,
                                                           dpct::queue_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
प्रतिरूपम् <ggml_type T>
स्थिर शून्य dequantize_mul_mat_vec_glu_reorder_esimd(ध्रुव शून्य * vx, ध्रुव शून्य * vgate,
                                                     ध्रुव भिन्न * y, भिन्न * dst,
                                                     ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows,
                                                     ध्रुव ggml_glu_op glu_op,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<भिन्न, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_आकार), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_आकार)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

तार्किक ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(गणना ggml_type src0_type, गणना ggml_glu_op glu_op,
                                                        ध्रुव शून्य * vx, ध्रुव शून्य * vgate, ध्रुव भिन्न * y,
                                                        भिन्न * dst, पूर्णाङ्क ncols, पूर्णाङ्क nrows, dpct::queue_ptr stream) {
    विकल्प (src0_type) {
        पक्ष GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

#यदि_अन्तः // GGML_SYCL_DMMV_HAS_ESIMD

स्थिर शून्य dequantize_mul_mat_vec_q4_K_sycl_reorder(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                     भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                     ध्रुव पूर्णाङ्क nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q5_K_sycl_reorder(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                     भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                     ध्रुव पूर्णाङ्क nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

स्थिर शून्य dequantize_mul_mat_vec_q6_K_sycl_reorder(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                                     भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                                                     ध्रुव पूर्णाङ्क nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ध्रुव पूर्णाङ्क ny = 2 / K_QUANTS_प्रत्येकम्_ITERATION;
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_y = (nrows + ny - 1) / ny;
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, खण्डम्_अङ्कः_y);
    ध्रुव sycl::range<3> खण्डम्_dims(1, ny, WARP_आकार);
    stream->parallel_for(
        sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

शून्य ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ध्रुव ggml_tensor *src0, ध्रुव ggml_tensor *src1, ggml_tensor *dst,
    ध्रुव वर्ण *src0_dd_i, ध्रुव भिन्न *src1_ddf_i, ध्रुव वर्ण *src1_ddq_i,
    भिन्न *dst_dd_i, ध्रुव int64_t row_नीचः, ध्रुव int64_t row_उच्चः,
    ध्रुव int64_t src1_ncols, ध्रुव int64_t src1_padded_row_आकार,
    ध्रुव dpct::queue_ptr &stream) {

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t row_diff = row_उच्चः - row_नीचः;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#यदि_परिभाषितम् GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = शून्यसूचकः; // dfloat == half

#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    ध्रुव तार्किक q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#अन्यथा
    ध्रुव तार्किक q8_0_esimd = असत्य;
#यदि_अन्तः

    तार्किक src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    यदि (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        ध्रुव गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16_sycl = ggml_get_गन्तव्यम्_fp16_sycl(src1->type, dst);
        GGML_ASSERT(गन्तव्यम्_fp16_sycl != शून्यसूचकः);
        गन्तव्यम्_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#अन्यथा
    ध्रुव dfloat * src1_dfloat = (ध्रुव dfloat *) src1_ddf_i; // dfloat == float, no conversion
#यदि_अन्तः // GGML_SYCL_F16

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q1_0:
            यदि ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } अन्यथा {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            यदि ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } अन्यथा {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q8_0:
            यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
                यदि (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                } अन्यथा
#यदि_अन्तः
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
                }
            } अन्यथा {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q2_K:
            यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
                यदि (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                अन्यथा
#यदि_अन्तः
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } अन्यथा {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q3_K:
            यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
                यदि (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                अन्यथा
#यदि_अन्तः
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } अन्यथा {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q4_K:
            यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
                यदि (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                अन्यथा
#यदि_अन्तः
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } अन्यथा {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q5_K:
            यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
                यदि (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                अन्यथा
#यदि_अन्तः
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } अन्यथा {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_Q6_K:
            यदि ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
                यदि (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                अन्यथा
#यदि_अन्तः
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } अन्यथा {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            भङ्ग;
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            भङ्ग;
#यदि_अन्तः
        स्वभाव:
            प्रारूप_लिख("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_अप्रयुक्तम्(src1);
    GGML_अप्रयुक्तम्(dst);
    GGML_अप्रयुक्तम्(src1_ddq_i);
    GGML_अप्रयुक्तम्(src1_ncols);
    GGML_अप्रयुक्तम्(src1_padded_row_आकार);
    GGML_अप्रयुक्तम्(ctx);
}
