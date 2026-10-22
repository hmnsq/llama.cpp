#அந்தர்பாவ "convert.hpp"
#அந்தர்பாவ "dmmv.hpp"
#அந்தர்பாவ "dequantize.hpp"
#அந்தர்பாவ "presets.hpp"

#எனில் defined(__INTEL_LLVM_COMPILER)
    #எனில் __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #அந்தர்பாவ <sycl/ext/oneapi/bfloat16.hpp>
        #define GGML_SYCL_DMMV_HAS_BF16
    #endif
    #அந்தர்பாவ <sycl/ext/intel/esimd.hpp>
    #அந்தர்பாவ "esimd.hpp"
#endif

நிலையான வெறுமை convert_f16(மாறிலி வெறுமை * vx, மாறிலி int64_t ib, மாறிலி முழு iqs, dfloat2 & v){
    மாறிலி sycl::half *x = (மாறிலி sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
நிலையான வெறுமை convert_bf16(மாறிலி வெறுமை * vx, மாறிலி int64_t ib, மாறிலி முழு iqs, dfloat2 & v){
    மாறிலி sycl::ext::oneapi::bfloat16 *x = (மாறிலி sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#endif

நிலையான வெறுமை convert_f32(மாறிலி வெறுமை * vx, மாறிலி int64_t ib, மாறிலி முழு iqs, dfloat2 & v){
    மாறிலி தசம * x = (மாறிலி தசம *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

template <முழு qk, முழு qr, dequantize_kernel_t dequantize_kernel>
நிலையான வெறுமை dequantize_mul_mat_vec(மாறிலி வெறுமை * __restrict__ vx, மாறிலி dfloat * __restrict__ y, தசம * __restrict__ dst, மாறிலி முழு ncols, மாறிலி முழு nrows,
                                   மாறிலி sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    எனில் (row >= nrows) {
        திரும்பு;
    }

    மாறிலி முழு tid = item_ct1.get_local_id(2);

    மாறிலி முழு iter_stride = 2*GGML_SYCL_DMMV_X;
    மாறிலி முழு vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    மாறிலி முழு y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#இல்லையெனில்
    தசம tmp = 0.0f;
#endif // GGML_SYCL_F16

    க்கு (முழு i = 0; i < ncols; i += iter_stride) {
        மாறிலி முழு col = i + vals_per_iter*tid;
        எனில் (col >= ncols) {
            தொடர்;
        }
        மாறிலி முழு ib = (row*ncols + col)/qk; // x block index
        மாறிலி முழு iqs = (col%qk)/qr; // x quant index
        மாறிலி முழு iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        க்கு (முழு j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#இல்லையெனில்
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    மாறிலி முழு mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    க்கு (முழு mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#இல்லையெனில்
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

template <முழு qk, முழு qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
நிலையான வெறுமை dequantize_mul_mat_vec_reorder(மாறிலி வெறுமை * __restrict__ vx, மாறிலி dfloat * __restrict__ y, தசம * __restrict__ dst, மாறிலி முழு ncols, மாறிலி முழு nrows,
                                   மாறிலி sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    எனில் (row >= nrows) {
        திரும்பு;
    }

    மாறிலி முழு tid = item_ct1.get_local_id(2);


    மாறிலி முழு ncols_இடது = ncols % (QK4_0*WARP_SIZE);
    மாறிலி முழு ncols_align = ncols - ncols_இடது;
    மாறிலி முழு iter_stride = 8*2*GGML_SYCL_DMMV_X;
    மாறிலி முழு vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    மாறிலி முழு y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#இல்லையெனில்
    தசம tmp = 0.0f;
#endif // GGML_SYCL_F16
    மாறிலி எழுத்து *d_ptr = (மாறிலி எழுத்து*)vx+ncols*nrows/2;
    முழு i=0;
    க்கு (i = 0; i < ncols_align; i += iter_stride) {
        மாறிலி முழு col = i + vals_per_iter*tid;
        மாறிலி முழு ib = (row*ncols + col)/qk; // x block index
        மாறிலி முழு iqs = (col%qk)/qr; // x quant index
        மாறிலி முழு iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        க்கு (முழு j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((மாறிலி வெறுமை *)d_ptr, ib, (மாறிலி வெறுமை *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#இல்லையெனில்
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    க்கு (; i < ncols; i += iter_stride) {
        எனில் (tid>=ncols_இடது/QK4_0) தொடர்;
        மாறிலி முழு col = i + vals_per_iter*tid;
        மாறிலி முழு ib = (row*ncols + col)/qk; // x block index
        மாறிலி முழு iqs = (col%qk)/qr; // x quant index
        மாறிலி முழு iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        க்கு (முழு j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((மாறிலி வெறுமை *)d_ptr, ib, (மாறிலி வெறுமை *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#இல்லையெனில்
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    மாறிலி முழு mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    க்கு (முழு mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#இல்லையெனில்
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

நிலையான வெறுமை convert_mul_mat_vec_f16_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                         தசம *dst, மாறிலி முழு ncols,
                                         மாறிலி முழு nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, item_ct1);
            });
    }
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
நிலையான வெறுமை convert_mul_mat_vec_bf16_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                          தசம *dst, மாறிலி முழு ncols,
                                          மாறிலி முழு nrows,
                                          dpct::queue_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, item_ct1);
            });
    }
}
#endif

நிலையான வெறுமை dequantize_mul_mat_vec_q2_k(மாறிலி வெறுமை *__restrict__ vx,
                                        மாறிலி தசம *__restrict__ yy,
                                        தசம *__restrict__ dst,
                                        மாறிலி முழு ncols, முழு nrows,
                                        மாறிலி sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;

    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    மாறிலி block_q2_K * x = (மாறிலி block_q2_K *)vx + ib0;

    தசம tmp = 0; // partial sum for thread in warp

#எனில் QK_K == 256
    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    மாறிலி முழு step = 16/K_QUANTS_PER_ITERATION;

    மாறிலி முழு in = tid % step;                           // 0...15 or 0...7

    மாறிலி முழு l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    மாறிலி uint8_t * d = (மாறிலி uint8_t *)aux;
    மாறிலி uint8_t * m = (மாறிலி uint8_t *)(aux + 2);

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        மாறிலி தசம dall = x[i].dm[0];
        மாறிலி தசம dmin = x[i].dm[1];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset = 32*im + l0;
            மாறிலி முழு s_offset = 8*im;
            மாறிலி முழு y_offset = 128*im + l0;

            மாறிலி தசம   * y = yy + i * QK_K + y_offset;
            மாறிலி uint8_t * q = x[i].qs + q_offset;

            மாறிலி uint32_t * a = (மாறிலி uint32_t *)(x[i].scales + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            தசம sum1 = 0, sum2 = 0;
            க்கு (முழு l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
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
#இல்லையெனில்
    மாறிலி முழு tid = item_ct1.get_local_id(2) /
                    (2 * K_QUANTS_PER_ITERATION); // 0...15 or 0...7
    மாறிலி முழு ix = item_ct1.get_local_id(2) %
                   (2 * K_QUANTS_PER_ITERATION); // 0....1 or 0...3
    மாறிலி முழு offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    மாறிலி uint8_t * d = (மாறிலி uint8_t *)uaux;


    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        மாறிலி தசம   * y = yy + i * QK_K + offset;
        மாறிலி uint8_t * q = x[i].qs + offset;
        மாறிலி uint32_t * s = (மாறிலி uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        மாறிலி sycl::float2 dall =
            x[i].dm.convert<தசம, sycl::rounding_mode::automatic>();

        தசம sum1 = 0, sum2 = 0;
        க்கு (முழு l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            மாறிலி uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q2_k_reorder(மாறிலி வெறுமை *__restrict__ vx,
                                                மாறிலி தசம *__restrict__ yy,
                                                தசம *__restrict__ dst,
                                                மாறிலி முழு ncols, முழு nrows,
                                                மாறிலி sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;

    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    மாறிலி முழு nb = nrows * இலக்கம்_blocks_per_row;
    மாறிலி uint8_t     * qs_base     = (மாறிலி uint8_t *)vx;
    மாறிலி uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 4);
    மாறிலி sycl::half2 * dm_base     = (மாறிலி sycl::half2 *)(scales_base + (size_t)nb * (QK_K / 16));

    தசம tmp = 0; // partial sum for thread in warp

#எனில் QK_K == 256
    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    மாறிலி முழு step = 16/K_QUANTS_PER_ITERATION;

    மாறிலி முழு in = tid % step;                           // 0...15 or 0...7

    மாறிலி முழு l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    மாறிலி uint8_t * d = (மாறிலி uint8_t *)aux;
    மாறிலி uint8_t * m = (மாறிலி uint8_t *)(aux + 2);

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி sycl::half2 dm_val = dm_base[bi];
        மாறிலி தசம dall = dm_val[0];
        மாறிலி தசம dmin = dm_val[1];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset = 32*im + l0;
            மாறிலி முழு s_offset = 8*im;
            மாறிலி முழு y_offset = 128*im + l0;

            மாறிலி தசம   * y = yy + i * QK_K + y_offset;
            மாறிலி uint8_t * q = qs_base + bi * (QK_K / 4) + q_offset;

            மாறிலி uint32_t * a = (மாறிலி uint32_t *)(scales_base + bi * (QK_K / 16) + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            தசம sum1 = 0, sum2 = 0;
            க்கு (முழு l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
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
#இல்லையெனில்
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q3_k(மாறிலி வெறுமை *__restrict__ vx,
                                        மாறிலி தசம *__restrict__ yy,
                                        தசம *__restrict__ dst,
                                        மாறிலி முழு ncols, முழு nrows,
                                        மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;

    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    மாறிலி block_q3_K * x = (மாறிலி block_q3_K *)vx + ib0;

    தசம tmp = 0; // partial sum for thread in warp

#எனில் QK_K == 256

    மாறிலி uint16_t kmask1 = 0x0303;
    மாறிலி uint16_t kmask2 = 0x0f0f;

    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    மாறிலி முழு n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    மாறிலி முழு step = 16/K_QUANTS_PER_ITERATION;
    மாறிலி முழு in = tid % step;                           // 0...15 or 0...7

    மாறிலி முழு l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    மாறிலி int8_t * s = (மாறிலி int8_t *)utmp;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        மாறிலி uint8_t * h = x[i].hmask + l0;
        மாறிலி தசம d = x[i].d;

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset =  32*im + l0;
            மாறிலி முழு y_offset = 128*im + l0;
            மாறிலி uint16_t s_shift = 4*im;
            மாறிலி uint8_t m = 1 << (4*im);

            மாறிலி தசம   * y  = yy + i * QK_K + y_offset;
            மாறிலி uint8_t * q = x[i].qs + q_offset;

            மாறிலி uint16_t * a = (மாறிலி uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            தசம sum = 0;
            க்கு (முழு l = 0; l < n; ++l) {
                sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * sum;
        }

    }
#இல்லையெனில்

    மாறிலி முழு tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    மாறிலி முழு ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    மாறிலி முழு offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    மாறிலி முழு in = offset/8;                                 // 0 or 1
    மாறிலி முழு im = offset%8;                                 // 0...7

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        மாறிலி தசம   * y = yy + i * QK_K + offset;
        மாறிலி uint8_t * q = x[i].qs + offset;
        மாறிலி uint8_t * s = x[i].scales;

        மாறிலி தசம dall = (தசம)x[i].d;

        தசம sum = 0;
        க்கு (முழு l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            மாறிலி uint8_t hl = x[i].hmask[im+l] >> in;
            மாறிலி uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q3_k_reorder(மாறிலி வெறுமை *__restrict__ vx,
                                                மாறிலி தசம *__restrict__ yy,
                                                தசம *__restrict__ dst,
                                                மாறிலி முழு ncols, முழு nrows,
                                                மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;

    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    மாறிலி முழு nb = nrows * இலக்கம்_blocks_per_row;
    மாறிலி uint8_t   * qs_base     = (மாறிலி uint8_t *)vx;
    மாறிலி uint8_t   * hmask_base  = qs_base + (size_t)nb * (QK_K / 4);
    மாறிலி uint8_t   * scales_base = hmask_base + (size_t)nb * (QK_K / 8);
    மாறிலி sycl::half * d_base     = (மாறிலி sycl::half *)(scales_base + (size_t)nb * 12);

    தசம tmp = 0; // partial sum for thread in warp

#எனில் QK_K == 256

    மாறிலி uint16_t kmask1 = 0x0303;
    மாறிலி uint16_t kmask2 = 0x0f0f;

    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    மாறிலி முழு n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    மாறிலி முழு step = 16/K_QUANTS_PER_ITERATION;
    மாறிலி முழு in = tid % step;                           // 0...15 or 0...7

    மாறிலி முழு l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    மாறிலி int8_t * s = (மாறிலி int8_t *)utmp;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி uint8_t * h  = hmask_base + bi * (QK_K / 8) + l0;

        மாறிலி தசம d = d_base[bi];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset =  32*im + l0;
            மாறிலி முழு y_offset = 128*im + l0;
            மாறிலி uint16_t s_shift = 4*im;
            மாறிலி uint8_t m = 1 << (4*im);

            மாறிலி தசம   * y  = yy + i * QK_K + y_offset;
            மாறிலி uint8_t * q  = qs_base + bi * (QK_K / 4) + q_offset;

            மாறிலி uint16_t * a = (மாறிலி uint16_t *)(scales_base + bi * 12);
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            தசம sum = 0;
            க்கு (முழு l = 0; l < n; ++l) {
                sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * sum;
        }
    }
#இல்லையெனில்
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q4_k(மாறிலி வெறுமை *__restrict__ vx,
                                        மாறிலி தசம *__restrict__ yy,
                                        தசம *__restrict__ dst,
                                        மாறிலி முழு ncols, முழு nrows,
                                        மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;
    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    மாறிலி block_q4_K * x = (மாறிலி block_q4_K *)vx + ib0;

#எனில் QK_K == 256
    மாறிலி uint16_t kmask1 = 0x3f3f;
    மாறிலி uint16_t kmask2 = 0x0f0f;
    மாறிலி uint16_t kmask3 = 0xc0c0;

    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    மாறிலி முழு step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    மாறிலி முழு il_base = tid/step;                        // 0 or 1 (was 0...3)
    மாறிலி முழு ir  = tid - step*il_base;                  // 0...7 or 0...3
    மாறிலி முழு n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    மாறிலி முழு in = il_base%2;

    மாறிலி முழு l0 = n*(2*ir + in);

    uint16_t aux[4];
    மாறிலி uint8_t * sc = (மாறிலி uint8_t *)aux;

#எனில் K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    மாறிலி uint8_t * q4 = (மாறிலி uint8_t *)q32;
#இல்லையெனில்
    uint16_t q16[4];
    மாறிலி uint8_t * q4 = (மாறிலி uint8_t *)q16;
#endif

    தசம tmp = 0; // partial sum for thread in warp

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        மாறிலி தசம dall = x[i].dm[0];
        மாறிலி தசம dmin = x[i].dm[1];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset = 32*im + l0;
            மாறிலி முழு y_offset = 64*im + l0;

            மாறிலி தசம   * y1 = yy + i*QK_K + y_offset;
            மாறிலி தசம   * y2 = y1 + 128;

            மாறிலி uint16_t * a = (மாறிலி uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#எனில் K_QUANTS_PER_ITERATION == 2
            மாறிலி uint32_t * q1 = (மாறிலி uint32_t *)(x[i].qs + q_offset);
            மாறிலி uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            தசம smin = 0;
            க்கு (முழு l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#இல்லையெனில்
            மாறிலி uint16_t * q1 = (மாறிலி uint16_t *)(x[i].qs + q_offset);
            மாறிலி uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            தசம smin = 0;
            க்கு (முழு l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#இல்லையெனில்
    மாறிலி முழு tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    மாறிலி முழு ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    மாறிலி முழு step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    மாறிலி uint8_t * s = (மாறிலி uint8_t *)aux16;

    தசம tmp = 0;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        மாறிலி uint8_t * q = x[i].qs + step;
        மாறிலி தசம   * y = yy + i*QK_K + step;
        மாறிலி uint16_t * a = (மாறிலி uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        மாறிலி தசம d = (தசம)x[i].dm[0];
        மாறிலி தசம m = (தசம)x[i].dm[1];
        தசம sum = 0.f;
        க்கு (முழு j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (tid == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q4_k_reorder(மாறிலி வெறுமை *__restrict__ vx,
                                                மாறிலி தசம *__restrict__ yy,
                                                தசம *__restrict__ dst,
                                                மாறிலி முழு ncols, முழு nrows,
                                                மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;
    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    மாறிலி முழு nb = nrows * இலக்கம்_blocks_per_row;
    மாறிலி uint8_t     * qs_base     = (மாறிலி uint8_t *)vx;
    மாறிலி uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 2);
    மாறிலி sycl::half2 * dm_base     = (மாறிலி sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

#எனில் QK_K == 256
    மாறிலி uint16_t kmask1 = 0x3f3f;
    மாறிலி uint16_t kmask2 = 0x0f0f;
    மாறிலி uint16_t kmask3 = 0xc0c0;

    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    மாறிலி முழு step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    மாறிலி முழு il_base = tid/step;                        // 0 or 1 (was 0...3)
    மாறிலி முழு ir  = tid - step*il_base;                  // 0...7 or 0...3
    மாறிலி முழு n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    மாறிலி முழு in = il_base%2;

    மாறிலி முழு l0 = n*(2*ir + in);

    uint16_t aux[4];
    மாறிலி uint8_t * sc = (மாறிலி uint8_t *)aux;

#எனில் K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    மாறிலி uint8_t * q4 = (மாறிலி uint8_t *)q32;
#இல்லையெனில்
    uint16_t q16[4];
    மாறிலி uint8_t * q4 = (மாறிலி uint8_t *)q16;
#endif

    தசம tmp = 0; // partial sum for thread in warp

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி sycl::half2 dm_val = dm_base[bi];
        மாறிலி தசம dall = dm_val[0];
        மாறிலி தசம dmin = dm_val[1];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset = 32*im + l0;
            மாறிலி முழு y_offset = 64*im + l0;

            மாறிலி தசம   * y1 = yy + i*QK_K + y_offset;
            மாறிலி தசம   * y2 = y1 + 128;

            மாறிலி uint16_t * a = (மாறிலி uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#எனில் K_QUANTS_PER_ITERATION == 2
            மாறிலி uint32_t * q1 = (மாறிலி uint32_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            மாறிலி uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            தசம smin = 0;
            க்கு (முழு l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#இல்லையெனில்
            மாறிலி uint16_t * q1 = (மாறிலி uint16_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            மாறிலி uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            தசம smin = 0;
            க்கு (முழு l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#இல்லையெனில்
    மாறிலி முழு tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    மாறிலி முழு ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    மாறிலி முழு step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    மாறிலி uint8_t * s = (மாறிலி uint8_t *)aux16;

    தசம tmp = 0;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி uint8_t * q = qs_base + bi * (QK_K / 2) + step;
        மாறிலி தசம   * y = yy + i*QK_K + step;
        மாறிலி uint16_t * a = (மாறிலி uint16_t *)(scales_base + bi * K_SCALE_SIZE);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        மாறிலி sycl::half2 dm_val = dm_base[bi];
        மாறிலி தசம d = (தசம)dm_val[0];
        மாறிலி தசம m = (தசம)dm_val[1];
        தசம sum = 0.f;
        க்கு (முழு j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (tid == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_k(மாறிலி வெறுமை *__restrict__ vx,
                                        மாறிலி தசம *__restrict__ yy,
                                        தசம *__restrict__ dst,
                                        மாறிலி முழு ncols, முழு nrows,
                                        மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;
    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    மாறிலி block_q5_K * x = (மாறிலி block_q5_K *)vx + ib0;

    தசம tmp = 0; // partial sum for thread in warp

#எனில் QK_K == 256
    மாறிலி uint16_t kmask1 = 0x3f3f;
    மாறிலி uint16_t kmask2 = 0x0f0f;
    மாறிலி uint16_t kmask3 = 0xc0c0;

    மாறிலி முழு tid = item_ct1.get_local_id(2) / 2; // 0...7
    மாறிலி முழு ix = item_ct1.get_local_id(2) % 2;

    மாறிலி முழு il_base = tid/4;     // 0 or 1 (was 0...3)
    மாறிலி முழு ir  = tid - 4*il_base;// 0...3
    மாறிலி முழு n   = 2;

    மாறிலி முழு in = il_base%2;

    மாறிலி முழு l0 = n*(2*ir + in);

    uint16_t aux[4];
    மாறிலி uint8_t * sc = (மாறிலி uint8_t *)aux;

    uint16_t q16[8];
    மாறிலி uint8_t * q4 = (மாறிலி uint8_t *)q16;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2) {

        மாறிலி uint8_t * qh  = x[i].qh + l0;
        மாறிலி தசம dall = x[i].dm[0];
        மாறிலி தசம dmin = x[i].dm[1];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset = 32*im + l0;
            மாறிலி முழு y_offset = 64*im + l0;

            மாறிலி uint8_t hm1  = 1 << (2*im);
            மாறிலி uint8_t hm2  = hm1 << 4;

            மாறிலி uint8_t * ql1 = x[i].qs + q_offset;
            மாறிலி தசம   * y1  = yy + i*QK_K + y_offset;
            மாறிலி தசம   * y2  = y1 + 128;

            மாறிலி uint16_t * a = (மாறிலி uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            தசம smin = 0;
            மாறிலி uint16_t * q1 = (மாறிலி uint16_t *)ql1;
            மாறிலி uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            க்கு (முழு l = 0; l < n; ++l) {
                sum.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                sum.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                sum.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                sum.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                           sum.w() * sc[5]) -
                   dmin * smin;
        }
    }

#இல்லையெனில்
    மாறிலி முழு tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    மாறிலி முழு ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);
    மாறிலி முழு step = tid * K_QUANTS_PER_ITERATION;
    மாறிலி முழு im = step/8;
    மாறிலி முழு in = step%8;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        மாறிலி uint8_t * q = x[i].qs + step;
        மாறிலி int8_t  * s = x[i].scales;
        மாறிலி தசம   * y = yy + i*QK_K + step;
        மாறிலி தசம     d = x[i].d;
        தசம sum = 0.f;
        க்கு (முழு j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            மாறிலி uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_k_reorder(மாறிலி வெறுமை *__restrict__ vx,
                                                மாறிலி தசம *__restrict__ yy,
                                                தசம *__restrict__ dst,
                                                மாறிலி முழு ncols, முழு nrows,
                                                மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;
    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    மாறிலி முழு nb = nrows * இலக்கம்_blocks_per_row;
    மாறிலி uint8_t     * qs_base     = (மாறிலி uint8_t *)vx;
    மாறிலி uint8_t     * qh_base     = qs_base + (size_t)nb * (QK_K / 2);
    மாறிலி uint8_t     * scales_base = qh_base + (size_t)nb * (QK_K / 8);
    மாறிலி sycl::half2 * dm_base     = (மாறிலி sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

    தசம tmp = 0; // partial sum for thread in warp

#எனில் QK_K == 256
    மாறிலி uint16_t kmask1 = 0x3f3f;
    மாறிலி uint16_t kmask2 = 0x0f0f;
    மாறிலி uint16_t kmask3 = 0xc0c0;

    மாறிலி முழு tid = item_ct1.get_local_id(2) / 2; // 0...15
    மாறிலி முழு ix = item_ct1.get_local_id(2) % 2;

    மாறிலி முழு il_base = tid/4;      // 0...3
    மாறிலி முழு ir  = tid - 4*il_base;// 0...3
    மாறிலி முழு n   = 2;

    மாறிலி முழு in = il_base%2;

    மாறிலி முழு l0 = n*(2*ir + in);

    uint16_t aux[4];
    மாறிலி uint8_t * sc = (மாறிலி uint8_t *)aux;

    uint16_t q16[8];
    மாறிலி uint8_t * q4 = (மாறிலி uint8_t *)q16;

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி uint8_t * qh  = qh_base + bi * (QK_K / 8) + l0;
        மாறிலி sycl::half2 dm_val = dm_base[bi];
        மாறிலி தசம dall = dm_val[0];
        மாறிலி தசம dmin = dm_val[1];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு q_offset = 32*im + l0;
            மாறிலி முழு y_offset = 64*im + l0;

            மாறிலி uint8_t hm1  = 1 << (2*im);
            மாறிலி uint8_t hm2  = hm1 << 4;

            மாறிலி uint8_t * ql1 = qs_base + bi * (QK_K / 2) + q_offset;
            மாறிலி தசம   * y1  = yy + i*QK_K + y_offset;
            மாறிலி தசம   * y2  = y1 + 128;

            மாறிலி uint16_t * a = (மாறிலி uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            தசம smin = 0;
            மாறிலி uint16_t * q1 = (மாறிலி uint16_t *)ql1;
            மாறிலி uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            க்கு (முழு l = 0; l < n; ++l) {
                sum.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                sum.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                sum.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                sum.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (sum.x() * sc[0] + sum.y() * sc[1] + sum.z() * sc[4] +
                           sum.w() * sc[5]) -
                   dmin * smin;
        }
    }
#இல்லையெனில்
    // The reordered Q5_K layout is only produced for QK_K == 256.
#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q6_k(மாறிலி வெறுமை * __restrict__ vx, மாறிலி தசம * __restrict__ yy, தசம * __restrict__ dst, மாறிலி முழு ncols, முழு nrows,
                                        மாறிலி sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;

    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    மாறிலி block_q6_K * x = (மாறிலி block_q6_K *)vx + ib0;

#எனில் QK_K == 256

    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    மாறிலி முழு step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    மாறிலி முழு in = tid % step;                           // 0...15 or 0...7

#எனில் K_QUANTS_PER_ITERATION == 1
    மாறிலி முழு l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    மாறிலி முழு is = 0;
#இல்லையெனில்
    மாறிலி முழு l0 = 4 * in;                               // 0, 4, 8, ..., 28
    மாறிலி முழு is = in / 4;
#endif

    தசம tmp = 0; // partial sum for thread in warp

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        மாறிலி தசம d = x[i].d;

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு ql_offset = 64*im + l0;
            மாறிலி முழு qh_offset = 32*im + l0;
            மாறிலி முழு s_offset  =  8*im + is;
            மாறிலி முழு y_offset = 128*im + l0;

            மாறிலி தசம   * y  = yy + i * QK_K + y_offset;
            மாறிலி uint8_t * ql = x[i].ql + ql_offset;
            மாறிலி uint8_t * qh = x[i].qh + qh_offset;
            மாறிலி int8_t  * s  = x[i].scales + s_offset;

#எனில் K_QUANTS_PER_ITERATION == 1
            தசம sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#இல்லையெனில்
            தசம sum = 0;
            க்கு (முழு l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#இல்லையெனில்

    மாறிலி முழு tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    மாறிலி முழு ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    மாறிலி முழு step = tid * K_QUANTS_PER_ITERATION;

    தசம tmp = 0; // partial sum for thread in warp

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        மாறிலி தசம   * y  = yy + i * QK_K + step;
        மாறிலி uint8_t * ql = x[i].ql + step;
        மாறிலி uint8_t * qh = x[i].qh + step;
        மாறிலி int8_t  * s  = x[i].scales;

        மாறிலி தசம d = x[i+0].d;

        தசம sum = 0;
        க்கு (முழு j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (tid == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q6_k_reorder(மாறிலி வெறுமை * __restrict__ vx, மாறிலி தசம * __restrict__ yy, தசம * __restrict__ dst, மாறிலி முழு ncols, முழு nrows,
                                                மாறிலி sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    எனில் (row >= nrows) திரும்பு;

    மாறிலி முழு இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி முழு ib0 = row*இலக்கம்_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    மாறிலி முழு nb = nrows * இலக்கம்_blocks_per_row;
    மாறிலி uint8_t   * ql_base     = (மாறிலி uint8_t *)vx;
    மாறிலி uint8_t   * qh_base     = ql_base + (size_t)nb * (QK_K / 2);
    மாறிலி int8_t    * scales_base = (மாறிலி int8_t *)(qh_base + (size_t)nb * (QK_K / 4));
    மாறிலி sycl::half * d_base     = (மாறிலி sycl::half *)((மாறிலி uint8_t *)scales_base + (size_t)nb * (QK_K / 16));

#எனில் QK_K == 256

    மாறிலி முழு tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    மாறிலி முழு ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    மாறிலி முழு step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    மாறிலி முழு in = tid % step;                           // 0...15 or 0...7

#எனில் K_QUANTS_PER_ITERATION == 1
    மாறிலி முழு l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    மாறிலி முழு is = 0;
#இல்லையெனில்
    மாறிலி முழு l0 = 4 * in;                               // 0, 4, 8, ..., 28
    மாறிலி முழு is = in / 4;
#endif

    தசம tmp = 0; // partial sum for thread in warp

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி தசம d = d_base[bi];

        க்கு (முழு im = 0; im < 2; ++im) {
            மாறிலி முழு ql_offset = 64*im + l0;
            மாறிலி முழு qh_offset = 32*im + l0;
            மாறிலி முழு s_offset  =  8*im + is;
            மாறிலி முழு y_offset = 128*im + l0;

            மாறிலி தசம   * y  = yy + i * QK_K + y_offset;
            மாறிலி uint8_t * ql = ql_base + bi * (QK_K / 2) + ql_offset;
            மாறிலி uint8_t * qh = qh_base + bi * (QK_K / 4) + qh_offset;
            மாறிலி int8_t  * s  = scales_base + bi * (QK_K / 16) + s_offset;

#எனில் K_QUANTS_PER_ITERATION == 1
            தசம sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#இல்லையெனில்
            தசம sum = 0;
            க்கு (முழு l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#இல்லையெனில்

    மாறிலி முழு tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    மாறிலி முழு ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    மாறிலி முழு step = tid * K_QUANTS_PER_ITERATION;

    தசம tmp = 0; // partial sum for thread in warp

    க்கு (முழு i = ix; i < இலக்கம்_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        மாறிலி முழு bi = ib0 + i;

        மாறிலி தசம   * y  = yy + i * QK_K + step;
        மாறிலி uint8_t * ql = ql_base + bi * (QK_K / 2) + step;
        மாறிலி uint8_t * qh = qh_base + bi * (QK_K / 4) + step;
        மாறிலி int8_t  * s  = scales_base + bi * (QK_K / 16);

        மாறிலி தசம d = d_base[bi];

        தசம sum = 0;
        க்கு (முழு j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    எனில் (tid == 0) {
        dst[row] = tmp;
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q4_0_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}


நிலையான வெறுமை dequantize_mul_mat_vec_q4_0_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q1_0_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q1_0_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q4_1_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_0_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_1_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q8_0_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                மாறிலி முழு row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
                எனில் (row >= nrows) திரும்பு;

                மாறிலி முழு tid = item_ct1.get_local_id(2);
                மாறிலி முழு iter_stride = 8*2*GGML_SYCL_DMMV_X;
                மாறிலி முழு vals_per_iter = iter_stride / WARP_SIZE;
                மாறிலி முழு ncols_இடது = ncols % (QK8_0*WARP_SIZE);
                மாறிலி முழு ncols_align = ncols - ncols_இடது;

#ifdef GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#இல்லையெனில்
                தசம tmp = 0.0f;
#endif
                மாறிலி எழுத்து *d_ptr = (மாறிலி எழுத்து*)vx + ncols*nrows;  // d after all qs

                முழு i = 0;
                க்கு (i = 0; i < ncols_align; i += iter_stride) {
                    மாறிலி முழு col = i + vals_per_iter*tid;
                    மாறிலி முழு ib = (row*ncols + col)/QK8_0;
                    மாறிலி முழு iqs = col % QK8_0;

#pragma unroll
                    க்கு (முழு j = 0; j < vals_per_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((மாறிலி வெறுமை *)d_ptr, ib, (மாறிலி வெறுமை *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ifdef GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#இல்லையெனில்
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#endif
                    }
                }

                // handle remaining columns
                க்கு (; i < ncols; i += iter_stride) {
                    எனில் (tid >= ncols_இடது/QK8_0) தொடர்;
                    மாறிலி முழு col = i + vals_per_iter*tid;
                    மாறிலி முழு ib = (row*ncols + col)/QK8_0;
                    மாறிலி முழு iqs = col % QK8_0;

#pragma unroll
                    க்கு (முழு j = 0; j < vals_per_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((மாறிலி வெறுமை *)d_ptr, ib, (மாறிலி வெறுமை *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ifdef GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#இல்லையெனில்
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#endif
                    }
                }

                // reduce
                மாறிலி முழு mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
                க்கு (முழு mask = mask_start; mask > 0; mask >>= 1) {
                    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
                }

                எனில் (tid == 0) {
#ifdef GGML_SYCL_F16
                    dst[row] = tmp.x() + tmp.y();
#இல்லையெனில்
                    dst[row] = tmp;
#endif
                }
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q8_0_sycl(மாறிலி வெறுமை *vx, மாறிலி dfloat *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    மாறிலி முழு block_இலக்கம்_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, item_ct1);
            });
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q2_K_sycl(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q2_K_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                     தசம *dst, மாறிலி முழு ncols,
                                                     மாறிலி முழு nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q3_K_sycl(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q3_K_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                     தசம *dst, மாறிலி முழு ncols,
                                                     மாறிலி முழு nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q4_K_sycl(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_K_sycl(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q6_K_sycl(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                             தசம *dst, மாறிலி முழு ncols,
                                             மாறிலி முழு nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
using ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_SIZE;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
கட்டமைப்பு esimd_dmmv_row_pair_policy {
    மாறிலி வெறுமை * vx;
    தசம *      dst;

    ESIMD_INLINE மாறிலி வெறுமை * w_a() மாறிலி { திரும்பு vx; }
    ESIMD_INLINE மாறிலி வெறுமை * w_b() மாறிலி { திரும்பு vx; }

    ESIMD_INLINE முழு  row_a(முழு g) மாறிலி { திரும்பு g * 2; }
    ESIMD_INLINE முழு  row_b(முழு g) மாறிலி { திரும்பு g * 2 + 1; }
    ESIMD_INLINE தருக்க has_b(முழு g, முழு nrows) மாறிலி { திரும்பு row_b(g) < nrows; }

    ESIMD_INLINE வெறுமை epilogue(முழு g, முழு nrows, தசம sum_a, தசம sum_b) மாறிலி {
        dst[row_a(g)] = sum_a;
        எனில் (has_b(g, nrows)) {
            dst[row_b(g)] = sum_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
கட்டமைப்பு esimd_dmmv_glu_policy {
    மாறிலி வெறுமை * vgate;
    மாறிலி வெறுமை * vup;
    தசம *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE மாறிலி வெறுமை * w_a() மாறிலி { திரும்பு vgate; }
    ESIMD_INLINE மாறிலி வெறுமை * w_b() மாறிலி { திரும்பு vup; }

    ESIMD_INLINE முழு  row_a(முழு g) மாறிலி { திரும்பு g; }
    ESIMD_INLINE முழு  row_b(முழு g) மாறிலி { திரும்பு g; }
    ESIMD_INLINE தருக்க has_b(முழு, முழு) மாறிலி { திரும்பு உண்மை; } // up always contributes

    ESIMD_INLINE வெறுமை epilogue(முழு g, முழு, தசம gate, தசம up) மாறிலி {
        using நாமாகாஶ sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        மாறிலி தசம GELU_COEF_A    = 0.044715f;
        மாறிலி தசம SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        மாறிலி தசம NEG_LOG2E      = -1.44269504088896341f;

        மாறிலி தசம arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<தசம, 16> t = arg * NEG_LOG2E;
        simd<தசம, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * up;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
template <ggml_type T, typename Policy>
ESIMD_INLINE வெறுமை dequantize_mul_mat_vec_reorder_esimd(
        மாறிலி Policy & policy, மாறிலி தசம * y,
        மாறிலி முழு ncols, மாறிலி முழு nrows,
        sycl::local_accessor<தசம, 1> lmem,
        மாறிலி sycl::nd_item<1> & it) {
    using நாமாகாஶ sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    மாறிலி முழு    இலக்கம்_blocks_per_row = ncols / QK_K;
    மாறிலி size_t nb = (size_t) nrows * இலக்கம்_blocks_per_row;
    மாறிலி auto   pa = traits::make_ptrs(policy.w_a(), nb);
    மாறிலி auto   pb = traits::make_ptrs(policy.w_b(), nb);

    மாறிலி முழு  tid   = it.get_local_id(0);
    மாறிலி முழு  g     = it.get_group(0);
    மாறிலி முழு  row_a = policy.row_a(g);
    மாறிலி முழு  row_b = policy.row_b(g);
    மாறிலி தருக்க has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<தசம, 32> acc_a = 0.0f;
    simd<தசம, 32> acc_b = 0.0f;

    க்கு (முழு ib = tid; ib < இலக்கம்_blocks_per_row; ib += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        simd<தசம, 256> y_vec = block_load<தசம, 256>(y + (size_t) ib * QK_K);

        மாறிலி size_t bi_a = (size_t) row_a * இலக்கம்_blocks_per_row + ib;
        மாறிலி size_t bi_b = (size_t) row_b * இலக்கம்_blocks_per_row + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<தசம>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<தசம>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    எனில் (tid == 0) {
        தசம sum_a = 0.0f;
        தசம sum_b = 0.0f;
        க்கு (முழு p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_SIZE; ++p) {
            sum_a += lmem[p * 2 + 0];
            sum_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, sum_a, sum_b);
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                           தசம *dst, மாறிலி முழு ncols,
                                                           மாறிலி முழு nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                           தசம *dst, மாறிலி முழு ncols,
                                                           மாறிலி முழு nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                           தசம *dst, மாறிலி முழு ncols,
                                                           மாறிலி முழு nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                           தசம *dst, மாறிலி முழு ncols,
                                                           மாறிலி முழு nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                           தசம *dst, மாறிலி முழு ncols,
                                                           மாறிலி முழு nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
template <முழு NBLK>
ESIMD_INLINE வெறுமை q8_0_mac_stripe(
        மாறிலி int8_t * qs_a, மாறிலி int8_t * qs_b,
        மாறிலி sycl::half * d_a, மாறிலி sycl::half * d_b, தருக்க has_b,
        sycl::ext::intel::esimd::simd<தசம, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<தசம, 32> & acc_a,
        sycl::ext::intel::esimd::simd<தசம, 32> & acc_b) {
    using நாமாகாஶ sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = block_load<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = block_load<sycl::half, NBLK>(d_a, element_aligned_tag{});
    simd<sycl::half, NBLK>  db = 0;
    எனில் (has_b) {
        qb = block_load<int8_t, 32 * NBLK>(qs_b);
        db = block_load<sycl::half, NBLK>(d_b, element_aligned_tag{});
    }

    simd<தசம, NBLK> da_f = convert<தசம>(da);
    simd<தசம, NBLK> db_f = convert<தசம>(db);

#pragma unroll
    க்கு (முழு s = 0; s < NBLK; ++s) {
        simd<தசம, 32>  y_s  = y_vec.template select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.template select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.template select<32, 1>(s * 32);
        மாறிலி தசம sa = da_f[s];
        மாறிலி தசம sb = db_f[s];
        acc_a += y_s * (convert<தசம>(qa_s) * sa);
        acc_b += y_s * (convert<தசம>(qb_s) * sb);
    }
}

template <முழு WG, typename Policy>
ESIMD_INLINE வெறுமை dequantize_mul_mat_vec_q8_0_reorder_esimd(
        மாறிலி Policy & policy, மாறிலி தசம * y,
        மாறிலி முழு ncols, மாறிலி முழு nrows,
        sycl::local_accessor<தசம, 1> lmem,
        மாறிலி sycl::nd_item<1> & it) {
    using நாமாகாஶ sycl::ext::intel::esimd;

    constexpr முழு STRIPE = 8;

    மாறிலி முழு          nblk_row = ncols / QK8_0;
    மாறிலி size_t       nb       = (size_t) nrows * nblk_row;
    மாறிலி int8_t *     qs_a     = (மாறிலி int8_t *) policy.w_a();
    மாறிலி int8_t *     qs_b     = (மாறிலி int8_t *) policy.w_b();
    மாறிலி sycl::half * d_a      = (மாறிலி sycl::half *) (qs_a + nb * QK8_0);
    மாறிலி sycl::half * d_b      = (மாறிலி sycl::half *) (qs_b + nb * QK8_0);

    மாறிலி முழு  tid   = it.get_local_id(0);
    மாறிலி முழு  g     = it.get_group(0);
    மாறிலி தருக்க has_b = policy.has_b(g, nrows);

    மாறிலி size_t base_a = (size_t) policy.row_a(g) * nblk_row;
    மாறிலி size_t base_b = (size_t) policy.row_b(g) * nblk_row;

    simd<தசம, 32> acc_a = 0.0f;
    simd<தசம, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    முழு ib = 0;
    க்கு (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        மாறிலி முழு b = ib + tid * STRIPE;
        simd<தசம, 256> y_vec = block_load<தசம, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (base_a + b) * QK8_0, qs_b + (base_b + b) * QK8_0,
                                d_a + base_a + b, d_b + base_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    க்கு (முழு b = ib + tid; b < nblk_row; b += WG) {
        simd<தசம, 32> y_vec = block_load<தசம, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (base_a + b) * QK8_0, qs_b + (base_b + b) * QK8_0,
                           d_a + base_a + b, d_b + base_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<தசம>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<தசம>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    எனில் (tid == 0) {
        தசம sum_a = 0.0f;
        தசம sum_b = 0.0f;
        க்கு (முழு p = 0; p < WG; ++p) {
            sum_a += lmem[p * 2 + 0];
            sum_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, sum_a, sum_b);
    }
}

template <முழு WG, typename Policy>
நிலையான வெறுமை q8_0_esimd_launch(மாறிலி Policy & policy, மாறிலி தசம * y, மாறிலி முழு ncols, மாறிலி முழு nrows,
                              மாறிலி முழு workgroups, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
template <typename Policy>
நிலையான வெறுமை q8_0_esimd_launch_scaled(மாறிலி Policy & policy, மாறிலி தசம * y, மாறிலி முழு ncols, மாறிலி முழு nrows,
                                     மாறிலி முழு workgroups, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    மாறிலி முழு nblk_row = ncols / QK8_0;
    எனில் (nblk_row >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } இல்லையெனில் எனில் (nblk_row >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } இல்லையெனில் எனில் (nblk_row >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } இல்லையெனில் {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

நிலையான வெறுமை dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                           தசம *dst, மாறிலி முழு ncols,
                                                           மாறிலி முழு nrows,
                                                           dpct::queue_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
template <ggml_type T>
நிலையான வெறுமை dequantize_mul_mat_vec_glu_reorder_esimd(மாறிலி வெறுமை * vx, மாறிலி வெறுமை * vgate,
                                                     மாறிலி தசம * y, தசம * dst,
                                                     மாறிலி முழு ncols, மாறிலி முழு nrows,
                                                     மாறிலி ggml_glu_op glu_op,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<தசம, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

தருக்க ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(எண்ணிக்கை ggml_type src0_type, எண்ணிக்கை ggml_glu_op glu_op,
                                                        மாறிலி வெறுமை * vx, மாறிலி வெறுமை * vgate, மாறிலி தசம * y,
                                                        தசம * dst, முழு ncols, முழு nrows, dpct::queue_ptr stream) {
    தேர்வு (src0_type) {
        நிலை GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            திரும்பு உண்மை;
        நிலை GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            திரும்பு உண்மை;
        நிலை GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            திரும்பு உண்மை;
        நிலை GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            திரும்பு உண்மை;
        நிலை GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            திரும்பு உண்மை;
        நிலை GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            திரும்பு உண்மை;
        இயல்பு:
            திரும்பு பொய்;
    }
}

#endif // GGML_SYCL_DMMV_HAS_ESIMD

நிலையான வெறுமை dequantize_mul_mat_vec_q4_K_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                     தசம *dst, மாறிலி முழு ncols,
                                                     மாறிலி முழு nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q5_K_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                     தசம *dst, மாறிலி முழு ncols,
                                                     மாறிலி முழு nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

நிலையான வெறுமை dequantize_mul_mat_vec_q6_K_sycl_reorder(மாறிலி வெறுமை *vx, மாறிலி தசம *y,
                                                     தசம *dst, மாறிலி முழு ncols,
                                                     மாறிலி முழு nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    மாறிலி முழு ny = 2 / K_QUANTS_PER_ITERATION;
    மாறிலி முழு block_இலக்கம்_y = (nrows + ny - 1) / ny;
    மாறிலி sycl::range<3> block_nums(1, 1, block_இலக்கம்_y);
    மாறிலி sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

வெறுமை ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    மாறிலி ggml_tensor *src0, மாறிலி ggml_tensor *src1, ggml_tensor *dst,
    மாறிலி எழுத்து *src0_dd_i, மாறிலி தசம *src1_ddf_i, மாறிலி எழுத்து *src1_ddq_i,
    தசம *dst_dd_i, மாறிலி int64_t row_கீழ், மாறிலி int64_t row_உயர்வு,
    மாறிலி int64_t src1_ncols, மாறிலி int64_t src1_padded_row_size,
    மாறிலி dpct::queue_ptr &stream) {

    மாறிலி int64_t ne00 = src0->ne[0];
    மாறிலி int64_t row_diff = row_உயர்வு - row_கீழ்;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = nullptr; // dfloat == half

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    மாறிலி தருக்க q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#இல்லையெனில்
    மாறிலி தருக்க q8_0_esimd = பொய்;
#endif

    தருக்க src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    எனில் (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        மாறிலி to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#இல்லையெனில்
    மாறிலி dfloat * src1_dfloat = (மாறிலி dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_Q1_0:
            எனில் ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q4_0:
            எனில் ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q8_0:
            எனில் ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                எனில் (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                } இல்லையெனில்
#endif
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
                }
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q2_K:
            எனில் ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                எனில் (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                இல்லையெனில்
#endif
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q3_K:
            எனில் ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                எனில் (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                இல்லையெனில்
#endif
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q4_K:
            எனில் ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                எனில் (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                இல்லையெனில்
#endif
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q5_K:
            எனில் ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                எனில் (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                இல்லையெனில்
#endif
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_Q6_K:
            எனில் ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                எனில் (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                இல்லையெனில்
#endif
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } இல்லையெனில் {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            நிறுத்து;
#ifdef GGML_SYCL_DMMV_HAS_BF16
        நிலை GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            நிறுத்து;
#endif
        இயல்பு:
            printf("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(ctx);
}
