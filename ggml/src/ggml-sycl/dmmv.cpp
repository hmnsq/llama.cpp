#അന്തര്ഭാവ "convert.hpp"
#അന്തര്ഭാവ "dmmv.hpp"
#അന്തര്ഭാവ "dequantize.hpp"
#അന്തര്ഭാവ "presets.hpp"

#എങ്കിൽ defined(__INTEL_LLVM_COMPILER)
    #എങ്കിൽ __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #അന്തര്ഭാവ <sycl/ext/oneapi/bfloat16.hpp>
        #define GGML_SYCL_DMMV_HAS_BF16
    #endif
    #അന്തര്ഭാവ <sycl/ext/intel/esimd.hpp>
    #അന്തര്ഭാവ "esimd.hpp"
#endif

സ്ഥിതം ശൂന്യം convert_f16(സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം int64_t ib, സ്ഥിരം പൂർണ്ണസംഖ്യ iqs, dfloat2 & v){
    സ്ഥിരം sycl::half *x = (സ്ഥിരം sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ifdef GGML_SYCL_DMMV_HAS_BF16
സ്ഥിതം ശൂന്യം convert_bf16(സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം int64_t ib, സ്ഥിരം പൂർണ്ണസംഖ്യ iqs, dfloat2 & v){
    സ്ഥിരം sycl::ext::oneapi::bfloat16 *x = (സ്ഥിരം sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#endif

സ്ഥിതം ശൂന്യം convert_f32(സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം int64_t ib, സ്ഥിരം പൂർണ്ണസംഖ്യ iqs, dfloat2 & v){
    സ്ഥിരം ദശാംശം * x = (സ്ഥിരം ദശാംശം *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

template <പൂർണ്ണസംഖ്യ qk, പൂർണ്ണസംഖ്യ qr, dequantize_kernel_t dequantize_kernel>
സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec(സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം dfloat * __restrict__ y, ദശാംശം * __restrict__ dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                   സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    എങ്കിൽ (row >= nrows) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2);

    സ്ഥിരം പൂർണ്ണസംഖ്യ iter_stride = 2*GGML_SYCL_DMMV_X;
    സ്ഥിരം പൂർണ്ണസംഖ്യ vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#അല്ലെങ്കിൽ
    ദശാംശം tmp = 0.0f;
#endif // GGML_SYCL_F16

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ncols; i += iter_stride) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col = i + vals_per_iter*tid;
        എങ്കിൽ (col >= ncols) {
            തുടരുക;
        }
        സ്ഥിരം പൂർണ്ണസംഖ്യ ib = (row*ncols + col)/qk; // x block index
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs = (col%qk)/qr; // x quant index
        സ്ഥിരം പൂർണ്ണസംഖ്യ iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < vals_per_iter; j += 2) {
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
#അല്ലെങ്കിൽ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    സ്ഥിരം പൂർണ്ണസംഖ്യ mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#അല്ലെങ്കിൽ
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

template <പൂർണ്ണസംഖ്യ qk, പൂർണ്ണസംഖ്യ qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_reorder(സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം dfloat * __restrict__ y, ദശാംശം * __restrict__ dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                   സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    എങ്കിൽ (row >= nrows) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2);


    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_ഇടത് = ncols % (QK4_0*WARP_SIZE);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_align = ncols - ncols_ഇടത്;
    സ്ഥിരം പൂർണ്ണസംഖ്യ iter_stride = 8*2*GGML_SYCL_DMMV_X;
    സ്ഥിരം പൂർണ്ണസംഖ്യ vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#അല്ലെങ്കിൽ
    ദശാംശം tmp = 0.0f;
#endif // GGML_SYCL_F16
    സ്ഥിരം അക്ഷരം *d_ptr = (സ്ഥിരം അക്ഷരം*)vx+ncols*nrows/2;
    പൂർണ്ണസംഖ്യ i=0;
    വേണ്ടി (i = 0; i < ncols_align; i += iter_stride) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col = i + vals_per_iter*tid;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ib = (row*ncols + col)/qk; // x block index
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs = (col%qk)/qr; // x quant index
        സ്ഥിരം പൂർണ്ണസംഖ്യ iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((സ്ഥിരം ശൂന്യം *)d_ptr, ib, (സ്ഥിരം ശൂന്യം *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#അല്ലെങ്കിൽ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    വേണ്ടി (; i < ncols; i += iter_stride) {
        എങ്കിൽ (tid>=ncols_ഇടത്/QK4_0) തുടരുക;
        സ്ഥിരം പൂർണ്ണസംഖ്യ col = i + vals_per_iter*tid;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ib = (row*ncols + col)/qk; // x block index
        സ്ഥിരം പൂർണ്ണസംഖ്യ iqs = (col%qk)/qr; // x quant index
        സ്ഥിരം പൂർണ്ണസംഖ്യ iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((സ്ഥിരം ശൂന്യം *)d_ptr, ib, (സ്ഥിരം ശൂന്യം *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_offset]};

            tmp += v * t1;
#അല്ലെങ്കിൽ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
#endif // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    സ്ഥിരം പൂർണ്ണസംഖ്യ mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = mask_start; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (tid == 0) {
#ifdef GGML_SYCL_F16
        dst[row] = tmp.x() + tmp.y();
#അല്ലെങ്കിൽ
        dst[row] = tmp;
#endif // GGML_SYCL_F16
    }
}

സ്ഥിതം ശൂന്യം convert_mul_mat_vec_f16_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                         ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                         സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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
സ്ഥിതം ശൂന്യം convert_mul_mat_vec_bf16_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                          ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                          സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                          dpct::queue_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q2_k(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                        സ്ഥിരം ദശാംശം *__restrict__ yy,
                                        ദശാംശം *__restrict__ dst,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                        സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    സ്ഥിരം block_q2_K * x = (സ്ഥിരം block_q2_K *)vx + ib0;

    ദശാംശം tmp = 0; // partial sum for thread in warp

#എങ്കിൽ QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 16/K_QUANTS_PER_ITERATION;

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = tid % step;                           // 0...15 or 0...7

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    സ്ഥിരം uint8_t * d = (സ്ഥിരം uint8_t *)aux;
    സ്ഥിരം uint8_t * m = (സ്ഥിരം uint8_t *)(aux + 2);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        സ്ഥിരം ദശാംശം dall = x[i].dm[0];
        സ്ഥിരം ദശാംശം dmin = x[i].dm[1];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ s_offset = 8*im;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 128*im + l0;

            സ്ഥിരം ദശാംശം   * y = yy + i * QK_K + y_offset;
            സ്ഥിരം uint8_t * q = x[i].qs + q_offset;

            സ്ഥിരം uint32_t * a = (സ്ഥിരം uint32_t *)(x[i].scales + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            ദശാംശം sum1 = 0, sum2 = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
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
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2) /
                    (2 * K_QUANTS_PER_ITERATION); // 0...15 or 0...7
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = item_ct1.get_local_id(2) %
                   (2 * K_QUANTS_PER_ITERATION); // 0....1 or 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    സ്ഥിരം uint8_t * d = (സ്ഥിരം uint8_t *)uaux;


    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        സ്ഥിരം ദശാംശം   * y = yy + i * QK_K + offset;
        സ്ഥിരം uint8_t * q = x[i].qs + offset;
        സ്ഥിരം uint32_t * s = (സ്ഥിരം uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        സ്ഥിരം sycl::float2 dall =
            x[i].dm.convert<ദശാംശം, sycl::rounding_mode::automatic>();

        ദശാംശം sum1 = 0, sum2 = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            സ്ഥിരം uint8_t ql = q[l];
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q2_k_reorder(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                                സ്ഥിരം ദശാംശം *__restrict__ yy,
                                                ദശാംശം *__restrict__ dst,
                                                സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                                സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = nrows * അക്കം_blocks_per_row;
    സ്ഥിരം uint8_t     * qs_base     = (സ്ഥിരം uint8_t *)vx;
    സ്ഥിരം uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 4);
    സ്ഥിരം sycl::half2 * dm_base     = (സ്ഥിരം sycl::half2 *)(scales_base + (size_t)nb * (QK_K / 16));

    ദശാംശം tmp = 0; // partial sum for thread in warp

#എങ്കിൽ QK_K == 256
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 16/K_QUANTS_PER_ITERATION;

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = tid % step;                           // 0...15 or 0...7

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    സ്ഥിരം uint8_t * d = (സ്ഥിരം uint8_t *)aux;
    സ്ഥിരം uint8_t * m = (സ്ഥിരം uint8_t *)(aux + 2);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം sycl::half2 dm_val = dm_base[bi];
        സ്ഥിരം ദശാംശം dall = dm_val[0];
        സ്ഥിരം ദശാംശം dmin = dm_val[1];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ s_offset = 8*im;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 128*im + l0;

            സ്ഥിരം ദശാംശം   * y = yy + i * QK_K + y_offset;
            സ്ഥിരം uint8_t * q = qs_base + bi * (QK_K / 4) + q_offset;

            സ്ഥിരം uint32_t * a = (സ്ഥിരം uint32_t *)(scales_base + bi * (QK_K / 16) + s_offset);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            ദശാംശം sum1 = 0, sum2 = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
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
#അല്ലെങ്കിൽ
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#endif

    // sum up partial sums and write back result
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q3_k(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                        സ്ഥിരം ദശാംശം *__restrict__ yy,
                                        ദശാംശം *__restrict__ dst,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                        സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    സ്ഥിരം block_q3_K * x = (സ്ഥിരം block_q3_K *)vx + ib0;

    ദശാംശം tmp = 0; // partial sum for thread in warp

#എങ്കിൽ QK_K == 256

    സ്ഥിരം uint16_t kmask1 = 0x0303;
    സ്ഥിരം uint16_t kmask2 = 0x0f0f;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 16/K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ in = tid % step;                           // 0...15 or 0...7

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    സ്ഥിരം int8_t * s = (സ്ഥിരം int8_t *)utmp;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        സ്ഥിരം uint8_t * h = x[i].hmask + l0;
        സ്ഥിരം ദശാംശം d = x[i].d;

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset =  32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 128*im + l0;
            സ്ഥിരം uint16_t s_shift = 4*im;
            സ്ഥിരം uint8_t m = 1 << (4*im);

            സ്ഥിരം ദശാംശം   * y  = yy + i * QK_K + y_offset;
            സ്ഥിരം uint8_t * q = x[i].qs + q_offset;

            സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            ദശാംശം sum = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < n; ++l) {
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
#അല്ലെങ്കിൽ

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    സ്ഥിരം പൂർണ്ണസംഖ്യ in = offset/8;                                 // 0 or 1
    സ്ഥിരം പൂർണ്ണസംഖ്യ im = offset%8;                                 // 0...7

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        സ്ഥിരം ദശാംശം   * y = yy + i * QK_K + offset;
        സ്ഥിരം uint8_t * q = x[i].qs + offset;
        സ്ഥിരം uint8_t * s = x[i].scales;

        സ്ഥിരം ദശാംശം dall = (ദശാംശം)x[i].d;

        ദശാംശം sum = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            സ്ഥിരം uint8_t hl = x[i].hmask[im+l] >> in;
            സ്ഥിരം uint8_t ql = q[l];
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q3_k_reorder(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                                സ്ഥിരം ദശാംശം *__restrict__ yy,
                                                ദശാംശം *__restrict__ dst,
                                                സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                                സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = nrows * അക്കം_blocks_per_row;
    സ്ഥിരം uint8_t   * qs_base     = (സ്ഥിരം uint8_t *)vx;
    സ്ഥിരം uint8_t   * hmask_base  = qs_base + (size_t)nb * (QK_K / 4);
    സ്ഥിരം uint8_t   * scales_base = hmask_base + (size_t)nb * (QK_K / 8);
    സ്ഥിരം sycl::half * d_base     = (സ്ഥിരം sycl::half *)(scales_base + (size_t)nb * 12);

    ദശാംശം tmp = 0; // partial sum for thread in warp

#എങ്കിൽ QK_K == 256

    സ്ഥിരം uint16_t kmask1 = 0x0303;
    സ്ഥിരം uint16_t kmask2 = 0x0f0f;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    സ്ഥിരം പൂർണ്ണസംഖ്യ n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 16/K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ in = tid % step;                           // 0...15 or 0...7

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = n*in;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    സ്ഥിരം int8_t * s = (സ്ഥിരം int8_t *)utmp;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം uint8_t * h  = hmask_base + bi * (QK_K / 8) + l0;

        സ്ഥിരം ദശാംശം d = d_base[bi];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset =  32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 128*im + l0;
            സ്ഥിരം uint16_t s_shift = 4*im;
            സ്ഥിരം uint8_t m = 1 << (4*im);

            സ്ഥിരം ദശാംശം   * y  = yy + i * QK_K + y_offset;
            സ്ഥിരം uint8_t * q  = qs_base + bi * (QK_K / 4) + q_offset;

            സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)(scales_base + bi * 12);
            utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

            ദശാംശം sum = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < n; ++l) {
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
#അല്ലെങ്കിൽ
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(ncols);
    GGML_UNUSED(item_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#endif

    // sum up partial sums and write back result
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_k(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                        സ്ഥിരം ദശാംശം *__restrict__ yy,
                                        ദശാംശം *__restrict__ dst,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                        സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    സ്ഥിരം block_q4_K * x = (സ്ഥിരം block_q4_K *)vx + ib0;

#എങ്കിൽ QK_K == 256
    സ്ഥിരം uint16_t kmask1 = 0x3f3f;
    സ്ഥിരം uint16_t kmask2 = 0x0f0f;
    സ്ഥിരം uint16_t kmask3 = 0xc0c0;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    സ്ഥിരം പൂർണ്ണസംഖ്യ il_base = tid/step;                        // 0 or 1 (was 0...3)
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir  = tid - step*il_base;                  // 0...7 or 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = il_base%2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = n*(2*ir + in);

    uint16_t aux[4];
    സ്ഥിരം uint8_t * sc = (സ്ഥിരം uint8_t *)aux;

#എങ്കിൽ K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    സ്ഥിരം uint8_t * q4 = (സ്ഥിരം uint8_t *)q32;
#അല്ലെങ്കിൽ
    uint16_t q16[4];
    സ്ഥിരം uint8_t * q4 = (സ്ഥിരം uint8_t *)q16;
#endif

    ദശാംശം tmp = 0; // partial sum for thread in warp

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        സ്ഥിരം ദശാംശം dall = x[i].dm[0];
        സ്ഥിരം ദശാംശം dmin = x[i].dm[1];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 64*im + l0;

            സ്ഥിരം ദശാംശം   * y1 = yy + i*QK_K + y_offset;
            സ്ഥിരം ദശാംശം   * y2 = y1 + 128;

            സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#എങ്കിൽ K_QUANTS_PER_ITERATION == 2
            സ്ഥിരം uint32_t * q1 = (സ്ഥിരം uint32_t *)(x[i].qs + q_offset);
            സ്ഥിരം uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ദശാംശം smin = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#അല്ലെങ്കിൽ
            സ്ഥിരം uint16_t * q1 = (സ്ഥിരം uint16_t *)(x[i].qs + q_offset);
            സ്ഥിരം uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ദശാംശം smin = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    സ്ഥിരം uint8_t * s = (സ്ഥിരം uint8_t *)aux16;

    ദശാംശം tmp = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        സ്ഥിരം uint8_t * q = x[i].qs + step;
        സ്ഥിരം ദശാംശം   * y = yy + i*QK_K + step;
        സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        സ്ഥിരം ദശാംശം d = (ദശാംശം)x[i].dm[0];
        സ്ഥിരം ദശാംശം m = (ദശാംശം)x[i].dm[1];
        ദശാംശം sum = 0.f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (tid == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_k_reorder(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                                സ്ഥിരം ദശാംശം *__restrict__ yy,
                                                ദശാംശം *__restrict__ dst,
                                                സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                                സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = nrows * അക്കം_blocks_per_row;
    സ്ഥിരം uint8_t     * qs_base     = (സ്ഥിരം uint8_t *)vx;
    സ്ഥിരം uint8_t     * scales_base = qs_base + (size_t)nb * (QK_K / 2);
    സ്ഥിരം sycl::half2 * dm_base     = (സ്ഥിരം sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

#എങ്കിൽ QK_K == 256
    സ്ഥിരം uint16_t kmask1 = 0x3f3f;
    സ്ഥിരം uint16_t kmask2 = 0x0f0f;
    സ്ഥിരം uint16_t kmask3 = 0xc0c0;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0,1

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    സ്ഥിരം പൂർണ്ണസംഖ്യ il_base = tid/step;                        // 0 or 1 (was 0...3)
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir  = tid - step*il_base;                  // 0...7 or 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = il_base%2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = n*(2*ir + in);

    uint16_t aux[4];
    സ്ഥിരം uint8_t * sc = (സ്ഥിരം uint8_t *)aux;

#എങ്കിൽ K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    സ്ഥിരം uint8_t * q4 = (സ്ഥിരം uint8_t *)q32;
#അല്ലെങ്കിൽ
    uint16_t q16[4];
    സ്ഥിരം uint8_t * q4 = (സ്ഥിരം uint8_t *)q16;
#endif

    ദശാംശം tmp = 0; // partial sum for thread in warp

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം sycl::half2 dm_val = dm_base[bi];
        സ്ഥിരം ദശാംശം dall = dm_val[0];
        സ്ഥിരം ദശാംശം dmin = dm_val[1];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 64*im + l0;

            സ്ഥിരം ദശാംശം   * y1 = yy + i*QK_K + y_offset;
            സ്ഥിരം ദശാംശം   * y2 = y1 + 128;

            സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#എങ്കിൽ K_QUANTS_PER_ITERATION == 2
            സ്ഥിരം uint32_t * q1 = (സ്ഥിരം uint32_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            സ്ഥിരം uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ദശാംശം smin = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#അല്ലെങ്കിൽ
            സ്ഥിരം uint16_t * q1 = (സ്ഥിരം uint16_t *)(qs_base + bi * (QK_K / 2) + q_offset);
            സ്ഥിരം uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ദശാംശം smin = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#endif
        }

    }
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    സ്ഥിരം uint8_t * s = (സ്ഥിരം uint8_t *)aux16;

    ദശാംശം tmp = 0;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം uint8_t * q = qs_base + bi * (QK_K / 2) + step;
        സ്ഥിരം ദശാംശം   * y = yy + i*QK_K + step;
        സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)(scales_base + bi * K_SCALE_SIZE);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        സ്ഥിരം sycl::half2 dm_val = dm_base[bi];
        സ്ഥിരം ദശാംശം d = (ദശാംശം)dm_val[0];
        സ്ഥിരം ദശാംശം m = (ദശാംശം)dm_val[1];
        ദശാംശം sum = 0.f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (tid == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_k(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                        സ്ഥിരം ദശാംശം *__restrict__ yy,
                                        ദശാംശം *__restrict__ dst,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                        സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    സ്ഥിരം block_q5_K * x = (സ്ഥിരം block_q5_K *)vx + ib0;

    ദശാംശം tmp = 0; // partial sum for thread in warp

#എങ്കിൽ QK_K == 256
    സ്ഥിരം uint16_t kmask1 = 0x3f3f;
    സ്ഥിരം uint16_t kmask2 = 0x0f0f;
    സ്ഥിരം uint16_t kmask3 = 0xc0c0;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2) / 2; // 0...7
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = item_ct1.get_local_id(2) % 2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ il_base = tid/4;     // 0 or 1 (was 0...3)
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir  = tid - 4*il_base;// 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ n   = 2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = il_base%2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = n*(2*ir + in);

    uint16_t aux[4];
    സ്ഥിരം uint8_t * sc = (സ്ഥിരം uint8_t *)aux;

    uint16_t q16[8];
    സ്ഥിരം uint8_t * q4 = (സ്ഥിരം uint8_t *)q16;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2) {

        സ്ഥിരം uint8_t * qh  = x[i].qh + l0;
        സ്ഥിരം ദശാംശം dall = x[i].dm[0];
        സ്ഥിരം ദശാംശം dmin = x[i].dm[1];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 64*im + l0;

            സ്ഥിരം uint8_t hm1  = 1 << (2*im);
            സ്ഥിരം uint8_t hm2  = hm1 << 4;

            സ്ഥിരം uint8_t * ql1 = x[i].qs + q_offset;
            സ്ഥിരം ദശാംശം   * y1  = yy + i*QK_K + y_offset;
            സ്ഥിരം ദശാംശം   * y2  = y1 + 128;

            സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            ദശാംശം smin = 0;
            സ്ഥിരം uint16_t * q1 = (സ്ഥിരം uint16_t *)ql1;
            സ്ഥിരം uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < n; ++l) {
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

#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);
    സ്ഥിരം പൂർണ്ണസംഖ്യ step = tid * K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ im = step/8;
    സ്ഥിരം പൂർണ്ണസംഖ്യ in = step%8;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        സ്ഥിരം uint8_t * q = x[i].qs + step;
        സ്ഥിരം int8_t  * s = x[i].scales;
        സ്ഥിരം ദശാംശം   * y = yy + i*QK_K + step;
        സ്ഥിരം ദശാംശം     d = x[i].d;
        ദശാംശം sum = 0.f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            സ്ഥിരം uint8_t h = x[i].qh[in+j] >> im;
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_k_reorder(സ്ഥിരം ശൂന്യം *__restrict__ vx,
                                                സ്ഥിരം ദശാംശം *__restrict__ yy,
                                                ദശാംശം *__restrict__ dst,
                                                സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                                സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = nrows * അക്കം_blocks_per_row;
    സ്ഥിരം uint8_t     * qs_base     = (സ്ഥിരം uint8_t *)vx;
    സ്ഥിരം uint8_t     * qh_base     = qs_base + (size_t)nb * (QK_K / 2);
    സ്ഥിരം uint8_t     * scales_base = qh_base + (size_t)nb * (QK_K / 8);
    സ്ഥിരം sycl::half2 * dm_base     = (സ്ഥിരം sycl::half2 *)(scales_base + (size_t)nb * K_SCALE_SIZE);

    ദശാംശം tmp = 0; // partial sum for thread in warp

#എങ്കിൽ QK_K == 256
    സ്ഥിരം uint16_t kmask1 = 0x3f3f;
    സ്ഥിരം uint16_t kmask2 = 0x0f0f;
    സ്ഥിരം uint16_t kmask3 = 0xc0c0;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2) / 2; // 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix = item_ct1.get_local_id(2) % 2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ il_base = tid/4;      // 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ ir  = tid - 4*il_base;// 0...3
    സ്ഥിരം പൂർണ്ണസംഖ്യ n   = 2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = il_base%2;

    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = n*(2*ir + in);

    uint16_t aux[4];
    സ്ഥിരം uint8_t * sc = (സ്ഥിരം uint8_t *)aux;

    uint16_t q16[8];
    സ്ഥിരം uint8_t * q4 = (സ്ഥിരം uint8_t *)q16;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം uint8_t * qh  = qh_base + bi * (QK_K / 8) + l0;
        സ്ഥിരം sycl::half2 dm_val = dm_base[bi];
        സ്ഥിരം ദശാംശം dall = dm_val[0];
        സ്ഥിരം ദശാംശം dmin = dm_val[1];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ q_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 64*im + l0;

            സ്ഥിരം uint8_t hm1  = 1 << (2*im);
            സ്ഥിരം uint8_t hm2  = hm1 << 4;

            സ്ഥിരം uint8_t * ql1 = qs_base + bi * (QK_K / 2) + q_offset;
            സ്ഥിരം ദശാംശം   * y1  = yy + i*QK_K + y_offset;
            സ്ഥിരം ദശാംശം   * y2  = y1 + 128;

            സ്ഥിരം uint16_t * a = (സ്ഥിരം uint16_t *)(scales_base + bi * K_SCALE_SIZE);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 sum = {0.f, 0.f, 0.f, 0.f};
            ദശാംശം smin = 0;
            സ്ഥിരം uint16_t * q1 = (സ്ഥിരം uint16_t *)ql1;
            സ്ഥിരം uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < n; ++l) {
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
#അല്ലെങ്കിൽ
    // The reordered Q5_K layout is only produced for QK_K == 256.
#endif

    // sum up partial sums and write back result
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q6_k(സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ദശാംശം * __restrict__ yy, ദശാംശം * __restrict__ dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                        സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    സ്ഥിരം block_q6_K * x = (സ്ഥിരം block_q6_K *)vx + ib0;

#എങ്കിൽ QK_K == 256

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = tid % step;                           // 0...15 or 0...7

#എങ്കിൽ K_QUANTS_PER_ITERATION == 1
    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ is = 0;
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = 4 * in;                               // 0, 4, 8, ..., 28
    സ്ഥിരം പൂർണ്ണസംഖ്യ is = in / 4;
#endif

    ദശാംശം tmp = 0; // partial sum for thread in warp

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        സ്ഥിരം ദശാംശം d = x[i].d;

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ ql_offset = 64*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ qh_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ s_offset  =  8*im + is;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 128*im + l0;

            സ്ഥിരം ദശാംശം   * y  = yy + i * QK_K + y_offset;
            സ്ഥിരം uint8_t * ql = x[i].ql + ql_offset;
            സ്ഥിരം uint8_t * qh = x[i].qh + qh_offset;
            സ്ഥിരം int8_t  * s  = x[i].scales + s_offset;

#എങ്കിൽ K_QUANTS_PER_ITERATION == 1
            ദശാംശം sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#അല്ലെങ്കിൽ
            ദശാംശം sum = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#അല്ലെങ്കിൽ

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = tid * K_QUANTS_PER_ITERATION;

    ദശാംശം tmp = 0; // partial sum for thread in warp

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        സ്ഥിരം ദശാംശം   * y  = yy + i * QK_K + step;
        സ്ഥിരം uint8_t * ql = x[i].ql + step;
        സ്ഥിരം uint8_t * qh = x[i].qh + step;
        സ്ഥിരം int8_t  * s  = x[i].scales;

        സ്ഥിരം ദശാംശം d = x[i+0].d;

        ദശാംശം sum = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (tid == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q6_k_reorder(സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ദശാംശം * __restrict__ yy, ദശാംശം * __restrict__ dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows,
                                                സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
    എങ്കിൽ (row >= nrows) തിരികെ;

    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം പൂർണ്ണസംഖ്യ ib0 = row*അക്കം_blocks_per_row;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb = nrows * അക്കം_blocks_per_row;
    സ്ഥിരം uint8_t   * ql_base     = (സ്ഥിരം uint8_t *)vx;
    സ്ഥിരം uint8_t   * qh_base     = ql_base + (size_t)nb * (QK_K / 2);
    സ്ഥിരം int8_t    * scales_base = (സ്ഥിരം int8_t *)(qh_base + (size_t)nb * (QK_K / 4));
    സ്ഥിരം sycl::half * d_base     = (സ്ഥിരം sycl::half *)((സ്ഥിരം uint8_t *)scales_base + (size_t)nb * (QK_K / 16));

#എങ്കിൽ QK_K == 256

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid =
        item_ct1.get_local_id(2) / K_QUANTS_PER_ITERATION; // 0...7 or 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix =
        item_ct1.get_local_id(2) % K_QUANTS_PER_ITERATION; // 0 or 0, 1

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    സ്ഥിരം പൂർണ്ണസംഖ്യ in = tid % step;                           // 0...15 or 0...7

#എങ്കിൽ K_QUANTS_PER_ITERATION == 1
    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    സ്ഥിരം പൂർണ്ണസംഖ്യ is = 0;
#അല്ലെങ്കിൽ
    സ്ഥിരം പൂർണ്ണസംഖ്യ l0 = 4 * in;                               // 0, 4, 8, ..., 28
    സ്ഥിരം പൂർണ്ണസംഖ്യ is = in / 4;
#endif

    ദശാംശം tmp = 0; // partial sum for thread in warp

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += K_QUANTS_PER_ITERATION) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം ദശാംശം d = d_base[bi];

        വേണ്ടി (പൂർണ്ണസംഖ്യ im = 0; im < 2; ++im) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ ql_offset = 64*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ qh_offset = 32*im + l0;
            സ്ഥിരം പൂർണ്ണസംഖ്യ s_offset  =  8*im + is;
            സ്ഥിരം പൂർണ്ണസംഖ്യ y_offset = 128*im + l0;

            സ്ഥിരം ദശാംശം   * y  = yy + i * QK_K + y_offset;
            സ്ഥിരം uint8_t * ql = ql_base + bi * (QK_K / 2) + ql_offset;
            സ്ഥിരം uint8_t * qh = qh_base + bi * (QK_K / 4) + qh_offset;
            സ്ഥിരം int8_t  * s  = scales_base + bi * (QK_K / 16) + s_offset;

#എങ്കിൽ K_QUANTS_PER_ITERATION == 1
            ദശാംശം sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += sum;
#അല്ലെങ്കിൽ
            ദശാംശം sum = 0;
            വേണ്ടി (പൂർണ്ണസംഖ്യ l = 0; l < 4; ++l) {
                sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += sum;
#endif
        }

    }

#അല്ലെങ്കിൽ

    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2)/(2*K_QUANTS_PER_ITERATION);  // 0...7
    സ്ഥിരം പൂർണ്ണസംഖ്യ ix  = item_ct1.get_local_id(2)%(2*K_QUANTS_PER_ITERATION);  // 0...3

    സ്ഥിരം പൂർണ്ണസംഖ്യ step = tid * K_QUANTS_PER_ITERATION;

    ദശാംശം tmp = 0; // partial sum for thread in warp

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = ix; i < അക്കം_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ bi = ib0 + i;

        സ്ഥിരം ദശാംശം   * y  = yy + i * QK_K + step;
        സ്ഥിരം uint8_t * ql = ql_base + bi * (QK_K / 2) + step;
        സ്ഥിരം uint8_t * qh = qh_base + bi * (QK_K / 4) + step;
        സ്ഥിരം int8_t  * s  = scales_base + bi * (QK_K / 16);

        സ്ഥിരം ദശാംശം d = d_base[bi];

        ദശാംശം sum = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (tid == 0) {
        dst[row] = tmp;
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_0_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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


സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_0_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q1_0_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q1_0_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_0_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_1_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q8_0_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);
                എങ്കിൽ (row >= nrows) തിരികെ;

                സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2);
                സ്ഥിരം പൂർണ്ണസംഖ്യ iter_stride = 8*2*GGML_SYCL_DMMV_X;
                സ്ഥിരം പൂർണ്ണസംഖ്യ vals_per_iter = iter_stride / WARP_SIZE;
                സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_ഇടത് = ncols % (QK8_0*WARP_SIZE);
                സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_align = ncols - ncols_ഇടത്;

#ifdef GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#അല്ലെങ്കിൽ
                ദശാംശം tmp = 0.0f;
#endif
                സ്ഥിരം അക്ഷരം *d_ptr = (സ്ഥിരം അക്ഷരം*)vx + ncols*nrows;  // d after all qs

                പൂർണ്ണസംഖ്യ i = 0;
                വേണ്ടി (i = 0; i < ncols_align; i += iter_stride) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ col = i + vals_per_iter*tid;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ ib = (row*ncols + col)/QK8_0;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ iqs = col % QK8_0;

#pragma unroll
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < vals_per_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((സ്ഥിരം ശൂന്യം *)d_ptr, ib, (സ്ഥിരം ശൂന്യം *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ifdef GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#അല്ലെങ്കിൽ
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#endif
                    }
                }

                // handle remaining columns
                വേണ്ടി (; i < ncols; i += iter_stride) {
                    എങ്കിൽ (tid >= ncols_ഇടത്/QK8_0) തുടരുക;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ col = i + vals_per_iter*tid;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ ib = (row*ncols + col)/QK8_0;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ iqs = col % QK8_0;

#pragma unroll
                    വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < vals_per_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((സ്ഥിരം ശൂന്യം *)d_ptr, ib, (സ്ഥിരം ശൂന്യം *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ifdef GGML_SYCL_F16
                        dfloat2 t1{y[col + j + 0], y[col + j + 1]};
                        tmp += v * t1;
#അല്ലെങ്കിൽ
                        tmp += v.x() * y[col + j + 0];
                        tmp += v.y() * y[col + j + 1];
#endif
                    }
                }

                // reduce
                സ്ഥിരം പൂർണ്ണസംഖ്യ mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
                വേണ്ടി (പൂർണ്ണസംഖ്യ mask = mask_start; mask > 0; mask >>= 1) {
                    tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
                }

                എങ്കിൽ (tid == 0) {
#ifdef GGML_SYCL_F16
                    dst[row] = tmp.x() + tmp.y();
#അല്ലെങ്കിൽ
                    dst[row] = tmp;
#endif
                }
            });
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q8_0_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം dfloat *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q2_K_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q2_K_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                     ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                     സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q3_K_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q3_K_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                     ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                     സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_K_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_K_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q6_K_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                             ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                             സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
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
ഘടന esimd_dmmv_row_pair_policy {
    സ്ഥിരം ശൂന്യം * vx;
    ദശാംശം *      dst;

    ESIMD_INLINE സ്ഥിരം ശൂന്യം * w_a() സ്ഥിരം { തിരികെ vx; }
    ESIMD_INLINE സ്ഥിരം ശൂന്യം * w_b() സ്ഥിരം { തിരികെ vx; }

    ESIMD_INLINE പൂർണ്ണസംഖ്യ  row_a(പൂർണ്ണസംഖ്യ g) സ്ഥിരം { തിരികെ g * 2; }
    ESIMD_INLINE പൂർണ്ണസംഖ്യ  row_b(പൂർണ്ണസംഖ്യ g) സ്ഥിരം { തിരികെ g * 2 + 1; }
    ESIMD_INLINE തർക്കം has_b(പൂർണ്ണസംഖ്യ g, പൂർണ്ണസംഖ്യ nrows) സ്ഥിരം { തിരികെ row_b(g) < nrows; }

    ESIMD_INLINE ശൂന്യം epilogue(പൂർണ്ണസംഖ്യ g, പൂർണ്ണസംഖ്യ nrows, ദശാംശം sum_a, ദശാംശം sum_b) സ്ഥിരം {
        dst[row_a(g)] = sum_a;
        എങ്കിൽ (has_b(g, nrows)) {
            dst[row_b(g)] = sum_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
ഘടന esimd_dmmv_glu_policy {
    സ്ഥിരം ശൂന്യം * vgate;
    സ്ഥിരം ശൂന്യം * vup;
    ദശാംശം *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE സ്ഥിരം ശൂന്യം * w_a() സ്ഥിരം { തിരികെ vgate; }
    ESIMD_INLINE സ്ഥിരം ശൂന്യം * w_b() സ്ഥിരം { തിരികെ vup; }

    ESIMD_INLINE പൂർണ്ണസംഖ്യ  row_a(പൂർണ്ണസംഖ്യ g) സ്ഥിരം { തിരികെ g; }
    ESIMD_INLINE പൂർണ്ണസംഖ്യ  row_b(പൂർണ്ണസംഖ്യ g) സ്ഥിരം { തിരികെ g; }
    ESIMD_INLINE തർക്കം has_b(പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ) സ്ഥിരം { തിരികെ സത്യം; } // up always contributes

    ESIMD_INLINE ശൂന്യം epilogue(പൂർണ്ണസംഖ്യ g, പൂർണ്ണസംഖ്യ, ദശാംശം gate, ദശാംശം up) സ്ഥിരം {
        using നാമാകാശ sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        സ്ഥിരം ദശാംശം GELU_COEF_A    = 0.044715f;
        സ്ഥിരം ദശാംശം SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        സ്ഥിരം ദശാംശം NEG_LOG2E      = -1.44269504088896341f;

        സ്ഥിരം ദശാംശം arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<ദശാംശം, 16> t = arg * NEG_LOG2E;
        simd<ദശാംശം, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * up;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
template <ggml_type T, typename Policy>
ESIMD_INLINE ശൂന്യം dequantize_mul_mat_vec_reorder_esimd(
        സ്ഥിരം Policy & policy, സ്ഥിരം ദശാംശം * y,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
        sycl::local_accessor<ദശാംശം, 1> lmem,
        സ്ഥിരം sycl::nd_item<1> & it) {
    using നാമാകാശ sycl::ext::intel::esimd;
    using traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    സ്ഥിരം പൂർണ്ണസംഖ്യ    അക്കം_blocks_per_row = ncols / QK_K;
    സ്ഥിരം size_t nb = (size_t) nrows * അക്കം_blocks_per_row;
    സ്ഥിരം auto   pa = traits::make_ptrs(policy.w_a(), nb);
    സ്ഥിരം auto   pb = traits::make_ptrs(policy.w_b(), nb);

    സ്ഥിരം പൂർണ്ണസംഖ്യ  tid   = it.get_local_id(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ  g     = it.get_group(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ  row_a = policy.row_a(g);
    സ്ഥിരം പൂർണ്ണസംഖ്യ  row_b = policy.row_b(g);
    സ്ഥിരം തർക്കം has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<ദശാംശം, 32> acc_a = 0.0f;
    simd<ദശാംശം, 32> acc_b = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ ib = tid; ib < അക്കം_blocks_per_row; ib += GGML_SYCL_DMMV_ESIMD_WG_SIZE) {
        simd<ദശാംശം, 256> y_vec = block_load<ദശാംശം, 256>(y + (size_t) ib * QK_K);

        സ്ഥിരം size_t bi_a = (size_t) row_a * അക്കം_blocks_per_row + ib;
        സ്ഥിരം size_t bi_b = (size_t) row_b * അക്കം_blocks_per_row + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<ദശാംശം>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<ദശാംശം>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    എങ്കിൽ (tid == 0) {
        ദശാംശം sum_a = 0.0f;
        ദശാംശം sum_b = 0.0f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_SIZE; ++p) {
            sum_a += lmem[p * 2 + 0];
            sum_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, sum_a, sum_b);
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
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
template <പൂർണ്ണസംഖ്യ NBLK>
ESIMD_INLINE ശൂന്യം q8_0_mac_stripe(
        സ്ഥിരം int8_t * qs_a, സ്ഥിരം int8_t * qs_b,
        സ്ഥിരം sycl::half * d_a, സ്ഥിരം sycl::half * d_b, തർക്കം has_b,
        sycl::ext::intel::esimd::simd<ദശാംശം, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<ദശാംശം, 32> & acc_a,
        sycl::ext::intel::esimd::simd<ദശാംശം, 32> & acc_b) {
    using നാമാകാശ sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = block_load<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = block_load<sycl::half, NBLK>(d_a, element_aligned_tag{});
    simd<sycl::half, NBLK>  db = 0;
    എങ്കിൽ (has_b) {
        qb = block_load<int8_t, 32 * NBLK>(qs_b);
        db = block_load<sycl::half, NBLK>(d_b, element_aligned_tag{});
    }

    simd<ദശാംശം, NBLK> da_f = convert<ദശാംശം>(da);
    simd<ദശാംശം, NBLK> db_f = convert<ദശാംശം>(db);

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ s = 0; s < NBLK; ++s) {
        simd<ദശാംശം, 32>  y_s  = y_vec.template select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.template select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.template select<32, 1>(s * 32);
        സ്ഥിരം ദശാംശം sa = da_f[s];
        സ്ഥിരം ദശാംശം sb = db_f[s];
        acc_a += y_s * (convert<ദശാംശം>(qa_s) * sa);
        acc_b += y_s * (convert<ദശാംശം>(qb_s) * sb);
    }
}

template <പൂർണ്ണസംഖ്യ WG, typename Policy>
ESIMD_INLINE ശൂന്യം dequantize_mul_mat_vec_q8_0_reorder_esimd(
        സ്ഥിരം Policy & policy, സ്ഥിരം ദശാംശം * y,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
        sycl::local_accessor<ദശാംശം, 1> lmem,
        സ്ഥിരം sycl::nd_item<1> & it) {
    using നാമാകാശ sycl::ext::intel::esimd;

    constexpr പൂർണ്ണസംഖ്യ STRIPE = 8;

    സ്ഥിരം പൂർണ്ണസംഖ്യ          nblk_row = ncols / QK8_0;
    സ്ഥിരം size_t       nb       = (size_t) nrows * nblk_row;
    സ്ഥിരം int8_t *     qs_a     = (സ്ഥിരം int8_t *) policy.w_a();
    സ്ഥിരം int8_t *     qs_b     = (സ്ഥിരം int8_t *) policy.w_b();
    സ്ഥിരം sycl::half * d_a      = (സ്ഥിരം sycl::half *) (qs_a + nb * QK8_0);
    സ്ഥിരം sycl::half * d_b      = (സ്ഥിരം sycl::half *) (qs_b + nb * QK8_0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ  tid   = it.get_local_id(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ  g     = it.get_group(0);
    സ്ഥിരം തർക്കം has_b = policy.has_b(g, nrows);

    സ്ഥിരം size_t base_a = (size_t) policy.row_a(g) * nblk_row;
    സ്ഥിരം size_t base_b = (size_t) policy.row_b(g) * nblk_row;

    simd<ദശാംശം, 32> acc_a = 0.0f;
    simd<ദശാംശം, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    പൂർണ്ണസംഖ്യ ib = 0;
    വേണ്ടി (; ib + WG * STRIPE <= nblk_row; ib += WG * STRIPE) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ b = ib + tid * STRIPE;
        simd<ദശാംശം, 256> y_vec = block_load<ദശാംശം, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (base_a + b) * QK8_0, qs_b + (base_b + b) * QK8_0,
                                d_a + base_a + b, d_b + base_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    വേണ്ടി (പൂർണ്ണസംഖ്യ b = ib + tid; b < nblk_row; b += WG) {
        simd<ദശാംശം, 32> y_vec = block_load<ദശാംശം, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (base_a + b) * QK8_0, qs_b + (base_b + b) * QK8_0,
                           d_a + base_a + b, d_b + base_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<ദശാംശം>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<ദശാംശം>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    എങ്കിൽ (tid == 0) {
        ദശാംശം sum_a = 0.0f;
        ദശാംശം sum_b = 0.0f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ p = 0; p < WG; ++p) {
            sum_a += lmem[p * 2 + 0];
            sum_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, sum_a, sum_b);
    }
}

template <പൂർണ്ണസംഖ്യ WG, typename Policy>
സ്ഥിതം ശൂന്യം q8_0_esimd_launch(സ്ഥിരം Policy & policy, സ്ഥിരം ദശാംശം * y, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                              സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups, dpct::queue_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
template <typename Policy>
സ്ഥിതം ശൂന്യം q8_0_esimd_launch_scaled(സ്ഥിരം Policy & policy, സ്ഥിരം ദശാംശം * y, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                     സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblk_row = ncols / QK8_0;
    എങ്കിൽ (nblk_row >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } അല്ലെങ്കിൽ എങ്കിൽ (nblk_row >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } അല്ലെങ്കിൽ എങ്കിൽ (nblk_row >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } അല്ലെങ്കിൽ {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                           dpct::queue_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_row_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
template <ggml_type T>
സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_glu_reorder_esimd(സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vgate,
                                                     സ്ഥിരം ദശാംശം * y, ദശാംശം * dst,
                                                     സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                     സ്ഥിരം ggml_glu_op glu_op,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<ദശാംശം, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_SIZE), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_SIZE)),
            [=](sycl::nd_item<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

തർക്കം ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(എണ്ണൽ ggml_type src0_type, എണ്ണൽ ggml_glu_op glu_op,
                                                        സ്ഥിരം ശൂന്യം * vx, സ്ഥിരം ശൂന്യം * vgate, സ്ഥിരം ദശാംശം * y,
                                                        ദശാംശം * dst, പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ nrows, dpct::queue_ptr stream) {
    തിരഞ്ഞെടുപ്പ് (src0_type) {
        അവസ്ഥ GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            തിരികെ സത്യം;
        അവസ്ഥ GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            തിരികെ സത്യം;
        അവസ്ഥ GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            തിരികെ സത്യം;
        അവസ്ഥ GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            തിരികെ സത്യം;
        അവസ്ഥ GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            തിരികെ സത്യം;
        അവസ്ഥ GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

#endif // GGML_SYCL_DMMV_HAS_ESIMD

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q4_K_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                     ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                     സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q5_K_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                     ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                     സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

സ്ഥിതം ശൂന്യം dequantize_mul_mat_vec_q6_K_sycl_reorder(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                                     ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                                     സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
                                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ny = 2 / K_QUANTS_PER_ITERATION;
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_y = (nrows + ny - 1) / ny;
    സ്ഥിരം sycl::range<3> block_nums(1, 1, block_അക്കം_y);
    സ്ഥിരം sycl::range<3> block_dims(1, ny, WARP_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, item_ct1);
        });
}

ശൂന്യം ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor *src0, സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
    സ്ഥിരം അക്ഷരം *src0_dd_i, സ്ഥിരം ദശാംശം *src1_ddf_i, സ്ഥിരം അക്ഷരം *src1_ddq_i,
    ദശാംശം *dst_dd_i, സ്ഥിരം int64_t row_താഴെയുള്ള, സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols, സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം dpct::queue_ptr &stream) {

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t row_diff = row_മുകളിലെ - row_താഴെയുള്ള;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ifdef GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = nullptr; // dfloat == half

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    സ്ഥിരം തർക്കം q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#അല്ലെങ്കിൽ
    സ്ഥിരം തർക്കം q8_0_esimd = അസത്യം;
#endif

    തർക്കം src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    എങ്കിൽ (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        സ്ഥിരം to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#അല്ലെങ്കിൽ
    സ്ഥിരം dfloat * src1_dfloat = (സ്ഥിരം dfloat *) src1_ddf_i; // dfloat == float, no conversion
#endif // GGML_SYCL_F16

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
            എങ്കിൽ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_0:
            എങ്കിൽ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q8_0:
            എങ്കിൽ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                എങ്കിൽ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                } അല്ലെങ്കിൽ
#endif
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
                }
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q2_K:
            എങ്കിൽ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                എങ്കിൽ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                അല്ലെങ്കിൽ
#endif
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q3_K:
            എങ്കിൽ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                എങ്കിൽ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                അല്ലെങ്കിൽ
#endif
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q4_K:
            എങ്കിൽ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                എങ്കിൽ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                അല്ലെങ്കിൽ
#endif
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q5_K:
            എങ്കിൽ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                എങ്കിൽ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                അല്ലെങ്കിൽ
#endif
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_Q6_K:
            എങ്കിൽ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
                എങ്കിൽ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
                അല്ലെങ്കിൽ
#endif
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
                }
            } അല്ലെങ്കിൽ {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stream);
            }
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            നിർത്തുക;
#ifdef GGML_SYCL_DMMV_HAS_BF16
        അവസ്ഥ GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, row_diff, stream);
            നിർത്തുക;
#endif
        സ്ഥിരസ്ഥിതി:
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
