#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dmmv.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dequantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "presets.hpp"

#ਜੇ defined(__INTEL_LLVM_COMPILER)
    #ਜੇ __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/bfloat16.hpp>
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DMMV_HAS_BF16
    #ਜੇ_ਅੰਤ
    #ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/intel/esimd.hpp>
    #ਅਨ੍ਤਰ੍ਭਾਵ "esimd.hpp"
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ convert_f16(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ int64_t ib, ਅਚਲ ਪੂਰਨ iqs, dfloat2 & v){
    ਅਚਲ sycl::half *x = (ਅਚਲ sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_BF16
ਸਥਿਰ ਖਾਲੀ convert_bf16(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ int64_t ib, ਅਚਲ ਪੂਰਨ iqs, dfloat2 & v){
    ਅਚਲ sycl::ext::oneapi::bfloat16 *x = (ਅਚਲ sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ convert_f32(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ int64_t ib, ਅਚਲ ਪੂਰਨ iqs, dfloat2 & v){
    ਅਚਲ ਦਸ਼ਮਲਵ * x = (ਅਚਲ ਦਸ਼ਮਲਵ *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dequantize_kernel>
ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ dfloat * __restrict__ y, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                   ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ ਪੂਰਨ iter_stride = 2*GGML_SYCL_DMMV_X;
    ਅਚਲ ਪੂਰਨ vals_ਪ੍ਰਤੀ_iter = iter_stride / WARP_ਮਾਪ; // num quantized vals per thread and i iter
    ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#ਨਹੀਂਤਾਂ
    ਦਸ਼ਮਲਵ tmp = 0.0f;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    ਲਈ (ਪੂਰਨ i = 0; i < ncols; i += iter_stride) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ = i + vals_ਪ੍ਰਤੀ_iter*tid;
        ਜੇ (ਖਾਨਾ >= ncols) {
            ਜਾਰੀ;
        }
        ਅਚਲ ਪੂਰਨ ib = (ਪੰਕਤੀ*ncols + ਖਾਨਾ)/qk; // x block index
        ਅਚਲ ਪੂਰਨ iqs = (ਖਾਨਾ%qk)/qr; // x quant index
        ਅਚਲ ਪੂਰਨ iybs = ਖਾਨਾ - ਖਾਨਾ%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < vals_ਪ੍ਰਤੀ_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_ਵਿੱਥ]};

            tmp += v * t1;
#ਨਹੀਂਤਾਂ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_ਵਿੱਥ];
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ਅਚਲ ਪੂਰਨ ਨਕਾਬ_ਸ਼ੁਰੂ = ncols > GGML_SYCL_DMMV_X ? WARP_ਮਾਪ >> 1 : WARP_ਮਾਪ >> 2;
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = ਨਕਾਬ_ਸ਼ੁਰੂ; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (tid == 0) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
        dst[ਪੰਕਤੀ] = tmp.x() + tmp.y();
#ਨਹੀਂਤਾਂ
        dst[ਪੰਕਤੀ] = tmp;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    }
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ dfloat * __restrict__ y, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                   ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (ਪੰਕਤੀ >= nrows) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);


    ਅਚਲ ਪੂਰਨ ncols_ਖੱਬਾ = ncols % (QK4_0*WARP_ਮਾਪ);
    ਅਚਲ ਪੂਰਨ ncols_align = ncols - ncols_ਖੱਬਾ;
    ਅਚਲ ਪੂਰਨ iter_stride = 8*2*GGML_SYCL_DMMV_X;
    ਅਚਲ ਪੂਰਨ vals_ਪ੍ਰਤੀ_iter = iter_stride / WARP_ਮਾਪ; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#ਨਹੀਂਤਾਂ
    ਦਸ਼ਮਲਵ tmp = 0.0f;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    ਅਚਲ ਅੱਖਰ *d_ptr = (ਅਚਲ ਅੱਖਰ*)vx+ncols*nrows/2;
    ਪੂਰਨ i=0;
    ਲਈ (i = 0; i < ncols_align; i += iter_stride) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ = i + vals_ਪ੍ਰਤੀ_iter*tid;
        ਅਚਲ ਪੂਰਨ ib = (ਪੰਕਤੀ*ncols + ਖਾਨਾ)/qk; // x block index
        ਅਚਲ ਪੂਰਨ iqs = (ਖਾਨਾ%qk)/qr; // x quant index
        ਅਚਲ ਪੂਰਨ iybs = ਖਾਨਾ - ਖਾਨਾ%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < vals_ਪ੍ਰਤੀ_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ਅਚਲ ਖਾਲੀ *)d_ptr, ib, (ਅਚਲ ਖਾਲੀ *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_ਵਿੱਥ]};

            tmp += v * t1;
#ਨਹੀਂਤਾਂ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_ਵਿੱਥ];
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }
    }

    ਲਈ (; i < ncols; i += iter_stride) {
        ਜੇ (tid>=ncols_ਖੱਬਾ/QK4_0) ਜਾਰੀ;
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ = i + vals_ਪ੍ਰਤੀ_iter*tid;
        ਅਚਲ ਪੂਰਨ ib = (ਪੰਕਤੀ*ncols + ਖਾਨਾ)/qk; // x block index
        ਅਚਲ ਪੂਰਨ iqs = (ਖਾਨਾ%qk)/qr; // x quant index
        ਅਚਲ ਪੂਰਨ iybs = ਖਾਨਾ - ਖਾਨਾ%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < vals_ਪ੍ਰਤੀ_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ਅਚਲ ਖਾਲੀ *)d_ptr, ib, (ਅਚਲ ਖਾਲੀ *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_ਵਿੱਥ]};

            tmp += v * t1;
#ਨਹੀਂਤਾਂ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_ਵਿੱਥ];
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ਅਚਲ ਪੂਰਨ ਨਕਾਬ_ਸ਼ੁਰੂ = ncols > GGML_SYCL_DMMV_X ? WARP_ਮਾਪ >> 1 : WARP_ਮਾਪ >> 2;
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = ਨਕਾਬ_ਸ਼ੁਰੂ; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (tid == 0) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
        dst[ਪੰਕਤੀ] = tmp.x() + tmp.y();
#ਨਹੀਂਤਾਂ
        dst[ਪੰਕਤੀ] = tmp;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
    }
}

ਸਥਿਰ ਖਾਲੀ convert_mul_mat_vec_f16_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                         ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                         ਅਚਲ ਪੂਰਨ nrows,
                                         dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, ਵਸਤੂ_ct1);
            });
    }
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_BF16
ਸਥਿਰ ਖਾਲੀ convert_mul_mat_vec_bf16_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                          ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                          ਅਚਲ ਪੂਰਨ nrows,
                                          dpct::ਕਤਾਰ_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, ਵਸਤੂ_ct1);
            });
    }
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q2_k(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                        ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                        ਦਸ਼ਮਲਵ *__restrict__ dst,
                                        ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    static_assert(16%K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਅਚਲ ਖੰਡ_q2_K * x = (ਅਚਲ ਖੰਡ_q2_K *)vx + ib0;

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

#ਜੇ QK_K == 256
    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0,1

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 16/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = tid % ਕਦਮ;                           // 0...15 or 0...7

    ਅਚਲ ਪੂਰਨ l0 = K_QUANTS_ਪ੍ਰਤੀ_ITERATION*ਵਿੱਚ;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ਅਚਲ uint8_t * d = (ਅਚਲ uint8_t *)aux;
    ਅਚਲ uint8_t * m = (ਅਚਲ uint8_t *)(aux + 2);

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ ਦਸ਼ਮਲਵ dall = x[i].dm[0];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[i].dm[1];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ s_ਵਿੱਥ = 8*im;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 128*im + l0;

            ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i * QK_K + y_ਵਿੱਥ;
            ਅਚਲ uint8_t * q = x[i].qs + q_ਵਿੱਥ;

            ਅਚਲ uint32_t * a = (ਅਚਲ uint32_t *)(x[i].scales + s_ਵਿੱਥ);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            ਦਸ਼ਮਲਵ sum1 = 0, sum2 = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++l) {
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
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2) /
                    (2 * K_QUANTS_ਪ੍ਰਤੀ_ITERATION); // 0...15 or 0...7
    ਅਚਲ ਪੂਰਨ ix = ਵਸਤੂ_ct1.get_local_id(2) %
                   (2 * K_QUANTS_ਪ੍ਰਤੀ_ITERATION); // 0....1 or 0...3
    ਅਚਲ ਪੂਰਨ ਵਿੱਥ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    uint32_t uaux[2];
    ਅਚਲ uint8_t * d = (ਅਚਲ uint8_t *)uaux;


    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i * QK_K + ਵਿੱਥ;
        ਅਚਲ uint8_t * q = x[i].qs + ਵਿੱਥ;
        ਅਚਲ uint32_t * s = (ਅਚਲ uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        ਅਚਲ sycl::float2 dall =
            x[i].dm.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

        ਦਸ਼ਮਲਵ sum1 = 0, sum2 = 0;
        ਲਈ (ਪੂਰਨ l = 0; l < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++l) {
            ਅਚਲ uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q2_k_reorder(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                                ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                                ਦਸ਼ਮਲਵ *__restrict__ dst,
                                                ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                                ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    static_assert(16%K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    ਅਚਲ ਪੂਰਨ nb = nrows * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ uint8_t     * qs_ਅਧਾਰ     = (ਅਚਲ uint8_t *)vx;
    ਅਚਲ uint8_t     * scales_ਅਧਾਰ = qs_ਅਧਾਰ + (size_t)nb * (QK_K / 4);
    ਅਚਲ sycl::half2 * dm_ਅਧਾਰ     = (ਅਚਲ sycl::half2 *)(scales_ਅਧਾਰ + (size_t)nb * (QK_K / 16));

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

#ਜੇ QK_K == 256
    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0,1

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 16/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = tid % ਕਦਮ;                           // 0...15 or 0...7

    ਅਚਲ ਪੂਰਨ l0 = K_QUANTS_ਪ੍ਰਤੀ_ITERATION*ਵਿੱਚ;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ਅਚਲ uint8_t * d = (ਅਚਲ uint8_t *)aux;
    ਅਚਲ uint8_t * m = (ਅਚਲ uint8_t *)(aux + 2);

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ sycl::half2 dm_val = dm_ਅਧਾਰ[bi];
        ਅਚਲ ਦਸ਼ਮਲਵ dall = dm_val[0];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm_val[1];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ s_ਵਿੱਥ = 8*im;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 128*im + l0;

            ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i * QK_K + y_ਵਿੱਥ;
            ਅਚਲ uint8_t * q = qs_ਅਧਾਰ + bi * (QK_K / 4) + q_ਵਿੱਥ;

            ਅਚਲ uint32_t * a = (ਅਚਲ uint32_t *)(scales_ਅਧਾਰ + bi * (QK_K / 16) + s_ਵਿੱਥ);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            ਦਸ਼ਮਲਵ sum1 = 0, sum2 = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++l) {
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
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(vx);
    GGML_ਅਣਵਰਤਿਆ(yy);
    GGML_ਅਣਵਰਤਿਆ(ncols);
    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q3_k(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                        ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                        ਦਸ਼ਮਲਵ *__restrict__ dst,
                                        ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਅਚਲ ਖੰਡ_q3_K * x = (ਅਚਲ ਖੰਡ_q3_K *)vx + ib0;

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

#ਜੇ QK_K == 256

    ਅਚਲ uint16_t kmask1 = 0x0303;
    ਅਚਲ uint16_t kmask2 = 0x0f0f;

    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0,1

    ਅਚਲ ਪੂਰਨ n  = K_QUANTS_ਪ੍ਰਤੀ_ITERATION;               // iterations in the inner loop
    ਅਚਲ ਪੂਰਨ ਕਦਮ = 16/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = tid % ਕਦਮ;                           // 0...15 or 0...7

    ਅਚਲ ਪੂਰਨ l0 = n*ਵਿੱਚ;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ਅਚਲ int8_t * s = (ਅਚਲ int8_t *)utmp;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ uint8_t * h = x[i].hmask + l0;
        ਅਚਲ ਦਸ਼ਮਲਵ d = x[i].d;

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ =  32*im + l0;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 128*im + l0;
            ਅਚਲ uint16_t s_ਖਿਸਕਾਓ = 4*im;
            ਅਚਲ uint8_t m = 1 << (4*im);

            ਅਚਲ ਦਸ਼ਮਲਵ   * y  = yy + i * QK_K + y_ਵਿੱਥ;
            ਅਚਲ uint8_t * q = x[i].qs + q_ਵਿੱਥ;

            ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[4] >> (s_ਖਿਸਕਾਓ + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[5] >> (s_ਖਿਸਕਾਓ + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[4] >> (s_ਖਿਸਕਾਓ + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[5] >> (s_ਖਿਸਕਾਓ + 2)) & kmask1) << 4);

            ਦਸ਼ਮਲਵ ਜੋੜ = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < n; ++l) {
                ਜੋੜ += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                ਜੋੜ += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * ਜੋੜ;
        }

    }
#ਨਹੀਂਤਾਂ

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2)/(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...15 or 0...7
    ਅਚਲ ਪੂਰਨ ix  = ਵਸਤੂ_ct1.get_local_id(2)%(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0....1 or 0...3
    ਅਚਲ ਪੂਰਨ ਵਿੱਥ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;         // 0...15 or 0...14
    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = ਵਿੱਥ/8;                                 // 0 or 1
    ਅਚਲ ਪੂਰਨ im = ਵਿੱਥ%8;                                 // 0...7

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i * QK_K + ਵਿੱਥ;
        ਅਚਲ uint8_t * q = x[i].qs + ਵਿੱਥ;
        ਅਚਲ uint8_t * s = x[i].scales;

        ਅਚਲ ਦਸ਼ਮਲਵ dall = (ਦਸ਼ਮਲਵ)x[i].d;

        ਦਸ਼ਮਲਵ ਜੋੜ = 0;
        ਲਈ (ਪੂਰਨ l = 0; l < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++l) {
            ਅਚਲ uint8_t hl = x[i].hmask[im+l] >> ਵਿੱਚ;
            ਅਚਲ uint8_t ql = q[l];
            ਜੋੜ += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += ਜੋੜ;
    }
#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q3_k_reorder(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                                ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                                ਦਸ਼ਮਲਵ *__restrict__ dst,
                                                ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                                ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    ਅਚਲ ਪੂਰਨ nb = nrows * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ uint8_t   * qs_ਅਧਾਰ     = (ਅਚਲ uint8_t *)vx;
    ਅਚਲ uint8_t   * hmask_ਅਧਾਰ  = qs_ਅਧਾਰ + (size_t)nb * (QK_K / 4);
    ਅਚਲ uint8_t   * scales_ਅਧਾਰ = hmask_ਅਧਾਰ + (size_t)nb * (QK_K / 8);
    ਅਚਲ sycl::half * d_ਅਧਾਰ     = (ਅਚਲ sycl::half *)(scales_ਅਧਾਰ + (size_t)nb * 12);

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

#ਜੇ QK_K == 256

    ਅਚਲ uint16_t kmask1 = 0x0303;
    ਅਚਲ uint16_t kmask2 = 0x0f0f;

    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0,1

    ਅਚਲ ਪੂਰਨ n  = K_QUANTS_ਪ੍ਰਤੀ_ITERATION;               // iterations in the inner loop
    ਅਚਲ ਪੂਰਨ ਕਦਮ = 16/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = tid % ਕਦਮ;                           // 0...15 or 0...7

    ਅਚਲ ਪੂਰਨ l0 = n*ਵਿੱਚ;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ਅਚਲ int8_t * s = (ਅਚਲ int8_t *)utmp;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ uint8_t * h  = hmask_ਅਧਾਰ + bi * (QK_K / 8) + l0;

        ਅਚਲ ਦਸ਼ਮਲਵ d = d_ਅਧਾਰ[bi];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ =  32*im + l0;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 128*im + l0;
            ਅਚਲ uint16_t s_ਖਿਸਕਾਓ = 4*im;
            ਅਚਲ uint8_t m = 1 << (4*im);

            ਅਚਲ ਦਸ਼ਮਲਵ   * y  = yy + i * QK_K + y_ਵਿੱਥ;
            ਅਚਲ uint8_t * q  = qs_ਅਧਾਰ + bi * (QK_K / 4) + q_ਵਿੱਥ;

            ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)(scales_ਅਧਾਰ + bi * 12);
            utmp[0] = ((a[0] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[4] >> (s_ਖਿਸਕਾਓ + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[5] >> (s_ਖਿਸਕਾਓ + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[4] >> (s_ਖਿਸਕਾਓ + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_ਖਿਸਕਾਓ) & kmask2) | (((a[5] >> (s_ਖਿਸਕਾਓ + 2)) & kmask1) << 4);

            ਦਸ਼ਮਲਵ ਜੋੜ = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < n; ++l) {
                ਜੋੜ += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                ਜੋੜ += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * ਜੋੜ;
        }
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(vx);
    GGML_ਅਣਵਰਤਿਆ(yy);
    GGML_ਅਣਵਰਤਿਆ(ncols);
    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_k(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                        ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                        ਦਸ਼ਮਲਵ *__restrict__ dst,
                                        ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਅਚਲ ਖੰਡ_q4_K * x = (ਅਚਲ ਖੰਡ_q4_K *)vx + ib0;

#ਜੇ QK_K == 256
    ਅਚਲ uint16_t kmask1 = 0x3f3f;
    ਅਚਲ uint16_t kmask2 = 0x0f0f;
    ਅਚਲ uint16_t kmask3 = 0xc0c0;

    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0,1

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 8/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;           // 8 or 4

    ਅਚਲ ਪੂਰਨ il_ਅਧਾਰ = tid/ਕਦਮ;                        // 0 or 1 (was 0...3)
    ਅਚਲ ਪੂਰਨ ir  = tid - ਕਦਮ*il_ਅਧਾਰ;                  // 0...7 or 0...3
    ਅਚਲ ਪੂਰਨ n   = 2 * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;          // 2 or 4

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = il_ਅਧਾਰ%2;

    ਅਚਲ ਪੂਰਨ l0 = n*(2*ir + ਵਿੱਚ);

    uint16_t aux[4];
    ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *)aux;

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 2
    uint32_t q32[4];
    ਅਚਲ uint8_t * q4 = (ਅਚਲ uint8_t *)q32;
#ਨਹੀਂਤਾਂ
    uint16_t q16[4];
    ਅਚਲ uint8_t * q4 = (ਅਚਲ uint8_t *)q16;
#ਜੇ_ਅੰਤ

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ ਦਸ਼ਮਲਵ dall = x[i].dm[0];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[i].dm[1];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 64*im + l0;

            ਅਚਲ ਦਸ਼ਮਲਵ   * y1 = yy + i*QK_K + y_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ   * y2 = y1 + 128;

            ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 2
            ਅਚਲ uint32_t * q1 = (ਅਚਲ uint32_t *)(x[i].qs + q_ਵਿੱਥ);
            ਅਚਲ uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ਦਸ਼ਮਲਵ smin = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#ਨਹੀਂਤਾਂ
            ਅਚਲ uint16_t * q1 = (ਅਚਲ uint16_t *)(x[i].qs + q_ਵਿੱਥ);
            ਅਚਲ uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ਦਸ਼ਮਲਵ smin = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#ਜੇ_ਅੰਤ
        }

    }
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2)/(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...15
    ਅਚਲ ਪੂਰਨ ix  = ਵਸਤੂ_ct1.get_local_id(2)%(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);

    ਅਚਲ ਪੂਰਨ ਕਦਮ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    uint16_t aux16[2];
    ਅਚਲ uint8_t * s = (ਅਚਲ uint8_t *)aux16;

    ਦਸ਼ਮਲਵ tmp = 0;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ uint8_t * q = x[i].qs + ਕਦਮ;
        ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i*QK_K + ਕਦਮ;
        ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].dm[0];
        ਅਚਲ ਦਸ਼ਮਲਵ m = (ਦਸ਼ਮਲਵ)x[i].dm[1];
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.f;
        ਲਈ (ਪੂਰਨ j = 0; j < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++j) {
            ਜੋੜ += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += ਜੋੜ;
    }

#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (tid == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_k_reorder(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                                ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                                ਦਸ਼ਮਲਵ *__restrict__ dst,
                                                ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                                ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ਅਚਲ ਪੂਰਨ nb = nrows * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ uint8_t     * qs_ਅਧਾਰ     = (ਅਚਲ uint8_t *)vx;
    ਅਚਲ uint8_t     * scales_ਅਧਾਰ = qs_ਅਧਾਰ + (size_t)nb * (QK_K / 2);
    ਅਚਲ sycl::half2 * dm_ਅਧਾਰ     = (ਅਚਲ sycl::half2 *)(scales_ਅਧਾਰ + (size_t)nb * K_SCALE_ਮਾਪ);

#ਜੇ QK_K == 256
    ਅਚਲ uint16_t kmask1 = 0x3f3f;
    ਅਚਲ uint16_t kmask2 = 0x0f0f;
    ਅਚਲ uint16_t kmask3 = 0xc0c0;

    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0,1

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 8/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;           // 8 or 4

    ਅਚਲ ਪੂਰਨ il_ਅਧਾਰ = tid/ਕਦਮ;                        // 0 or 1 (was 0...3)
    ਅਚਲ ਪੂਰਨ ir  = tid - ਕਦਮ*il_ਅਧਾਰ;                  // 0...7 or 0...3
    ਅਚਲ ਪੂਰਨ n   = 2 * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;          // 2 or 4

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = il_ਅਧਾਰ%2;

    ਅਚਲ ਪੂਰਨ l0 = n*(2*ir + ਵਿੱਚ);

    uint16_t aux[4];
    ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *)aux;

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 2
    uint32_t q32[4];
    ਅਚਲ uint8_t * q4 = (ਅਚਲ uint8_t *)q32;
#ਨਹੀਂਤਾਂ
    uint16_t q16[4];
    ਅਚਲ uint8_t * q4 = (ਅਚਲ uint8_t *)q16;
#ਜੇ_ਅੰਤ

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ sycl::half2 dm_val = dm_ਅਧਾਰ[bi];
        ਅਚਲ ਦਸ਼ਮਲਵ dall = dm_val[0];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm_val[1];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 64*im + l0;

            ਅਚਲ ਦਸ਼ਮਲਵ   * y1 = yy + i*QK_K + y_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ   * y2 = y1 + 128;

            ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)(scales_ਅਧਾਰ + bi * K_SCALE_ਮਾਪ);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 2
            ਅਚਲ uint32_t * q1 = (ਅਚਲ uint32_t *)(qs_ਅਧਾਰ + bi * (QK_K / 2) + q_ਵਿੱਥ);
            ਅਚਲ uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ਦਸ਼ਮਲਵ smin = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#ਨਹੀਂਤਾਂ
            ਅਚਲ uint16_t * q1 = (ਅਚਲ uint16_t *)(qs_ਅਧਾਰ + bi * (QK_K / 2) + q_ਵਿੱਥ);
            ਅਚਲ uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ਦਸ਼ਮਲਵ smin = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#ਜੇ_ਅੰਤ
        }

    }
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2)/(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...15
    ਅਚਲ ਪੂਰਨ ix  = ਵਸਤੂ_ct1.get_local_id(2)%(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);

    ਅਚਲ ਪੂਰਨ ਕਦਮ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    uint16_t aux16[2];
    ਅਚਲ uint8_t * s = (ਅਚਲ uint8_t *)aux16;

    ਦਸ਼ਮਲਵ tmp = 0;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ uint8_t * q = qs_ਅਧਾਰ + bi * (QK_K / 2) + ਕਦਮ;
        ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i*QK_K + ਕਦਮ;
        ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)(scales_ਅਧਾਰ + bi * K_SCALE_ਮਾਪ);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ਅਚਲ sycl::half2 dm_val = dm_ਅਧਾਰ[bi];
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)dm_val[0];
        ਅਚਲ ਦਸ਼ਮਲਵ m = (ਦਸ਼ਮਲਵ)dm_val[1];
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.f;
        ਲਈ (ਪੂਰਨ j = 0; j < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++j) {
            ਜੋੜ += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += ਜੋੜ;
    }

#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (tid == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_k(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                        ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                        ਦਸ਼ਮਲਵ *__restrict__ dst,
                                        ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਅਚਲ ਖੰਡ_q5_K * x = (ਅਚਲ ਖੰਡ_q5_K *)vx + ib0;

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

#ਜੇ QK_K == 256
    ਅਚਲ uint16_t kmask1 = 0x3f3f;
    ਅਚਲ uint16_t kmask2 = 0x0f0f;
    ਅਚਲ uint16_t kmask3 = 0xc0c0;

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2) / 2; // 0...7
    ਅਚਲ ਪੂਰਨ ix = ਵਸਤੂ_ct1.get_local_id(2) % 2;

    ਅਚਲ ਪੂਰਨ il_ਅਧਾਰ = tid/4;     // 0 or 1 (was 0...3)
    ਅਚਲ ਪੂਰਨ ir  = tid - 4*il_ਅਧਾਰ;// 0...3
    ਅਚਲ ਪੂਰਨ n   = 2;

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = il_ਅਧਾਰ%2;

    ਅਚਲ ਪੂਰਨ l0 = n*(2*ir + ਵਿੱਚ);

    uint16_t aux[4];
    ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *)aux;

    uint16_t q16[8];
    ਅਚਲ uint8_t * q4 = (ਅਚਲ uint8_t *)q16;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2) {

        ਅਚਲ uint8_t * qh  = x[i].qh + l0;
        ਅਚਲ ਦਸ਼ਮਲਵ dall = x[i].dm[0];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[i].dm[1];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 64*im + l0;

            ਅਚਲ uint8_t hm1  = 1 << (2*im);
            ਅਚਲ uint8_t hm2  = hm1 << 4;

            ਅਚਲ uint8_t * ql1 = x[i].qs + q_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ   * y1  = yy + i*QK_K + y_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ   * y2  = y1 + 128;

            ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 ਜੋੜ = {0.f, 0.f, 0.f, 0.f};
            ਦਸ਼ਮਲਵ smin = 0;
            ਅਚਲ uint16_t * q1 = (ਅਚਲ uint16_t *)ql1;
            ਅਚਲ uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            ਲਈ (ਪੂਰਨ l = 0; l < n; ++l) {
                ਜੋੜ.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                ਜੋੜ.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                ਜੋੜ.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                ਜੋੜ.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (ਜੋੜ.x() * sc[0] + ਜੋੜ.y() * sc[1] + ਜੋੜ.z() * sc[4] +
                           ਜੋੜ.w() * sc[5]) -
                   dmin * smin;
        }
    }

#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2)/(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...15
    ਅਚਲ ਪੂਰਨ ix  = ਵਸਤੂ_ct1.get_local_id(2)%(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);
    ਅਚਲ ਪੂਰਨ ਕਦਮ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ im = ਕਦਮ/8;
    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = ਕਦਮ%8;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ uint8_t * q = x[i].qs + ਕਦਮ;
        ਅਚਲ int8_t  * s = x[i].scales;
        ਅਚਲ ਦਸ਼ਮਲਵ   * y = yy + i*QK_K + ਕਦਮ;
        ਅਚਲ ਦਸ਼ਮਲਵ     d = x[i].d;
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.f;
        ਲਈ (ਪੂਰਨ j = 0; j < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++j) {
            ਅਚਲ uint8_t h = x[i].qh[ਵਿੱਚ+j] >> im;
            ਜੋੜ += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += ਜੋੜ;
    }
#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_k_reorder(ਅਚਲ ਖਾਲੀ *__restrict__ vx,
                                                ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ yy,
                                                ਦਸ਼ਮਲਵ *__restrict__ dst,
                                                ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                                ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ਅਚਲ ਪੂਰਨ nb = nrows * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ uint8_t     * qs_ਅਧਾਰ     = (ਅਚਲ uint8_t *)vx;
    ਅਚਲ uint8_t     * qh_ਅਧਾਰ     = qs_ਅਧਾਰ + (size_t)nb * (QK_K / 2);
    ਅਚਲ uint8_t     * scales_ਅਧਾਰ = qh_ਅਧਾਰ + (size_t)nb * (QK_K / 8);
    ਅਚਲ sycl::half2 * dm_ਅਧਾਰ     = (ਅਚਲ sycl::half2 *)(scales_ਅਧਾਰ + (size_t)nb * K_SCALE_ਮਾਪ);

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

#ਜੇ QK_K == 256
    ਅਚਲ uint16_t kmask1 = 0x3f3f;
    ਅਚਲ uint16_t kmask2 = 0x0f0f;
    ਅਚਲ uint16_t kmask3 = 0xc0c0;

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2) / 2; // 0...15
    ਅਚਲ ਪੂਰਨ ix = ਵਸਤੂ_ct1.get_local_id(2) % 2;

    ਅਚਲ ਪੂਰਨ il_ਅਧਾਰ = tid/4;      // 0...3
    ਅਚਲ ਪੂਰਨ ir  = tid - 4*il_ਅਧਾਰ;// 0...3
    ਅਚਲ ਪੂਰਨ n   = 2;

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = il_ਅਧਾਰ%2;

    ਅਚਲ ਪੂਰਨ l0 = n*(2*ir + ਵਿੱਚ);

    uint16_t aux[4];
    ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *)aux;

    uint16_t q16[8];
    ਅਚਲ uint8_t * q4 = (ਅਚਲ uint8_t *)q16;

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ uint8_t * qh  = qh_ਅਧਾਰ + bi * (QK_K / 8) + l0;
        ਅਚਲ sycl::half2 dm_val = dm_ਅਧਾਰ[bi];
        ਅਚਲ ਦਸ਼ਮਲਵ dall = dm_val[0];
        ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm_val[1];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ q_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 64*im + l0;

            ਅਚਲ uint8_t hm1  = 1 << (2*im);
            ਅਚਲ uint8_t hm2  = hm1 << 4;

            ਅਚਲ uint8_t * ql1 = qs_ਅਧਾਰ + bi * (QK_K / 2) + q_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ   * y1  = yy + i*QK_K + y_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ   * y2  = y1 + 128;

            ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)(scales_ਅਧਾਰ + bi * K_SCALE_ਮਾਪ);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 ਜੋੜ = {0.f, 0.f, 0.f, 0.f};
            ਦਸ਼ਮਲਵ smin = 0;
            ਅਚਲ uint16_t * q1 = (ਅਚਲ uint16_t *)ql1;
            ਅਚਲ uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            ਲਈ (ਪੂਰਨ l = 0; l < n; ++l) {
                ਜੋੜ.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                ਜੋੜ.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                ਜੋੜ.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                ਜੋੜ.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (ਜੋੜ.x() * sc[0] + ਜੋੜ.y() * sc[1] + ਜੋੜ.z() * sc[4] +
                           ਜੋੜ.w() * sc[5]) -
                   dmin * smin;
        }
    }
#ਨਹੀਂਤਾਂ
    // The reordered Q5_K layout is only produced for QK_K == 256.
#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q6_k(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ yy, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    static_assert(16%K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    ਅਚਲ ਖੰਡ_q6_K * x = (ਅਚਲ ਖੰਡ_q6_K *)vx + ib0;

#ਜੇ QK_K == 256

    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0, 1

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 16/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;          // 16 or 8

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = tid % ਕਦਮ;                           // 0...15 or 0...7

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 1
    ਅਚਲ ਪੂਰਨ l0 = K_QUANTS_ਪ੍ਰਤੀ_ITERATION*ਵਿੱਚ;            // 0...15
    ਅਚਲ ਪੂਰਨ is = 0;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ l0 = 4 * ਵਿੱਚ;                               // 0, 4, 8, ..., 28
    ਅਚਲ ਪੂਰਨ is = ਵਿੱਚ / 4;
#ਜੇ_ਅੰਤ

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ ਦਸ਼ਮਲਵ d = x[i].d;

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ ql_ਵਿੱਥ = 64*im + l0;
            ਅਚਲ ਪੂਰਨ qh_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ s_ਵਿੱਥ  =  8*im + is;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 128*im + l0;

            ਅਚਲ ਦਸ਼ਮਲਵ   * y  = yy + i * QK_K + y_ਵਿੱਥ;
            ਅਚਲ uint8_t * ql = x[i].ql + ql_ਵਿੱਥ;
            ਅਚਲ uint8_t * qh = x[i].qh + qh_ਵਿੱਥ;
            ਅਚਲ int8_t  * s  = x[i].scales + s_ਵਿੱਥ;

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 1
            ਦਸ਼ਮਲਵ ਜੋੜ = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += ਜੋੜ;
#ਨਹੀਂਤਾਂ
            ਦਸ਼ਮਲਵ ਜੋੜ = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਜੋੜ += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += ਜੋੜ;
#ਜੇ_ਅੰਤ
        }

    }

#ਨਹੀਂਤਾਂ

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2)/(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...7
    ਅਚਲ ਪੂਰਨ ix  = ਵਸਤੂ_ct1.get_local_id(2)%(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...3

    ਅਚਲ ਪੂਰਨ ਕਦਮ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {

        ਅਚਲ ਦਸ਼ਮਲਵ   * y  = yy + i * QK_K + ਕਦਮ;
        ਅਚਲ uint8_t * ql = x[i].ql + ਕਦਮ;
        ਅਚਲ uint8_t * qh = x[i].qh + ਕਦਮ;
        ਅਚਲ int8_t  * s  = x[i].scales;

        ਅਚਲ ਦਸ਼ਮਲਵ d = x[i+0].d;

        ਦਸ਼ਮਲਵ ਜੋੜ = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++j) {
            ਜੋੜ += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += ਜੋੜ;

    }

#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (tid == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q6_k_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ yy, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ nrows,
                                                ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    static_assert(16%K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ ਪੂਰਨ ib0 = ਪੰਕਤੀ*ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    ਅਚਲ ਪੂਰਨ nb = nrows * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ uint8_t   * ql_ਅਧਾਰ     = (ਅਚਲ uint8_t *)vx;
    ਅਚਲ uint8_t   * qh_ਅਧਾਰ     = ql_ਅਧਾਰ + (size_t)nb * (QK_K / 2);
    ਅਚਲ int8_t    * scales_ਅਧਾਰ = (ਅਚਲ int8_t *)(qh_ਅਧਾਰ + (size_t)nb * (QK_K / 4));
    ਅਚਲ sycl::half * d_ਅਧਾਰ     = (ਅਚਲ sycl::half *)((ਅਚਲ uint8_t *)scales_ਅਧਾਰ + (size_t)nb * (QK_K / 16));

#ਜੇ QK_K == 256

    ਅਚਲ ਪੂਰਨ tid =
        ਵਸਤੂ_ct1.get_local_id(2) / K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0...7 or 0...15
    ਅਚਲ ਪੂਰਨ ix =
        ਵਸਤੂ_ct1.get_local_id(2) % K_QUANTS_ਪ੍ਰਤੀ_ITERATION; // 0 or 0, 1

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 16/K_QUANTS_ਪ੍ਰਤੀ_ITERATION;          // 16 or 8

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = tid % ਕਦਮ;                           // 0...15 or 0...7

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 1
    ਅਚਲ ਪੂਰਨ l0 = K_QUANTS_ਪ੍ਰਤੀ_ITERATION*ਵਿੱਚ;            // 0...15
    ਅਚਲ ਪੂਰਨ is = 0;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਪੂਰਨ l0 = 4 * ਵਿੱਚ;                               // 0, 4, 8, ..., 28
    ਅਚਲ ਪੂਰਨ is = ਵਿੱਚ / 4;
#ਜੇ_ਅੰਤ

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ ਦਸ਼ਮਲਵ d = d_ਅਧਾਰ[bi];

        ਲਈ (ਪੂਰਨ im = 0; im < 2; ++im) {
            ਅਚਲ ਪੂਰਨ ql_ਵਿੱਥ = 64*im + l0;
            ਅਚਲ ਪੂਰਨ qh_ਵਿੱਥ = 32*im + l0;
            ਅਚਲ ਪੂਰਨ s_ਵਿੱਥ  =  8*im + is;
            ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = 128*im + l0;

            ਅਚਲ ਦਸ਼ਮਲਵ   * y  = yy + i * QK_K + y_ਵਿੱਥ;
            ਅਚਲ uint8_t * ql = ql_ਅਧਾਰ + bi * (QK_K / 2) + ql_ਵਿੱਥ;
            ਅਚਲ uint8_t * qh = qh_ਅਧਾਰ + bi * (QK_K / 4) + qh_ਵਿੱਥ;
            ਅਚਲ int8_t  * s  = scales_ਅਧਾਰ + bi * (QK_K / 16) + s_ਵਿੱਥ;

#ਜੇ K_QUANTS_ਪ੍ਰਤੀ_ITERATION == 1
            ਦਸ਼ਮਲਵ ਜੋੜ = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += ਜੋੜ;
#ਨਹੀਂਤਾਂ
            ਦਸ਼ਮਲਵ ਜੋੜ = 0;
            ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
                ਜੋੜ += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += ਜੋੜ;
#ਜੇ_ਅੰਤ
        }

    }

#ਨਹੀਂਤਾਂ

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2)/(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...7
    ਅਚਲ ਪੂਰਨ ix  = ਵਸਤੂ_ct1.get_local_id(2)%(2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION);  // 0...3

    ਅਚਲ ਪੂਰਨ ਕਦਮ = tid * K_QUANTS_ਪ੍ਰਤੀ_ITERATION;

    ਦਸ਼ਮਲਵ tmp = 0; // partial sum for thread in warp

    ਲਈ (ਪੂਰਨ i = ix; i < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; i += 2*K_QUANTS_ਪ੍ਰਤੀ_ITERATION) {
        ਅਚਲ ਪੂਰਨ bi = ib0 + i;

        ਅਚਲ ਦਸ਼ਮਲਵ   * y  = yy + i * QK_K + ਕਦਮ;
        ਅਚਲ uint8_t * ql = ql_ਅਧਾਰ + bi * (QK_K / 2) + ਕਦਮ;
        ਅਚਲ uint8_t * qh = qh_ਅਧਾਰ + bi * (QK_K / 4) + ਕਦਮ;
        ਅਚਲ int8_t  * s  = scales_ਅਧਾਰ + bi * (QK_K / 16);

        ਅਚਲ ਦਸ਼ਮਲਵ d = d_ਅਧਾਰ[bi];

        ਦਸ਼ਮਲਵ ਜੋੜ = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < K_QUANTS_ਪ੍ਰਤੀ_ITERATION; ++j) {
            ਜੋੜ += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += ਜੋੜ;

    }

#ਜੇ_ਅੰਤ

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (tid == 0) {
        dst[ਪੰਕਤੀ] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_0_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}


ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_0_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q1_0_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q1_0_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_0_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_1_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q8_0_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
                ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

                ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
                ਅਚਲ ਪੂਰਨ iter_stride = 8*2*GGML_SYCL_DMMV_X;
                ਅਚਲ ਪੂਰਨ vals_ਪ੍ਰਤੀ_iter = iter_stride / WARP_ਮਾਪ;
                ਅਚਲ ਪੂਰਨ ncols_ਖੱਬਾ = ncols % (QK8_0*WARP_ਮਾਪ);
                ਅਚਲ ਪੂਰਨ ncols_align = ncols - ncols_ਖੱਬਾ;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#ਨਹੀਂਤਾਂ
                ਦਸ਼ਮਲਵ tmp = 0.0f;
#ਜੇ_ਅੰਤ
                ਅਚਲ ਅੱਖਰ *d_ptr = (ਅਚਲ ਅੱਖਰ*)vx + ncols*nrows;  // d after all qs

                ਪੂਰਨ i = 0;
                ਲਈ (i = 0; i < ncols_align; i += iter_stride) {
                    ਅਚਲ ਪੂਰਨ ਖਾਨਾ = i + vals_ਪ੍ਰਤੀ_iter*tid;
                    ਅਚਲ ਪੂਰਨ ib = (ਪੰਕਤੀ*ncols + ਖਾਨਾ)/QK8_0;
                    ਅਚਲ ਪੂਰਨ iqs = ਖਾਨਾ % QK8_0;

#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < vals_ਪ੍ਰਤੀ_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ਅਚਲ ਖਾਲੀ *)d_ptr, ib, (ਅਚਲ ਖਾਲੀ *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
                        dfloat2 t1{y[ਖਾਨਾ + j + 0], y[ਖਾਨਾ + j + 1]};
                        tmp += v * t1;
#ਨਹੀਂਤਾਂ
                        tmp += v.x() * y[ਖਾਨਾ + j + 0];
                        tmp += v.y() * y[ਖਾਨਾ + j + 1];
#ਜੇ_ਅੰਤ
                    }
                }

                // handle remaining columns
                ਲਈ (; i < ncols; i += iter_stride) {
                    ਜੇ (tid >= ncols_ਖੱਬਾ/QK8_0) ਜਾਰੀ;
                    ਅਚਲ ਪੂਰਨ ਖਾਨਾ = i + vals_ਪ੍ਰਤੀ_iter*tid;
                    ਅਚਲ ਪੂਰਨ ib = (ਪੰਕਤੀ*ncols + ਖਾਨਾ)/QK8_0;
                    ਅਚਲ ਪੂਰਨ iqs = ਖਾਨਾ % QK8_0;

#ਹਦਾਇਤ unroll
                    ਲਈ (ਪੂਰਨ j = 0; j < vals_ਪ੍ਰਤੀ_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ਅਚਲ ਖਾਲੀ *)d_ptr, ib, (ਅਚਲ ਖਾਲੀ *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
                        dfloat2 t1{y[ਖਾਨਾ + j + 0], y[ਖਾਨਾ + j + 1]};
                        tmp += v * t1;
#ਨਹੀਂਤਾਂ
                        tmp += v.x() * y[ਖਾਨਾ + j + 0];
                        tmp += v.y() * y[ਖਾਨਾ + j + 1];
#ਜੇ_ਅੰਤ
                    }
                }

                // reduce
                ਅਚਲ ਪੂਰਨ ਨਕਾਬ_ਸ਼ੁਰੂ = ncols > GGML_SYCL_DMMV_X ? WARP_ਮਾਪ >> 1 : WARP_ਮਾਪ >> 2;
                ਲਈ (ਪੂਰਨ ਨਕਾਬ = ਨਕਾਬ_ਸ਼ੁਰੂ; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
                    tmp += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
                }

                ਜੇ (tid == 0) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
                    dst[ਪੰਕਤੀ] = tmp.x() + tmp.y();
#ਨਹੀਂਤਾਂ
                    dst[ਪੰਕਤੀ] = tmp;
#ਜੇ_ਅੰਤ
                }
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q8_0_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ dfloat *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, GGML_SYCL_MMV_Y, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q2_K_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q2_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                     ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                     ਅਚਲ ਪੂਰਨ nrows,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q3_K_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q3_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                     ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                     ਅਚਲ ਪੂਰਨ nrows,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_K_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_K_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q6_K_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                             ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                             ਅਚਲ ਪੂਰਨ nrows,
                                             dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
ਵਰਤੋਂ ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
ਬਣਤਰ esimd_dmmv_ਪੰਕਤੀ_pair_policy {
    ਅਚਲ ਖਾਲੀ * vx;
    ਦਸ਼ਮਲਵ *      dst;

    ESIMD_INLINE ਅਚਲ ਖਾਲੀ * w_a() ਅਚਲ { ਵਾਪਸ vx; }
    ESIMD_INLINE ਅਚਲ ਖਾਲੀ * w_b() ਅਚਲ { ਵਾਪਸ vx; }

    ESIMD_INLINE ਪੂਰਨ  ਪੰਕਤੀ_a(ਪੂਰਨ g) ਅਚਲ { ਵਾਪਸ g * 2; }
    ESIMD_INLINE ਪੂਰਨ  ਪੰਕਤੀ_b(ਪੂਰਨ g) ਅਚਲ { ਵਾਪਸ g * 2 + 1; }
    ESIMD_INLINE ਤਾਰਕਿਕ has_b(ਪੂਰਨ g, ਪੂਰਨ nrows) ਅਚਲ { ਵਾਪਸ ਪੰਕਤੀ_b(g) < nrows; }

    ESIMD_INLINE ਖਾਲੀ epilogue(ਪੂਰਨ g, ਪੂਰਨ nrows, ਦਸ਼ਮਲਵ ਜੋੜ_a, ਦਸ਼ਮਲਵ ਜੋੜ_b) ਅਚਲ {
        dst[ਪੰਕਤੀ_a(g)] = ਜੋੜ_a;
        ਜੇ (has_b(g, nrows)) {
            dst[ਪੰਕਤੀ_b(g)] = ਜੋੜ_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
ਬਣਤਰ esimd_dmmv_glu_policy {
    ਅਚਲ ਖਾਲੀ * vgate;
    ਅਚਲ ਖਾਲੀ * vup;
    ਦਸ਼ਮਲਵ *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE ਅਚਲ ਖਾਲੀ * w_a() ਅਚਲ { ਵਾਪਸ vgate; }
    ESIMD_INLINE ਅਚਲ ਖਾਲੀ * w_b() ਅਚਲ { ਵਾਪਸ vup; }

    ESIMD_INLINE ਪੂਰਨ  ਪੰਕਤੀ_a(ਪੂਰਨ g) ਅਚਲ { ਵਾਪਸ g; }
    ESIMD_INLINE ਪੂਰਨ  ਪੰਕਤੀ_b(ਪੂਰਨ g) ਅਚਲ { ਵਾਪਸ g; }
    ESIMD_INLINE ਤਾਰਕਿਕ has_b(ਪੂਰਨ, ਪੂਰਨ) ਅਚਲ { ਵਾਪਸ ਸੱਚ; } // up always contributes

    ESIMD_INLINE ਖਾਲੀ epilogue(ਪੂਰਨ g, ਪੂਰਨ, ਦਸ਼ਮਲਵ gate, ਦਸ਼ਮਲਵ ਉੱਪਰ) ਅਚਲ {
        ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        ਅਚਲ ਦਸ਼ਮਲਵ GELU_COEF_A    = 0.044715f;
        ਅਚਲ ਦਸ਼ਮਲਵ SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        ਅਚਲ ਦਸ਼ਮਲਵ NEG_LOG2E      = -1.44269504088896341f;

        ਅਚਲ ਦਸ਼ਮਲਵ arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<ਦਸ਼ਮਲਵ, 16> t = arg * NEG_LOG2E;
        simd<ਦਸ਼ਮਲਵ, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * ਉੱਪਰ;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
ਸਾਂਚਾ <ggml_type T, ਕਿਸਮ_ਨਾਂ Policy>
ESIMD_INLINE ਖਾਲੀ dequantize_mul_mat_vec_reorder_esimd(
        ਅਚਲ Policy & policy, ਅਚਲ ਦਸ਼ਮਲਵ * y,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem,
        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl::ext::intel::esimd;
    ਵਰਤੋਂ traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    ਅਚਲ ਪੂਰਨ    ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = ncols / QK_K;
    ਅਚਲ size_t nb = (size_t) nrows * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
    ਅਚਲ ਸਵੈਚਾਲਿਤ   pa = traits::make_ptrs(policy.w_a(), nb);
    ਅਚਲ ਸਵੈਚਾਲਿਤ   pb = traits::make_ptrs(policy.w_b(), nb);

    ਅਚਲ ਪੂਰਨ  tid   = it.get_local_id(0);
    ਅਚਲ ਪੂਰਨ  g     = it.get_ਸਮੂਹ(0);
    ਅਚਲ ਪੂਰਨ  ਪੰਕਤੀ_a = policy.ਪੰਕਤੀ_a(g);
    ਅਚਲ ਪੂਰਨ  ਪੰਕਤੀ_b = policy.ਪੰਕਤੀ_b(g);
    ਅਚਲ ਤਾਰਕਿਕ has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<ਦਸ਼ਮਲਵ, 32> acc_a = 0.0f;
    simd<ਦਸ਼ਮਲਵ, 32> acc_b = 0.0f;

    ਲਈ (ਪੂਰਨ ib = tid; ib < ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ; ib += GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ) {
        simd<ਦਸ਼ਮਲਵ, 256> y_vec = ਖੰਡ_ਲੱਦੋ<ਦਸ਼ਮਲਵ, 256>(y + (size_t) ib * QK_K);

        ਅਚਲ size_t bi_a = (size_t) ਪੰਕਤੀ_a * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + ib;
        ਅਚਲ size_t bi_b = (size_t) ਪੰਕਤੀ_b * ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<ਦਸ਼ਮਲਵ>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<ਦਸ਼ਮਲਵ>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    ਜੇ (tid == 0) {
        ਦਸ਼ਮਲਵ ਜੋੜ_a = 0.0f;
        ਦਸ਼ਮਲਵ ਜੋੜ_b = 0.0f;
        ਲਈ (ਪੂਰਨ p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ; ++p) {
            ਜੋੜ_a += lmem[p * 2 + 0];
            ਜੋੜ_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, ਜੋੜ_a, ਜੋੜ_b);
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                           ਅਚਲ ਪੂਰਨ nrows,
                                                           dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_ਪੰਕਤੀ_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                           ਅਚਲ ਪੂਰਨ nrows,
                                                           dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_ਪੰਕਤੀ_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                           ਅਚਲ ਪੂਰਨ nrows,
                                                           dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_ਪੰਕਤੀ_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                           ਅਚਲ ਪੂਰਨ nrows,
                                                           dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_ਪੰਕਤੀ_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                           ਅਚਲ ਪੂਰਨ nrows,
                                                           dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_ਪੰਕਤੀ_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
ਸਾਂਚਾ <ਪੂਰਨ NBLK>
ESIMD_INLINE ਖਾਲੀ q8_0_mac_stripe(
        ਅਚਲ int8_t * qs_a, ਅਚਲ int8_t * qs_b,
        ਅਚਲ sycl::half * d_a, ਅਚਲ sycl::half * d_b, ਤਾਰਕਿਕ has_b,
        sycl::ext::intel::esimd::simd<ਦਸ਼ਮਲਵ, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<ਦਸ਼ਮਲਵ, 32> & acc_a,
        sycl::ext::intel::esimd::simd<ਦਸ਼ਮਲਵ, 32> & acc_b) {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = ਖੰਡ_ਲੱਦੋ<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = ਖੰਡ_ਲੱਦੋ<sycl::half, NBLK>(d_a, element_aligned_ਟੈਗ{});
    simd<sycl::half, NBLK>  db = 0;
    ਜੇ (has_b) {
        qb = ਖੰਡ_ਲੱਦੋ<int8_t, 32 * NBLK>(qs_b);
        db = ਖੰਡ_ਲੱਦੋ<sycl::half, NBLK>(d_b, element_aligned_ਟੈਗ{});
    }

    simd<ਦਸ਼ਮਲਵ, NBLK> da_f = convert<ਦਸ਼ਮਲਵ>(da);
    simd<ਦਸ਼ਮਲਵ, NBLK> db_f = convert<ਦਸ਼ਮਲਵ>(db);

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ s = 0; s < NBLK; ++s) {
        simd<ਦਸ਼ਮਲਵ, 32>  y_s  = y_vec.ਸਾਂਚਾ select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.ਸਾਂਚਾ select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.ਸਾਂਚਾ select<32, 1>(s * 32);
        ਅਚਲ ਦਸ਼ਮਲਵ sa = da_f[s];
        ਅਚਲ ਦਸ਼ਮਲਵ sb = db_f[s];
        acc_a += y_s * (convert<ਦਸ਼ਮਲਵ>(qa_s) * sa);
        acc_b += y_s * (convert<ਦਸ਼ਮਲਵ>(qb_s) * sb);
    }
}

ਸਾਂਚਾ <ਪੂਰਨ WG, ਕਿਸਮ_ਨਾਂ Policy>
ESIMD_INLINE ਖਾਲੀ dequantize_mul_mat_vec_q8_0_reorder_esimd(
        ਅਚਲ Policy & policy, ਅਚਲ ਦਸ਼ਮਲਵ * y,
        ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem,
        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) {
    ਵਰਤੋਂ ਨਾਮਾਕਾਸ਼ sycl::ext::intel::esimd;

    constexpr ਪੂਰਨ STRIPE = 8;

    ਅਚਲ ਪੂਰਨ          nblk_ਪੰਕਤੀ = ncols / QK8_0;
    ਅਚਲ size_t       nb       = (size_t) nrows * nblk_ਪੰਕਤੀ;
    ਅਚਲ int8_t *     qs_a     = (ਅਚਲ int8_t *) policy.w_a();
    ਅਚਲ int8_t *     qs_b     = (ਅਚਲ int8_t *) policy.w_b();
    ਅਚਲ sycl::half * d_a      = (ਅਚਲ sycl::half *) (qs_a + nb * QK8_0);
    ਅਚਲ sycl::half * d_b      = (ਅਚਲ sycl::half *) (qs_b + nb * QK8_0);

    ਅਚਲ ਪੂਰਨ  tid   = it.get_local_id(0);
    ਅਚਲ ਪੂਰਨ  g     = it.get_ਸਮੂਹ(0);
    ਅਚਲ ਤਾਰਕਿਕ has_b = policy.has_b(g, nrows);

    ਅਚਲ size_t ਅਧਾਰ_a = (size_t) policy.ਪੰਕਤੀ_a(g) * nblk_ਪੰਕਤੀ;
    ਅਚਲ size_t ਅਧਾਰ_b = (size_t) policy.ਪੰਕਤੀ_b(g) * nblk_ਪੰਕਤੀ;

    simd<ਦਸ਼ਮਲਵ, 32> acc_a = 0.0f;
    simd<ਦਸ਼ਮਲਵ, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    ਪੂਰਨ ib = 0;
    ਲਈ (; ib + WG * STRIPE <= nblk_ਪੰਕਤੀ; ib += WG * STRIPE) {
        ਅਚਲ ਪੂਰਨ b = ib + tid * STRIPE;
        simd<ਦਸ਼ਮਲਵ, 256> y_vec = ਖੰਡ_ਲੱਦੋ<ਦਸ਼ਮਲਵ, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (ਅਧਾਰ_a + b) * QK8_0, qs_b + (ਅਧਾਰ_b + b) * QK8_0,
                                d_a + ਅਧਾਰ_a + b, d_b + ਅਧਾਰ_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    ਲਈ (ਪੂਰਨ b = ib + tid; b < nblk_ਪੰਕਤੀ; b += WG) {
        simd<ਦਸ਼ਮਲਵ, 32> y_vec = ਖੰਡ_ਲੱਦੋ<ਦਸ਼ਮਲਵ, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (ਅਧਾਰ_a + b) * QK8_0, qs_b + (ਅਧਾਰ_b + b) * QK8_0,
                           d_a + ਅਧਾਰ_a + b, d_b + ਅਧਾਰ_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<ਦਸ਼ਮਲਵ>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<ਦਸ਼ਮਲਵ>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    ਜੇ (tid == 0) {
        ਦਸ਼ਮਲਵ ਜੋੜ_a = 0.0f;
        ਦਸ਼ਮਲਵ ਜੋੜ_b = 0.0f;
        ਲਈ (ਪੂਰਨ p = 0; p < WG; ++p) {
            ਜੋੜ_a += lmem[p * 2 + 0];
            ਜੋੜ_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, ਜੋੜ_a, ਜੋੜ_b);
    }
}

ਸਾਂਚਾ <ਪੂਰਨ WG, ਕਿਸਮ_ਨਾਂ Policy>
ਸਥਿਰ ਖਾਲੀ q8_0_esimd_launch(ਅਚਲ Policy & policy, ਅਚਲ ਦਸ਼ਮਲਵ * y, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                              ਅਚਲ ਪੂਰਨ workgroups, dpct::ਕਤਾਰ_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Policy>
ਸਥਿਰ ਖਾਲੀ q8_0_esimd_launch_scaled(ਅਚਲ Policy & policy, ਅਚਲ ਦਸ਼ਮਲਵ * y, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                     ਅਚਲ ਪੂਰਨ workgroups, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    ਅਚਲ ਪੂਰਨ nblk_ਪੰਕਤੀ = ncols / QK8_0;
    ਜੇ (nblk_ਪੰਕਤੀ >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } ਨਹੀਂਤਾਂ ਜੇ (nblk_ਪੰਕਤੀ >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } ਨਹੀਂਤਾਂ ਜੇ (nblk_ਪੰਕਤੀ >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } ਨਹੀਂਤਾਂ {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                           ਅਚਲ ਪੂਰਨ nrows,
                                                           dpct::ਕਤਾਰ_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_ਪੰਕਤੀ_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
ਸਾਂਚਾ <ggml_type T>
ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_glu_reorder_esimd(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vgate,
                                                     ਅਚਲ ਦਸ਼ਮਲਵ * y, ਦਸ਼ਮਲਵ * dst,
                                                     ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows,
                                                     ਅਚਲ ggml_glu_op glu_op,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

ਤਾਰਕਿਕ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(ਗਿਣਤੀ ggml_type src0_type, ਗਿਣਤੀ ggml_glu_op glu_op,
                                                        ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ ਖਾਲੀ * vgate, ਅਚਲ ਦਸ਼ਮਲਵ * y,
                                                        ਦਸ਼ਮਲਵ * dst, ਪੂਰਨ ncols, ਪੂਰਨ nrows, dpct::ਕਤਾਰ_ptr stream) {
    ਚੋਣ (src0_type) {
        ਹਾਲਤ GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

#ਜੇ_ਅੰਤ // GGML_SYCL_DMMV_HAS_ESIMD

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q4_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                     ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                     ਅਚਲ ਪੂਰਨ nrows,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q5_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                     ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                     ਅਚਲ ਪੂਰਨ nrows,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ dequantize_mul_mat_vec_q6_K_sycl_reorder(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                                     ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                                     ਅਚਲ ਪੂਰਨ nrows,
                                                     dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ny = 2 / K_QUANTS_ਪ੍ਰਤੀ_ITERATION;
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_y = (nrows + ny - 1) / ny;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਖੰਡ_ਅੰਕ_y);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, ny, WARP_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, ਵਸਤੂ_ct1);
        });
}

ਖਾਲੀ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
    ਅਚਲ ਅੱਖਰ *src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ *src1_ddf_i, ਅਚਲ ਅੱਖਰ *src1_ddq_i,
    ਦਸ਼ਮਲਵ *dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ dpct::ਕਤਾਰ_ptr &stream) {

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ਪੰਕਤੀ_diff = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = ਖਾਲੀ_ਸੂਚਕ; // dfloat == half

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    ਅਚਲ ਤਾਰਕਿਕ q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਤਾਰਕਿਕ q8_0_esimd = ਝੂਠ;
#ਜੇ_ਅੰਤ

    ਤਾਰਕਿਕ src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    ਜੇ (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        ਅਚਲ ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(src1->type, dst);
        GGML_ASSERT(ਮੰਜ਼ਿਲ_fp16_sycl != ਖਾਲੀ_ਸੂਚਕ);
        ਮੰਜ਼ਿਲ_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#ਨਹੀਂਤਾਂ
    ਅਚਲ dfloat * src1_dfloat = (ਅਚਲ dfloat *) src1_ddf_i; // dfloat == float, no conversion
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
            ਜੇ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਜੇ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
                ਜੇ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
                ਜੇ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
                ਜੇ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
                ਜੇ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
                ਜੇ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਜੇ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
                ਜੇ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
                ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
                }
            } ਨਹੀਂਤਾਂ {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            ਰੋਕੋ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_BF16
        ਹਾਲਤ GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ਪੰਕਤੀ_diff, stream);
            ਰੋਕੋ;
#ਜੇ_ਅੰਤ
        ਮੂਲ:
            ਲਿਖੋ("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(src1_ddq_i);
    GGML_ਅਣਵਰਤਿਆ(src1_ncols);
    GGML_ਅਣਵਰਤਿਆ(src1_padded_ਪੰਕਤੀ_ਮਾਪ);
    GGML_ਅਣਵਰਤਿਆ(ctx);
}
