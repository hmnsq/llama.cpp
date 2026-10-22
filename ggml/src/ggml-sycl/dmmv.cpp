#ಅನ್ತರ್ಭಾವ "convert.hpp"
#ಅನ್ತರ್ಭಾವ "dmmv.hpp"
#ಅನ್ತರ್ಭಾವ "dequantize.hpp"
#ಅನ್ತರ್ಭಾವ "presets.hpp"

#ಆದರೆ defined(__INTEL_LLVM_COMPILER)
    #ಆದರೆ __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
        #ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/bfloat16.hpp>
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DMMV_HAS_BF16
    #ಆದರೆ_ಅಂತ್ಯ
    #ಅನ್ತರ್ಭಾವ <sycl/ext/intel/esimd.hpp>
    #ಅನ್ತರ್ಭಾವ "esimd.hpp"
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ convert_f16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 & v){
    ಸ್ಥಿರಾಂಕ sycl::half *x = (ಸ್ಥಿರಾಂಕ sycl::half *)vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_BF16
ಸ್ಥಿರ ಶೂನ್ಯ convert_bf16(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 & v){
    ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *x = (ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *)vx;

    // automatic bfloat16 -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ convert_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 & v){
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) vx;

    // automatic half -> float type cast if dfloat == float
    v.x() = x[ib + iqs + 0];
    v.y() = x[ib + iqs + 1];
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dequantize_kernel>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ dfloat * __restrict__ y, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                   ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_stride = 2*GGML_SYCL_DMMV_X;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vals_ಪ್ರತಿ_iter = iter_stride / WARP_ಗಾತ್ರ; // num quantized vals per thread and i iter
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#ಇಲ್ಲದಿದ್ದರೆ
    ದಶಮಾಂಶ tmp = 0.0f;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ncols; i += iter_stride) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = i + vals_ಪ್ರತಿ_iter*tid;
        ಆದರೆ (ಲಂಬಸಾಲು >= ncols) {
            ಮುಂದುವರಿ;
        }
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = (ಅಡ್ಡಸಾಲು*ncols + ಲಂಬಸಾಲು)/qk; // x block index
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = (ಲಂಬಸಾಲು%qk)/qr; // x quant index
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iybs = ಲಂಬಸಾಲು - ಲಂಬಸಾಲು%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < vals_ಪ್ರತಿ_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_ಸ್ಥಾನಾಂತರ]};

            tmp += v * t1;
#ಇಲ್ಲದಿದ್ದರೆ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_ಸ್ಥಾನಾಂತರ];
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ_ಆರಂಭ = ncols > GGML_SYCL_DMMV_X ? WARP_ಗಾತ್ರ >> 1 : WARP_ಗಾತ್ರ >> 2;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = ಮುಖವಾಡ_ಆರಂಭ; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (tid == 0) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
        dst[ಅಡ್ಡಸಾಲು] = tmp.x() + tmp.y();
#ಇಲ್ಲದಿದ್ದರೆ
        dst[ಅಡ್ಡಸಾಲು] = tmp;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t_reorder dequantize_kernel_reorder>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ dfloat * __restrict__ y, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                   ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);


    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_ಎಡ = ncols % (QK4_0*WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_align = ncols - ncols_ಎಡ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_stride = 8*2*GGML_SYCL_DMMV_X;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vals_ಪ್ರತಿ_iter = iter_stride / WARP_ಗಾತ್ರ; // num quantized vals per thread and i iter //64/16=4, 512/16/2= 16
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    sycl::half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#ಇಲ್ಲದಿದ್ದರೆ
    ದಶಮಾಂಶ tmp = 0.0f;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *d_ptr = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)vx+ncols*nrows/2;
    ಪೂರ್ಣಾಂಕ i=0;
    ಗಾಗಿ (i = 0; i < ncols_align; i += iter_stride) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = i + vals_ಪ್ರತಿ_iter*tid;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = (ಅಡ್ಡಸಾಲು*ncols + ಲಂಬಸಾಲು)/qk; // x block index
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = (ಲಂಬಸಾಲು%qk)/qr; // x quant index
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iybs = ಲಂಬಸಾಲು - ಲಂಬಸಾಲು%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < vals_ಪ್ರತಿ_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)d_ptr, ib, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_ಸ್ಥಾನಾಂತರ]};

            tmp += v * t1;
#ಇಲ್ಲದಿದ್ದರೆ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_ಸ್ಥಾನಾಂತರ];
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }
    }

    ಗಾಗಿ (; i < ncols; i += iter_stride) {
        ಆದರೆ (tid>=ncols_ಎಡ/QK4_0) ಮುಂದುವರಿ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = i + vals_ಪ್ರತಿ_iter*tid;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = (ಅಡ್ಡಸಾಲು*ncols + ಲಂಬಸಾಲು)/qk; // x block index
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = (ಲಂಬಸಾಲು%qk)/qr; // x quant index
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iybs = ಲಂಬಸಾಲು - ಲಂಬಸಾಲು%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < vals_ಪ್ರತಿ_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel_reorder((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)d_ptr, ib, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)vx, ib * QK4_0 / 2 +iqs+j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
            dfloat2 t1{y[iybs + iqs + j / qr + 0],
                        y[iybs + iqs + j / qr + y_ಸ್ಥಾನಾಂತರ]};

            tmp += v * t1;
#ಇಲ್ಲದಿದ್ದರೆ
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_ಸ್ಥಾನಾಂತರ];
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
        }
    }

    // sum up partial sums and write back result
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ_ಆರಂಭ = ncols > GGML_SYCL_DMMV_X ? WARP_ಗಾತ್ರ >> 1 : WARP_ಗಾತ್ರ >> 2;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = ಮುಖವಾಡ_ಆರಂಭ; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (tid == 0) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
        dst[ಅಡ್ಡಸಾಲು] = tmp.x() + tmp.y();
#ಇಲ್ಲದಿದ್ದರೆ
        dst[ಅಡ್ಡಸಾಲು] = tmp;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ convert_mul_mat_vec_f16_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                         ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                         dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<1, 1, convert_f16>(vx, y, dst, ncols,
                                                          nrows, ಅಂಶ_ct1);
            });
    }
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_BF16
ಸ್ಥಿರ ಶೂನ್ಯ convert_mul_mat_vec_bf16_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                          ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                          dpct::ಸರತಿ_ptr stream) {
    // The qk=1 kernel iterates with stride 2*GGML_SYCL_DMMV_X, so ncols must be a
    // multiple of that — not just GGML_SYCL_DMMV_X — to avoid out-of-bounds reads.
    GGML_ASSERT(ncols % (2*GGML_SYCL_DMMV_X) == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<1, 1, convert_bf16>(vx, y, dst, ncols,
                                                           nrows, ಅಂಶ_ct1);
            });
    }
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q2_k(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                        ದಶಮಾಂಶ *__restrict__ dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    static_assert(16%K_QUANTS_ಪ್ರತಿ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *)vx + ib0;

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0,1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 16/K_QUANTS_ಪ್ರತಿ_ITERATION;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = tid % ಹೆಜ್ಜೆ;                           // 0...15 or 0...7

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = K_QUANTS_ಪ್ರತಿ_ITERATION*ಒಳ;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ಸ್ಥಿರಾಂಕ uint8_t * d = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;
    ಸ್ಥಿರಾಂಕ uint8_t * m = (ಸ್ಥಿರಾಂಕ uint8_t *)(aux + 2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[i].dm[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[i].dm[1];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s_ಸ್ಥಾನಾಂತರ = 8*im;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 128*im + l0;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i * QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + q_ಸ್ಥಾನಾಂತರ;

            ಸ್ಥಿರಾಂಕ uint32_t * a = (ಸ್ಥಿರಾಂಕ uint32_t *)(x[i].scales + s_ಸ್ಥಾನಾಂತರ);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            ದಶಮಾಂಶ sum1 = 0, sum2 = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < K_QUANTS_ಪ್ರತಿ_ITERATION; ++l) {
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
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2) /
                    (2 * K_QUANTS_ಪ್ರತಿ_ITERATION); // 0...15 or 0...7
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = ಅಂಶ_ct1.get_local_id(2) %
                   (2 * K_QUANTS_ಪ್ರತಿ_ITERATION); // 0....1 or 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;

    uint32_t uaux[2];
    ಸ್ಥಿರಾಂಕ uint8_t * d = (ಸ್ಥಿರಾಂಕ uint8_t *)uaux;


    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i * QK_K + ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ uint32_t * s = (ಸ್ಥಿರಾಂಕ uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        ಸ್ಥಿರಾಂಕ sycl::float2 dall =
            x[i].dm.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

        ದಶಮಾಂಶ sum1 = 0, sum2 = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < K_QUANTS_ಪ್ರತಿ_ITERATION; ++l) {
            ಸ್ಥಿರಾಂಕ uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x() * sum1 - dall.y() * sum2;
    }

#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q2_k_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                                ದಶಮಾಂಶ *__restrict__ dst,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                                ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    static_assert(16%K_QUANTS_ಪ್ರತಿ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [scales: nb * (QK_K/16)] [dm: nb * sizeof(half2)]
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = nrows * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ uint8_t     * qs_ತಳ     = (ಸ್ಥಿರಾಂಕ uint8_t *)vx;
    ಸ್ಥಿರಾಂಕ uint8_t     * scales_ತಳ = qs_ತಳ + (size_t)nb * (QK_K / 4);
    ಸ್ಥಿರಾಂಕ sycl::half2 * dm_ತಳ     = (ಸ್ಥಿರಾಂಕ sycl::half2 *)(scales_ತಳ + (size_t)nb * (QK_K / 16));

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0,1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 16/K_QUANTS_ಪ್ರತಿ_ITERATION;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = tid % ಹೆಜ್ಜೆ;                           // 0...15 or 0...7

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = K_QUANTS_ಪ್ರತಿ_ITERATION*ಒಳ;            // 0...15 or 0...14 in steps of 2

    uint32_t aux[4];
    ಸ್ಥಿರಾಂಕ uint8_t * d = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;
    ಸ್ಥಿರಾಂಕ uint8_t * m = (ಸ್ಥಿರಾಂಕ uint8_t *)(aux + 2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ sycl::half2 dm_val = dm_ತಳ[bi];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm_val[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm_val[1];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s_ಸ್ಥಾನಾಂತರ = 8*im;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 128*im + l0;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i * QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * q = qs_ತಳ + bi * (QK_K / 4) + q_ಸ್ಥಾನಾಂತರ;

            ಸ್ಥಿರಾಂಕ uint32_t * a = (ಸ್ಥಿರಾಂಕ uint32_t *)(scales_ತಳ + bi * (QK_K / 16) + s_ಸ್ಥಾನಾಂತರ);
            aux[0] = a[0] & 0x0f0f0f0f;
            aux[1] = a[1] & 0x0f0f0f0f;
            aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
            aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

            ದಶಮಾಂಶ sum1 = 0, sum2 = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < K_QUANTS_ಪ್ರತಿ_ITERATION; ++l) {
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
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(vx);
    GGML_ಬಳಸದ(yy);
    GGML_ಬಳಸದ(ncols);
    GGML_ಬಳಸದ(ಅಂಶ_ct1);
    GGML_ABORT("Q2_K reorder DMMV not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q3_k(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                        ದಶಮಾಂಶ *__restrict__ dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *)vx + ib0;

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

#ಆದರೆ QK_K == 256

    ಸ್ಥಿರಾಂಕ uint16_t kmask1 = 0x0303;
    ಸ್ಥಿರಾಂಕ uint16_t kmask2 = 0x0f0f;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0,1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = K_QUANTS_ಪ್ರತಿ_ITERATION;               // iterations in the inner loop
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 16/K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = tid % ಹೆಜ್ಜೆ;                           // 0...15 or 0...7

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = n*ಒಳ;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ಸ್ಥಿರಾಂಕ int8_t * s = (ಸ್ಥಿರಾಂಕ int8_t *)utmp;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ uint8_t * h = x[i].hmask + l0;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[i].d;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ =  32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 128*im + l0;
            ಸ್ಥಿರಾಂಕ uint16_t s_ಸರಿಸು = 4*im;
            ಸ್ಥಿರಾಂಕ uint8_t m = 1 << (4*im);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y  = yy + i * QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + q_ಸ್ಥಾನಾಂತರ;

            ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;
            utmp[0] = ((a[0] >> s_ಸರಿಸು) & kmask2) | (((a[4] >> (s_ಸರಿಸು + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_ಸರಿಸು) & kmask2) | (((a[5] >> (s_ಸರಿಸು + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_ಸರಿಸು) & kmask2) | (((a[4] >> (s_ಸರಿಸು + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_ಸರಿಸು) & kmask2) | (((a[5] >> (s_ಸರಿಸು + 2)) & kmask1) << 4);

            ದಶಮಾಂಶ ಮೊತ್ತ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < n; ++l) {
                ಮೊತ್ತ += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                ಮೊತ್ತ += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * ಮೊತ್ತ;
        }

    }
#ಇಲ್ಲದಿದ್ದರೆ

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2)/(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...15 or 0...7
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix  = ಅಂಶ_ct1.get_local_id(2)%(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0....1 or 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;         // 0...15 or 0...14
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = ಸ್ಥಾನಾಂತರ/8;                                 // 0 or 1
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im = ಸ್ಥಾನಾಂತರ%8;                                 // 0...7

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i * QK_K + ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ uint8_t * s = x[i].scales;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = (ದಶಮಾಂಶ)x[i].d;

        ದಶಮಾಂಶ ಮೊತ್ತ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < K_QUANTS_ಪ್ರತಿ_ITERATION; ++l) {
            ಸ್ಥಿರಾಂಕ uint8_t hl = x[i].hmask[im+l] >> ಒಳ;
            ಸ್ಥಿರಾಂಕ uint8_t ql = q[l];
            ಮೊತ್ತ += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += ಮೊತ್ತ;
    }
#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q3_k_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                                ದಶಮಾಂಶ *__restrict__ dst,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                                ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * (QK_K/4)] [hmask: nb * (QK_K/8)] [scales: nb * 12] [d: nb * sizeof(half)]
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = nrows * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ uint8_t   * qs_ತಳ     = (ಸ್ಥಿರಾಂಕ uint8_t *)vx;
    ಸ್ಥಿರಾಂಕ uint8_t   * hmask_ತಳ  = qs_ತಳ + (size_t)nb * (QK_K / 4);
    ಸ್ಥಿರಾಂಕ uint8_t   * scales_ತಳ = hmask_ತಳ + (size_t)nb * (QK_K / 8);
    ಸ್ಥಿರಾಂಕ sycl::half * d_ತಳ     = (ಸ್ಥಿರಾಂಕ sycl::half *)(scales_ತಳ + (size_t)nb * 12);

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

#ಆದರೆ QK_K == 256

    ಸ್ಥಿರಾಂಕ uint16_t kmask1 = 0x0303;
    ಸ್ಥಿರಾಂಕ uint16_t kmask2 = 0x0f0f;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0,1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n  = K_QUANTS_ಪ್ರತಿ_ITERATION;               // iterations in the inner loop
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 16/K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = tid % ಹೆಜ್ಜೆ;                           // 0...15 or 0...7

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = n*ಒಳ;                                 // 0...15 or 0...14 in steps of 2

    uint16_t utmp[4];
    ಸ್ಥಿರಾಂಕ int8_t * s = (ಸ್ಥಿರಾಂಕ int8_t *)utmp;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ uint8_t * h  = hmask_ತಳ + bi * (QK_K / 8) + l0;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = d_ತಳ[bi];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ =  32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 128*im + l0;
            ಸ್ಥಿರಾಂಕ uint16_t s_ಸರಿಸು = 4*im;
            ಸ್ಥಿರಾಂಕ uint8_t m = 1 << (4*im);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y  = yy + i * QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * q  = qs_ತಳ + bi * (QK_K / 4) + q_ಸ್ಥಾನಾಂತರ;

            ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)(scales_ತಳ + bi * 12);
            utmp[0] = ((a[0] >> s_ಸರಿಸು) & kmask2) | (((a[4] >> (s_ಸರಿಸು + 0)) & kmask1) << 4);
            utmp[1] = ((a[1] >> s_ಸರಿಸು) & kmask2) | (((a[5] >> (s_ಸರಿಸು + 0)) & kmask1) << 4);
            utmp[2] = ((a[2] >> s_ಸರಿಸು) & kmask2) | (((a[4] >> (s_ಸರಿಸು + 2)) & kmask1) << 4);
            utmp[3] = ((a[3] >> s_ಸರಿಸು) & kmask2) | (((a[5] >> (s_ಸರಿಸು + 2)) & kmask1) << 4);

            ದಶಮಾಂಶ ಮೊತ್ತ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < n; ++l) {
                ಮೊತ್ತ += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                     + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                     + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                     + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
                ಮೊತ್ತ += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                     + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                     + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                    + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
            }
            tmp += d * ಮೊತ್ತ;
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(vx);
    GGML_ಬಳಸದ(yy);
    GGML_ಬಳಸದ(ncols);
    GGML_ಬಳಸದ(ಅಂಶ_ct1);
    GGML_ABORT("Q3_K reorder DMMV not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_k(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                        ದಶಮಾಂಶ *__restrict__ dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *)vx + ib0;

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ uint16_t kmask1 = 0x3f3f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask2 = 0x0f0f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask3 = 0xc0c0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0,1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 8/K_QUANTS_ಪ್ರತಿ_ITERATION;           // 8 or 4

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il_ತಳ = tid/ಹೆಜ್ಜೆ;                        // 0 or 1 (was 0...3)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir  = tid - ಹೆಜ್ಜೆ*il_ತಳ;                  // 0...7 or 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n   = 2 * K_QUANTS_ಪ್ರತಿ_ITERATION;          // 2 or 4

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = il_ತಳ%2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = n*(2*ir + ಒಳ);

    uint16_t aux[4];
    ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 2
    uint32_t q32[4];
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = (ಸ್ಥಿರಾಂಕ uint8_t *)q32;
#ಇಲ್ಲದಿದ್ದರೆ
    uint16_t q16[4];
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = (ಸ್ಥಿರಾಂಕ uint8_t *)q16;
#ಆದರೆ_ಅಂತ್ಯ

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[i].dm[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[i].dm[1];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 64*im + l0;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y1 = yy + i*QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y2 = y1 + 128;

            ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 2
            ಸ್ಥಿರಾಂಕ uint32_t * q1 = (ಸ್ಥಿರಾಂಕ uint32_t *)(x[i].qs + q_ಸ್ಥಾನಾಂತರ);
            ಸ್ಥಿರಾಂಕ uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ದಶಮಾಂಶ smin = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#ಇಲ್ಲದಿದ್ದರೆ
            ಸ್ಥಿರಾಂಕ uint16_t * q1 = (ಸ್ಥಿರಾಂಕ uint16_t *)(x[i].qs + q_ಸ್ಥಾನಾಂತರ);
            ಸ್ಥಿರಾಂಕ uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ದಶಮಾಂಶ smin = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#ಆದರೆ_ಅಂತ್ಯ
        }

    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2)/(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix  = ಅಂಶ_ct1.get_local_id(2)%(2*K_QUANTS_ಪ್ರತಿ_ITERATION);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;

    uint16_t aux16[2];
    ಸ್ಥಿರಾಂಕ uint8_t * s = (ಸ್ಥಿರಾಂಕ uint8_t *)aux16;

    ದಶಮಾಂಶ tmp = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i*QK_K + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].dm[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m = (ದಶಮಾಂಶ)x[i].dm[1];
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_QUANTS_ಪ್ರತಿ_ITERATION; ++j) {
            ಮೊತ್ತ += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += ಮೊತ್ತ;
    }

#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (tid == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_k_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                                ದಶಮಾಂಶ *__restrict__ dst,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                                ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = nrows * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ uint8_t     * qs_ತಳ     = (ಸ್ಥಿರಾಂಕ uint8_t *)vx;
    ಸ್ಥಿರಾಂಕ uint8_t     * scales_ತಳ = qs_ತಳ + (size_t)nb * (QK_K / 2);
    ಸ್ಥಿರಾಂಕ sycl::half2 * dm_ತಳ     = (ಸ್ಥಿರಾಂಕ sycl::half2 *)(scales_ತಳ + (size_t)nb * K_SCALE_ಗಾತ್ರ);

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ uint16_t kmask1 = 0x3f3f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask2 = 0x0f0f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask3 = 0xc0c0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0,1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 8/K_QUANTS_ಪ್ರತಿ_ITERATION;           // 8 or 4

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il_ತಳ = tid/ಹೆಜ್ಜೆ;                        // 0 or 1 (was 0...3)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir  = tid - ಹೆಜ್ಜೆ*il_ತಳ;                  // 0...7 or 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n   = 2 * K_QUANTS_ಪ್ರತಿ_ITERATION;          // 2 or 4

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = il_ತಳ%2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = n*(2*ir + ಒಳ);

    uint16_t aux[4];
    ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 2
    uint32_t q32[4];
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = (ಸ್ಥಿರಾಂಕ uint8_t *)q32;
#ಇಲ್ಲದಿದ್ದರೆ
    uint16_t q16[4];
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = (ಸ್ಥಿರಾಂಕ uint8_t *)q16;
#ಆದರೆ_ಅಂತ್ಯ

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ sycl::half2 dm_val = dm_ತಳ[bi];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm_val[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm_val[1];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 64*im + l0;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y1 = yy + i*QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y2 = y1 + 128;

            ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)(scales_ತಳ + bi * K_SCALE_ಗಾತ್ರ);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 2
            ಸ್ಥಿರಾಂಕ uint32_t * q1 = (ಸ್ಥಿರಾಂಕ uint32_t *)(qs_ತಳ + bi * (QK_K / 2) + q_ಸ್ಥಾನಾಂತರ);
            ಸ್ಥಿರಾಂಕ uint32_t * q2 = q1 + 16;

            q32[0] = q1[0] & 0x0f0f0f0f;
            q32[1] = q1[0] & 0xf0f0f0f0;
            q32[2] = q2[0] & 0x0f0f0f0f;
            q32[3] = q2[0] & 0xf0f0f0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ದಶಮಾಂಶ smin = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
                s.x() += y1[l] * q4[l + 0]; s.y() += y1[l + 32] * q4[l + 4];
                s.z() += y2[l] * q4[l + 8]; s.w() += y2[l + 32] * q4[l + 12];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f / 16.f +
                           s.z() * sc[4] + s.w() * sc[5] * 1.f / 16.f) -
                   dmin * smin;
#ಇಲ್ಲದಿದ್ದರೆ
            ಸ್ಥಿರಾಂಕ uint16_t * q1 = (ಸ್ಥಿರಾಂಕ uint16_t *)(qs_ತಳ + bi * (QK_K / 2) + q_ಸ್ಥಾನಾಂತರ);
            ಸ್ಥಿರಾಂಕ uint16_t * q2 = q1 + 32;

            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[0] & 0xf0f0;
            q16[2] = q2[0] & 0x0f0f;
            q16[3] = q2[0] & 0xf0f0;

            sycl::float4 s = {0.f, 0.f, 0.f, 0.f};
            ದಶಮಾಂಶ smin = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 2; ++l) {
                s.x() += y1[l] * q4[l+0]; s.y() += y1[l+32] * q4[l+2];
                s.z() += y2[l] * q4[l+4]; s.w() += y2[l+32] * q4[l+6];
                smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
            }
            tmp += dall * (s.x() * sc[0] + s.y() * sc[1] * 1.f/16.f + s.z() * sc[4] + s.w() * sc[5] * 1.f/16.f) - dmin * smin;
#ಆದರೆ_ಅಂತ್ಯ
        }

    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2)/(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix  = ಅಂಶ_ct1.get_local_id(2)%(2*K_QUANTS_ಪ್ರತಿ_ITERATION);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;

    uint16_t aux16[2];
    ಸ್ಥಿರಾಂಕ uint8_t * s = (ಸ್ಥಿರಾಂಕ uint8_t *)aux16;

    ದಶಮಾಂಶ tmp = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ uint8_t * q = qs_ತಳ + bi * (QK_K / 2) + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i*QK_K + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)(scales_ತಳ + bi * K_SCALE_ಗಾತ್ರ);
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        ಸ್ಥಿರಾಂಕ sycl::half2 dm_val = dm_ತಳ[bi];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)dm_val[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m = (ದಶಮಾಂಶ)dm_val[1];
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_QUANTS_ಪ್ರತಿ_ITERATION; ++j) {
            ಮೊತ್ತ += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += ಮೊತ್ತ;
    }

#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (tid == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_k(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                        ದಶಮಾಂಶ *__restrict__ dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *)vx + ib0;

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ uint16_t kmask1 = 0x3f3f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask2 = 0x0f0f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask3 = 0xc0c0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2) / 2; // 0...7
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = ಅಂಶ_ct1.get_local_id(2) % 2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il_ತಳ = tid/4;     // 0 or 1 (was 0...3)
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir  = tid - 4*il_ತಳ;// 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n   = 2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = il_ತಳ%2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = n*(2*ir + ಒಳ);

    uint16_t aux[4];
    ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;

    uint16_t q16[8];
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = (ಸ್ಥಿರಾಂಕ uint8_t *)q16;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2) {

        ಸ್ಥಿರಾಂಕ uint8_t * qh  = x[i].qh + l0;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[i].dm[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[i].dm[1];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 64*im + l0;

            ಸ್ಥಿರಾಂಕ uint8_t hm1  = 1 << (2*im);
            ಸ್ಥಿರಾಂಕ uint8_t hm2  = hm1 << 4;

            ಸ್ಥಿರಾಂಕ uint8_t * ql1 = x[i].qs + q_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y1  = yy + i*QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y2  = y1 + 128;

            ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 ಮೊತ್ತ = {0.f, 0.f, 0.f, 0.f};
            ದಶಮಾಂಶ smin = 0;
            ಸ್ಥಿರಾಂಕ uint16_t * q1 = (ಸ್ಥಿರಾಂಕ uint16_t *)ql1;
            ಸ್ಥಿರಾಂಕ uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < n; ++l) {
                ಮೊತ್ತ.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                ಮೊತ್ತ.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                ಮೊತ್ತ.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                ಮೊತ್ತ.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (ಮೊತ್ತ.x() * sc[0] + ಮೊತ್ತ.y() * sc[1] + ಮೊತ್ತ.z() * sc[4] +
                           ಮೊತ್ತ.w() * sc[5]) -
                   dmin * smin;
        }
    }

#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2)/(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix  = ಅಂಶ_ct1.get_local_id(2)%(2*K_QUANTS_ಪ್ರತಿ_ITERATION);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im = ಹೆಜ್ಜೆ/8;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = ಹೆಜ್ಜೆ%8;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ int8_t  * s = x[i].scales;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y = yy + i*QK_K + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d = x[i].d;
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_QUANTS_ಪ್ರತಿ_ITERATION; ++j) {
            ಸ್ಥಿರಾಂಕ uint8_t h = x[i].qh[ಒಳ+j] >> im;
            ಮೊತ್ತ += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += ಮೊತ್ತ;
    }
#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_k_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx,
                                                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ yy,
                                                ದಶಮಾಂಶ *__restrict__ dst,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                                ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    // SOA base pointers for the reordered layout:
    //   [qs: nb * QK_K/2] [qh: nb * QK_K/8] [scales: nb * K_SCALE_SIZE] [dm: nb * sizeof(half2)]
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = nrows * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ uint8_t     * qs_ತಳ     = (ಸ್ಥಿರಾಂಕ uint8_t *)vx;
    ಸ್ಥಿರಾಂಕ uint8_t     * qh_ತಳ     = qs_ತಳ + (size_t)nb * (QK_K / 2);
    ಸ್ಥಿರಾಂಕ uint8_t     * scales_ತಳ = qh_ತಳ + (size_t)nb * (QK_K / 8);
    ಸ್ಥಿರಾಂಕ sycl::half2 * dm_ತಳ     = (ಸ್ಥಿರಾಂಕ sycl::half2 *)(scales_ತಳ + (size_t)nb * K_SCALE_ಗಾತ್ರ);

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ uint16_t kmask1 = 0x3f3f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask2 = 0x0f0f;
    ಸ್ಥಿರಾಂಕ uint16_t kmask3 = 0xc0c0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2) / 2; // 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = ಅಂಶ_ct1.get_local_id(2) % 2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il_ತಳ = tid/4;      // 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir  = tid - 4*il_ತಳ;// 0...3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n   = 2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = il_ತಳ%2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = n*(2*ir + ಒಳ);

    uint16_t aux[4];
    ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;

    uint16_t q16[8];
    ಸ್ಥಿರಾಂಕ uint8_t * q4 = (ಸ್ಥಿರಾಂಕ uint8_t *)q16;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ uint8_t * qh  = qh_ತಳ + bi * (QK_K / 8) + l0;
        ಸ್ಥಿರಾಂಕ sycl::half2 dm_val = dm_ತಳ[bi];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm_val[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm_val[1];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 64*im + l0;

            ಸ್ಥಿರಾಂಕ uint8_t hm1  = 1 << (2*im);
            ಸ್ಥಿರಾಂಕ uint8_t hm2  = hm1 << 4;

            ಸ್ಥಿರಾಂಕ uint8_t * ql1 = qs_ತಳ + bi * (QK_K / 2) + q_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y1  = yy + i*QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y2  = y1 + 128;

            ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)(scales_ತಳ + bi * K_SCALE_ಗಾತ್ರ);
            aux[0] = a[im+0] & kmask1;
            aux[1] = a[im+2] & kmask1;
            aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
            aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

            sycl::float4 ಮೊತ್ತ = {0.f, 0.f, 0.f, 0.f};
            ದಶಮಾಂಶ smin = 0;
            ಸ್ಥಿರಾಂಕ uint16_t * q1 = (ಸ್ಥಿರಾಂಕ uint16_t *)ql1;
            ಸ್ಥಿರಾಂಕ uint16_t * q2 = q1 + 32;
            q16[0] = q1[0] & 0x0f0f;
            q16[1] = q1[8] & 0x0f0f;
            q16[2] = (q1[0] >> 4) & 0x0f0f;
            q16[3] = (q1[8] >> 4) & 0x0f0f;
            q16[4] = q2[0] & 0x0f0f;
            q16[5] = q2[8] & 0x0f0f;
            q16[6] = (q2[0] >> 4) & 0x0f0f;
            q16[7] = (q2[8] >> 4) & 0x0f0f;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < n; ++l) {
                ಮೊತ್ತ.x() +=
                    y1[l + 0] * (q4[l + 0] + (qh[l + 0] & (hm1 << 0) ? 16 : 0)) +
                    y1[l + 16] * (q4[l + 2] + (qh[l + 16] & (hm1 << 0) ? 16 : 0));
                ಮೊತ್ತ.y() +=
                    y1[l + 32] * (q4[l + 4] + (qh[l + 0] & (hm1 << 1) ? 16 : 0)) +
                    y1[l + 48] * (q4[l + 6] + (qh[l + 16] & (hm1 << 1) ? 16 : 0));
                ಮೊತ್ತ.z() +=
                    y2[l + 0] * (q4[l + 8] + (qh[l + 0] & (hm2 << 0) ? 16 : 0)) +
                    y2[l + 16] * (q4[l + 10] + (qh[l + 16] & (hm2 << 0) ? 16 : 0));
                ಮೊತ್ತ.w() +=
                    y2[l + 32] * (q4[l + 12] + (qh[l + 0] & (hm2 << 1) ? 16 : 0)) +
                    y2[l + 48] * (q4[l + 14] + (qh[l + 16] & (hm2 << 1) ? 16 : 0));
                smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                      + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
            }
            tmp += dall * (ಮೊತ್ತ.x() * sc[0] + ಮೊತ್ತ.y() * sc[1] + ಮೊತ್ತ.z() * sc[4] +
                           ಮೊತ್ತ.w() * sc[5]) -
                   dmin * smin;
        }
    }
#ಇಲ್ಲದಿದ್ದರೆ
    // The reordered Q5_K layout is only produced for QK_K == 256.
#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q6_k(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ yy, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    static_assert(16%K_QUANTS_ಪ್ರತಿ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *)vx + ib0;

#ಆದರೆ QK_K == 256

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0, 1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 16/K_QUANTS_ಪ್ರತಿ_ITERATION;          // 16 or 8

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = tid % ಹೆಜ್ಜೆ;                           // 0...15 or 0...7

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 1
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = K_QUANTS_ಪ್ರತಿ_ITERATION*ಒಳ;            // 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 0;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = 4 * ಒಳ;                               // 0, 4, 8, ..., 28
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = ಒಳ / 4;
#ಆದರೆ_ಅಂತ್ಯ

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[i].d;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql_ಸ್ಥಾನಾಂತರ = 64*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s_ಸ್ಥಾನಾಂತರ  =  8*im + is;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 128*im + l0;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y  = yy + i * QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * ql = x[i].ql + ql_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * qh = x[i].qh + qh_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ int8_t  * s  = x[i].scales + s_ಸ್ಥಾನಾಂತರ;

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 1
            ದಶಮಾಂಶ ಮೊತ್ತ = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += ಮೊತ್ತ;
#ಇಲ್ಲದಿದ್ದರೆ
            ದಶಮಾಂಶ ಮೊತ್ತ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
                ಮೊತ್ತ += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += ಮೊತ್ತ;
#ಆದರೆ_ಅಂತ್ಯ
        }

    }

#ಇಲ್ಲದಿದ್ದರೆ

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2)/(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...7
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix  = ಅಂಶ_ct1.get_local_id(2)%(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...3

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y  = yy + i * QK_K + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ uint8_t * ql = x[i].ql + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ uint8_t * qh = x[i].qh + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ int8_t  * s  = x[i].scales;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[i+0].d;

        ದಶಮಾಂಶ ಮೊತ್ತ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_QUANTS_ಪ್ರತಿ_ITERATION; ++j) {
            ಮೊತ್ತ += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += ಮೊತ್ತ;

    }

#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (tid == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q6_k_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ yy, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows,
                                                ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    static_assert(16%K_QUANTS_ಪ್ರತಿ_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib0 = ಅಡ್ಡಸಾಲು*ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    // SOA base pointers for the reordered layout:
    //   [ql: nb * QK_K/2] [qh: nb * QK_K/4] [scales: nb * QK_K/16] [d: nb * sizeof(half)]
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb = nrows * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ uint8_t   * ql_ತಳ     = (ಸ್ಥಿರಾಂಕ uint8_t *)vx;
    ಸ್ಥಿರಾಂಕ uint8_t   * qh_ತಳ     = ql_ತಳ + (size_t)nb * (QK_K / 2);
    ಸ್ಥಿರಾಂಕ int8_t    * scales_ತಳ = (ಸ್ಥಿರಾಂಕ int8_t *)(qh_ತಳ + (size_t)nb * (QK_K / 4));
    ಸ್ಥಿರಾಂಕ sycl::half * d_ತಳ     = (ಸ್ಥಿರಾಂಕ sycl::half *)((ಸ್ಥಿರಾಂಕ uint8_t *)scales_ತಳ + (size_t)nb * (QK_K / 16));

#ಆದರೆ QK_K == 256

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid =
        ಅಂಶ_ct1.get_local_id(2) / K_QUANTS_ಪ್ರತಿ_ITERATION; // 0...7 or 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix =
        ಅಂಶ_ct1.get_local_id(2) % K_QUANTS_ಪ್ರತಿ_ITERATION; // 0 or 0, 1

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 16/K_QUANTS_ಪ್ರತಿ_ITERATION;          // 16 or 8

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = tid % ಹೆಜ್ಜೆ;                           // 0...15 or 0...7

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 1
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = K_QUANTS_ಪ್ರತಿ_ITERATION*ಒಳ;            // 0...15
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 0;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l0 = 4 * ಒಳ;                               // 0, 4, 8, ..., 28
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = ಒಳ / 4;
#ಆದರೆ_ಅಂತ್ಯ

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = d_ತಳ[bi];

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ im = 0; im < 2; ++im) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ql_ಸ್ಥಾನಾಂತರ = 64*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qh_ಸ್ಥಾನಾಂತರ = 32*im + l0;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s_ಸ್ಥಾನಾಂತರ  =  8*im + is;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = 128*im + l0;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y  = yy + i * QK_K + y_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * ql = ql_ತಳ + bi * (QK_K / 2) + ql_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ uint8_t * qh = qh_ತಳ + bi * (QK_K / 4) + qh_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ int8_t  * s  = scales_ತಳ + bi * (QK_K / 16) + s_ಸ್ಥಾನಾಂತರ;

#ಆದರೆ K_QUANTS_ಪ್ರತಿ_ITERATION == 1
            ದಶಮಾಂಶ ಮೊತ್ತ = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                      + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                      + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                      + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                      + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                      + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                      + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                      +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
            tmp += ಮೊತ್ತ;
#ಇಲ್ಲದಿದ್ದರೆ
            ದಶಮಾಂಶ ಮೊತ್ತ = 0;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
                ಮೊತ್ತ += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                     + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                     + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                     + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            tmp += ಮೊತ್ತ;
#ಆದರೆ_ಅಂತ್ಯ
        }

    }

#ಇಲ್ಲದಿದ್ದರೆ

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2)/(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...7
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix  = ಅಂಶ_ct1.get_local_id(2)%(2*K_QUANTS_ಪ್ರತಿ_ITERATION);  // 0...3

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = tid * K_QUANTS_ಪ್ರತಿ_ITERATION;

    ದಶಮಾಂಶ tmp = 0; // partial sum for thread in warp

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ix; i < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; i += 2*K_QUANTS_ಪ್ರತಿ_ITERATION) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bi = ib0 + i;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   * y  = yy + i * QK_K + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ uint8_t * ql = ql_ತಳ + bi * (QK_K / 2) + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ uint8_t * qh = qh_ತಳ + bi * (QK_K / 4) + ಹೆಜ್ಜೆ;
        ಸ್ಥಿರಾಂಕ int8_t  * s  = scales_ತಳ + bi * (QK_K / 16);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = d_ತಳ[bi];

        ದಶಮಾಂಶ ಮೊತ್ತ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_QUANTS_ಪ್ರತಿ_ITERATION; ++j) {
            ಮೊತ್ತ += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += ಮೊತ್ತ;

    }

#ಆದರೆ_ಅಂತ್ಯ

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (tid == 0) {
        dst[ಅಡ್ಡಸಾಲು] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_0_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}


ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_0_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q1_0_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec_reorder<QK1_0, QR1_0, dequantize_q1_0_reorder>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q1_0_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // the number of rows may exceed maximum grid size in the y or z dimensions, use the x dimension instead
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<QK1_0, QR1_0, dequantize_q1_0>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_0_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_1_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q8_0_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                // Q8_0 reorder layout: [all qs (ncols*nrows bytes)][all d values]
                // Cannot reuse dequantize_mul_mat_vec_reorder template because it has
                // Q4_0-specific constants hardcoded (d_ptr offset and qs stride).
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(1) +
                    ಅಂಶ_ct1.get_local_id(1);
                ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iter_stride = 8*2*GGML_SYCL_DMMV_X;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vals_ಪ್ರತಿ_iter = iter_stride / WARP_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_ಎಡ = ncols % (QK8_0*WARP_ಗಾತ್ರ);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_align = ncols - ncols_ಎಡ;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
                sycl::half2 tmp = {0.0f, 0.0f};
#ಇಲ್ಲದಿದ್ದರೆ
                ದಶಮಾಂಶ tmp = 0.0f;
#ಆದರೆ_ಅಂತ್ಯ
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *d_ptr = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ*)vx + ncols*nrows;  // d after all qs

                ಪೂರ್ಣಾಂಕ i = 0;
                ಗಾಗಿ (i = 0; i < ncols_align; i += iter_stride) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = i + vals_ಪ್ರತಿ_iter*tid;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = (ಅಡ್ಡಸಾಲು*ncols + ಲಂಬಸಾಲು)/QK8_0;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = ಲಂಬಸಾಲು % QK8_0;

#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < vals_ಪ್ರತಿ_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)d_ptr, ib, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
                        dfloat2 t1{y[ಲಂಬಸಾಲು + j + 0], y[ಲಂಬಸಾಲು + j + 1]};
                        tmp += v * t1;
#ಇಲ್ಲದಿದ್ದರೆ
                        tmp += v.x() * y[ಲಂಬಸಾಲು + j + 0];
                        tmp += v.y() * y[ಲಂಬಸಾಲು + j + 1];
#ಆದರೆ_ಅಂತ್ಯ
                    }
                }

                // handle remaining columns
                ಗಾಗಿ (; i < ncols; i += iter_stride) {
                    ಆದರೆ (tid >= ncols_ಎಡ/QK8_0) ಮುಂದುವರಿ;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = i + vals_ಪ್ರತಿ_iter*tid;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = (ಅಡ್ಡಸಾಲು*ncols + ಲಂಬಸಾಲು)/QK8_0;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = ಲಂಬಸಾಲು % QK8_0;

#ಸೂಚನೆ unroll
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < vals_ಪ್ರತಿ_iter; j += 2) {
                        dfloat2 v;
                        dequantize_q8_0_reorder((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)d_ptr, ib, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)vx,
                                                ib * QK8_0 + iqs + j, v);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
                        dfloat2 t1{y[ಲಂಬಸಾಲು + j + 0], y[ಲಂಬಸಾಲು + j + 1]};
                        tmp += v * t1;
#ಇಲ್ಲದಿದ್ದರೆ
                        tmp += v.x() * y[ಲಂಬಸಾಲು + j + 0];
                        tmp += v.y() * y[ಲಂಬಸಾಲು + j + 1];
#ಆದರೆ_ಅಂತ್ಯ
                    }
                }

                // reduce
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ_ಆರಂಭ = ncols > GGML_SYCL_DMMV_X ? WARP_ಗಾತ್ರ >> 1 : WARP_ಗಾತ್ರ >> 2;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = ಮುಖವಾಡ_ಆರಂಭ; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
                    tmp += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
                }

                ಆದರೆ (tid == 0) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
                    dst[ಅಡ್ಡಸಾಲು] = tmp.x() + tmp.y();
#ಇಲ್ಲದಿದ್ದರೆ
                    dst[ಅಡ್ಡಸಾಲು] = tmp;
#ಆದರೆ_ಅಂತ್ಯ
                }
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q8_0_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ dfloat *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % GGML_SYCL_DMMV_X == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, GGML_SYCL_MMV_Y, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(
                    vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q2_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2; // very slightly faster than 1 even when K_QUANTS_PER_ITERATION = 2
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q2_k(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q2_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                     ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q2_k_reorder(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q3_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q3_k(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q3_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                     ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q3_k_reorder(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q4_k(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q5_k(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q6_K_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                             ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                             dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q6_k(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
ಬಳಸಿ ggml_sycl_esimd::GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ;

// policies for dequantize_mul_mat_vec_reorder_esimd: row mapping and what to do with the two sums
// plain mat-vec: a work-group owns two consecutive rows of one matrix
ರಚನೆ esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy {
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx;
    ದಶಮಾಂಶ *      dst;

    ESIMD_INLINE ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * w_a() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ vx; }
    ESIMD_INLINE ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * w_b() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ vx; }

    ESIMD_INLINE ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು_a(ಪೂರ್ಣಾಂಕ g) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ g * 2; }
    ESIMD_INLINE ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು_b(ಪೂರ್ಣಾಂಕ g) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ g * 2 + 1; }
    ESIMD_INLINE ತಾರ್ಕಿಕ has_b(ಪೂರ್ಣಾಂಕ g, ಪೂರ್ಣಾಂಕ nrows) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಅಡ್ಡಸಾಲು_b(g) < nrows; }

    ESIMD_INLINE ಶೂನ್ಯ epilogue(ಪೂರ್ಣಾಂಕ g, ಪೂರ್ಣಾಂಕ nrows, ದಶಮಾಂಶ ಮೊತ್ತ_a, ದಶಮಾಂಶ ಮೊತ್ತ_b) ಸ್ಥಿರಾಂಕ {
        dst[ಅಡ್ಡಸಾಲು_a(g)] = ಮೊತ್ತ_a;
        ಆದರೆ (has_b(g, nrows)) {
            dst[ಅಡ್ಡಸಾಲು_b(g)] = ಮೊತ್ತ_b;
        }
    }
};

// fused gate+up+GLU: a work-group owns one row of both matrices; a is the gate, b the up
ರಚನೆ esimd_dmmv_glu_policy {
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vup;
    ದಶಮಾಂಶ *      dst;
    ggml_glu_op  glu_op;

    ESIMD_INLINE ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * w_a() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ vgate; }
    ESIMD_INLINE ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * w_b() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ vup; }

    ESIMD_INLINE ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು_a(ಪೂರ್ಣಾಂಕ g) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ g; }
    ESIMD_INLINE ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು_b(ಪೂರ್ಣಾಂಕ g) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ g; }
    ESIMD_INLINE ತಾರ್ಕಿಕ has_b(ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ನಿಜ; } // up always contributes

    ESIMD_INLINE ಶೂನ್ಯ epilogue(ಪೂರ್ಣಾಂಕ g, ಪೂರ್ಣಾಂಕ, ದಶಮಾಂಶ gate, ದಶಮಾಂಶ ಮೇಲೆ) ಸ್ಥಿರಾಂಕ {
        ಬಳಸಿ ನಾಮಾಕಾಶ sycl::ext::intel::esimd;

        // SwiGLU: silu(gate)*up = gate*sigmoid(gate)*up. GEGLU's tanh form reduces
        // to the same shape via tanh(y) = 2*sigmoid(2y) - 1, so only the sigmoid
        // argument differs. sigmoid via exp2 (width-1 simd is a degenerate case for
        // the EM pipe, so broadcast to 16 lanes); lane 0 holds the result.
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ GELU_COEF_A    = 0.044715f;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ NEG_LOG2E      = -1.44269504088896341f;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ arg = (glu_op == GGML_GLU_OP_GEGLU) ?
            2.0f * SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate) :
            gate;

        simd<ದಶಮಾಂಶ, 16> t = arg * NEG_LOG2E;
        simd<ದಶಮಾಂಶ, 16> e = exp2(t);
        dst[g] = gate / (1.0f + e[0]) * ಮೇಲೆ;
    }
};

// generic reordered dequantize-matvec: each work-group owns a pair of rows chosen by the policy
// and updates one 32-wide accumulator per row
ಮಾದರಿ <ggml_type T, ಬಗೆ_ಹೆಸರು Policy>
ESIMD_INLINE ಶೂನ್ಯ dequantize_mul_mat_vec_reorder_esimd(
        ಸ್ಥಿರಾಂಕ Policy & policy, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) {
    ಬಳಸಿ ನಾಮಾಕಾಶ sycl::ext::intel::esimd;
    ಬಳಸಿ traits = ggml_sycl_esimd::esimd_reorder_q_traits<T>;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = ncols / QK_K;
    ಸ್ಥಿರಾಂಕ size_t nb = (size_t) nrows * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ   pa = traits::make_ptrs(policy.w_a(), nb);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ   pb = traits::make_ptrs(policy.w_b(), nb);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  tid   = it.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  g     = it.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು_a = policy.ಅಡ್ಡಸಾಲು_a(g);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  ಅಡ್ಡಸಾಲು_b = policy.ಅಡ್ಡಸಾಲು_b(g);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_b = policy.has_b(g, nrows);

    // one 32-wide accumulator per row (small footprint, no spill)
    simd<ದಶಮಾಂಶ, 32> acc_a = 0.0f;
    simd<ದಶಮಾಂಶ, 32> acc_b = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ib = tid; ib < ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು; ib += GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ) {
        simd<ದಶಮಾಂಶ, 256> y_vec = ಖಂಡ_ತುಂಬು<ದಶಮಾಂಶ, 256>(y + (size_t) ib * QK_K);

        ಸ್ಥಿರಾಂಕ size_t bi_a = (size_t) ಅಡ್ಡಸಾಲು_a * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + ib;
        ಸ್ಥಿರಾಂಕ size_t bi_b = (size_t) ಅಡ್ಡಸಾಲು_b * ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + ib;

        traits::mac_pair(pa, bi_a, pb, bi_b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<ದಶಮಾಂಶ>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<ದಶಮಾಂಶ>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    ಆದರೆ (tid == 0) {
        ದಶಮಾಂಶ ಮೊತ್ತ_a = 0.0f;
        ದಶಮಾಂಶ ಮೊತ್ತ_b = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ p = 0; p < GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ; ++p) {
            ಮೊತ್ತ_a += lmem[p * 2 + 0];
            ಮೊತ್ತ_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, ಮೊತ್ತ_a, ಮೊತ್ತ_b);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                           dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q2_K>(
                    esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                           dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q3_K>(
                    esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                           dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q4_K>(
                    esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                           dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q5_K>(
                    esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                           dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups = (nrows + 1) / 2;
    stream->submit([&](sycl::handler &h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)workgroups * GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<GGML_TYPE_Q6_K>(
                    esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy{ vx, dst }, y, ncols, nrows, lmem, it);
            });
    });
}

// Q8_0 SOA reorder layout: [qs: nb*QK8_0] [d: nb*sizeof(half)].
// Process eight blocks per stripe and process remaining blocks one at a time.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ NBLK>
ESIMD_INLINE ಶೂನ್ಯ q8_0_mac_stripe(
        ಸ್ಥಿರಾಂಕ int8_t * qs_a, ಸ್ಥಿರಾಂಕ int8_t * qs_b,
        ಸ್ಥಿರಾಂಕ sycl::half * d_a, ಸ್ಥಿರಾಂಕ sycl::half * d_b, ತಾರ್ಕಿಕ has_b,
        sycl::ext::intel::esimd::simd<ದಶಮಾಂಶ, 32 * NBLK> & y_vec,
        sycl::ext::intel::esimd::simd<ದಶಮಾಂಶ, 32> & acc_a,
        sycl::ext::intel::esimd::simd<ದಶಮಾಂಶ, 32> & acc_b) {
    ಬಳಸಿ ನಾಮಾಕಾಶ sycl::ext::intel::esimd;

    simd<int8_t, 32 * NBLK> qa = ಖಂಡ_ತುಂಬು<int8_t, 32 * NBLK>(qs_a);
    simd<int8_t, 32 * NBLK> qb = 0;
    // Scale rows can be only 2-byte aligned when nblk_row is odd.
    simd<sycl::half, NBLK>  da = ಖಂಡ_ತುಂಬು<sycl::half, NBLK>(d_a, element_aligned_ಟ್ಯಾಗ್{});
    simd<sycl::half, NBLK>  db = 0;
    ಆದರೆ (has_b) {
        qb = ಖಂಡ_ತುಂಬು<int8_t, 32 * NBLK>(qs_b);
        db = ಖಂಡ_ತುಂಬು<sycl::half, NBLK>(d_b, element_aligned_ಟ್ಯಾಗ್{});
    }

    simd<ದಶಮಾಂಶ, NBLK> da_f = convert<ದಶಮಾಂಶ>(da);
    simd<ದಶಮಾಂಶ, NBLK> db_f = convert<ದಶಮಾಂಶ>(db);

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < NBLK; ++s) {
        simd<ದಶಮಾಂಶ, 32>  y_s  = y_vec.ಮಾದರಿ select<32, 1>(s * 32);
        simd<int8_t, 32> qa_s = qa.ಮಾದರಿ select<32, 1>(s * 32);
        simd<int8_t, 32> qb_s = qb.ಮಾದರಿ select<32, 1>(s * 32);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sa = da_f[s];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sb = db_f[s];
        acc_a += y_s * (convert<ದಶಮಾಂಶ>(qa_s) * sa);
        acc_b += y_s * (convert<ದಶಮಾಂಶ>(qb_s) * sb);
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ WG, ಬಗೆ_ಹೆಸರು Policy>
ESIMD_INLINE ಶೂನ್ಯ dequantize_mul_mat_vec_q8_0_reorder_esimd(
        ಸ್ಥಿರಾಂಕ Policy & policy, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) {
    ಬಳಸಿ ನಾಮಾಕಾಶ sycl::ext::intel::esimd;

    constexpr ಪೂರ್ಣಾಂಕ STRIPE = 8;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          nblk_ಅಡ್ಡಸಾಲು = ncols / QK8_0;
    ಸ್ಥಿರಾಂಕ size_t       nb       = (size_t) nrows * nblk_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ int8_t *     qs_a     = (ಸ್ಥಿರಾಂಕ int8_t *) policy.w_a();
    ಸ್ಥಿರಾಂಕ int8_t *     qs_b     = (ಸ್ಥಿರಾಂಕ int8_t *) policy.w_b();
    ಸ್ಥಿರಾಂಕ sycl::half * d_a      = (ಸ್ಥಿರಾಂಕ sycl::half *) (qs_a + nb * QK8_0);
    ಸ್ಥಿರಾಂಕ sycl::half * d_b      = (ಸ್ಥಿರಾಂಕ sycl::half *) (qs_b + nb * QK8_0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  tid   = it.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  g     = it.get_ಗುಂಪು(0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_b = policy.has_b(g, nrows);

    ಸ್ಥಿರಾಂಕ size_t ತಳ_a = (size_t) policy.ಅಡ್ಡಸಾಲು_a(g) * nblk_ಅಡ್ಡಸಾಲು;
    ಸ್ಥಿರಾಂಕ size_t ತಳ_b = (size_t) policy.ಅಡ್ಡಸಾಲು_b(g) * nblk_ಅಡ್ಡಸಾಲು;

    simd<ದಶಮಾಂಶ, 32> acc_a = 0.0f;
    simd<ದಶಮಾಂಶ, 32> acc_b = 0.0f;

    // Each thread processes one contiguous stripe.
    ಪೂರ್ಣಾಂಕ ib = 0;
    ಗಾಗಿ (; ib + WG * STRIPE <= nblk_ಅಡ್ಡಸಾಲು; ib += WG * STRIPE) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b = ib + tid * STRIPE;
        simd<ದಶಮಾಂಶ, 256> y_vec = ಖಂಡ_ತುಂಬು<ದಶಮಾಂಶ, 256>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<STRIPE>(qs_a + (ತಳ_a + b) * QK8_0, qs_b + (ತಳ_b + b) * QK8_0,
                                d_a + ತಳ_a + b, d_b + ತಳ_b + b, has_b, y_vec, acc_a, acc_b);
    }

    // Distribute remaining blocks across the work-group.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = ib + tid; b < nblk_ಅಡ್ಡಸಾಲು; b += WG) {
        simd<ದಶಮಾಂಶ, 32> y_vec = ಖಂಡ_ತುಂಬು<ದಶಮಾಂಶ, 32>(y + (size_t) b * QK8_0);
        q8_0_mac_stripe<1>(qs_a + (ತಳ_a + b) * QK8_0, qs_b + (ತಳ_b + b) * QK8_0,
                           d_a + ತಳ_a + b, d_b + ತಳ_b + b, has_b, y_vec, acc_a, acc_b);
    }

    lmem[tid * 2 + 0] = reduce<ದಶಮಾಂಶ>(acc_a, std::plus<>{});
    lmem[tid * 2 + 1] = reduce<ದಶಮಾಂಶ>(acc_b, std::plus<>{});
    it.barrier(sycl::access::fence_space::local_space);

    ಆದರೆ (tid == 0) {
        ದಶಮಾಂಶ ಮೊತ್ತ_a = 0.0f;
        ದಶಮಾಂಶ ಮೊತ್ತ_b = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ p = 0; p < WG; ++p) {
            ಮೊತ್ತ_a += lmem[p * 2 + 0];
            ಮೊತ್ತ_b += lmem[p * 2 + 1];
        }

        policy.epilogue(g, nrows, ಮೊತ್ತ_a, ಮೊತ್ತ_b);
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ WG, ಬಗೆ_ಹೆಸರು Policy>
ಸ್ಥಿರ ಶೂನ್ಯ q8_0_esimd_launch(ಸ್ಥಿರಾಂಕ Policy & policy, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups, dpct::ಸರತಿ_ptr stream) {
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(WG * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * WG), sycl::range<1>(WG)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_q8_0_reorder_esimd<WG>(policy, y, ncols, nrows, lmem, it);
            });
    });
}

// Scale the work-group with the number of blocks per row.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Policy>
ಸ್ಥಿರ ಶೂನ್ಯ q8_0_esimd_launch_scaled(ಸ್ಥಿರಾಂಕ Policy & policy, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblk_ಅಡ್ಡಸಾಲು = ncols / QK8_0;
    ಆದರೆ (nblk_ಅಡ್ಡಸಾಲು >= 64) {
        q8_0_esimd_launch<8>(policy, y, ncols, nrows, workgroups, stream);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (nblk_ಅಡ್ಡಸಾಲು >= 32) {
        q8_0_esimd_launch<4>(policy, y, ncols, nrows, workgroups, stream);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (nblk_ಅಡ್ಡಸಾಲು >= 16) {
        q8_0_esimd_launch<2>(policy, y, ncols, nrows, workgroups, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        q8_0_esimd_launch<1>(policy, y, ncols, nrows, workgroups, stream);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                           dpct::ಸರತಿ_ptr stream) {
    q8_0_esimd_launch_scaled(esimd_dmmv_ಅಡ್ಡಸಾಲು_pair_policy{ vx, dst }, y, ncols, nrows, (nrows + 1) / 2, stream);
}

// ESIMD fused gate+up+GLU on reordered K-quant SOA weights: one output row (gate
// and up) per work-group, two independent 32-wide accumulators per super-block.
// Reuses the DMMV kernels' dequant+MAC primitive (esimd_reorder_q_traits<T>::mac_pair).
ಮಾದರಿ <ggml_type T>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_glu_reorder_esimd(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate,
                                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y, ದಶಮಾಂಶ * dst,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                     ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op,
                                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ workgroups = nrows; // one output row (gate+up) per work-group
    stream->submit([&](sycl::handler & h) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> lmem(sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ * 2), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) workgroups * GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ), sycl::range<1>(GGML_SYCL_DMMV_ESIMD_WG_ಗಾತ್ರ)),
            [=](sycl::nd_ಅಂಶ<1> it) [[intel::sycl_explicit_simd]] {
                dequantize_mul_mat_vec_reorder_esimd<T>(
                    esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, lmem, it);
            });
    });
}

ತಾರ್ಕಿಕ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(ಎಣಿಕೆ ggml_type src0_type, ಎಣಿಕೆ ggml_glu_op glu_op,
                                                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vgate, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * y,
                                                        ದಶಮಾಂಶ * dst, ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ nrows, dpct::ಸರತಿ_ptr stream) {
    ಆಯ್ಕೆ (src0_type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q2_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q3_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q4_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q5_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            dequantize_mul_mat_vec_glu_reorder_esimd<GGML_TYPE_Q6_K>(vx, vgate, y, dst, ncols, nrows, glu_op, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            q8_0_esimd_launch_scaled(esimd_dmmv_glu_policy{ vgate, vx, dst, glu_op }, y, ncols, nrows, nrows, stream);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_DMMV_HAS_ESIMD

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q4_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                     ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q4_k_reorder(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q5_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                     ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q5_k_reorder(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_mul_mat_vec_q6_K_sycl_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                                     ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
                                                     dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ny = 2 / K_QUANTS_ಪ್ರತಿ_ITERATION;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_y = (nrows + ny - 1) / ny;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಖಂಡ_ಅಂಕೆ_y);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, ny, WARP_ಗಾತ್ರ);
    stream->parallel_for(
        sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            dequantize_mul_mat_vec_q6_k_reorder(vx, y, dst, ncols, nrows, ಅಂಶ_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_ddf_i, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1_ddq_i,
    ದಶಮಾಂಶ *dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr &stream) {

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_diff = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    // on some GPUs it is faster to convert src1 to half and to use half precision intrinsics
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ggml_sycl_pool_alloc<sycl::half> src1_dfloat_a(ctx.pool());
    sycl::half *src1_dfloat = ಶೂನ್ಯಸೂಚಕ; // dfloat == half

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
    // The ESIMD Q8_0 kernel reads F32 activations.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ q8_0_esimd = src0->type == GGML_TYPE_Q8_0 && g_ggml_sycl_enable_esimd &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra) &&
                            ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ q8_0_esimd = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    ತಾರ್ಕಿಕ src1_convert_f16 =
        src0->type == GGML_TYPE_Q1_0 ||
        src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_1 ||
        src0->type == GGML_TYPE_Q5_0 || src0->type == GGML_TYPE_Q5_1 ||
        (src0->type == GGML_TYPE_Q8_0 && !q8_0_esimd) || src0->type == GGML_TYPE_F16 ||
        src0->type == GGML_TYPE_BF16;

    ಆದರೆ (src1_convert_f16) {
        scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                             " : converting src1 to fp16");
        src1_dfloat = src1_dfloat_a.alloc(ne00);
        ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16_sycl = ggml_get_ಗಮ್ಯ_fp16_sycl(src1->type, dst);
        GGML_ASSERT(ಗಮ್ಯ_fp16_sycl != ಶೂನ್ಯಸೂಚಕ);
        ಗಮ್ಯ_fp16_sycl(src1_ddf_i, src1_dfloat, ne00, stream);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ dfloat * src1_dfloat = (ಸ್ಥಿರಾಂಕ dfloat *) src1_ddf_i; // dfloat == float, no conversion
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            ಆದರೆ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q1_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q1_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಆದರೆ ((ggml_tensor_extra_gpu*)dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                dequantize_mul_mat_vec_q4_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q4_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            dequantize_mul_mat_vec_q4_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            dequantize_mul_mat_vec_q5_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            dequantize_mul_mat_vec_q5_1_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
                ಆದರೆ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
                {
                    dequantize_mul_mat_vec_q8_0_sycl_reorder(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q8_0_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
                ಆದರೆ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
                {
                    dequantize_mul_mat_vec_q2_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q2_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
                ಆದರೆ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
                {
                    dequantize_mul_mat_vec_q3_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q3_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
                ಆದರೆ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
                {
                    dequantize_mul_mat_vec_q4_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q4_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
                ಆದರೆ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
                {
                    dequantize_mul_mat_vec_q5_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q5_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಆದರೆ ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
                ಆದರೆ (g_ggml_sycl_enable_esimd) {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder_esimd(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
                ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
                {
                    dequantize_mul_mat_vec_q6_K_sycl_reorder(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                dequantize_mul_mat_vec_q6_K_sycl(src0_dd_i, src1_ddf_i, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            convert_mul_mat_vec_f16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            ನಿಲ್ಲಿಸು;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_BF16
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            convert_mul_mat_vec_bf16_sycl(src0_dd_i, src1_dfloat, dst_dd_i, ne00, ಅಡ್ಡಸಾಲು_diff, stream);
            ನಿಲ್ಲಿಸು;
#ಆದರೆ_ಅಂತ್ಯ
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಬರೆ("ggml_sycl_op_dequantize_mul_mat_vec unsupported GGML_TYPE %d\n", src0->type);
            GGML_ABORT("fatal error");
    }

    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(src1_ddq_i);
    GGML_ಬಳಸದ(src1_ncols);
    GGML_ಬಳಸದ(src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
    GGML_ಬಳಸದ(ctx);
}
