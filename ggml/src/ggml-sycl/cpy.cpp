#ಅನ್ತರ್ಭಾವ "cpy.hpp"

#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <vector>

#ಅನ್ತರ್ಭಾವ "dequantize.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/presets.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f32_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    ಭಿನ್ನ *       dsti = (ಭಿನ್ನ *) cdsti;

    *dsti = *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f32_f16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<ಭಿನ್ನ, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f16_f16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::half * xi   = (ಧ್ರುವ sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f16_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::half * xi   = (ಧ್ರುವ sycl::half *) cxi;
    ಭಿನ್ನ *            dsti = (ಭಿನ್ನ *) cdsti;

    *dsti = *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_i16_i16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ int16_t * xi   = (ಧ್ರುವ int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_i32_i32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ int32_t * xi   = (ಧ್ರುವ int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f32_i32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi     = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_i32_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ int32_t * xi   = (ಧ್ರುವ int32_t *) cxi;
    ಭಿನ್ನ *         dsti = (ಭಿನ್ನ *) cdsti;

    *dsti = (ಭಿನ್ನ) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f32_bf16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ ಭಿನ್ನ * xi   = (ಧ್ರುವ ಭಿನ್ನ *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_bf16_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::ext::oneapi::bfloat16 * xi   = (ಧ್ರುವ sycl::ext::oneapi::bfloat16 *) cxi;
    ಭಿನ್ನ *            dsti = (ಭಿನ್ನ *) cdsti;

    *dsti = static_cast<ಭಿನ್ನ>(*xi);
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_bf16_bf16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::ext::oneapi::bfloat16 * xi   = (ಧ್ರುವ sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_f16_bf16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::half * xi   = (ಧ್ರುವ sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<ಭಿನ್ನ>(*xi));
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_1_bf16_f16(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::ext::oneapi::bfloat16 * xi   = (ಧ್ರುವ sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<ಭಿನ್ನ>(*xi));
}
#endif

template <cpy_kernel_t cpy_1>
ಸ್ಥಿರ ಶೂನ್ಯ cpy_f32_f16(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02,
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11,
                        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13,
                        ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    ಯದಿ (i >= ne) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03      = i / (ne00 * ne01 * ne02);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13        = i / (ne10 * ne11 * ne12);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_q_q(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ T * xi = (ಧ್ರುವ T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_q8_0_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಭಿನ್ನ * cdstf = (ಭಿನ್ನ *) (cdsti);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_q2_0_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ block_q2_0 * xi = (ಧ್ರುವ block_q2_0 *) cxi;
    ಭಿನ್ನ * cdstf = (ಭಿನ್ನ *) cdsti;

    ಧ್ರುವ ಭಿನ್ನ d = xi->d;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK2_0; ++j) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ byte_index = j / 4;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_offset = (j % 4) * 2;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ q = (xi->qs[byte_index] >> bit_offset) & 0x3;
        cdstf[j] = (ಭಿನ್ನ) (q - 1) * d;
    }
}



template <dequantize_kernel_t dequant, ಪೂರ್ಣಾಙ್ಕ qk> ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_q_f32(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಭಿನ್ನ * cdstf = (ಭಿನ್ನ *) (cdsti);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, ಪೂರ್ಣಾಙ್ಕ qk>
ಸ್ಥಿರ ಶೂನ್ಯ cpy_q_q(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13,
                      ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    ಯದಿ (i >= ne) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03      = i / (ne00 * ne01 * ne02);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13        = i / (ne10 * ne11 * ne12);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, ಪೂರ್ಣಾಙ್ಕ qk>
ಸ್ಥಿರ ಶೂನ್ಯ cpy_f32_q(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13,
                      ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    ಯದಿ (i >= ne) {
        ಪ್ರತ್ಯಾಗಮ;
    }


    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03      = i / (ne00 * ne01 * ne02);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13        = i / (ne10 * ne11 * ne12);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, ಪೂರ್ಣಾಙ್ಕ qk>
ಸ್ಥಿರ ಶೂನ್ಯ cpy_q_f32(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11,
                      ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13,
                      ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    ಯದಿ (i >= ne) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i03      = i / (ne00 * ne01 * ne02);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i13        = i / (ne10 * ne11 * ne12);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f16_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_f16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_i32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_i32_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_q8_0_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q8_0_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q2_0_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_q4_0_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q4_0_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_q4_1_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q4_1_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_q5_0_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q5_0_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_q5_1_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q5_1_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_mxfp4_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_iq4_nl_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_f16_q4_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::half * xi = (ಧ್ರುವ sycl::half *) cxi;
    ಭಿನ್ನ xf[QK4_0];

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_0; ++j) {
        xf[j] = (ಭಿನ್ನ) xi[j];
    }

    cpy_blck_f32_q4_0((ಧ್ರುವ ವರ್ಣ *) xf, cdsti);
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_f16_q4_1(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::half * xi = (ಧ್ರುವ sycl::half *) cxi;
    ಭಿನ್ನ xf[QK4_1];

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK4_1; ++j) {
        xf[j] = (ಭಿನ್ನ) xi[j];
    }

    cpy_blck_f32_q4_1((ಧ್ರುವ ವರ್ಣ *) xf, cdsti);
}

ಸ್ಥಿರ ಶೂನ್ಯ cpy_blck_f16_q5_0(ಧ್ರುವ ವರ್ಣ * cxi, ವರ್ಣ * cdsti) {
    ಧ್ರುವ sycl::half * xi = (ಧ್ರುವ sycl::half *) cxi;
    ಭಿನ್ನ xf[QK5_0];

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK5_0; ++j) {
        xf[j] = (ಭಿನ್ನ) xi[j];
    }

    cpy_blck_f32_q5_0((ಧ್ರುವ ವರ್ಣ *) xf, cdsti);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f16_q4_0_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f16_q4_1_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f16_q5_0_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_is_quantized_type(ಗಣನಾ ggml_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_NVFP4:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_IQ2_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ2_XS:
        ಪಕ್ಷ GGML_TYPE_IQ2_S:
        ಪಕ್ಷ GGML_TYPE_IQ3_XXS:
        ಪಕ್ಷ GGML_TYPE_IQ3_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_S:
        ಪಕ್ಷ GGML_TYPE_IQ1_M:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_can_quantize_rows_sycl(ಗಣನಾ ggml_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_Q1_0:
        ಪಕ್ಷ GGML_TYPE_Q2_0:
        ಪಕ್ಷ GGML_TYPE_Q4_0:
        ಪಕ್ಷ GGML_TYPE_Q4_1:
        ಪಕ್ಷ GGML_TYPE_Q5_0:
        ಪಕ್ಷ GGML_TYPE_Q5_1:
        ಪಕ್ಷ GGML_TYPE_Q8_0:
        ಪಕ್ಷ GGML_TYPE_MXFP4:
        ಪಕ್ಷ GGML_TYPE_NVFP4:
        ಪಕ್ಷ GGML_TYPE_Q2_K:
        ಪಕ್ಷ GGML_TYPE_Q3_K:
        ಪಕ್ಷ GGML_TYPE_Q4_K:
        ಪಕ್ಷ GGML_TYPE_Q5_K:
        ಪಕ್ಷ GGML_TYPE_Q6_K:
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
        ಪಕ್ಷ GGML_TYPE_IQ4_XS:
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}

template <typename SrcScalar>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_sycl_src_to_f32(ಧ್ರುವ SrcScalar & x) {
    ಪ್ರತ್ಯಾಗಮ (ಭಿನ್ನ) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(ಧ್ರುವ sycl::ext::oneapi::bfloat16 & x) {
    ಪ್ರತ್ಯಾಗಮ static_cast<ಭಿನ್ನ>(x);
}

template <>
ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_sycl_src_to_f32<ggml_bf16_t>(ಧ್ರುವ ggml_bf16_t & x) {
    ಸಙ್ಘ {
        uint32_t u32;
        ಭಿನ್ನ f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    ಪ್ರತ್ಯಾಗಮ value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, ಪೂರ್ಣಾಙ್ಕ qk>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_quantize_rows_q(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ int64_t ne,
                                      ಧ್ರುವ int64_t ne00, ಧ್ರುವ int64_t ne01, ಧ್ರುವ int64_t ne02,
                                      ಧ್ರುವ size_t nb00, ಧ್ರುವ size_t nb01, ಧ್ರುವ size_t nb02, ಧ್ರುವ size_t nb03,
                                      ಧ್ರುವ int64_t ne10, ಧ್ರುವ int64_t ne11, ಧ್ರುವ int64_t ne12,
                                      ಧ್ರುವ size_t nb10, ಧ್ರುವ size_t nb11, ಧ್ರುವ size_t nb12, ಧ್ರುವ size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    ಧ್ರುವ int64_t total_blocks = ne / qk;
    constexpr ಪೂರ್ಣಾಙ್ಕ block_size = 256;
    ಧ್ರುವ int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        ಧ್ರುವ int64_t block_idx = item_ct1.get_global_linear_id();
        ಯದಿ (block_idx >= total_blocks) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಧ್ರುವ int64_t i = block_idx * qk;

        ಧ್ರುವ int64_t i03 = i / (ne00 * ne01 * ne02);
        ಧ್ರುವ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        ಧ್ರುವ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        ಧ್ರುವ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        ಧ್ರುವ size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        ಧ್ರುವ int64_t i13 = i / (ne10 * ne11 * ne12);
        ಧ್ರುವ int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        ಧ್ರುವ int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        ಧ್ರುವ int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        ಧ್ರುವ size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        ಭಿನ್ನ xf[qk];
        ಯದಿ (nb00 == ಪರಿಮಾಣ(SrcScalar)) {
            ಧ್ರುವ SrcScalar * src_row = (ಧ್ರುವ SrcScalar *) (cx + x_offset);
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } ಅನ್ಯಥಾ {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < qk; ++j) {
                ಧ್ರುವ SrcScalar * src_val = (ಧ್ರುವ SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((ಧ್ರುವ ವರ್ಣ *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_quantize_rows_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1,
                                         ಧ್ರುವ int64_t ne, ಧ್ರುವ int64_t ne00, ಧ್ರುವ int64_t ne01, ಧ್ರುವ int64_t ne02,
                                         ಧ್ರುವ size_t nb00, ಧ್ರುವ size_t nb01, ಧ್ರುವ size_t nb02, ಧ್ರುವ size_t nb03,
                                         ಧ್ರುವ int64_t ne10, ಧ್ರುವ int64_t ne11, ಧ್ರುವ int64_t ne12, ಧ್ರುವ size_t nb10,
                                         ಧ್ರುವ size_t nb11, ಧ್ರುವ size_t nb12, ಧ್ರುವ size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    ವಿಕಲ್ಪ (src1->type) {
        ಪಕ್ಷ GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f16_f16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_i16_i16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_i32_i32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q8_0_q8_0(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q5_0_q5_0(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q5_1_q5_1(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q4_0_q4_0(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q4_1_q4_1(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q1_0_q1_0(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q2_0_q2_0(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_mxfp4_mxfp4(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_nvfp4_nvfp4(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q2_K_q2_K(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q3_K_q3_K(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q4_K_q4_K(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q5_K_q5_K(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_q6_K_q6_K(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq2_xxs_iq2_xxs(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq2_xs_iq2_xs(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq2_s_iq2_s(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq3_xxs_iq3_xxs(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq1_s_iq1_s(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq1_m_iq1_m(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq4_nl_iq4_nl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq3_s_iq3_s(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_iq4_xs_iq4_xs(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f32_bf16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_bf16_f32_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_bf16_bf16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_f16_bf16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_cpy_bf16_f16_sycl(ಧ್ರುವ ವರ್ಣ * cx, ವರ್ಣ * cdst, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne01,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb00, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb01, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb02, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb03,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne11, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ne12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb10, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb11,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb12, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nb13, queue_ptr stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

ಶೂನ್ಯ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ಧ್ರುವ ggml_tensor * src0, ಧ್ರುವ ggml_tensor * src1) ಯತ್ನ {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    ಧ್ರುವ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    ವರ್ಣ * src0_ddc = (ವರ್ಣ *) src0->data;
    ವರ್ಣ * src1_ddc = (ವರ್ಣ *) src1->data;
    ಯದಿ ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ಭಿನ್ನ>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ಅನ್ಯಥಾ ಯದಿ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } ಅನ್ಯಥಾ {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} ಗೃಹಾಣ (ಧ್ರುವ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ಶೂನ್ಯ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
