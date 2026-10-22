#ଅନ୍ତର୍ଭାଵ "cpy.hpp"

#ଅନ୍ତର୍ଭାଵ <float.h>
#ଅନ୍ତର୍ଭାଵ <vector>

#ଅନ୍ତର୍ଭାଵ "dequantize.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/presets.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f32_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    ଦଶମିକ *       dsti = (ଦଶମିକ *) cdsti;

    *dsti = *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f32_f16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<ଦଶମିକ, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f16_f16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::half * xi   = (ଅଚଳ sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f16_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::half * xi   = (ଅଚଳ sycl::half *) cxi;
    ଦଶମିକ *            dsti = (ଦଶମିକ *) cdsti;

    *dsti = *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_i16_i16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ int16_t * xi   = (ଅଚଳ int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_i32_i32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ int32_t * xi   = (ଅଚଳ int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f32_i32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi     = (ଅଚଳ ଦଶମିକ *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_i32_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ int32_t * xi   = (ଅଚଳ int32_t *) cxi;
    ଦଶମିକ *         dsti = (ଦଶମିକ *) cdsti;

    *dsti = (ଦଶମିକ) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f32_bf16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ ଦଶମିକ * xi   = (ଅଚଳ ଦଶମିକ *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_bf16_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::ext::oneapi::bfloat16 * xi   = (ଅଚଳ sycl::ext::oneapi::bfloat16 *) cxi;
    ଦଶମିକ *            dsti = (ଦଶମିକ *) cdsti;

    *dsti = static_cast<ଦଶମିକ>(*xi);
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_bf16_bf16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::ext::oneapi::bfloat16 * xi   = (ଅଚଳ sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_f16_bf16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::half * xi   = (ଅଚଳ sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<ଦଶମିକ>(*xi));
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_1_bf16_f16(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::ext::oneapi::bfloat16 * xi   = (ଅଚଳ sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<ଦଶମିକ>(*xi));
}
#endif

template <cpy_kernel_t cpy_1>
ସ୍ଥିର ଶୂନ୍ୟ cpy_f32_f16(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13,
                        ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    ଯଦି (i >= ne) {
        ଫେରନ୍ତୁ;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03      = i / (ne00 * ne01 * ne02);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13        = i / (ne10 * ne11 * ne12);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_q_q(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ T * xi = (ଅଚଳ T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_q8_0_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଦଶମିକ * cdstf = (ଦଶମିକ *) (cdsti);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_q2_0_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ block_q2_0 * xi = (ଅଚଳ block_q2_0 *) cxi;
    ଦଶମିକ * cdstf = (ଦଶମିକ *) cdsti;

    ଅଚଳ ଦଶମିକ d = xi->d;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK2_0; ++j) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte_index = j / 4;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା bit_offset = (j % 4) * 2;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା q = (xi->qs[byte_index] >> bit_offset) & 0x3;
        cdstf[j] = (ଦଶମିକ) (q - 1) * d;
    }
}



template <dequantize_kernel_t dequant, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk> ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_q_f32(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଦଶମିକ * cdstf = (ଦଶମିକ *) (cdsti);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk>
ସ୍ଥିର ଶୂନ୍ୟ cpy_q_q(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13,
                      ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    ଯଦି (i >= ne) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03      = i / (ne00 * ne01 * ne02);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13        = i / (ne10 * ne11 * ne12);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk>
ସ୍ଥିର ଶୂନ୍ୟ cpy_f32_q(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13,
                      ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    ଯଦି (i >= ne) {
        ଫେରନ୍ତୁ;
    }


    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03      = i / (ne00 * ne01 * ne02);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13        = i / (ne10 * ne11 * ne12);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk>
ସ୍ଥିର ଶୂନ୍ୟ cpy_q_f32(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11,
                      ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13,
                      ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    ଯଦି (i >= ne) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i03      = i / (ne00 * ne01 * ne02);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i13        = i / (ne10 * ne11 * ne12);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f16_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_f16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_i32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_i32_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_q8_0_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q8_0_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q2_0_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_q4_0_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q4_0_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_q4_1_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q4_1_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_q5_0_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q5_0_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_q5_1_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q5_1_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_mxfp4_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_iq4_nl_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                     ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_f16_q4_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::half * xi = (ଅଚଳ sycl::half *) cxi;
    ଦଶମିକ xf[QK4_0];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_0; ++j) {
        xf[j] = (ଦଶମିକ) xi[j];
    }

    cpy_blck_f32_q4_0((ଅଚଳ ଅକ୍ଷର *) xf, cdsti);
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_f16_q4_1(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::half * xi = (ଅଚଳ sycl::half *) cxi;
    ଦଶମିକ xf[QK4_1];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_1; ++j) {
        xf[j] = (ଦଶମିକ) xi[j];
    }

    cpy_blck_f32_q4_1((ଅଚଳ ଅକ୍ଷର *) xf, cdsti);
}

ସ୍ଥିର ଶୂନ୍ୟ cpy_blck_f16_q5_0(ଅଚଳ ଅକ୍ଷର * cxi, ଅକ୍ଷର * cdsti) {
    ଅଚଳ sycl::half * xi = (ଅଚଳ sycl::half *) cxi;
    ଦଶମିକ xf[QK5_0];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK5_0; ++j) {
        xf[j] = (ଦଶମିକ) xi[j];
    }

    cpy_blck_f32_q5_0((ଅଚଳ ଅକ୍ଷର *) xf, cdsti);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f16_q4_0_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f16_q4_1_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f16_q5_0_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_is_quantized_type(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_can_quantize_rows_sycl(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q2_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_MXFP4:
        ସ୍ଥିତି GGML_TYPE_NVFP4:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

template <typename SrcScalar>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_sycl_src_to_f32(ଅଚଳ SrcScalar & x) {
    ଫେରନ୍ତୁ (ଦଶମିକ) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(ଅଚଳ sycl::ext::oneapi::bfloat16 & x) {
    ଫେରନ୍ତୁ static_cast<ଦଶମିକ>(x);
}

template <>
ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_sycl_src_to_f32<ggml_bf16_t>(ଅଚଳ ggml_bf16_t & x) {
    ସଂଘ {
        uint32_t u32;
        ଦଶମିକ f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    ଫେରନ୍ତୁ value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk>
ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_quantize_rows_q(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ int64_t ne,
                                      ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02,
                                      ଅଚଳ size_t nb00, ଅଚଳ size_t nb01, ଅଚଳ size_t nb02, ଅଚଳ size_t nb03,
                                      ଅଚଳ int64_t ne10, ଅଚଳ int64_t ne11, ଅଚଳ int64_t ne12,
                                      ଅଚଳ size_t nb10, ଅଚଳ size_t nb11, ଅଚଳ size_t nb12, ଅଚଳ size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    ଅଚଳ int64_t total_blocks = ne / qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = 256;
    ଅଚଳ int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        ଅଚଳ int64_t block_idx = item_ct1.get_global_linear_id();
        ଯଦି (block_idx >= total_blocks) {
            ଫେରନ୍ତୁ;
        }

        ଅଚଳ int64_t i = block_idx * qk;

        ଅଚଳ int64_t i03 = i / (ne00 * ne01 * ne02);
        ଅଚଳ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        ଅଚଳ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        ଅଚଳ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        ଅଚଳ size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        ଅଚଳ int64_t i13 = i / (ne10 * ne11 * ne12);
        ଅଚଳ int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        ଅଚଳ int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        ଅଚଳ int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        ଅଚଳ size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        ଦଶମିକ xf[qk];
        ଯଦି (nb00 == ଆକାର(SrcScalar)) {
            ଅଚଳ SrcScalar * src_row = (ଅଚଳ SrcScalar *) (cx + x_offset);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < qk; ++j) {
                ଅଚଳ SrcScalar * src_val = (ଅଚଳ SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((ଅଚଳ ଅକ୍ଷର *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_quantize_rows_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1,
                                         ଅଚଳ int64_t ne, ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02,
                                         ଅଚଳ size_t nb00, ଅଚଳ size_t nb01, ଅଚଳ size_t nb02, ଅଚଳ size_t nb03,
                                         ଅଚଳ int64_t ne10, ଅଚଳ int64_t ne11, ଅଚଳ int64_t ne12, ଅଚଳ size_t nb10,
                                         ଅଚଳ size_t nb11, ଅଚଳ size_t nb12, ଅଚଳ size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    ବାଛ (src1->type) {
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f16_f16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_i16_i16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_i32_i32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q8_0_q8_0(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q5_0_q5_0(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q5_1_q5_1(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q4_0_q4_0(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q4_1_q4_1(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q1_0_q1_0(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q2_0_q2_0(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_mxfp4_mxfp4(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_nvfp4_nvfp4(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q2_K_q2_K(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q3_K_q3_K(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q4_K_q4_K(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q5_K_q5_K(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_q6_K_q6_K(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq2_xxs_iq2_xxs(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq2_xs_iq2_xs(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq2_s_iq2_s(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq3_xxs_iq3_xxs(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq1_s_iq1_s(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq1_m_iq1_m(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq4_nl_iq4_nl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq3_s_iq3_s(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_iq4_xs_iq4_xs(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f32_bf16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_bf16_f32_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_bf16_bf16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_f16_bf16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_cpy_bf16_f16_sycl(ଅଚଳ ଅକ୍ଷର * cx, ଅକ୍ଷର * cdst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne01,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb00, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb01, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb02, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb03,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne11, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb10, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb11,
                                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb12, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb13, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

ଶୂନ୍ୟ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1) ଚେଷ୍ଟା {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    ଅଚଳ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    ଅକ୍ଷର * src0_ddc = (ଅକ୍ଷର *) src0->data;
    ଅକ୍ଷର * src1_ddc = (ଅକ୍ଷର *) src1->data;
    ଯଦି ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ଦଶମିକ>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } ଅନ୍ୟଥା ଯଦି (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } ଅନ୍ୟଥା {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} ଧର (ଅଚଳ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ଶୂନ୍ୟ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
