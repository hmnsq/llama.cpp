#ਅਨ੍ਤਰ੍ਭਾਵ "cpy.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਅਨ੍ਤਰ੍ਭਾਵ "dequantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/presets.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

ਸਥਿਰ ਖਾਲੀ cpy_1_f32_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    ਦਸ਼ਮਲਵ *       dsti = (ਦਸ਼ਮਲਵ *) cdsti;

    *dsti = *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_f32_f16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<ਦਸ਼ਮਲਵ, 1>(*xi).convert<sycl::half, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
}

ਸਥਿਰ ਖਾਲੀ cpy_1_f16_f16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::half * xi   = (ਅਚਲ sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_f16_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::half * xi   = (ਅਚਲ sycl::half *) cxi;
    ਦਸ਼ਮਲਵ *            dsti = (ਦਸ਼ਮਲਵ *) cdsti;

    *dsti = *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_i16_i16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ int16_t * xi   = (ਅਚਲ int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_i32_i32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ int32_t * xi   = (ਅਚਲ int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_f32_i32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi     = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_i32_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ int32_t * xi   = (ਅਚਲ int32_t *) cxi;
    ਦਸ਼ਮਲਵ *         dsti = (ਦਸ਼ਮਲਵ *) cdsti;

    *dsti = (ਦਸ਼ਮਲਵ) *xi;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
ਸਥਿਰ ਖਾਲੀ cpy_1_f32_bf16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਦਸ਼ਮਲਵ * xi   = (ਅਚਲ ਦਸ਼ਮਲਵ *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

ਸਥਿਰ ਖਾਲੀ cpy_1_bf16_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::ext::oneapi::bfloat16 * xi   = (ਅਚਲ sycl::ext::oneapi::bfloat16 *) cxi;
    ਦਸ਼ਮਲਵ *            dsti = (ਦਸ਼ਮਲਵ *) cdsti;

    *dsti = static_cast<ਦਸ਼ਮਲਵ>(*xi);
}

ਸਥਿਰ ਖਾਲੀ cpy_1_bf16_bf16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::ext::oneapi::bfloat16 * xi   = (ਅਚਲ sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

ਸਥਿਰ ਖਾਲੀ cpy_1_f16_bf16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::half * xi   = (ਅਚਲ sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<ਦਸ਼ਮਲਵ>(*xi));
}

ਸਥਿਰ ਖਾਲੀ cpy_1_bf16_f16(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::ext::oneapi::bfloat16 * xi   = (ਅਚਲ sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<ਦਸ਼ਮਲਵ>(*xi));
}
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <cpy_kernel_t cpy_1>
ਸਥਿਰ ਖਾਲੀ cpy_f32_f16(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01, ਅਚਲ ਪੂਰਨ ne02,
                        ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03, ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11,
                        ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11, ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13,
                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2);

    ਜੇ (i >= ne) {
        ਵਾਪਸ;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    ਅਚਲ ਪੂਰਨ i03      = i / (ne00 * ne01 * ne02);
    ਅਚਲ ਪੂਰਨ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ਅਚਲ ਪੂਰਨ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ਅਚਲ ਪੂਰਨ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ਅਚਲ ਪੂਰਨ x_ਵਿੱਥ = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ਅਚਲ ਪੂਰਨ i13        = i / (ne10 * ne11 * ne12);
    ਅਚਲ ਪੂਰਨ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ਅਚਲ ਪੂਰਨ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ਅਚਲ ਪੂਰਨ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ਅਚਲ ਪੂਰਨ dst_ਵਿੱਥ = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_ਵਿੱਥ, cdst + dst_ਵਿੱਥ);
}


/* quantized type same copy */
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ cpy_blck_q_q(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ T * xi = (ਅਚਲ T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


ਸਥਿਰ ਖਾਲੀ cpy_blck_q8_0_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਦਸ਼ਮਲਵ * cdstf = (ਦਸ਼ਮਲਵ *) (cdsti);

    ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

ਸਥਿਰ ਖਾਲੀ cpy_blck_q2_0_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ ਖੰਡ_q2_0 * xi = (ਅਚਲ ਖੰਡ_q2_0 *) cxi;
    ਦਸ਼ਮਲਵ * cdstf = (ਦਸ਼ਮਲਵ *) cdsti;

    ਅਚਲ ਦਸ਼ਮਲਵ d = xi->d;

    ਲਈ (ਪੂਰਨ j = 0; j < QK2_0; ++j) {
        ਅਚਲ ਪੂਰਨ byte_ਸੂਚਕਾਂਕ = j / 4;
        ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_ਵਿੱਥ = (j % 4) * 2;
        ਅਚਲ ਪੂਰਨ q = (xi->qs[byte_ਸੂਚਕਾਂਕ] >> ਦੋਆਧਾਰੀ_ਅੰਕ_ਵਿੱਥ) & 0x3;
        cdstf[j] = (ਦਸ਼ਮਲਵ) (q - 1) * d;
    }
}



ਸਾਂਚਾ <dequantize_kernel_t dequant, ਪੂਰਨ qk> ਸਥਿਰ ਖਾਲੀ cpy_blck_q_f32(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਦਸ਼ਮਲਵ * cdstf = (ਦਸ਼ਮਲਵ *) (cdsti);

    ਲਈ (ਪੂਰਨ j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ qk>
ਸਥਿਰ ਖਾਲੀ cpy_q_q(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01, ਅਚਲ ਪੂਰਨ ne02,
                      ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03, ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11,
                      ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11, ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13,
                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i = (ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2)) * qk;

    ਜੇ (i >= ne) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i03      = i / (ne00 * ne01 * ne02);
    ਅਚਲ ਪੂਰਨ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ਅਚਲ ਪੂਰਨ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ਅਚਲ ਪੂਰਨ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ਅਚਲ ਪੂਰਨ x_ਵਿੱਥ = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    ਅਚਲ ਪੂਰਨ i13        = i / (ne10 * ne11 * ne12);
    ਅਚਲ ਪੂਰਨ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ਅਚਲ ਪੂਰਨ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ਅਚਲ ਪੂਰਨ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ਅਚਲ ਪੂਰਨ dst_ਵਿੱਥ = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_ਵਿੱਥ, cdst + dst_ਵਿੱਥ);
}

ਸਾਂਚਾ <cpy_kernel_t cpy_blck, ਪੂਰਨ qk>
ਸਥਿਰ ਖਾਲੀ cpy_f32_q(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01, ਅਚਲ ਪੂਰਨ ne02,
                      ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03, ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11,
                      ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11, ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13,
                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i = (ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2)) * qk;

    ਜੇ (i >= ne) {
        ਵਾਪਸ;
    }


    ਅਚਲ ਪੂਰਨ i03      = i / (ne00 * ne01 * ne02);
    ਅਚਲ ਪੂਰਨ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ਅਚਲ ਪੂਰਨ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ਅਚਲ ਪੂਰਨ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ਅਚਲ ਪੂਰਨ x_ਵਿੱਥ = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ਅਚਲ ਪੂਰਨ i13        = i / (ne10 * ne11 * ne12);
    ਅਚਲ ਪੂਰਨ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ਅਚਲ ਪੂਰਨ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ਅਚਲ ਪੂਰਨ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ਅਚਲ ਪੂਰਨ dst_ਵਿੱਥ = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_ਵਿੱਥ, cdst + dst_ਵਿੱਥ);
}

ਸਾਂਚਾ <cpy_kernel_t cpy_blck, ਪੂਰਨ qk>
ਸਥਿਰ ਖਾਲੀ cpy_q_f32(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01, ਅਚਲ ਪੂਰਨ ne02,
                      ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03, ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11,
                      ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11, ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13,
                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i = (ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) + ਵਸਤੂ_ct1.get_local_id(2)) * qk;

    ਜੇ (i >= ne) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i03      = i / (ne00 * ne01 * ne02);
    ਅਚਲ ਪੂਰਨ i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ਅਚਲ ਪੂਰਨ i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ਅਚਲ ਪੂਰਨ i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ਅਚਲ ਪੂਰਨ x_ਵਿੱਥ = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ਅਚਲ ਪੂਰਨ i13        = i / (ne10 * ne11 * ne12);
    ਅਚਲ ਪੂਰਨ i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ਅਚਲ ਪੂਰਨ i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ਅਚਲ ਪੂਰਨ i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ਅਚਲ ਪੂਰਨ dst_ਵਿੱਥ = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_ਵਿੱਥ, cdst + dst_ਵਿੱਥ);
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f16_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_f16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_i32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_i32_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_q8_0_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK8_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q8_0_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK8_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q2_0_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK2_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_q4_0_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q4_0_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_q4_1_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q4_1_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_q5_0_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q5_0_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_q5_1_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q5_1_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_mxfp4_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                    ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                    ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                    ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_MXFP4, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_iq4_nl_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                     ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                     ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                     ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_NL, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ cpy_blck_f16_q4_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::half * xi = (ਅਚਲ sycl::half *) cxi;
    ਦਸ਼ਮਲਵ xf[QK4_0];

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_0; ++j) {
        xf[j] = (ਦਸ਼ਮਲਵ) xi[j];
    }

    cpy_blck_f32_q4_0((ਅਚਲ ਅੱਖਰ *) xf, cdsti);
}

ਸਥਿਰ ਖਾਲੀ cpy_blck_f16_q4_1(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::half * xi = (ਅਚਲ sycl::half *) cxi;
    ਦਸ਼ਮਲਵ xf[QK4_1];

    ਲਈ (ਪੂਰਨ j = 0; j < QK4_1; ++j) {
        xf[j] = (ਦਸ਼ਮਲਵ) xi[j];
    }

    cpy_blck_f32_q4_1((ਅਚਲ ਅੱਖਰ *) xf, cdsti);
}

ਸਥਿਰ ਖਾਲੀ cpy_blck_f16_q5_0(ਅਚਲ ਅੱਖਰ * cxi, ਅੱਖਰ * cdsti) {
    ਅਚਲ sycl::half * xi = (ਅਚਲ sycl::half *) cxi;
    ਦਸ਼ਮਲਵ xf[QK5_0];

    ਲਈ (ਪੂਰਨ j = 0; j < QK5_0; ++j) {
        xf[j] = (ਦਸ਼ਮਲਵ) xi[j];
    }

    cpy_blck_f32_q5_0((ਅਚਲ ਅੱਖਰ *) xf, cdsti);
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f16_q4_0_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f16_q4_1_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f16_q5_0_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                                           sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_is_quantized_type(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_can_quantize_ਪੰਕਤੀਆਂ_sycl(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q2_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_MXFP4:
        ਹਾਲਤ GGML_TYPE_NVFP4:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SrcScalar>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_sycl_src_ਮੰਜ਼ਿਲ_f32(ਅਚਲ SrcScalar & x) {
    ਵਾਪਸ (ਦਸ਼ਮਲਵ) x;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_sycl_src_ਮੰਜ਼ਿਲ_f32<sycl::ext::oneapi::bfloat16>(ਅਚਲ sycl::ext::oneapi::bfloat16 & x) {
    ਵਾਪਸ static_cast<ਦਸ਼ਮਲਵ>(x);
}

ਸਾਂਚਾ <>
ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_sycl_src_ਮੰਜ਼ਿਲ_f32<ggml_bf16_t>(ਅਚਲ ggml_bf16_t & x) {
    ਸੰਘ {
        uint32_t u32;
        ਦਸ਼ਮਲਵ f32;
    } value;

    value.u32 = (uint32_t) x.ਬਿੱਟ << 16;
    ਵਾਪਸ value.f32;
}
#ਜੇ_ਅੰਤ

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SrcScalar, cpy_kernel_t quantize_ਖੰਡ, ਪੂਰਨ qk>
ਸਥਿਰ ਖਾਲੀ ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ int64_t ne,
                                      ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02,
                                      ਅਚਲ size_t nb00, ਅਚਲ size_t nb01, ਅਚਲ size_t nb02, ਅਚਲ size_t nb03,
                                      ਅਚਲ int64_t ne10, ਅਚਲ int64_t ne11, ਅਚਲ int64_t ne12,
                                      ਅਚਲ size_t nb10, ਅਚਲ size_t nb11, ਅਚਲ size_t nb12, ਅਚਲ size_t nb13,
                                      ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    ਅਚਲ int64_t ਕੁੱਲ_ਖੰਡਾਂ = ne / qk;
    constexpr ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = 256;
    ਅਚਲ int64_t grid_ਮਾਪ = ceil_div(ਕੁੱਲ_ਖੰਡਾਂ, (int64_t) ਖੰਡ_ਮਾਪ_3);

    stream->parallel_for(sycl::nd_range<1>(grid_ਮਾਪ * ਖੰਡ_ਮਾਪ_3, ਖੰਡ_ਮਾਪ_3),
                        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
        ਅਚਲ int64_t ਖੰਡ_idx = ਵਸਤੂ_ct1.get_global_ਰੇਖਿਕ_id();
        ਜੇ (ਖੰਡ_idx >= ਕੁੱਲ_ਖੰਡਾਂ) {
            ਵਾਪਸ;
        }

        ਅਚਲ int64_t i = ਖੰਡ_idx * qk;

        ਅਚਲ int64_t i03 = i / (ne00 * ne01 * ne02);
        ਅਚਲ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        ਅਚਲ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        ਅਚਲ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        ਅਚਲ size_t x_ਵਿੱਥ = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        ਅਚਲ int64_t i13 = i / (ne10 * ne11 * ne12);
        ਅਚਲ int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        ਅਚਲ int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        ਅਚਲ int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        ਅਚਲ size_t dst_ਵਿੱਥ = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        ਦਸ਼ਮਲਵ xf[qk];
        ਜੇ (nb00 == ਆਕਾਰ(SrcScalar)) {
            ਅਚਲ SrcScalar * src_ਪੰਕਤੀ = (ਅਚਲ SrcScalar *) (cx + x_ਵਿੱਥ);
            ਲਈ (ਪੂਰਨ j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_ਮੰਜ਼ਿਲ_f32(src_ਪੰਕਤੀ[j]);
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (ਪੂਰਨ j = 0; j < qk; ++j) {
                ਅਚਲ SrcScalar * src_val = (ਅਚਲ SrcScalar *) (cx + x_ਵਿੱਥ + j * nb00);
                xf[j] = ggml_sycl_src_ਮੰਜ਼ਿਲ_f32(*src_val);
            }
        }

        quantize_ਖੰਡ((ਅਚਲ ਅੱਖਰ *) xf, cdst + dst_ਵਿੱਥ);
    });
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ SrcScalar>
ਸਥਿਰ ਖਾਲੀ ggml_sycl_quantize_ਪੰਕਤੀਆਂ_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1,
                                         ਅਚਲ int64_t ne, ਅਚਲ int64_t ne00, ਅਚਲ int64_t ne01, ਅਚਲ int64_t ne02,
                                         ਅਚਲ size_t nb00, ਅਚਲ size_t nb01, ਅਚਲ size_t nb02, ਅਚਲ size_t nb03,
                                         ਅਚਲ int64_t ne10, ਅਚਲ int64_t ne11, ਅਚਲ int64_t ne12, ਅਚਲ size_t nb10,
                                         ਅਚਲ size_t nb11, ਅਚਲ size_t nb12, ਅਚਲ size_t nb13, ਕਤਾਰ_ptr stream) {
    GGML_ਅਣਵਰਤਿਆ(src0);
    GGML_ਅਣਵਰਤਿਆ(src1);

    ਚੋਣ (src1->type) {
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_0:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_NVFP4:
            ggml_sycl_quantize_ਪੰਕਤੀਆਂ_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f16_f16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_i16_i16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_i32_i32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q8_0_q8_0(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK8_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                              [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}


ਸਥਿਰ ਖਾਲੀ ggml_cpy_q5_0_q5_0(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                              [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}


ਸਥਿਰ ਖਾਲੀ ggml_cpy_q5_1_q5_1(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK5_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                              sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
                              [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}


ਸਥਿਰ ਖਾਲੀ ggml_cpy_q4_0_q4_0(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}


ਸਥਿਰ ਖਾਲੀ ggml_cpy_q4_1_q4_1(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_1, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q1_0_q1_0(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK1_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            cpy_q_q<ਖੰਡ_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q2_0_q2_0(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK2_0, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_mxfp4_mxfp4(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_MXFP4, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            cpy_q_q<ਖੰਡ_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_nvfp4_nvfp4(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_NVFP4, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q2_K_q2_K(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q3_K_q3_K(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q4_K_q4_K(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q5_K_q5_K(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_q6_K_q6_K(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq2_xxs_iq2_xxs(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq2_xs_iq2_xs(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq2_s_iq2_s(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq3_xxs_iq3_xxs(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq1_s_iq1_s(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq1_m_iq1_m(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq4_nl_iq4_nl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK4_NL, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq3_s_iq3_s(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_iq4_xs_iq4_xs(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                   ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                   ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                   ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = ceil_div(ne / QK_K, SYCL_CPY_ਖੰਡ_ਮਾਪ);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ), sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_q_q<ਖੰਡ_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
ਸਥਿਰ ਖਾਲੀ ggml_cpy_f32_bf16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_bf16_f32_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_bf16_bf16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_f16_bf16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}

ਸਥਿਰ ਖਾਲੀ ggml_cpy_bf16_f16_sycl(ਅਚਲ ਅੱਖਰ * cx, ਅੱਖਰ * cdst, ਅਚਲ ਪੂਰਨ ne, ਅਚਲ ਪੂਰਨ ne00, ਅਚਲ ਪੂਰਨ ne01,
                                  ਅਚਲ ਪੂਰਨ ne02, ਅਚਲ ਪੂਰਨ nb00, ਅਚਲ ਪੂਰਨ nb01, ਅਚਲ ਪੂਰਨ nb02, ਅਚਲ ਪੂਰਨ nb03,
                                  ਅਚਲ ਪੂਰਨ ne10, ਅਚਲ ਪੂਰਨ ne11, ਅਚਲ ਪੂਰਨ ne12, ਅਚਲ ਪੂਰਨ nb10, ਅਚਲ ਪੂਰਨ nb11,
                                  ਅਚਲ ਪੂਰਨ nb12, ਅਚਲ ਪੂਰਨ nb13, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_CPY_ਖੰਡ_ਮਾਪ - 1) / SYCL_CPY_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) * sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_CPY_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, ਵਸਤੂ_ct1);
        });
}
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1) ਕੋਸ਼ਿਸ਼ {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    ਅਚਲ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();

    ਅੱਖਰ * src0_ddc = (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
    ਅੱਖਰ * src1_ddc = (ਅੱਖਰ *) src1->ਅੰਕੜੇ;
    ਜੇ ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_ਧਾਰਾ->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_ਪੰਕਤੀਆਂ_sycl(src1->type));
        ggml_sycl_quantize_ਪੰਕਤੀਆਂ_sycl<ਦਸ਼ਮਲਵ>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_ਪੰਕਤੀਆਂ_sycl(src1->type));
        ggml_sycl_quantize_ਪੰਕਤੀਆਂ_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_ਧਾਰਾ);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_ਪੰਕਤੀਆਂ_sycl(src1->type));
        ggml_sycl_quantize_ਪੰਕਤੀਆਂ_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_ਧਾਰਾ);
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_ਧਾਰਾ);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_ਧਾਰਾ);
#ਜੇ_ਅੰਤ
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} ਫੜੋ (ਅਚਲ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਖਾਲੀ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
