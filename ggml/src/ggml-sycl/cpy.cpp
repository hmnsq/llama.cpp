#অন্তর্ভাব "cpy.hpp"

#অন্তর্ভাব <float.h>
#অন্তর্ভাব <vector>

#অন্তর্ভাব "dequantize.hpp"
#অন্তর্ভাব "ggml-sycl/common.hpp"
#অন্তর্ভাব "ggml-sycl/presets.hpp"
#অন্তর্ভাব "ggml.h"

স্থির শূন্য cpy_1_f32_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    দশমিক *       dsti = (দশমিক *) cdsti;

    *dsti = *xi;
}

স্থির শূন্য cpy_1_f32_f16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<দশমিক, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

স্থির শূন্য cpy_1_f16_f16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::half * xi   = (ধ্রুবক sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

স্থির শূন্য cpy_1_f16_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::half * xi   = (ধ্রুবক sycl::half *) cxi;
    দশমিক *            dsti = (দশমিক *) cdsti;

    *dsti = *xi;
}

স্থির শূন্য cpy_1_i16_i16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক int16_t * xi   = (ধ্রুবক int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

স্থির শূন্য cpy_1_i32_i32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক int32_t * xi   = (ধ্রুবক int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

স্থির শূন্য cpy_1_f32_i32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi     = (ধ্রুবক দশমিক *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

স্থির শূন্য cpy_1_i32_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক int32_t * xi   = (ধ্রুবক int32_t *) cxi;
    দশমিক *         dsti = (দশমিক *) cdsti;

    *dsti = (দশমিক) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
স্থির শূন্য cpy_1_f32_bf16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক দশমিক * xi   = (ধ্রুবক দশমিক *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

স্থির শূন্য cpy_1_bf16_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::ext::oneapi::bfloat16 * xi   = (ধ্রুবক sycl::ext::oneapi::bfloat16 *) cxi;
    দশমিক *            dsti = (দশমিক *) cdsti;

    *dsti = static_cast<দশমিক>(*xi);
}

স্থির শূন্য cpy_1_bf16_bf16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::ext::oneapi::bfloat16 * xi   = (ধ্রুবক sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

স্থির শূন্য cpy_1_f16_bf16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::half * xi   = (ধ্রুবক sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<দশমিক>(*xi));
}

স্থির শূন্য cpy_1_bf16_f16(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::ext::oneapi::bfloat16 * xi   = (ধ্রুবক sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<দশমিক>(*xi));
}
#endif

template <cpy_kernel_t cpy_1>
স্থির শূন্য cpy_f32_f16(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01, ধ্রুবক পূর্ণসংখ্যা ne02,
                        ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03, ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11,
                        ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11, ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13,
                        ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    যদি (i >= ne) {
        ফেরত;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    ধ্রুবক পূর্ণসংখ্যা i03      = i / (ne00 * ne01 * ne02);
    ধ্রুবক পূর্ণসংখ্যা i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ধ্রুবক পূর্ণসংখ্যা i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ধ্রুবক পূর্ণসংখ্যা i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ধ্রুবক পূর্ণসংখ্যা x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ধ্রুবক পূর্ণসংখ্যা i13        = i / (ne10 * ne11 * ne12);
    ধ্রুবক পূর্ণসংখ্যা i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ধ্রুবক পূর্ণসংখ্যা i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ধ্রুবক পূর্ণসংখ্যা i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ধ্রুবক পূর্ণসংখ্যা dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
স্থির শূন্য cpy_blck_q_q(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক T * xi = (ধ্রুবক T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


স্থির শূন্য cpy_blck_q8_0_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    দশমিক * cdstf = (দশমিক *) (cdsti);

    জন্য (পূর্ণসংখ্যা j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

স্থির শূন্য cpy_blck_q2_0_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক block_q2_0 * xi = (ধ্রুবক block_q2_0 *) cxi;
    দশমিক * cdstf = (দশমিক *) cdsti;

    ধ্রুবক দশমিক d = xi->d;

    জন্য (পূর্ণসংখ্যা j = 0; j < QK2_0; ++j) {
        ধ্রুবক পূর্ণসংখ্যা byte_সূচক = j / 4;
        ধ্রুবক পূর্ণসংখ্যা bit_offset = (j % 4) * 2;
        ধ্রুবক পূর্ণসংখ্যা q = (xi->qs[byte_সূচক] >> bit_offset) & 0x3;
        cdstf[j] = (দশমিক) (q - 1) * d;
    }
}



template <dequantize_kernel_t dequant, পূর্ণসংখ্যা qk> স্থির শূন্য cpy_blck_q_f32(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    দশমিক * cdstf = (দশমিক *) (cdsti);

    জন্য (পূর্ণসংখ্যা j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, পূর্ণসংখ্যা qk>
স্থির শূন্য cpy_q_q(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01, ধ্রুবক পূর্ণসংখ্যা ne02,
                      ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03, ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11,
                      ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11, ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13,
                      ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    যদি (i >= ne) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা i03      = i / (ne00 * ne01 * ne02);
    ধ্রুবক পূর্ণসংখ্যা i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ধ্রুবক পূর্ণসংখ্যা i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ধ্রুবক পূর্ণসংখ্যা i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ধ্রুবক পূর্ণসংখ্যা x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    ধ্রুবক পূর্ণসংখ্যা i13        = i / (ne10 * ne11 * ne12);
    ধ্রুবক পূর্ণসংখ্যা i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ধ্রুবক পূর্ণসংখ্যা i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ধ্রুবক পূর্ণসংখ্যা i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ধ্রুবক পূর্ণসংখ্যা dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, পূর্ণসংখ্যা qk>
স্থির শূন্য cpy_f32_q(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01, ধ্রুবক পূর্ণসংখ্যা ne02,
                      ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03, ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11,
                      ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11, ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13,
                      ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    যদি (i >= ne) {
        ফেরত;
    }


    ধ্রুবক পূর্ণসংখ্যা i03      = i / (ne00 * ne01 * ne02);
    ধ্রুবক পূর্ণসংখ্যা i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ধ্রুবক পূর্ণসংখ্যা i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ধ্রুবক পূর্ণসংখ্যা i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ধ্রুবক পূর্ণসংখ্যা x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ধ্রুবক পূর্ণসংখ্যা i13        = i / (ne10 * ne11 * ne12);
    ধ্রুবক পূর্ণসংখ্যা i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ধ্রুবক পূর্ণসংখ্যা i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ধ্রুবক পূর্ণসংখ্যা i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ধ্রুবক পূর্ণসংখ্যা dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, পূর্ণসংখ্যা qk>
স্থির শূন্য cpy_q_f32(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01, ধ্রুবক পূর্ণসংখ্যা ne02,
                      ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03, ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11,
                      ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11, ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13,
                      ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    যদি (i >= ne) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা i03      = i / (ne00 * ne01 * ne02);
    ধ্রুবক পূর্ণসংখ্যা i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ধ্রুবক পূর্ণসংখ্যা i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ধ্রুবক পূর্ণসংখ্যা i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ধ্রুবক পূর্ণসংখ্যা x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ধ্রুবক পূর্ণসংখ্যা i13        = i / (ne10 * ne11 * ne12);
    ধ্রুবক পূর্ণসংখ্যা i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ধ্রুবক পূর্ণসংখ্যা i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ধ্রুবক পূর্ণসংখ্যা i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ধ্রুবক পূর্ণসংখ্যা dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

স্থির শূন্য ggml_cpy_f16_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_f32_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_f32_f16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_f32_i32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_i32_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_f32_q8_0_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_q8_0_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_q2_0_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_f32_q4_0_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_q4_0_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

স্থির শূন্য ggml_cpy_f32_q4_1_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_q4_1_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

স্থির শূন্য ggml_cpy_f32_q5_0_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_q5_0_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

স্থির শূন্য ggml_cpy_f32_q5_1_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_q5_1_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

স্থির শূন্য ggml_cpy_mxfp4_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                    ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                    ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                    ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_f32_iq4_nl_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                     ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                     ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                     ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য cpy_blck_f16_q4_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::half * xi = (ধ্রুবক sycl::half *) cxi;
    দশমিক xf[QK4_0];

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_0; ++j) {
        xf[j] = (দশমিক) xi[j];
    }

    cpy_blck_f32_q4_0((ধ্রুবক অক্ষর *) xf, cdsti);
}

স্থির শূন্য cpy_blck_f16_q4_1(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::half * xi = (ধ্রুবক sycl::half *) cxi;
    দশমিক xf[QK4_1];

    জন্য (পূর্ণসংখ্যা j = 0; j < QK4_1; ++j) {
        xf[j] = (দশমিক) xi[j];
    }

    cpy_blck_f32_q4_1((ধ্রুবক অক্ষর *) xf, cdsti);
}

স্থির শূন্য cpy_blck_f16_q5_0(ধ্রুবক অক্ষর * cxi, অক্ষর * cdsti) {
    ধ্রুবক sycl::half * xi = (ধ্রুবক sycl::half *) cxi;
    দশমিক xf[QK5_0];

    জন্য (পূর্ণসংখ্যা j = 0; j < QK5_0; ++j) {
        xf[j] = (দশমিক) xi[j];
    }

    cpy_blck_f32_q5_0((ধ্রুবক অক্ষর *) xf, cdsti);
}

স্থির শূন্য ggml_cpy_f16_q4_0_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_f16_q4_1_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির শূন্য ggml_cpy_f16_q5_0_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

স্থির যৌক্তিক ggml_sycl_is_quantized_type(গণনা ggml_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_Q1_0:
        ক্ষেত্র GGML_TYPE_Q2_0:
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_Q4_1:
        ক্ষেত্র GGML_TYPE_Q5_0:
        ক্ষেত্র GGML_TYPE_Q5_1:
        ক্ষেত্র GGML_TYPE_Q8_0:
        ক্ষেত্র GGML_TYPE_MXFP4:
        ক্ষেত্র GGML_TYPE_NVFP4:
        ক্ষেত্র GGML_TYPE_Q2_K:
        ক্ষেত্র GGML_TYPE_Q3_K:
        ক্ষেত্র GGML_TYPE_Q4_K:
        ক্ষেত্র GGML_TYPE_Q5_K:
        ক্ষেত্র GGML_TYPE_Q6_K:
        ক্ষেত্র GGML_TYPE_IQ2_XXS:
        ক্ষেত্র GGML_TYPE_IQ2_XS:
        ক্ষেত্র GGML_TYPE_IQ2_S:
        ক্ষেত্র GGML_TYPE_IQ3_XXS:
        ক্ষেত্র GGML_TYPE_IQ3_S:
        ক্ষেত্র GGML_TYPE_IQ1_S:
        ক্ষেত্র GGML_TYPE_IQ1_M:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
        ক্ষেত্র GGML_TYPE_IQ4_XS:
            ফেরত সত্য;
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}

স্থির যৌক্তিক ggml_sycl_can_quantize_rows_sycl(গণনা ggml_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_Q1_0:
        ক্ষেত্র GGML_TYPE_Q2_0:
        ক্ষেত্র GGML_TYPE_Q4_0:
        ক্ষেত্র GGML_TYPE_Q4_1:
        ক্ষেত্র GGML_TYPE_Q5_0:
        ক্ষেত্র GGML_TYPE_Q5_1:
        ক্ষেত্র GGML_TYPE_Q8_0:
        ক্ষেত্র GGML_TYPE_MXFP4:
        ক্ষেত্র GGML_TYPE_NVFP4:
        ক্ষেত্র GGML_TYPE_Q2_K:
        ক্ষেত্র GGML_TYPE_Q3_K:
        ক্ষেত্র GGML_TYPE_Q4_K:
        ক্ষেত্র GGML_TYPE_Q5_K:
        ক্ষেত্র GGML_TYPE_Q6_K:
        ক্ষেত্র GGML_TYPE_IQ4_NL:
        ক্ষেত্র GGML_TYPE_IQ4_XS:
            ফেরত সত্য;
        পূর্বনির্ধারিত:
            ফেরত মিথ্যা;
    }
}

template <typename SrcScalar>
স্থির অন্তঃরেখ দশমিক ggml_sycl_src_to_f32(ধ্রুবক SrcScalar & x) {
    ফেরত (দশমিক) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
অন্তঃরেখ দশমিক ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(ধ্রুবক sycl::ext::oneapi::bfloat16 & x) {
    ফেরত static_cast<দশমিক>(x);
}

template <>
অন্তঃরেখ দশমিক ggml_sycl_src_to_f32<ggml_bf16_t>(ধ্রুবক ggml_bf16_t & x) {
    সংঘ {
        uint32_t u32;
        দশমিক f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    ফেরত value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, পূর্ণসংখ্যা qk>
স্থির শূন্য ggml_sycl_quantize_rows_q(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক int64_t ne,
                                      ধ্রুবক int64_t ne00, ধ্রুবক int64_t ne01, ধ্রুবক int64_t ne02,
                                      ধ্রুবক size_t nb00, ধ্রুবক size_t nb01, ধ্রুবক size_t nb02, ধ্রুবক size_t nb03,
                                      ধ্রুবক int64_t ne10, ধ্রুবক int64_t ne11, ধ্রুবক int64_t ne12,
                                      ধ্রুবক size_t nb10, ধ্রুবক size_t nb11, ধ্রুবক size_t nb12, ধ্রুবক size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    ধ্রুবক int64_t যোগ_blocks = ne / qk;
    constexpr পূর্ণসংখ্যা block_আকার_2 = 256;
    ধ্রুবক int64_t grid_আকার = ceil_div(যোগ_blocks, (int64_t) block_আকার_2);

    stream->parallel_for(sycl::nd_range<1>(grid_আকার * block_আকার_2, block_আকার_2),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
        ধ্রুবক int64_t block_idx = item_ct1.get_global_linear_id();
        যদি (block_idx >= যোগ_blocks) {
            ফেরত;
        }

        ধ্রুবক int64_t i = block_idx * qk;

        ধ্রুবক int64_t i03 = i / (ne00 * ne01 * ne02);
        ধ্রুবক int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        ধ্রুবক int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        ধ্রুবক int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        ধ্রুবক size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        ধ্রুবক int64_t i13 = i / (ne10 * ne11 * ne12);
        ধ্রুবক int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        ধ্রুবক int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        ধ্রুবক int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        ধ্রুবক size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        দশমিক xf[qk];
        যদি (nb00 == আকার(SrcScalar)) {
            ধ্রুবক SrcScalar * src_row = (ধ্রুবক SrcScalar *) (cx + x_offset);
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } অন্যথা {
            জন্য (পূর্ণসংখ্যা j = 0; j < qk; ++j) {
                ধ্রুবক SrcScalar * src_val = (ধ্রুবক SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((ধ্রুবক অক্ষর *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
স্থির শূন্য ggml_sycl_quantize_rows_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক ggml_tensor * src0, ধ্রুবক ggml_tensor * src1,
                                         ধ্রুবক int64_t ne, ধ্রুবক int64_t ne00, ধ্রুবক int64_t ne01, ধ্রুবক int64_t ne02,
                                         ধ্রুবক size_t nb00, ধ্রুবক size_t nb01, ধ্রুবক size_t nb02, ধ্রুবক size_t nb03,
                                         ধ্রুবক int64_t ne10, ধ্রুবক int64_t ne11, ধ্রুবক int64_t ne12, ধ্রুবক size_t nb10,
                                         ধ্রুবক size_t nb11, ধ্রুবক size_t nb12, ধ্রুবক size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    নির্বাচন (src1->type) {
        ক্ষেত্র GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            বিরতি;
        ক্ষেত্র GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            বিরতি;
        পূর্বনির্ধারিত:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

স্থির শূন্য ggml_cpy_f16_f16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_i16_i16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_i32_i32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

স্থির শূন্য ggml_cpy_q8_0_q8_0(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


স্থির শূন্য ggml_cpy_q5_0_q5_0(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


স্থির শূন্য ggml_cpy_q5_1_q5_1(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_আকার);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


স্থির শূন্য ggml_cpy_q4_0_q4_0(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


স্থির শূন্য ggml_cpy_q4_1_q4_1(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q1_0_q1_0(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q2_0_q2_0(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_mxfp4_mxfp4(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_nvfp4_nvfp4(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q2_K_q2_K(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q3_K_q3_K(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q4_K_q4_K(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q5_K_q5_K(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_q6_K_q6_K(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq2_xxs_iq2_xxs(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq2_xs_iq2_xs(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq2_s_iq2_s(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq3_xxs_iq3_xxs(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq1_s_iq1_s(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq1_m_iq1_m(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq4_nl_iq4_nl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq3_s_iq3_s(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_iq4_xs_iq4_xs(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                   ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                   ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                   ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_আকার);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
স্থির শূন্য ggml_cpy_f32_bf16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_bf16_f32_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_bf16_bf16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_f16_bf16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

স্থির শূন্য ggml_cpy_bf16_f16_sycl(ধ্রুবক অক্ষর * cx, অক্ষর * cdst, ধ্রুবক পূর্ণসংখ্যা ne, ধ্রুবক পূর্ণসংখ্যা ne00, ধ্রুবক পূর্ণসংখ্যা ne01,
                                  ধ্রুবক পূর্ণসংখ্যা ne02, ধ্রুবক পূর্ণসংখ্যা nb00, ধ্রুবক পূর্ণসংখ্যা nb01, ধ্রুবক পূর্ণসংখ্যা nb02, ধ্রুবক পূর্ণসংখ্যা nb03,
                                  ধ্রুবক পূর্ণসংখ্যা ne10, ধ্রুবক পূর্ণসংখ্যা ne11, ধ্রুবক পূর্ণসংখ্যা ne12, ধ্রুবক পূর্ণসংখ্যা nb10, ধ্রুবক পূর্ণসংখ্যা nb11,
                                  ধ্রুবক পূর্ণসংখ্যা nb12, ধ্রুবক পূর্ণসংখ্যা nb13, queue_ptr stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (ne + SYCL_CPY_BLOCK_আকার - 1) / SYCL_CPY_BLOCK_আকার;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, অঙ্ক_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_আকার(WARP_আকার)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

শূন্য ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ধ্রুবক ggml_tensor * src0, ধ্রুবক ggml_tensor * src1) চেষ্টা {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    ধ্রুবক int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    অক্ষর * src0_ddc = (অক্ষর *) src0->তথ্য;
    অক্ষর * src1_ddc = (অক্ষর *) src1->তথ্য;
    যদি ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<দশমিক>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } অন্যথা যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } অন্যথা {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} ধরো (ধ্রুবক sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

শূন্য ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
