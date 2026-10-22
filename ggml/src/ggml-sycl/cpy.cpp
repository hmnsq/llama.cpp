#अन्तर्भाव "cpy.hpp"

#अन्तर्भाव <float.h>
#अन्तर्भाव <vector>

#अन्तर्भाव "dequantize.hpp"
#अन्तर्भाव "ggml-sycl/common.hpp"
#अन्तर्भाव "ggml-sycl/presets.hpp"
#अन्तर्भाव "ggml.h"

स्थिर शून्य cpy_1_f32_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    दशांश *       dsti = (दशांश *) cdsti;

    *dsti = *xi;
}

स्थिर शून्य cpy_1_f32_f16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<दशांश, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

स्थिर शून्य cpy_1_f16_f16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::half * xi   = (स्थिरांक sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

स्थिर शून्य cpy_1_f16_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::half * xi   = (स्थिरांक sycl::half *) cxi;
    दशांश *            dsti = (दशांश *) cdsti;

    *dsti = *xi;
}

स्थिर शून्य cpy_1_i16_i16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक int16_t * xi   = (स्थिरांक int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

स्थिर शून्य cpy_1_i32_i32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक int32_t * xi   = (स्थिरांक int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

स्थिर शून्य cpy_1_f32_i32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi     = (स्थिरांक दशांश *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

स्थिर शून्य cpy_1_i32_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक int32_t * xi   = (स्थिरांक int32_t *) cxi;
    दशांश *         dsti = (दशांश *) cdsti;

    *dsti = (दशांश) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
स्थिर शून्य cpy_1_f32_bf16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक दशांश * xi   = (स्थिरांक दशांश *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

स्थिर शून्य cpy_1_bf16_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::ext::oneapi::bfloat16 * xi   = (स्थिरांक sycl::ext::oneapi::bfloat16 *) cxi;
    दशांश *            dsti = (दशांश *) cdsti;

    *dsti = static_cast<दशांश>(*xi);
}

स्थिर शून्य cpy_1_bf16_bf16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::ext::oneapi::bfloat16 * xi   = (स्थिरांक sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

स्थिर शून्य cpy_1_f16_bf16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::half * xi   = (स्थिरांक sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<दशांश>(*xi));
}

स्थिर शून्य cpy_1_bf16_f16(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::ext::oneapi::bfloat16 * xi   = (स्थिरांक sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<दशांश>(*xi));
}
#endif

ढाँचा <cpy_kernel_t cpy_1>
स्थिर शून्य cpy_f32_f16(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01, स्थिरांक पूर्णांक ne02,
                        स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03, स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11,
                        स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11, स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13,
                        स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक पूर्णांक i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    जर (i >= ne) {
        परत;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    स्थिरांक पूर्णांक i03      = i / (ne00 * ne01 * ne02);
    स्थिरांक पूर्णांक i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    स्थिरांक पूर्णांक i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    स्थिरांक पूर्णांक i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    स्थिरांक पूर्णांक x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    स्थिरांक पूर्णांक i13        = i / (ne10 * ne11 * ne12);
    स्थिरांक पूर्णांक i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    स्थिरांक पूर्णांक i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    स्थिरांक पूर्णांक i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    स्थिरांक पूर्णांक dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
ढाँचा<typename T>
स्थिर शून्य cpy_blck_q_q(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक T * xi = (स्थिरांक T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


स्थिर शून्य cpy_blck_q8_0_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    दशांश * cdstf = (दशांश *) (cdsti);

    साठी (पूर्णांक j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

स्थिर शून्य cpy_blck_q2_0_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक block_q2_0 * xi = (स्थिरांक block_q2_0 *) cxi;
    दशांश * cdstf = (दशांश *) cdsti;

    स्थिरांक दशांश d = xi->d;

    साठी (पूर्णांक j = 0; j < QK2_0; ++j) {
        स्थिरांक पूर्णांक byte_index = j / 4;
        स्थिरांक पूर्णांक bit_offset = (j % 4) * 2;
        स्थिरांक पूर्णांक q = (xi->qs[byte_index] >> bit_offset) & 0x3;
        cdstf[j] = (दशांश) (q - 1) * d;
    }
}



ढाँचा <dequantize_kernel_t dequant, पूर्णांक qk> स्थिर शून्य cpy_blck_q_f32(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    दशांश * cdstf = (दशांश *) (cdsti);

    साठी (पूर्णांक j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


ढाँचा <typename T, पूर्णांक qk>
स्थिर शून्य cpy_q_q(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01, स्थिरांक पूर्णांक ne02,
                      स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03, स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11,
                      स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11, स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13,
                      स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक पूर्णांक i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    जर (i >= ne) {
        परत;
    }

    स्थिरांक पूर्णांक i03      = i / (ne00 * ne01 * ne02);
    स्थिरांक पूर्णांक i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    स्थिरांक पूर्णांक i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    स्थिरांक पूर्णांक i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    स्थिरांक पूर्णांक x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    स्थिरांक पूर्णांक i13        = i / (ne10 * ne11 * ne12);
    स्थिरांक पूर्णांक i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    स्थिरांक पूर्णांक i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    स्थिरांक पूर्णांक i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    स्थिरांक पूर्णांक dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

ढाँचा <cpy_kernel_t cpy_blck, पूर्णांक qk>
स्थिर शून्य cpy_f32_q(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01, स्थिरांक पूर्णांक ne02,
                      स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03, स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11,
                      स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11, स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13,
                      स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक पूर्णांक i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    जर (i >= ne) {
        परत;
    }


    स्थिरांक पूर्णांक i03      = i / (ne00 * ne01 * ne02);
    स्थिरांक पूर्णांक i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    स्थिरांक पूर्णांक i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    स्थिरांक पूर्णांक i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    स्थिरांक पूर्णांक x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    स्थिरांक पूर्णांक i13        = i / (ne10 * ne11 * ne12);
    स्थिरांक पूर्णांक i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    स्थिरांक पूर्णांक i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    स्थिरांक पूर्णांक i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    स्थिरांक पूर्णांक dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

ढाँचा <cpy_kernel_t cpy_blck, पूर्णांक qk>
स्थिर शून्य cpy_q_f32(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01, स्थिरांक पूर्णांक ne02,
                      स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03, स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11,
                      स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11, स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13,
                      स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक पूर्णांक i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    जर (i >= ne) {
        परत;
    }

    स्थिरांक पूर्णांक i03      = i / (ne00 * ne01 * ne02);
    स्थिरांक पूर्णांक i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    स्थिरांक पूर्णांक i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    स्थिरांक पूर्णांक i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    स्थिरांक पूर्णांक x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    स्थिरांक पूर्णांक i13        = i / (ne10 * ne11 * ne12);
    स्थिरांक पूर्णांक i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    स्थिरांक पूर्णांक i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    स्थिरांक पूर्णांक i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    स्थिरांक पूर्णांक dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

स्थिर शून्य ggml_cpy_f16_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_f32_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_f32_f16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_f32_i32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_i32_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_f32_q8_0_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_q8_0_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_q2_0_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_f32_q4_0_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_q4_0_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

स्थिर शून्य ggml_cpy_f32_q4_1_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_q4_1_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

स्थिर शून्य ggml_cpy_f32_q5_0_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_q5_0_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

स्थिर शून्य ggml_cpy_f32_q5_1_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_q5_1_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

स्थिर शून्य ggml_cpy_mxfp4_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                    स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                    स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                    स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_f32_iq4_nl_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                     स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                     स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                     स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य cpy_blck_f16_q4_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::half * xi = (स्थिरांक sycl::half *) cxi;
    दशांश xf[QK4_0];

    साठी (पूर्णांक j = 0; j < QK4_0; ++j) {
        xf[j] = (दशांश) xi[j];
    }

    cpy_blck_f32_q4_0((स्थिरांक अक्षर *) xf, cdsti);
}

स्थिर शून्य cpy_blck_f16_q4_1(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::half * xi = (स्थिरांक sycl::half *) cxi;
    दशांश xf[QK4_1];

    साठी (पूर्णांक j = 0; j < QK4_1; ++j) {
        xf[j] = (दशांश) xi[j];
    }

    cpy_blck_f32_q4_1((स्थिरांक अक्षर *) xf, cdsti);
}

स्थिर शून्य cpy_blck_f16_q5_0(स्थिरांक अक्षर * cxi, अक्षर * cdsti) {
    स्थिरांक sycl::half * xi = (स्थिरांक sycl::half *) cxi;
    दशांश xf[QK5_0];

    साठी (पूर्णांक j = 0; j < QK5_0; ++j) {
        xf[j] = (दशांश) xi[j];
    }

    cpy_blck_f32_q5_0((स्थिरांक अक्षर *) xf, cdsti);
}

स्थिर शून्य ggml_cpy_f16_q4_0_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_f16_q4_1_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर शून्य ggml_cpy_f16_q5_0_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

स्थिर तार्किक ggml_sycl_is_quantized_type(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ2_S:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

स्थिर तार्किक ggml_sycl_can_quantize_rows_sycl(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

ढाँचा <typename SrcScalar>
स्थिर ओळीत दशांश ggml_sycl_src_to_f32(स्थिरांक SrcScalar & x) {
    परत (दशांश) x;
}

#ifdef GGML_SYCL_HAS_BF16
ढाँचा <>
ओळीत दशांश ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(स्थिरांक sycl::ext::oneapi::bfloat16 & x) {
    परत static_cast<दशांश>(x);
}

ढाँचा <>
ओळीत दशांश ggml_sycl_src_to_f32<ggml_bf16_t>(स्थिरांक ggml_bf16_t & x) {
    संघ {
        uint32_t u32;
        दशांश f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    परत value.f32;
}
#endif

ढाँचा <typename SrcScalar, cpy_kernel_t quantize_block, पूर्णांक qk>
स्थिर शून्य ggml_sycl_quantize_rows_q(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक int64_t ne,
                                      स्थिरांक int64_t ne00, स्थिरांक int64_t ne01, स्थिरांक int64_t ne02,
                                      स्थिरांक माप_प्रकार nb00, स्थिरांक माप_प्रकार nb01, स्थिरांक माप_प्रकार nb02, स्थिरांक माप_प्रकार nb03,
                                      स्थिरांक int64_t ne10, स्थिरांक int64_t ne11, स्थिरांक int64_t ne12,
                                      स्थिरांक माप_प्रकार nb10, स्थिरांक माप_प्रकार nb11, स्थिरांक माप_प्रकार nb12, स्थिरांक माप_प्रकार nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    स्थिरांक int64_t total_blocks = ne / qk;
    constexpr पूर्णांक block_size = 256;
    स्थिरांक int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        स्थिरांक int64_t block_idx = item_ct1.get_global_linear_id();
        जर (block_idx >= total_blocks) {
            परत;
        }

        स्थिरांक int64_t i = block_idx * qk;

        स्थिरांक int64_t i03 = i / (ne00 * ne01 * ne02);
        स्थिरांक int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        स्थिरांक int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        स्थिरांक int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        स्थिरांक माप_प्रकार x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        स्थिरांक int64_t i13 = i / (ne10 * ne11 * ne12);
        स्थिरांक int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        स्थिरांक int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        स्थिरांक int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        स्थिरांक माप_प्रकार dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        दशांश xf[qk];
        जर (nb00 == आकार(SrcScalar)) {
            स्थिरांक SrcScalar * src_row = (स्थिरांक SrcScalar *) (cx + x_offset);
            साठी (पूर्णांक j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } अन्यथा {
            साठी (पूर्णांक j = 0; j < qk; ++j) {
                स्थिरांक SrcScalar * src_val = (स्थिरांक SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((स्थिरांक अक्षर *) xf, cdst + dst_offset);
    });
}

ढाँचा <typename SrcScalar>
स्थिर शून्य ggml_sycl_quantize_rows_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * src1,
                                         स्थिरांक int64_t ne, स्थिरांक int64_t ne00, स्थिरांक int64_t ne01, स्थिरांक int64_t ne02,
                                         स्थिरांक माप_प्रकार nb00, स्थिरांक माप_प्रकार nb01, स्थिरांक माप_प्रकार nb02, स्थिरांक माप_प्रकार nb03,
                                         स्थिरांक int64_t ne10, स्थिरांक int64_t ne11, स्थिरांक int64_t ne12, स्थिरांक माप_प्रकार nb10,
                                         स्थिरांक माप_प्रकार nb11, स्थिरांक माप_प्रकार nb12, स्थिरांक माप_प्रकार nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    निवड (src1->type) {
        स्थिती GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            थांबा;
        स्थिती GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

स्थिर शून्य ggml_cpy_f16_f16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_i16_i16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_i32_i32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

स्थिर शून्य ggml_cpy_q8_0_q8_0(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


स्थिर शून्य ggml_cpy_q5_0_q5_0(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


स्थिर शून्य ggml_cpy_q5_1_q5_1(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


स्थिर शून्य ggml_cpy_q4_0_q4_0(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


स्थिर शून्य ggml_cpy_q4_1_q4_1(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q1_0_q1_0(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q2_0_q2_0(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_mxfp4_mxfp4(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_nvfp4_nvfp4(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q2_K_q2_K(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q3_K_q3_K(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q4_K_q4_K(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q5_K_q5_K(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_q6_K_q6_K(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq2_xxs_iq2_xxs(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq2_xs_iq2_xs(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq2_s_iq2_s(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq3_xxs_iq3_xxs(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq1_s_iq1_s(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq1_m_iq1_m(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq4_nl_iq4_nl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq3_s_iq3_s(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_iq4_xs_iq4_xs(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                   स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                   स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                   स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    स्थिरांक पूर्णांक अंक_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
स्थिर शून्य ggml_cpy_f32_bf16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_bf16_f32_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_bf16_bf16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_f16_bf16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

स्थिर शून्य ggml_cpy_bf16_f16_sycl(स्थिरांक अक्षर * cx, अक्षर * cdst, स्थिरांक पूर्णांक ne, स्थिरांक पूर्णांक ne00, स्थिरांक पूर्णांक ne01,
                                  स्थिरांक पूर्णांक ne02, स्थिरांक पूर्णांक nb00, स्थिरांक पूर्णांक nb01, स्थिरांक पूर्णांक nb02, स्थिरांक पूर्णांक nb03,
                                  स्थिरांक पूर्णांक ne10, स्थिरांक पूर्णांक ne11, स्थिरांक पूर्णांक ne12, स्थिरांक पूर्णांक nb10, स्थिरांक पूर्णांक nb11,
                                  स्थिरांक पूर्णांक nb12, स्थिरांक पूर्णांक nb13, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

शून्य ggml_sycl_cpy(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * src1) प्रयत्न {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    स्थिरांक int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_द्विमान_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    अक्षर * src0_ddc = (अक्षर *) src0->data;
    अक्षर * src1_ddc = (अक्षर *) src1->data;
    जर ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->स_नकल(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<दशांश>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } अन्यथा जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } अन्यथा जर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } अन्यथा {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} पकडा (स्थिरांक sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::निकास(1);
}

शून्य ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
