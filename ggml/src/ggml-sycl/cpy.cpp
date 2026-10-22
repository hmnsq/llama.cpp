#அந்தர்பாவ "cpy.hpp"

#அந்தர்பாவ <float.h>
#அந்தர்பாவ <vector>

#அந்தர்பாவ "dequantize.hpp"
#அந்தர்பாவ "ggml-sycl/common.hpp"
#அந்தர்பாவ "ggml-sycl/presets.hpp"
#அந்தர்பாவ "ggml.h"

நிலையான வெறுமை cpy_1_f32_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    தசம *       dsti = (தசம *) cdsti;

    *dsti = *xi;
}

நிலையான வெறுமை cpy_1_f32_f16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<தசம, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

நிலையான வெறுமை cpy_1_f16_f16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::half * xi   = (மாறிலி sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

நிலையான வெறுமை cpy_1_f16_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::half * xi   = (மாறிலி sycl::half *) cxi;
    தசம *            dsti = (தசம *) cdsti;

    *dsti = *xi;
}

நிலையான வெறுமை cpy_1_i16_i16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி int16_t * xi   = (மாறிலி int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

நிலையான வெறுமை cpy_1_i32_i32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி int32_t * xi   = (மாறிலி int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

நிலையான வெறுமை cpy_1_f32_i32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi     = (மாறிலி தசம *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

நிலையான வெறுமை cpy_1_i32_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி int32_t * xi   = (மாறிலி int32_t *) cxi;
    தசம *         dsti = (தசம *) cdsti;

    *dsti = (தசம) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
நிலையான வெறுமை cpy_1_f32_bf16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி தசம * xi   = (மாறிலி தசம *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

நிலையான வெறுமை cpy_1_bf16_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::ext::oneapi::bfloat16 * xi   = (மாறிலி sycl::ext::oneapi::bfloat16 *) cxi;
    தசம *            dsti = (தசம *) cdsti;

    *dsti = static_cast<தசம>(*xi);
}

நிலையான வெறுமை cpy_1_bf16_bf16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::ext::oneapi::bfloat16 * xi   = (மாறிலி sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

நிலையான வெறுமை cpy_1_f16_bf16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::half * xi   = (மாறிலி sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<தசம>(*xi));
}

நிலையான வெறுமை cpy_1_bf16_f16(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::ext::oneapi::bfloat16 * xi   = (மாறிலி sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<தசம>(*xi));
}
#endif

template <cpy_kernel_t cpy_1>
நிலையான வெறுமை cpy_f32_f16(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01, மாறிலி முழு ne02,
                        மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03, மாறிலி முழு ne10, மாறிலி முழு ne11,
                        மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11, மாறிலி முழு nb12, மாறிலி முழு nb13,
                        மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    எனில் (i >= ne) {
        திரும்பு;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    மாறிலி முழு i03      = i / (ne00 * ne01 * ne02);
    மாறிலி முழு i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    மாறிலி முழு i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    மாறிலி முழு i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    மாறிலி முழு x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    மாறிலி முழு i13        = i / (ne10 * ne11 * ne12);
    மாறிலி முழு i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    மாறிலி முழு i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    மாறிலி முழு i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    மாறிலி முழு dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
நிலையான வெறுமை cpy_blck_q_q(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி T * xi = (மாறிலி T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


நிலையான வெறுமை cpy_blck_q8_0_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    தசம * cdstf = (தசம *) (cdsti);

    க்கு (முழு j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

நிலையான வெறுமை cpy_blck_q2_0_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி block_q2_0 * xi = (மாறிலி block_q2_0 *) cxi;
    தசம * cdstf = (தசம *) cdsti;

    மாறிலி தசம d = xi->d;

    க்கு (முழு j = 0; j < QK2_0; ++j) {
        மாறிலி முழு byte_index = j / 4;
        மாறிலி முழு bit_offset = (j % 4) * 2;
        மாறிலி முழு q = (xi->qs[byte_index] >> bit_offset) & 0x3;
        cdstf[j] = (தசம) (q - 1) * d;
    }
}



template <dequantize_kernel_t dequant, முழு qk> நிலையான வெறுமை cpy_blck_q_f32(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    தசம * cdstf = (தசம *) (cdsti);

    க்கு (முழு j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, முழு qk>
நிலையான வெறுமை cpy_q_q(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01, மாறிலி முழு ne02,
                      மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03, மாறிலி முழு ne10, மாறிலி முழு ne11,
                      மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11, மாறிலி முழு nb12, மாறிலி முழு nb13,
                      மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    எனில் (i >= ne) {
        திரும்பு;
    }

    மாறிலி முழு i03      = i / (ne00 * ne01 * ne02);
    மாறிலி முழு i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    மாறிலி முழு i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    மாறிலி முழு i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    மாறிலி முழு x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    மாறிலி முழு i13        = i / (ne10 * ne11 * ne12);
    மாறிலி முழு i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    மாறிலி முழு i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    மாறிலி முழு i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    மாறிலி முழு dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, முழு qk>
நிலையான வெறுமை cpy_f32_q(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01, மாறிலி முழு ne02,
                      மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03, மாறிலி முழு ne10, மாறிலி முழு ne11,
                      மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11, மாறிலி முழு nb12, மாறிலி முழு nb13,
                      மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    எனில் (i >= ne) {
        திரும்பு;
    }


    மாறிலி முழு i03      = i / (ne00 * ne01 * ne02);
    மாறிலி முழு i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    மாறிலி முழு i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    மாறிலி முழு i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    மாறிலி முழு x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    மாறிலி முழு i13        = i / (ne10 * ne11 * ne12);
    மாறிலி முழு i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    மாறிலி முழு i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    மாறிலி முழு i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    மாறிலி முழு dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, முழு qk>
நிலையான வெறுமை cpy_q_f32(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01, மாறிலி முழு ne02,
                      மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03, மாறிலி முழு ne10, மாறிலி முழு ne11,
                      மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11, மாறிலி முழு nb12, மாறிலி முழு nb13,
                      மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    எனில் (i >= ne) {
        திரும்பு;
    }

    மாறிலி முழு i03      = i / (ne00 * ne01 * ne02);
    மாறிலி முழு i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    மாறிலி முழு i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    மாறிலி முழு i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    மாறிலி முழு x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    மாறிலி முழு i13        = i / (ne10 * ne11 * ne12);
    மாறிலி முழு i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    மாறிலி முழு i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    மாறிலி முழு i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    மாறிலி முழு dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

நிலையான வெறுமை ggml_cpy_f16_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_f32_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_f32_f16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_f32_i32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_i32_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_f32_q8_0_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_q8_0_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_q2_0_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_f32_q4_0_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_q4_0_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_f32_q4_1_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_q4_1_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_f32_q5_0_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_q5_0_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_f32_q5_1_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_q5_1_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_mxfp4_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                    மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                    மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                    மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_f32_iq4_nl_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                     மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                     மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                     மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை cpy_blck_f16_q4_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::half * xi = (மாறிலி sycl::half *) cxi;
    தசம xf[QK4_0];

    க்கு (முழு j = 0; j < QK4_0; ++j) {
        xf[j] = (தசம) xi[j];
    }

    cpy_blck_f32_q4_0((மாறிலி எழுத்து *) xf, cdsti);
}

நிலையான வெறுமை cpy_blck_f16_q4_1(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::half * xi = (மாறிலி sycl::half *) cxi;
    தசம xf[QK4_1];

    க்கு (முழு j = 0; j < QK4_1; ++j) {
        xf[j] = (தசம) xi[j];
    }

    cpy_blck_f32_q4_1((மாறிலி எழுத்து *) xf, cdsti);
}

நிலையான வெறுமை cpy_blck_f16_q5_0(மாறிலி எழுத்து * cxi, எழுத்து * cdsti) {
    மாறிலி sycl::half * xi = (மாறிலி sycl::half *) cxi;
    தசம xf[QK5_0];

    க்கு (முழு j = 0; j < QK5_0; ++j) {
        xf[j] = (தசம) xi[j];
    }

    cpy_blck_f32_q5_0((மாறிலி எழுத்து *) xf, cdsti);
}

நிலையான வெறுமை ggml_cpy_f16_q4_0_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_f16_q4_1_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான வெறுமை ggml_cpy_f16_q5_0_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

நிலையான தருக்க ggml_sycl_is_quantized_type(எண்ணிக்கை ggml_type type) {
    தேர்வு (type) {
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ2_S:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
            திரும்பு உண்மை;
        இயல்பு:
            திரும்பு பொய்;
    }
}

நிலையான தருக்க ggml_sycl_can_quantize_rows_sycl(எண்ணிக்கை ggml_type type) {
    தேர்வு (type) {
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
            திரும்பு உண்மை;
        இயல்பு:
            திரும்பு பொய்;
    }
}

template <typename SrcScalar>
நிலையான உள்வரி தசம ggml_sycl_src_to_f32(மாறிலி SrcScalar & x) {
    திரும்பு (தசம) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
உள்வரி தசம ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(மாறிலி sycl::ext::oneapi::bfloat16 & x) {
    திரும்பு static_cast<தசம>(x);
}

template <>
உள்வரி தசம ggml_sycl_src_to_f32<ggml_bf16_t>(மாறிலி ggml_bf16_t & x) {
    ஒன்றிப்பு {
        uint32_t u32;
        தசம f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    திரும்பு value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, முழு qk>
நிலையான வெறுமை ggml_sycl_quantize_rows_q(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி int64_t ne,
                                      மாறிலி int64_t ne00, மாறிலி int64_t ne01, மாறிலி int64_t ne02,
                                      மாறிலி size_t nb00, மாறிலி size_t nb01, மாறிலி size_t nb02, மாறிலி size_t nb03,
                                      மாறிலி int64_t ne10, மாறிலி int64_t ne11, மாறிலி int64_t ne12,
                                      மாறிலி size_t nb10, மாறிலி size_t nb11, மாறிலி size_t nb12, மாறிலி size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    மாறிலி int64_t total_blocks = ne / qk;
    constexpr முழு block_size = 256;
    மாறிலி int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        மாறிலி int64_t block_idx = item_ct1.get_global_linear_id();
        எனில் (block_idx >= total_blocks) {
            திரும்பு;
        }

        மாறிலி int64_t i = block_idx * qk;

        மாறிலி int64_t i03 = i / (ne00 * ne01 * ne02);
        மாறிலி int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        மாறிலி int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        மாறிலி int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        மாறிலி size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        மாறிலி int64_t i13 = i / (ne10 * ne11 * ne12);
        மாறிலி int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        மாறிலி int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        மாறிலி int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        மாறிலி size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        தசம xf[qk];
        எனில் (nb00 == அளவு(SrcScalar)) {
            மாறிலி SrcScalar * src_row = (மாறிலி SrcScalar *) (cx + x_offset);
            க்கு (முழு j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } இல்லையெனில் {
            க்கு (முழு j = 0; j < qk; ++j) {
                மாறிலி SrcScalar * src_val = (மாறிலி SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((மாறிலி எழுத்து *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
நிலையான வெறுமை ggml_sycl_quantize_rows_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி ggml_tensor * src0, மாறிலி ggml_tensor * src1,
                                         மாறிலி int64_t ne, மாறிலி int64_t ne00, மாறிலி int64_t ne01, மாறிலி int64_t ne02,
                                         மாறிலி size_t nb00, மாறிலி size_t nb01, மாறிலி size_t nb02, மாறிலி size_t nb03,
                                         மாறிலி int64_t ne10, மாறிலி int64_t ne11, மாறிலி int64_t ne12, மாறிலி size_t nb10,
                                         மாறிலி size_t nb11, மாறிலி size_t nb12, மாறிலி size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    தேர்வு (src1->type) {
        நிலை GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            நிறுத்து;
        நிலை GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

நிலையான வெறுமை ggml_cpy_f16_f16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_i16_i16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_i32_i32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

நிலையான வெறுமை ggml_cpy_q8_0_q8_0(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


நிலையான வெறுமை ggml_cpy_q5_0_q5_0(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


நிலையான வெறுமை ggml_cpy_q5_1_q5_1(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


நிலையான வெறுமை ggml_cpy_q4_0_q4_0(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


நிலையான வெறுமை ggml_cpy_q4_1_q4_1(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q1_0_q1_0(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q2_0_q2_0(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_mxfp4_mxfp4(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_nvfp4_nvfp4(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q2_K_q2_K(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q3_K_q3_K(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q4_K_q4_K(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q5_K_q5_K(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_q6_K_q6_K(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq2_xxs_iq2_xxs(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq2_xs_iq2_xs(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq2_s_iq2_s(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq3_xxs_iq3_xxs(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq1_s_iq1_s(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq1_m_iq1_m(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq4_nl_iq4_nl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq3_s_iq3_s(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_iq4_xs_iq4_xs(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                   மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                   மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                   மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    மாறிலி முழு இலக்கம்_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
நிலையான வெறுமை ggml_cpy_f32_bf16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_bf16_f32_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_bf16_bf16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_f16_bf16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

நிலையான வெறுமை ggml_cpy_bf16_f16_sycl(மாறிலி எழுத்து * cx, எழுத்து * cdst, மாறிலி முழு ne, மாறிலி முழு ne00, மாறிலி முழு ne01,
                                  மாறிலி முழு ne02, மாறிலி முழு nb00, மாறிலி முழு nb01, மாறிலி முழு nb02, மாறிலி முழு nb03,
                                  மாறிலி முழு ne10, மாறிலி முழு ne11, மாறிலி முழு ne12, மாறிலி முழு nb10, மாறிலி முழு nb11,
                                  மாறிலி முழு nb12, மாறிலி முழு nb13, queue_ptr stream) {
    மாறிலி முழு இலக்கம்_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, இலக்கம்_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

வெறுமை ggml_sycl_cpy(ggml_backend_sycl_context & ctx, மாறிலி ggml_tensor * src0, மாறிலி ggml_tensor * src1) முயல் {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    மாறிலி int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_இருமம்_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    எழுத்து * src0_ddc = (எழுத்து *) src0->data;
    எழுத்து * src1_ddc = (எழுத்து *) src1->data;
    எனில் ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<தசம>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } இல்லையெனில் {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} பிடி (மாறிலி sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

வெறுமை ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
