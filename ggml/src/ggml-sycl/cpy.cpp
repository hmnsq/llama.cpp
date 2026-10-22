#ضمن "cpy.hpp"

#ضمن <float.h>
#ضمن <vector>

#ضمن "dequantize.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml-sycl/presets.hpp"
#ضمن "ggml.h"

ساكن فراغ cpy_1_f32_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    عائم *       dsti = (عائم *) cdsti;

    *dsti = *xi;
}

ساكن فراغ cpy_1_f32_f16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    sycl::half *  dsti = (sycl::half *) cdsti;

    *dsti = sycl::vec<عائم, 1>(*xi).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

ساكن فراغ cpy_1_f16_f16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi   = (ثابت sycl::half *) cxi;
    sycl::half *       dsti = (sycl::half *) cdsti;

    *dsti = *xi;
}

ساكن فراغ cpy_1_f16_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi   = (ثابت sycl::half *) cxi;
    عائم *            dsti = (عائم *) cdsti;

    *dsti = *xi;
}

ساكن فراغ cpy_1_i16_i16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت int16_t * xi   = (ثابت int16_t *) cxi;
    int16_t *       dsti = (int16_t *) cdsti;

    *dsti = *xi;
}

ساكن فراغ cpy_1_i32_i32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت int32_t * xi   = (ثابت int32_t *) cxi;
    int32_t *       dsti = (int32_t *) cdsti;

    *dsti = *xi;
}

ساكن فراغ cpy_1_f32_i32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi     = (ثابت عائم *) cxi;
    int32_t *     dsti   = (int32_t *) cdsti;

    *dsti = (int32_t) *xi;
}

ساكن فراغ cpy_1_i32_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت int32_t * xi   = (ثابت int32_t *) cxi;
    عائم *         dsti = (عائم *) cdsti;

    *dsti = (عائم) *xi;
}

#ifdef GGML_SYCL_HAS_BF16
ساكن فراغ cpy_1_f32_bf16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت عائم * xi   = (ثابت عائم *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(*xi);
}

ساكن فراغ cpy_1_bf16_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::ext::oneapi::bfloat16 * xi   = (ثابت sycl::ext::oneapi::bfloat16 *) cxi;
    عائم *            dsti = (عائم *) cdsti;

    *dsti = static_cast<عائم>(*xi);
}

ساكن فراغ cpy_1_bf16_bf16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::ext::oneapi::bfloat16 * xi   = (ثابت sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::ext::oneapi::bfloat16 *       dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = *xi;
}

ساكن فراغ cpy_1_f16_bf16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi   = (ثابت sycl::half *) cxi;
    sycl::ext::oneapi::bfloat16 *  dsti = (sycl::ext::oneapi::bfloat16 *) cdsti;

    *dsti = sycl::ext::oneapi::bfloat16(static_cast<عائم>(*xi));
}

ساكن فراغ cpy_1_bf16_f16(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::ext::oneapi::bfloat16 * xi   = (ثابت sycl::ext::oneapi::bfloat16 *) cxi;
    sycl::half *            dsti = (sycl::half *) cdsti;

    *dsti = sycl::half(static_cast<عائم>(*xi));
}
#endif

template <cpy_kernel_t cpy_1>
ساكن فراغ cpy_f32_f16(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01, ثابت صحيح ne02,
                        ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03, ثابت صحيح ne10, ثابت صحيح ne11,
                        ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11, ثابت صحيح nb12, ثابت صحيح nb13,
                        ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح i = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    إذا (i >= ne) {
        أرجع;
    }

    // determine indices i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    ثابت صحيح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحيح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحيح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحيح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحيح x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ثابت صحيح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحيح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحيح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحيح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحيح dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}


/* quantized type same copy */
template<typename T>
ساكن فراغ cpy_blck_q_q(ثابت حرف * cxi, حرف * cdsti) {
    ثابت T * xi = (ثابت T *) cxi;
    T * dsti = (T *) cdsti;
    *dsti = *xi;
}


ساكن فراغ cpy_blck_q8_0_f32(ثابت حرف * cxi, حرف * cdsti) {
    عائم * cdstf = (عائم *) (cdsti);

    لكل (صحيح j = 0; j < QK8_0; j += 2) {
        dfloat2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j)     = dq.x();
        *(cdstf + j + 1) = dq.y();
    }
}

ساكن فراغ cpy_blck_q2_0_f32(ثابت حرف * cxi, حرف * cdsti) {
    ثابت block_q2_0 * xi = (ثابت block_q2_0 *) cxi;
    عائم * cdstf = (عائم *) cdsti;

    ثابت عائم d = xi->d;

    لكل (صحيح j = 0; j < QK2_0; ++j) {
        ثابت صحيح byte_index = j / 4;
        ثابت صحيح bit_offset = (j % 4) * 2;
        ثابت صحيح q = (xi->qs[byte_index] >> bit_offset) & 0x3;
        cdstf[j] = (عائم) (q - 1) * d;
    }
}



template <dequantize_kernel_t dequant, صحيح qk> ساكن فراغ cpy_blck_q_f32(ثابت حرف * cxi, حرف * cdsti) {
    عائم * cdstf = (عائم *) (cdsti);

    لكل (صحيح j = 0; j < qk / 2; j++) {
        dfloat2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j)          = dq.x();
        *(cdstf + j + qk / 2) = dq.y();
    }
}


template <typename T, صحيح qk>
ساكن فراغ cpy_q_q(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01, ثابت صحيح ne02,
                      ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03, ثابت صحيح ne10, ثابت صحيح ne11,
                      ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11, ثابت صحيح nb12, ثابت صحيح nb13,
                      ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    إذا (i >= ne) {
        أرجع;
    }

    ثابت صحيح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحيح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحيح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحيح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحيح x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;


    ثابت صحيح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحيح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحيح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحيح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحيح dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck_q_q<T>(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, صحيح qk>
ساكن فراغ cpy_f32_q(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01, ثابت صحيح ne02,
                      ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03, ثابت صحيح ne10, ثابت صحيح ne11,
                      ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11, ثابت صحيح nb12, ثابت صحيح nb13,
                      ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    إذا (i >= ne) {
        أرجع;
    }


    ثابت صحيح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحيح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحيح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحيح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحيح x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ثابت صحيح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحيح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحيح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحيح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحيح dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, صحيح qk>
ساكن فراغ cpy_q_f32(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01, ثابت صحيح ne02,
                      ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03, ثابت صحيح ne10, ثابت صحيح ne11,
                      ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11, ثابت صحيح nb12, ثابت صحيح nb13,
                      ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح i = (item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2)) * qk;

    إذا (i >= ne) {
        أرجع;
    }

    ثابت صحيح i03      = i / (ne00 * ne01 * ne02);
    ثابت صحيح i02      = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ثابت صحيح i01      = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
    ثابت صحيح i00      = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
    ثابت صحيح x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

    ثابت صحيح i13        = i / (ne10 * ne11 * ne12);
    ثابت صحيح i12        = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    ثابت صحيح i11        = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
    ثابت صحيح i10        = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
    ثابت صحيح dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

ساكن فراغ ggml_cpy_f16_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_f32_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_f32_f16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_f32_i32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_i32_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_f32_q8_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_q8_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_q_f32<cpy_blck_q8_0_f32, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_q2_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q2_0_f32, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_f32_q4_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_q4_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ساكن فراغ ggml_cpy_f32_q4_1_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_q4_1_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ساكن فراغ ggml_cpy_f32_q5_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             cpy_f32_q<cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_q5_0_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ساكن فراغ ggml_cpy_f32_q5_1_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_q5_1_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02,
                                                                     nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                                     item_ct1);
        });
}

ساكن فراغ ggml_cpy_mxfp4_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                    ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                    ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                    ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_f32<cpy_blck_q_f32<dequantize_mxfp4, QK_MXFP4>, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                             nb01, nb02, nb03, ne10, ne11, ne12,
                                                                             nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_f32_iq4_nl_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                     ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                     ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                     ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11,
                                                   ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ cpy_blck_f16_q4_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi = (ثابت sycl::half *) cxi;
    عائم xf[QK4_0];

    لكل (صحيح j = 0; j < QK4_0; ++j) {
        xf[j] = (عائم) xi[j];
    }

    cpy_blck_f32_q4_0((ثابت حرف *) xf, cdsti);
}

ساكن فراغ cpy_blck_f16_q4_1(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi = (ثابت sycl::half *) cxi;
    عائم xf[QK4_1];

    لكل (صحيح j = 0; j < QK4_1; ++j) {
        xf[j] = (عائم) xi[j];
    }

    cpy_blck_f32_q4_1((ثابت حرف *) xf, cdsti);
}

ساكن فراغ cpy_blck_f16_q5_0(ثابت حرف * cxi, حرف * cdsti) {
    ثابت sycl::half * xi = (ثابت sycl::half *) cxi;
    عائم xf[QK5_0];

    لكل (صحيح j = 0; j < QK5_0; ++j) {
        xf[j] = (عائم) xi[j];
    }

    cpy_blck_f32_q5_0((ثابت حرف *) xf, cdsti);
}

ساكن فراغ ggml_cpy_f16_q4_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_f16_q4_1_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن فراغ ggml_cpy_f16_q5_0_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                             cpy_f32_q<cpy_blck_f16_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02,
                                                                 nb00, nb01, nb02, nb03,
                                                                 ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
                         });
}

ساكن منطقي ggml_sycl_is_quantized_type(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
            أرجع true;
        افتراضي:
            أرجع خطأ;
    }
}

ساكن منطقي ggml_sycl_can_quantize_rows_sycl(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q2_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_MXFP4:
        حالة GGML_TYPE_NVFP4:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ4_NL:
        حالة GGML_TYPE_IQ4_XS:
            أرجع true;
        افتراضي:
            أرجع خطأ;
    }
}

template <typename SrcScalar>
ساكن مضمن عائم ggml_sycl_src_to_f32(ثابت SrcScalar & x) {
    أرجع (عائم) x;
}

#ifdef GGML_SYCL_HAS_BF16
template <>
مضمن عائم ggml_sycl_src_to_f32<sycl::ext::oneapi::bfloat16>(ثابت sycl::ext::oneapi::bfloat16 & x) {
    أرجع static_cast<عائم>(x);
}

template <>
مضمن عائم ggml_sycl_src_to_f32<ggml_bf16_t>(ثابت ggml_bf16_t & x) {
    اتحاد {
        uint32_t u32;
        عائم f32;
    } value;

    value.u32 = (uint32_t) x.bits << 16;
    أرجع value.f32;
}
#endif

template <typename SrcScalar, cpy_kernel_t quantize_block, صحيح qk>
ساكن فراغ ggml_sycl_quantize_rows_q(ثابت حرف * cx, حرف * cdst, ثابت int64_t ne,
                                      ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
                                      ثابت size_t nb00, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
                                      ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12,
                                      ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12, ثابت size_t nb13,
                                      queue_ptr stream) {
    GGML_ASSERT(ne % qk == 0);
    GGML_ASSERT(ne00 % qk == 0);

    ثابت int64_t total_blocks = ne / qk;
    constexpr صحيح block_size = 256;
    ثابت int64_t grid_size = ceil_div(total_blocks, (int64_t) block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size),
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
        ثابت int64_t block_idx = item_ct1.get_global_linear_id();
        إذا (block_idx >= total_blocks) {
            أرجع;
        }

        ثابت int64_t i = block_idx * qk;

        ثابت int64_t i03 = i / (ne00 * ne01 * ne02);
        ثابت int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        ثابت int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
        ثابت int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
        ثابت size_t x_offset = i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

        ثابت int64_t i13 = i / (ne10 * ne11 * ne12);
        ثابت int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
        ثابت int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
        ثابت int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
        ثابت size_t dst_offset = (i10 / qk) * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

        عائم xf[qk];
        إذا (nb00 == حجم(SrcScalar)) {
            ثابت SrcScalar * src_row = (ثابت SrcScalar *) (cx + x_offset);
            لكل (صحيح j = 0; j < qk; ++j) {
                xf[j] = ggml_sycl_src_to_f32(src_row[j]);
            }
        } وإلا {
            لكل (صحيح j = 0; j < qk; ++j) {
                ثابت SrcScalar * src_val = (ثابت SrcScalar *) (cx + x_offset + j * nb00);
                xf[j] = ggml_sycl_src_to_f32(*src_val);
            }
        }

        quantize_block((ثابت حرف *) xf, cdst + dst_offset);
    });
}

template <typename SrcScalar>
ساكن فراغ ggml_sycl_quantize_rows_sycl(ثابت حرف * cx, حرف * cdst, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1,
                                         ثابت int64_t ne, ثابت int64_t ne00, ثابت int64_t ne01, ثابت int64_t ne02,
                                         ثابت size_t nb00, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
                                         ثابت int64_t ne10, ثابت int64_t ne11, ثابت int64_t ne12, ثابت size_t nb10,
                                         ثابت size_t nb11, ثابت size_t nb12, ثابت size_t nb13, queue_ptr stream) {
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);

    بدل (src1->type) {
        حالة GGML_TYPE_Q8_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_Q1_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_Q2_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_Q5_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_Q5_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_Q4_1:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_Q4_0:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01,
                                                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                                            nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_IQ4_NL:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                               nb01, nb02, nb03, ne10, ne11, ne12,
                                                                               nb10, nb11, nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_MXFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            اكسر;
        حالة GGML_TYPE_NVFP4:
            ggml_sycl_quantize_rows_q<SrcScalar, cpy_blck_f32_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00,
                                                                                nb01, nb02, nb03, ne10, ne11, ne12,
                                                                                nb10, nb11, nb12, nb13, stream);
            اكسر;
        افتراضي:
            GGML_ABORT("unsupported quantized target type in sycl quantizer src1->type=%s\n",
                       ggml_type_name(src1->type));
    }
}

ساكن فراغ ggml_cpy_f16_f16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_f16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_i16_i16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i16_i16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_i32_i32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    {
        // dpct::has_capability_or_fail(stream->get_device(),
        //                              {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
                cpy_f32_f16<cpy_1_i32_i32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
            });
    }
}

ساكن فراغ ggml_cpy_q8_0_q8_0(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK8_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK8_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q8_0, QK8_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ساكن فراغ ggml_cpy_q5_0_q5_0(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_0, QK5_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ساكن فراغ ggml_cpy_q5_1_q5_1(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK5_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK5_1, SYCL_CPY_BLOCK_SIZE);

    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                              sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
                              [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_1, QK5_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ساكن فراغ ggml_cpy_q4_0_q4_0(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_0, QK4_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}


ساكن فراغ ggml_cpy_q4_1_q4_1(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_1, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_1, QK4_1>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q1_0_q1_0(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK1_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK1_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_q1_0, QK1_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q2_0_q2_0(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK2_0 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK2_0, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_0, QK2_0>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_mxfp4_mxfp4(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_MXFP4 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_MXFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            cpy_q_q<block_mxfp4, QK_MXFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_nvfp4_nvfp4(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_NVFP4 == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_NVFP4, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_nvfp4, QK_NVFP4>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q2_K_q2_K(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q2_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q3_K_q3_K(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q3_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q4_K_q4_K(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q4_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q5_K_q5_K(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q5_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_q6_K_q6_K(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_q6_K, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq2_xxs_iq2_xxs(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq2_xs_iq2_xs(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq2_s_iq2_s(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq2_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq3_xxs_iq3_xxs(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_xxs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq1_s_iq1_s(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq1_m_iq1_m(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq1_m, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq4_nl_iq4_nl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK4_NL == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK4_NL, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_nl, QK4_NL>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq3_s_iq3_s(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq3_s, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_iq4_xs_iq4_xs(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                   ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                   ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                   ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    GGML_ASSERT(ne % QK_K == 0);
    ثابت صحيح اڠک_blocks = ceil_div(ne / QK_K, SYCL_CPY_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE), sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_q_q<block_iq4_xs, QK_K>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, item_ct1);
        });
}

#ifdef GGML_SYCL_HAS_BF16
ساكن فراغ ggml_cpy_f32_bf16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f32_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_bf16_f32_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f32>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_bf16_bf16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                            nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_f16_bf16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_f16_bf16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}

ساكن فراغ ggml_cpy_bf16_f16_sycl(ثابت حرف * cx, حرف * cdst, ثابت صحيح ne, ثابت صحيح ne00, ثابت صحيح ne01,
                                  ثابت صحيح ne02, ثابت صحيح nb00, ثابت صحيح nb01, ثابت صحيح nb02, ثابت صحيح nb03,
                                  ثابت صحيح ne10, ثابت صحيح ne11, ثابت صحيح ne12, ثابت صحيح nb10, ثابت صحيح nb11,
                                  ثابت صحيح nb12, ثابت صحيح nb13, queue_ptr stream) {
    ثابت صحيح اڠک_blocks = (ne + SYCL_CPY_BLOCK_SIZE - 1) / SYCL_CPY_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, اڠک_blocks) * sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_CPY_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            cpy_f32_f16<cpy_1_bf16_f16>(cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                           nb10, nb11, nb12, nb13, item_ct1);
        });
}
#endif

فراغ ggml_sycl_cpy(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1) حاول {
    // Unlike other operators ggml_sycl_cpy takes 2 distinct tensors instead of a dst ggml_tensor and rely on its src field
    GGML_SYCL_DEBUG("ggml_sycl_cpy: src0->type=%s, src1->type=%s\n",
        ggml_type_name(src0->type), ggml_type_name(src1->type));
    scope_op_debug_print scope_dbg_print(__func__, src1, /*num_src=*/0, debug_get_tensor_str("\tsrc0", src0));
    ثابت int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));
    // the cpy kernels index in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    GGML_TENSOR_بيناري_OP_LOCALS01;

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    حرف * src0_ddc = (حرف *) src0->data;
    حرف * src1_ddc = (حرف *) src1->data;
    إذا ((src0->type == src1->type) && (ggml_is_contiguous(src0) && ggml_is_contiguous(src1))) {
        GGML_SYCL_DEBUG("%s: memcpy path\n", __func__);
        main_stream->memcpy(src1_ddc, src0_ddc, ggml_nbytes(src0));
    } وإلا إذا (src0->type == GGML_TYPE_F32 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<عائم>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00, nb01,
                                            nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<sycl::half>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02, nb00,
                                                 nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13,
                                                 main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && ggml_sycl_is_quantized_type(src1->type)) {
        GGML_ASSERT(ggml_sycl_can_quantize_rows_sycl(src1->type));
        ggml_sycl_quantize_rows_sycl<ggml_bf16_t>(src0_ddc, src1_ddc, src0, src1, ne, ne00, ne01, ne02,
                                                  nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
                                                  nb12, nb13, main_stream);
#endif
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f32_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_f32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_f16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_f16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f16_q4_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f16_q4_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f16_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02,
                               nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16) {
        ggml_cpy_i16_i16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        ggml_cpy_i32_i32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_i32_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q2_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                               nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_mxfp4_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12,
                                 nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_q8_0_q8_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_q5_0_q5_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_q5_1_q5_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_q4_0_q4_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_q4_1_q4_1(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q1_0 && src1->type == GGML_TYPE_Q1_0) {
        ggml_cpy_q1_0_q1_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q2_0 && src1->type == GGML_TYPE_Q2_0) {
        ggml_cpy_q2_0_q2_0(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_MXFP4) {
        ggml_cpy_mxfp4_mxfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_NVFP4 && src1->type == GGML_TYPE_NVFP4) {
        ggml_cpy_nvfp4_nvfp4(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q2_K && src1->type == GGML_TYPE_Q2_K) {
        ggml_cpy_q2_K_q2_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q3_K && src1->type == GGML_TYPE_Q3_K) {
        ggml_cpy_q3_K_q3_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_Q4_K) {
        ggml_cpy_q4_K_q4_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_Q5_K) {
        ggml_cpy_q5_K_q5_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_Q6_K) {
        ggml_cpy_q6_K_q6_K(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ2_XXS && src1->type == GGML_TYPE_IQ2_XXS) {
        ggml_cpy_iq2_xxs_iq2_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ2_XS && src1->type == GGML_TYPE_IQ2_XS) {
        ggml_cpy_iq2_xs_iq2_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ2_S && src1->type == GGML_TYPE_IQ2_S) {
        ggml_cpy_iq2_s_iq2_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ3_XXS && src1->type == GGML_TYPE_IQ3_XXS) {
        ggml_cpy_iq3_xxs_iq3_xxs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ1_S && src1->type == GGML_TYPE_IQ1_S) {
        ggml_cpy_iq1_s_iq1_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ1_M && src1->type == GGML_TYPE_IQ1_M) {
        ggml_cpy_iq1_m_iq1_m(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ4_NL && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_iq4_nl_iq4_nl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ3_S && src1->type == GGML_TYPE_IQ3_S) {
        ggml_cpy_iq3_s_iq3_s(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_IQ4_XS && src1->type == GGML_TYPE_IQ4_XS) {
        ggml_cpy_iq4_xs_iq4_xs(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } وإلا إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f32_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_bf16_f32_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_bf16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                               nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        ggml_cpy_f16_bf16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        ggml_cpy_bf16_f16_sycl(src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10,
                              nb11, nb12, nb13, main_stream);
#endif
    } وإلا {
        GGML_LOG_ERROR("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(src0->type),
                       ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
} امسك (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

فراغ ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_cpy(ctx, dst->src[0], dst);
}
