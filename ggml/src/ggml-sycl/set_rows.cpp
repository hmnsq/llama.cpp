#अन्तर्भाव "set_rows.hpp"
#अन्तर्भाव "cpy.hpp"

#अन्तर्भाव "ggml-quants.h"

#अन्तर्भाव <vector>

नामाकाश utils {
ढाँचा<typename T>
स्थिर constexpr तार्किक is_arithmetic_v() {
    फर्काऊ std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ifdef GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#endif
        ;
}
}

ढाँचा<typename TIn, typename TOut>
स्थिर अन्तःस्थ std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), शून्य>
convert (अचर वर्ण* src, वर्ण* dst) {
    स्वतः src_val = *reinterpret_cast<अचर TIn*>(src);
    स्वतः dst_val = sycl::vec<TIn, 1>(src_val).ढाँचा convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#ifdef GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
ढाँचा<>
अन्तःस्थ शून्य convert<sycl::half, sycl::ext::oneapi::bfloat16>(अचर वर्ण* src, वर्ण* dst) {
    अचर भिन्न tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<अचर sycl::half*>(src))
                          .ढाँचा convert<भिन्न, sycl::rounding_mode::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#endif

ढाँचा <typename TIn, typename TIdx, typename blockType, पूर्णाङ्क qk, cpy_kernel_t cpyblck>
स्थिर शून्य set_rows_sycl_q(अचर वर्ण * __restrict__ src0_d,
                            अचर TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            अचर int64_t ne00,
                            अचर int64_t ne01,
                            अचर int64_t ne02,
                            अचर int64_t ne03,
                            अचर int64_t ne10,
                            अचर int64_t ne11,
                            अचर int64_t ne12,
                            अचर int64_t ne13,
                            // strides for src0
                            अचर माप_प्रकार  nb00,
                            अचर माप_प्रकार  nb01,
                            अचर माप_प्रकार  nb02,
                            अचर माप_प्रकार  nb03,
                            // strides for src1
                            अचर माप_प्रकार  nb10,
                            अचर माप_प्रकार  nb11,
                            अचर माप_प्रकार  nb12,
                            अचर माप_प्रकार  nb13,
                            // strides for dst
                            अचर माप_प्रकार  nb1,
                            अचर माप_प्रकार  nb2,
                            अचर माप_प्रकार  nb3,
                            queue_ptr     stream) {
    अचर int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr पूर्णाङ्क block_size   = 256;
    अचर int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        अचर int64_t i = item_ct1.get_global_linear_id();
        यदि (i >= total_blocks) {
            फर्काऊ;
        }
        अचर int64_t i_base      = i * qk;
        अचर int64_t i03         = i_base / (ne00 * ne01 * ne02);
        अचर int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        अचर int64_t i02         = rem1 / (ne00 * ne01);
        अचर int64_t rem2        = rem1 - i02 * ne00 * ne01;
        अचर int64_t i01         = rem2 / ne00;
        अचर int64_t i00         = rem2 - i01 * ne00;
        अचर int64_t i12         = i03 % ne12;
        अचर int64_t i11         = i02 % ne11;
        अचर int64_t i10         = i01;
        अचर माप_प्रकार  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        अचर वर्ण *  src_block   = src0_d + src_offset + i00 * परिमाण(TIn);
        अचर माप_प्रकार  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        अचर int64_t dst_row     = src1_d[src1_offset / परिमाण(TIdx)];
        अचर माप_प्रकार  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * परिमाण(blockType);
        वर्ण * dst_block = reinterpret_cast<वर्ण *>(reinterpret_cast<वर्ण *>(dst_d) + dst_offset);
        यदि constexpr (std::is_same_v<TIn, भिन्न>) {
            cpyblck(src_block, dst_block);
        } अन्यथा {
            भिन्न src_block_f32[qk];
            अचर TIn * src_block_t = reinterpret_cast<अचर TIn *>(src_block);
            लागि (पूर्णाङ्क j = 0; j < qk; ++j) {
                src_block_f32[j] = (भिन्न) src_block_t[j];
            }
            cpyblck(reinterpret_cast<अचर वर्ण *>(src_block_f32), dst_block);
        }
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

ढाँचा<typename blockType>
using quantize_row_qk_t = शून्य (*)(अचर भिन्न *, blockType *, int64_t);

using quantize_rows_f_t = माप_प्रकार (*)(अचर भिन्न *, शून्य *, int64_t, int64_t, अचर भिन्न *);

ढाँचा <typename TIn, typename TIdx, typename blockType, पूर्णाङ्क qk, quantize_row_qk_t<blockType> quantize_row>
स्थिर शून्य set_rows_sycl_qk_host(
        अचर ggml_tensor * src0,
        अचर ggml_tensor * src1,
        ggml_tensor * dst,
        अचर int64_t ne00,
        अचर int64_t ne01,
        अचर int64_t ne02,
        अचर int64_t ne03,
        अचर int64_t ne11,
        अचर int64_t ne12,
        अचर माप_प्रकार nb01,
        अचर माप_प्रकार nb02,
        अचर माप_प्रकार nb03,
        अचर माप_प्रकार nb10,
        अचर माप_प्रकार nb11,
        अचर माप_प्रकार nb12,
        अचर माप_प्रकार nb1,
        अचर माप_प्रकार nb2,
        अचर माप_प्रकार nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    अचर माप_प्रकार src0_bytes = ggml_nbytes(src0);
    अचर माप_प्रकार src1_bytes = ggml_nbytes(src1);

    std::vector<वर्ण> src0_host(src0_bytes);
    std::vector<वर्ण> src1_host(src1_bytes);

    stream->स_नकल(src0_host.data(), src0->data, src0_bytes);
    stream->स_नकल(src1_host.data(), src1->data, src1_bytes);
    stream->रुको();

    std::vector<भिन्न> src_row_f32(ne00);
    अचर int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    लागि (int64_t i03 = 0; i03 < ne03; ++i03) {
        लागि (int64_t i02 = 0; i02 < ne02; ++i02) {
            लागि (int64_t i01 = 0; i01 < ne01; ++i01) {
                अचर int64_t i12 = i03 % ne12;
                अचर int64_t i11 = i02 % ne11;
                अचर int64_t i10 = i01;

                अचर माप_प्रकार src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                अचर int64_t dst_row = *(अचर TIdx *) (src1_host.data() + src1_offset);

                अचर माप_प्रकार src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                अचर TIn * src_row = reinterpret_cast<अचर TIn *>(src0_host.data() + src0_row_offset);

                लागि (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (भिन्न) src_row[i00];
                }

                quantize_row(src_row_f32.data(), dst_row_q.data(), ne00);

                अचर माप_प्रकार dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->स_नकल((वर्ण *) dst->data + dst_offset, dst_row_q.data(), nblocks * परिमाण(blockType));
                stream->रुको();
            }
        }
    }
}

ढाँचा <typename TIn, typename TIdx, typename blockType, पूर्णाङ्क qk, quantize_rows_f_t quantize_rows>
स्थिर शून्य set_rows_sycl_iq_host(
        अचर ggml_tensor * src0,
        अचर ggml_tensor * src1,
        ggml_tensor * dst,
        अचर int64_t ne00,
        अचर int64_t ne01,
        अचर int64_t ne02,
        अचर int64_t ne03,
        अचर int64_t ne11,
        अचर int64_t ne12,
        अचर माप_प्रकार nb01,
        अचर माप_प्रकार nb02,
        अचर माप_प्रकार nb03,
        अचर माप_प्रकार nb10,
        अचर माप_प्रकार nb11,
        अचर माप_प्रकार nb12,
        अचर माप_प्रकार nb1,
        अचर माप_प्रकार nb2,
        अचर माप_प्रकार nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    अचर माप_प्रकार src0_bytes = ggml_nbytes(src0);
    अचर माप_प्रकार src1_bytes = ggml_nbytes(src1);

    std::vector<वर्ण> src0_host(src0_bytes);
    std::vector<वर्ण> src1_host(src1_bytes);

    stream->स_नकल(src0_host.data(), src0->data, src0_bytes);
    stream->स_नकल(src1_host.data(), src1->data, src1_bytes);
    stream->रुको();

    std::vector<भिन्न> src_row_f32(ne00);
    अचर int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    लागि (int64_t i03 = 0; i03 < ne03; ++i03) {
        लागि (int64_t i02 = 0; i02 < ne02; ++i02) {
            लागि (int64_t i01 = 0; i01 < ne01; ++i01) {
                अचर int64_t i12 = i03 % ne12;
                अचर int64_t i11 = i02 % ne11;
                अचर int64_t i10 = i01;

                अचर माप_प्रकार src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                अचर int64_t dst_row = *(अचर TIdx *) (src1_host.data() + src1_offset);

                अचर माप_प्रकार src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                अचर TIn * src_row = reinterpret_cast<अचर TIn *>(src0_host.data() + src0_row_offset);

                लागि (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (भिन्न) src_row[i00];
                }

                quantize_rows(src_row_f32.data(), dst_row_q.data(), 1, ne00, nullptr);

                अचर माप_प्रकार dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->स_नकल((वर्ण *) dst->data + dst_offset, dst_row_q.data(), nblocks * परिमाण(blockType));
                stream->रुको();
            }
        }
    }
}

ढाँचा<typename TIn, typename TIdx, typename TOut>
स्थिर शून्य k_set_rows(
        अचर वर्ण * __restrict__ src0, अचर TIdx * __restrict__ src1, वर्ण * __restrict__ dst,
        अचर int64_t ne00, अचर int64_t ne01, अचर int64_t ne02,
        अचर int64_t ne11, अचर int64_t ne12,
        अचर माप_प्रकार nb01, अचर माप_प्रकार nb02, अचर माप_प्रकार nb03,
        अचर माप_प्रकार nb10, अचर माप_प्रकार nb11, अचर माप_प्रकार nb12,
        अचर माप_प्रकार nb1, अचर माप_प्रकार nb2, अचर माप_प्रकार nb3,
        अचर माप_प्रकार src_type_size, अचर माप_प्रकार dst_type_size,
        अचर int64_t total_elements,
        अचर sycl::nd_item<1> & item_ct1) {

    अचर int64_t i = item_ct1.get_global_linear_id();
    यदि (i >= total_elements) {
        फर्काऊ;
    }

    अचर int64_t i03 = i / (ne00 * ne01 * ne02);
    अचर int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    अचर int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    अचर int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    अचर int64_t i12 = i03 % ne12;
    अचर int64_t i11 = i02 % ne11;
    अचर int64_t i10 = i01;

    अचर int64_t dst_row = *(अचर TIdx *)((अचर वर्ण *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    अचर वर्ण * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    अचर वर्ण * src_elem = src0_row + i00 * src_type_size;
    वर्ण * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    वर्ण * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

ढाँचा<typename TIn, typename TIdx, typename TOut>
स्थिर शून्य set_rows_sycl(
        अचर वर्ण * src0_d, अचर TIdx * src1_d, वर्ण * dst_d,
        अचर int64_t ne00, अचर int64_t ne01, अचर int64_t ne02, अचर int64_t ne03,
        अचर int64_t ne11, अचर int64_t ne12, अचर माप_प्रकार nb01, अचर माप_प्रकार nb02, अचर माप_प्रकार nb03,
        अचर माप_प्रकार nb10, अचर माप_प्रकार nb11, अचर माप_प्रकार nb12,
        अचर माप_प्रकार nb1, अचर माप_प्रकार nb2, अचर माप_प्रकार nb3,
        अचर माप_प्रकार src_type_size, अचर माप_प्रकार dst_type_size,
        queue_ptr stream) {

    अचर int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr पूर्णाङ्क block_size = 64;
    अचर int64_t grid_size = ceil_div(total_elements, block_size);

    stream->parallel_for(
        sycl::nd_range<1>(grid_size * block_size, block_size),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            k_set_rows<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_size, dst_type_size,
                total_elements,
                item_ct1
            );
        }
    );
}

ढाँचा<typename TIn, typename TIdx>
स्थिर शून्य set_rows_sycl(ggml_backend_sycl_context & ctx, अचर ggml_tensor * src0, अचर ggml_tensor * src1, ggml_tensor * dst) {
    अचर वर्ण * src0_d = (अचर वर्ण *)src0->data;
    अचर TIdx * src1_d = (अचर TIdx *)src1->data;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, भिन्न>(
                src0_d, src1_d, (वर्ण *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                परिमाण(TIn), परिमाण(भिन्न),
                stream
            );
            रोक;
        पक्ष GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (वर्ण *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                परिमाण(TIn), परिमाण(sycl::half),
                stream
            );
            रोक;
#ifdef GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (वर्ण *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                परिमाण(TIn), परिमाण(sycl::ext::oneapi::bfloat16),
                stream
            );
            रोक;
#endif
        पक्ष GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIn, TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (block_q8_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q1_0:
            set_rows_sycl_q<TIn, TIdx, block_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (block_q1_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q2_0:
            set_rows_sycl_q<TIn, TIdx, block_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (block_q2_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIn, TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (block_q5_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIn, TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (block_q5_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIn, TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (block_q4_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIn, TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (block_q4_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIn, TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (block_iq4_nl *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_MXFP4:
            set_rows_sycl_q<TIn, TIdx, block_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (block_mxfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_NVFP4:
            set_rows_sycl_q<TIn, TIdx, block_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (block_nvfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            रोक;
        पक्ष GGML_TYPE_Q2_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q2_K, QK_K, quantize_row_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_Q3_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q3_K, QK_K, quantize_row_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_Q4_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q4_K, QK_K, quantize_row_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_Q5_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q5_K, QK_K, quantize_row_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_Q6_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q6_K, QK_K, quantize_row_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ2_XXS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ2_XS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ2_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ3_XXS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_xxs, QK_K, quantize_row_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ3_S:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_s, QK_K, quantize_row_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ1_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ1_M:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        पक्ष GGML_TYPE_IQ4_XS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq4_xs, QK_K, quantize_row_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            रोक;
        स्वभाव:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            रोक;
    }
}

शून्य ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    अचर ggml_tensor * src0 = dst->src[0];
    अचर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    यदि (src0->type == GGML_TYPE_F16) {
        यदि (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } अन्यथा {
            set_rows_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } अन्यथा {
        यदि (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<भिन्न, int64_t>(ctx, src0, src1, dst);
        } अन्यथा {
            set_rows_sycl<भिन्न, int32_t>(ctx, src0, src1, dst);
        }
    }
}
