#समावेश "set_rows.hpp"
#समावेश "cpy.hpp"

#समावेश "ggml-quants.h"

#समावेश <vector>

namespace utils {
ढाँचा<typename T>
अटल constexpr bool is_arithmetic_v() {
    वापस std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#अगर_घोषित GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#पूर्ण_अगर
        ;
}
}

ढाँचा<typename TIn, typename TOut>
अटल अंतरभूत std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), व्योम>
convert (स्थिर अक्षर* src, अक्षर* dst) {
    स्वतः src_val = *reinterpret_cast<स्थिर TIn*>(src);
    स्वतः dst_val = sycl::vec<TIn, 1>(src_val).ढाँचा convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#अगर_घोषित GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
ढाँचा<>
अंतरभूत व्योम convert<sycl::half, sycl::ext::oneapi::bfloat16>(स्थिर अक्षर* src, अक्षर* dst) {
    स्थिर भग्न tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<स्थिर sycl::half*>(src))
                          .ढाँचा convert<भग्न, sycl::rounding_mode::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#पूर्ण_अगर

ढाँचा <typename TIn, typename TIdx, typename blockType, पूर्णांक qk, cpy_kernel_t cpyblck>
अटल व्योम set_rows_sycl_q(स्थिर अक्षर * __restrict__ src0_d,
                            स्थिर TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            स्थिर int64_t ne00,
                            स्थिर int64_t ne01,
                            स्थिर int64_t ne02,
                            स्थिर int64_t ne03,
                            स्थिर int64_t ne10,
                            स्थिर int64_t ne11,
                            स्थिर int64_t ne12,
                            स्थिर int64_t ne13,
                            // strides for src0
                            स्थिर माप_प्रकार  nb00,
                            स्थिर माप_प्रकार  nb01,
                            स्थिर माप_प्रकार  nb02,
                            स्थिर माप_प्रकार  nb03,
                            // strides for src1
                            स्थिर माप_प्रकार  nb10,
                            स्थिर माप_प्रकार  nb11,
                            स्थिर माप_प्रकार  nb12,
                            स्थिर माप_प्रकार  nb13,
                            // strides for dst
                            स्थिर माप_प्रकार  nb1,
                            स्थिर माप_प्रकार  nb2,
                            स्थिर माप_प्रकार  nb3,
                            queue_ptr     stream) {
    स्थिर int64_t योग_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr पूर्णांक block_आकार_2   = 256;
    स्थिर int64_t grid_आकार    = ceil_div(योग_blocks, block_आकार_2);

    stream->parallel_for(sycl::nd_range<1>(grid_आकार * block_आकार_2, block_आकार_2), [=](sycl::nd_item<1> item_ct1) {
        स्थिर int64_t i = item_ct1.get_global_linear_id();
        अगर (i >= योग_blocks) {
            वापस;
        }
        स्थिर int64_t i_base      = i * qk;
        स्थिर int64_t i03         = i_base / (ne00 * ne01 * ne02);
        स्थिर int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        स्थिर int64_t i02         = rem1 / (ne00 * ne01);
        स्थिर int64_t rem2        = rem1 - i02 * ne00 * ne01;
        स्थिर int64_t i01         = rem2 / ne00;
        स्थिर int64_t i00         = rem2 - i01 * ne00;
        स्थिर int64_t i12         = i03 % ne12;
        स्थिर int64_t i11         = i02 % ne11;
        स्थिर int64_t i10         = i01;
        स्थिर माप_प्रकार  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        स्थिर अक्षर *  src_block   = src0_d + src_offset + i00 * माप(TIn);
        स्थिर माप_प्रकार  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        स्थिर int64_t dst_row     = src1_d[src1_offset / माप(TIdx)];
        स्थिर माप_प्रकार  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * माप(blockType);
        अक्षर * dst_block = reinterpret_cast<अक्षर *>(reinterpret_cast<अक्षर *>(dst_d) + dst_offset);
        अगर constexpr (std::is_same_v<TIn, भग्न>) {
            cpyblck(src_block, dst_block);
        } अन्यथा {
            भग्न src_block_f32[qk];
            स्थिर TIn * src_block_t = reinterpret_cast<स्थिर TIn *>(src_block);
            क्रम (पूर्णांक j = 0; j < qk; ++j) {
                src_block_f32[j] = (भग्न) src_block_t[j];
            }
            cpyblck(reinterpret_cast<स्थिर अक्षर *>(src_block_f32), dst_block);
        }
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

ढाँचा<typename blockType>
using quantize_row_qk_t = व्योम (*)(स्थिर भग्न *, blockType *, int64_t);

using quantize_rows_f_t = माप_प्रकार (*)(स्थिर भग्न *, व्योम *, int64_t, int64_t, स्थिर भग्न *);

ढाँचा <typename TIn, typename TIdx, typename blockType, पूर्णांक qk, quantize_row_qk_t<blockType> quantize_row>
अटल व्योम set_rows_sycl_qk_host(
        स्थिर ggml_tensor * src0,
        स्थिर ggml_tensor * src1,
        ggml_tensor * dst,
        स्थिर int64_t ne00,
        स्थिर int64_t ne01,
        स्थिर int64_t ne02,
        स्थिर int64_t ne03,
        स्थिर int64_t ne11,
        स्थिर int64_t ne12,
        स्थिर माप_प्रकार nb01,
        स्थिर माप_प्रकार nb02,
        स्थिर माप_प्रकार nb03,
        स्थिर माप_प्रकार nb10,
        स्थिर माप_प्रकार nb11,
        स्थिर माप_प्रकार nb12,
        स्थिर माप_प्रकार nb1,
        स्थिर माप_प्रकार nb2,
        स्थिर माप_प्रकार nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    स्थिर माप_प्रकार src0_bytes = ggml_nbytes(src0);
    स्थिर माप_प्रकार src1_bytes = ggml_nbytes(src1);

    std::vector<अक्षर> src0_host(src0_bytes);
    std::vector<अक्षर> src1_host(src1_bytes);

    stream->स_नकल(src0_host.तथ्य(), src0->तथ्य, src0_bytes);
    stream->स_नकल(src1_host.तथ्य(), src1->तथ्य, src1_bytes);
    stream->रुको();

    std::vector<भग्न> src_row_f32(ne00);
    स्थिर int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    क्रम (int64_t i03 = 0; i03 < ne03; ++i03) {
        क्रम (int64_t i02 = 0; i02 < ne02; ++i02) {
            क्रम (int64_t i01 = 0; i01 < ne01; ++i01) {
                स्थिर int64_t i12 = i03 % ne12;
                स्थिर int64_t i11 = i02 % ne11;
                स्थिर int64_t i10 = i01;

                स्थिर माप_प्रकार src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                स्थिर int64_t dst_row = *(स्थिर TIdx *) (src1_host.तथ्य() + src1_offset);

                स्थिर माप_प्रकार src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                स्थिर TIn * src_row = reinterpret_cast<स्थिर TIn *>(src0_host.तथ्य() + src0_row_offset);

                क्रम (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (भग्न) src_row[i00];
                }

                quantize_row(src_row_f32.तथ्य(), dst_row_q.तथ्य(), ne00);

                स्थिर माप_प्रकार dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->स_नकल((अक्षर *) dst->तथ्य + dst_offset, dst_row_q.तथ्य(), nblocks * माप(blockType));
                stream->रुको();
            }
        }
    }
}

ढाँचा <typename TIn, typename TIdx, typename blockType, पूर्णांक qk, quantize_rows_f_t quantize_rows>
अटल व्योम set_rows_sycl_iq_host(
        स्थिर ggml_tensor * src0,
        स्थिर ggml_tensor * src1,
        ggml_tensor * dst,
        स्थिर int64_t ne00,
        स्थिर int64_t ne01,
        स्थिर int64_t ne02,
        स्थिर int64_t ne03,
        स्थिर int64_t ne11,
        स्थिर int64_t ne12,
        स्थिर माप_प्रकार nb01,
        स्थिर माप_प्रकार nb02,
        स्थिर माप_प्रकार nb03,
        स्थिर माप_प्रकार nb10,
        स्थिर माप_प्रकार nb11,
        स्थिर माप_प्रकार nb12,
        स्थिर माप_प्रकार nb1,
        स्थिर माप_प्रकार nb2,
        स्थिर माप_प्रकार nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    स्थिर माप_प्रकार src0_bytes = ggml_nbytes(src0);
    स्थिर माप_प्रकार src1_bytes = ggml_nbytes(src1);

    std::vector<अक्षर> src0_host(src0_bytes);
    std::vector<अक्षर> src1_host(src1_bytes);

    stream->स_नकल(src0_host.तथ्य(), src0->तथ्य, src0_bytes);
    stream->स_नकल(src1_host.तथ्य(), src1->तथ्य, src1_bytes);
    stream->रुको();

    std::vector<भग्न> src_row_f32(ne00);
    स्थिर int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    क्रम (int64_t i03 = 0; i03 < ne03; ++i03) {
        क्रम (int64_t i02 = 0; i02 < ne02; ++i02) {
            क्रम (int64_t i01 = 0; i01 < ne01; ++i01) {
                स्थिर int64_t i12 = i03 % ne12;
                स्थिर int64_t i11 = i02 % ne11;
                स्थिर int64_t i10 = i01;

                स्थिर माप_प्रकार src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                स्थिर int64_t dst_row = *(स्थिर TIdx *) (src1_host.तथ्य() + src1_offset);

                स्थिर माप_प्रकार src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                स्थिर TIn * src_row = reinterpret_cast<स्थिर TIn *>(src0_host.तथ्य() + src0_row_offset);

                क्रम (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (भग्न) src_row[i00];
                }

                quantize_rows(src_row_f32.तथ्य(), dst_row_q.तथ्य(), 1, ne00, nullptr);

                स्थिर माप_प्रकार dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->स_नकल((अक्षर *) dst->तथ्य + dst_offset, dst_row_q.तथ्य(), nblocks * माप(blockType));
                stream->रुको();
            }
        }
    }
}

ढाँचा<typename TIn, typename TIdx, typename TOut>
अटल व्योम k_set_rows(
        स्थिर अक्षर * __restrict__ src0, स्थिर TIdx * __restrict__ src1, अक्षर * __restrict__ dst,
        स्थिर int64_t ne00, स्थिर int64_t ne01, स्थिर int64_t ne02,
        स्थिर int64_t ne11, स्थिर int64_t ne12,
        स्थिर माप_प्रकार nb01, स्थिर माप_प्रकार nb02, स्थिर माप_प्रकार nb03,
        स्थिर माप_प्रकार nb10, स्थिर माप_प्रकार nb11, स्थिर माप_प्रकार nb12,
        स्थिर माप_प्रकार nb1, स्थिर माप_प्रकार nb2, स्थिर माप_प्रकार nb3,
        स्थिर माप_प्रकार src_type_आकार, स्थिर माप_प्रकार dst_type_आकार,
        स्थिर int64_t योग_elements,
        स्थिर sycl::nd_item<1> & item_ct1) {

    स्थिर int64_t i = item_ct1.get_global_linear_id();
    अगर (i >= योग_elements) {
        वापस;
    }

    स्थिर int64_t i03 = i / (ne00 * ne01 * ne02);
    स्थिर int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    स्थिर int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    स्थिर int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    स्थिर int64_t i12 = i03 % ne12;
    स्थिर int64_t i11 = i02 % ne11;
    स्थिर int64_t i10 = i01;

    स्थिर int64_t dst_row = *(स्थिर TIdx *)((स्थिर अक्षर *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    स्थिर अक्षर * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    स्थिर अक्षर * src_elem = src0_row + i00 * src_type_आकार;
    अक्षर * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    अक्षर * dst_elem = dst_row_ptr + i00 * dst_type_आकार;

    convert<TIn, TOut>(src_elem, dst_elem);
}

ढाँचा<typename TIn, typename TIdx, typename TOut>
अटल व्योम set_rows_sycl(
        स्थिर अक्षर * src0_d, स्थिर TIdx * src1_d, अक्षर * dst_d,
        स्थिर int64_t ne00, स्थिर int64_t ne01, स्थिर int64_t ne02, स्थिर int64_t ne03,
        स्थिर int64_t ne11, स्थिर int64_t ne12, स्थिर माप_प्रकार nb01, स्थिर माप_प्रकार nb02, स्थिर माप_प्रकार nb03,
        स्थिर माप_प्रकार nb10, स्थिर माप_प्रकार nb11, स्थिर माप_प्रकार nb12,
        स्थिर माप_प्रकार nb1, स्थिर माप_प्रकार nb2, स्थिर माप_प्रकार nb3,
        स्थिर माप_प्रकार src_type_आकार, स्थिर माप_प्रकार dst_type_आकार,
        queue_ptr stream) {

    स्थिर int64_t योग_elements = ne00 * ne01 * ne02 * ne03;

    constexpr पूर्णांक block_आकार_2 = 64;
    स्थिर int64_t grid_आकार = ceil_div(योग_elements, block_आकार_2);

    stream->parallel_for(
        sycl::nd_range<1>(grid_आकार * block_आकार_2, block_आकार_2),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            k_set_rows<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_आकार, dst_type_आकार,
                योग_elements,
                item_ct1
            );
        }
    );
}

ढाँचा<typename TIn, typename TIdx>
अटल व्योम set_rows_sycl(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    स्थिर अक्षर * src0_d = (स्थिर अक्षर *)src0->तथ्य;
    स्थिर TIdx * src1_d = (स्थिर TIdx *)src1->तथ्य;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    चयन (dst->type) {
        हाल GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, भग्न>(
                src0_d, src1_d, (अक्षर *)dst->तथ्य,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                माप(TIn), माप(भग्न),
                stream
            );
            अवरोध;
        हाल GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (अक्षर *)dst->तथ्य,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                माप(TIn), माप(sycl::half),
                stream
            );
            अवरोध;
#अगर_घोषित GGML_SYCL_HAS_BF16
        हाल GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (अक्षर *)dst->तथ्य,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                माप(TIn), माप(sycl::ext::oneapi::bfloat16),
                stream
            );
            अवरोध;
#पूर्ण_अगर
        हाल GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIn, TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (block_q8_0 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q1_0:
            set_rows_sycl_q<TIn, TIdx, block_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (block_q1_0 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q2_0:
            set_rows_sycl_q<TIn, TIdx, block_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (block_q2_0 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIn, TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (block_q5_1 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIn, TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (block_q5_0 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIn, TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (block_q4_1 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIn, TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (block_q4_0 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIn, TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (block_iq4_nl *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_MXFP4:
            set_rows_sycl_q<TIn, TIdx, block_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (block_mxfp4 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_NVFP4:
            set_rows_sycl_q<TIn, TIdx, block_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (block_nvfp4 *) dst->तथ्य, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            अवरोध;
        हाल GGML_TYPE_Q2_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q2_K, QK_K, quantize_row_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_Q3_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q3_K, QK_K, quantize_row_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_Q4_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q4_K, QK_K, quantize_row_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_Q5_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q5_K, QK_K, quantize_row_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_Q6_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q6_K, QK_K, quantize_row_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ2_XXS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ2_XS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ2_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ3_XXS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_xxs, QK_K, quantize_row_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ3_S:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_s, QK_K, quantize_row_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ1_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ1_M:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        हाल GGML_TYPE_IQ4_XS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq4_xs, QK_K, quantize_row_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            अवरोध;
        शेष:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            अवरोध;
    }
}

व्योम ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    अगर (src0->type == GGML_TYPE_F16) {
        अगर (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } अन्यथा {
            set_rows_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } अन्यथा {
        अगर (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<भग्न, int64_t>(ctx, src0, src1, dst);
        } अन्यथा {
            set_rows_sycl<भग्न, int32_t>(ctx, src0, src1, dst);
        }
    }
}
