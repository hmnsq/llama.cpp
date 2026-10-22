#ଅନ୍ତର୍ଭାଵ "set_rows.hpp"
#ଅନ୍ତର୍ଭାଵ "cpy.hpp"

#ଅନ୍ତର୍ଭାଵ "ggml-quants.h"

#ଅନ୍ତର୍ଭାଵ <vector>

ନାମାକାଶ utils {
template<typename T>
ସ୍ଥିର constexpr ତାର୍କିକ is_arithmetic_v() {
    ଫେରନ୍ତୁ std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ifdef GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#endif
        ;
}
}

template<typename TIn, typename TOut>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), ଶୂନ୍ୟ>
convert (ଅଚଳ ଅକ୍ଷର* src, ଅକ୍ଷର* dst) {
    auto src_val = *reinterpret_cast<ଅଚଳ TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#ifdef GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
template<>
ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ convert<sycl::half, sycl::ext::oneapi::bfloat16>(ଅଚଳ ଅକ୍ଷର* src, ଅକ୍ଷର* dst) {
    ଅଚଳ ଦଶମିକ tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ଅଚଳ sycl::half*>(src))
                          .template convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#endif

template <typename TIn, typename TIdx, typename blockType, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, cpy_kernel_t cpyblck>
ସ୍ଥିର ଶୂନ୍ୟ set_rows_sycl_q(ଅଚଳ ଅକ୍ଷର * __restrict__ src0_d,
                            ଅଚଳ TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            ଅଚଳ int64_t ne00,
                            ଅଚଳ int64_t ne01,
                            ଅଚଳ int64_t ne02,
                            ଅଚଳ int64_t ne03,
                            ଅଚଳ int64_t ne10,
                            ଅଚଳ int64_t ne11,
                            ଅଚଳ int64_t ne12,
                            ଅଚଳ int64_t ne13,
                            // strides for src0
                            ଅଚଳ size_t  nb00,
                            ଅଚଳ size_t  nb01,
                            ଅଚଳ size_t  nb02,
                            ଅଚଳ size_t  nb03,
                            // strides for src1
                            ଅଚଳ size_t  nb10,
                            ଅଚଳ size_t  nb11,
                            ଅଚଳ size_t  nb12,
                            ଅଚଳ size_t  nb13,
                            // strides for dst
                            ଅଚଳ size_t  nb1,
                            ଅଚଳ size_t  nb2,
                            ଅଚଳ size_t  nb3,
                            queue_ptr     stream) {
    ଅଚଳ int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size   = 256;
    ଅଚଳ int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        ଅଚଳ int64_t i = item_ct1.get_global_linear_id();
        ଯଦି (i >= total_blocks) {
            ଫେରନ୍ତୁ;
        }
        ଅଚଳ int64_t i_base      = i * qk;
        ଅଚଳ int64_t i03         = i_base / (ne00 * ne01 * ne02);
        ଅଚଳ int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        ଅଚଳ int64_t i02         = rem1 / (ne00 * ne01);
        ଅଚଳ int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ଅଚଳ int64_t i01         = rem2 / ne00;
        ଅଚଳ int64_t i00         = rem2 - i01 * ne00;
        ଅଚଳ int64_t i12         = i03 % ne12;
        ଅଚଳ int64_t i11         = i02 % ne11;
        ଅଚଳ int64_t i10         = i01;
        ଅଚଳ size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ଅଚଳ ଅକ୍ଷର *  src_block   = src0_d + src_offset + i00 * ଆକାର(TIn);
        ଅଚଳ size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ଅଚଳ int64_t dst_row     = src1_d[src1_offset / ଆକାର(TIdx)];
        ଅଚଳ size_t  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * ଆକାର(blockType);
        ଅକ୍ଷର * dst_block = reinterpret_cast<ଅକ୍ଷର *>(reinterpret_cast<ଅକ୍ଷର *>(dst_d) + dst_offset);
        ଯଦି constexpr (std::is_same_v<TIn, ଦଶମିକ>) {
            cpyblck(src_block, dst_block);
        } ଅନ୍ୟଥା {
            ଦଶମିକ src_block_f32[qk];
            ଅଚଳ TIn * src_block_t = reinterpret_cast<ଅଚଳ TIn *>(src_block);
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < qk; ++j) {
                src_block_f32[j] = (ଦଶମିକ) src_block_t[j];
            }
            cpyblck(reinterpret_cast<ଅଚଳ ଅକ୍ଷର *>(src_block_f32), dst_block);
        }
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template<typename blockType>
using quantize_row_qk_t = ଶୂନ୍ୟ (*)(ଅଚଳ ଦଶମିକ *, blockType *, int64_t);

using quantize_rows_f_t = size_t (*)(ଅଚଳ ଦଶମିକ *, ଶୂନ୍ୟ *, int64_t, int64_t, ଅଚଳ ଦଶମିକ *);

template <typename TIn, typename TIdx, typename blockType, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, quantize_row_qk_t<blockType> quantize_row>
ସ୍ଥିର ଶୂନ୍ୟ set_rows_sycl_qk_host(
        ଅଚଳ ggml_tensor * src0,
        ଅଚଳ ggml_tensor * src1,
        ggml_tensor * dst,
        ଅଚଳ int64_t ne00,
        ଅଚଳ int64_t ne01,
        ଅଚଳ int64_t ne02,
        ଅଚଳ int64_t ne03,
        ଅଚଳ int64_t ne11,
        ଅଚଳ int64_t ne12,
        ଅଚଳ size_t nb01,
        ଅଚଳ size_t nb02,
        ଅଚଳ size_t nb03,
        ଅଚଳ size_t nb10,
        ଅଚଳ size_t nb11,
        ଅଚଳ size_t nb12,
        ଅଚଳ size_t nb1,
        ଅଚଳ size_t nb2,
        ଅଚଳ size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ଅଚଳ size_t src0_bytes = ggml_nbytes(src0);
    ଅଚଳ size_t src1_bytes = ggml_nbytes(src1);

    std::vector<ଅକ୍ଷର> src0_host(src0_bytes);
    std::vector<ଅକ୍ଷର> src1_host(src1_bytes);

    stream->memcpy(src0_host.data(), src0->data, src0_bytes);
    stream->memcpy(src1_host.data(), src1->data, src1_bytes);
    stream->wait();

    std::vector<ଦଶମିକ> src_row_f32(ne00);
    ଅଚଳ int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ପାଇଁ (int64_t i01 = 0; i01 < ne01; ++i01) {
                ଅଚଳ int64_t i12 = i03 % ne12;
                ଅଚଳ int64_t i11 = i02 % ne11;
                ଅଚଳ int64_t i10 = i01;

                ଅଚଳ size_t src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ଅଚଳ int64_t dst_row = *(ଅଚଳ TIdx *) (src1_host.data() + src1_offset);

                ଅଚଳ size_t src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ଅଚଳ TIn * src_row = reinterpret_cast<ଅଚଳ TIn *>(src0_host.data() + src0_row_offset);

                ପାଇଁ (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (ଦଶମିକ) src_row[i00];
                }

                quantize_row(src_row_f32.data(), dst_row_q.data(), ne00);

                ଅଚଳ size_t dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->memcpy((ଅକ୍ଷର *) dst->data + dst_offset, dst_row_q.data(), nblocks * ଆକାର(blockType));
                stream->wait();
            }
        }
    }
}

template <typename TIn, typename TIdx, typename blockType, ପୂର୍ଣ୍ଣସଂଖ୍ୟା qk, quantize_rows_f_t quantize_rows>
ସ୍ଥିର ଶୂନ୍ୟ set_rows_sycl_iq_host(
        ଅଚଳ ggml_tensor * src0,
        ଅଚଳ ggml_tensor * src1,
        ggml_tensor * dst,
        ଅଚଳ int64_t ne00,
        ଅଚଳ int64_t ne01,
        ଅଚଳ int64_t ne02,
        ଅଚଳ int64_t ne03,
        ଅଚଳ int64_t ne11,
        ଅଚଳ int64_t ne12,
        ଅଚଳ size_t nb01,
        ଅଚଳ size_t nb02,
        ଅଚଳ size_t nb03,
        ଅଚଳ size_t nb10,
        ଅଚଳ size_t nb11,
        ଅଚଳ size_t nb12,
        ଅଚଳ size_t nb1,
        ଅଚଳ size_t nb2,
        ଅଚଳ size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ଅଚଳ size_t src0_bytes = ggml_nbytes(src0);
    ଅଚଳ size_t src1_bytes = ggml_nbytes(src1);

    std::vector<ଅକ୍ଷର> src0_host(src0_bytes);
    std::vector<ଅକ୍ଷର> src1_host(src1_bytes);

    stream->memcpy(src0_host.data(), src0->data, src0_bytes);
    stream->memcpy(src1_host.data(), src1->data, src1_bytes);
    stream->wait();

    std::vector<ଦଶମିକ> src_row_f32(ne00);
    ଅଚଳ int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    ପାଇଁ (int64_t i03 = 0; i03 < ne03; ++i03) {
        ପାଇଁ (int64_t i02 = 0; i02 < ne02; ++i02) {
            ପାଇଁ (int64_t i01 = 0; i01 < ne01; ++i01) {
                ଅଚଳ int64_t i12 = i03 % ne12;
                ଅଚଳ int64_t i11 = i02 % ne11;
                ଅଚଳ int64_t i10 = i01;

                ଅଚଳ size_t src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ଅଚଳ int64_t dst_row = *(ଅଚଳ TIdx *) (src1_host.data() + src1_offset);

                ଅଚଳ size_t src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ଅଚଳ TIn * src_row = reinterpret_cast<ଅଚଳ TIn *>(src0_host.data() + src0_row_offset);

                ପାଇଁ (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (ଦଶମିକ) src_row[i00];
                }

                quantize_rows(src_row_f32.data(), dst_row_q.data(), 1, ne00, nullptr);

                ଅଚଳ size_t dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->memcpy((ଅକ୍ଷର *) dst->data + dst_offset, dst_row_q.data(), nblocks * ଆକାର(blockType));
                stream->wait();
            }
        }
    }
}

template<typename TIn, typename TIdx, typename TOut>
ସ୍ଥିର ଶୂନ୍ୟ k_set_rows(
        ଅଚଳ ଅକ୍ଷର * __restrict__ src0, ଅଚଳ TIdx * __restrict__ src1, ଅକ୍ଷର * __restrict__ dst,
        ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02,
        ଅଚଳ int64_t ne11, ଅଚଳ int64_t ne12,
        ଅଚଳ size_t nb01, ଅଚଳ size_t nb02, ଅଚଳ size_t nb03,
        ଅଚଳ size_t nb10, ଅଚଳ size_t nb11, ଅଚଳ size_t nb12,
        ଅଚଳ size_t nb1, ଅଚଳ size_t nb2, ଅଚଳ size_t nb3,
        ଅଚଳ size_t src_type_size, ଅଚଳ size_t dst_type_size,
        ଅଚଳ int64_t total_elements,
        ଅଚଳ sycl::nd_item<1> & item_ct1) {

    ଅଚଳ int64_t i = item_ct1.get_global_linear_id();
    ଯଦି (i >= total_elements) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t i03 = i / (ne00 * ne01 * ne02);
    ଅଚଳ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ଅଚଳ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ଅଚଳ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ଅଚଳ int64_t i12 = i03 % ne12;
    ଅଚଳ int64_t i11 = i02 % ne11;
    ଅଚଳ int64_t i10 = i01;

    ଅଚଳ int64_t dst_row = *(ଅଚଳ TIdx *)((ଅଚଳ ଅକ୍ଷର *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ଅଚଳ ଅକ୍ଷର * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ଅଚଳ ଅକ୍ଷର * src_elem = src0_row + i00 * src_type_size;
    ଅକ୍ଷର * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    ଅକ୍ଷର * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template<typename TIn, typename TIdx, typename TOut>
ସ୍ଥିର ଶୂନ୍ୟ set_rows_sycl(
        ଅଚଳ ଅକ୍ଷର * src0_d, ଅଚଳ TIdx * src1_d, ଅକ୍ଷର * dst_d,
        ଅଚଳ int64_t ne00, ଅଚଳ int64_t ne01, ଅଚଳ int64_t ne02, ଅଚଳ int64_t ne03,
        ଅଚଳ int64_t ne11, ଅଚଳ int64_t ne12, ଅଚଳ size_t nb01, ଅଚଳ size_t nb02, ଅଚଳ size_t nb03,
        ଅଚଳ size_t nb10, ଅଚଳ size_t nb11, ଅଚଳ size_t nb12,
        ଅଚଳ size_t nb1, ଅଚଳ size_t nb2, ଅଚଳ size_t nb3,
        ଅଚଳ size_t src_type_size, ଅଚଳ size_t dst_type_size,
        queue_ptr stream) {

    ଅଚଳ int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = 64;
    ଅଚଳ int64_t grid_size = ceil_div(total_elements, block_size);

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

template<typename TIn, typename TIdx>
ସ୍ଥିର ଶୂନ୍ୟ set_rows_sycl(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1, ggml_tensor * dst) {
    ଅଚଳ ଅକ୍ଷର * src0_d = (ଅଚଳ ଅକ୍ଷର *)src0->data;
    ଅଚଳ TIdx * src1_d = (ଅଚଳ TIdx *)src1->data;

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, ଦଶମିକ>(
                src0_d, src1_d, (ଅକ୍ଷର *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ଆକାର(TIn), ଆକାର(ଦଶମିକ),
                stream
            );
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (ଅକ୍ଷର *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ଆକାର(TIn), ଆକାର(sycl::half),
                stream
            );
            ଭାଙ୍ଗ;
#ifdef GGML_SYCL_HAS_BF16
        ସ୍ଥିତି GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (ଅକ୍ଷର *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                ଆକାର(TIn), ଆକାର(sycl::ext::oneapi::bfloat16),
                stream
            );
            ଭାଙ୍ଗ;
#endif
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIn, TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (block_q8_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q1_0:
            set_rows_sycl_q<TIn, TIdx, block_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (block_q1_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_0:
            set_rows_sycl_q<TIn, TIdx, block_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (block_q2_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIn, TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (block_q5_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIn, TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (block_q5_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIn, TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (block_q4_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIn, TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (block_q4_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIn, TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (block_iq4_nl *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_MXFP4:
            set_rows_sycl_q<TIn, TIdx, block_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (block_mxfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_NVFP4:
            set_rows_sycl_q<TIn, TIdx, block_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (block_nvfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q2_K, QK_K, quantize_row_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q3_K, QK_K, quantize_row_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q4_K, QK_K, quantize_row_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q5_K, QK_K, quantize_row_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q6_K, QK_K, quantize_row_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_xxs, QK_K, quantize_row_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_s, QK_K, quantize_row_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq4_xs, QK_K, quantize_row_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            ଭାଙ୍ଗ;
    }
}

ଶୂନ୍ୟ ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    ଯଦି (src0->type == GGML_TYPE_F16) {
        ଯଦି (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } ଅନ୍ୟଥା {
            set_rows_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<ଦଶମିକ, int64_t>(ctx, src0, src1, dst);
        } ଅନ୍ୟଥା {
            set_rows_sycl<ଦଶମିକ, int32_t>(ctx, src0, src1, dst);
        }
    }
}
