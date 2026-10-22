#অন্তর্ভাৱ "set_rows.hpp"
#অন্তর্ভাৱ "cpy.hpp"

#অন্তর্ভাৱ "ggml-quants.h"

#অন্তর্ভাৱ <vector>

নামাকাশ utils {
template<typename T>
স্থির constexpr তার্কিক is_arithmetic_v() {
    ঘূৰাই std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#ifdef GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#endif
        ;
}
}

template<typename TIn, typename TOut>
স্থির অন্তঃস্থ std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), শূন্য>
convert (ধ্রুৱ ৱর্ণ* src, ৱর্ণ* dst) {
    auto src_val = *reinterpret_cast<ধ্রুৱ TIn*>(src);
    auto dst_val = sycl::vec<TIn, 1>(src_val).template convert<TOut, sycl::rounding_mode::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#ifdef GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
template<>
অন্তঃস্থ শূন্য convert<sycl::half, sycl::ext::oneapi::bfloat16>(ধ্রুৱ ৱর্ণ* src, ৱর্ণ* dst) {
    ধ্রুৱ ভিন্ন tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ধ্রুৱ sycl::half*>(src))
                          .template convert<ভিন্ন, sycl::rounding_mode::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#endif

template <typename TIn, typename TIdx, typename blockType, পূর্ণাঙ্ক qk, cpy_kernel_t cpyblck>
স্থির শূন্য set_rows_sycl_q(ধ্রুৱ ৱর্ণ * __restrict__ src0_d,
                            ধ্রুৱ TIdx * __restrict__ src1_d,
                            blockType * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            ধ্রুৱ int64_t ne00,
                            ধ্রুৱ int64_t ne01,
                            ধ্রুৱ int64_t ne02,
                            ধ্রুৱ int64_t ne03,
                            ধ্রুৱ int64_t ne10,
                            ধ্রুৱ int64_t ne11,
                            ধ্রুৱ int64_t ne12,
                            ধ্রুৱ int64_t ne13,
                            // strides for src0
                            ধ্রুৱ size_t  nb00,
                            ধ্রুৱ size_t  nb01,
                            ধ্রুৱ size_t  nb02,
                            ধ্রুৱ size_t  nb03,
                            // strides for src1
                            ধ্রুৱ size_t  nb10,
                            ধ্রুৱ size_t  nb11,
                            ধ্রুৱ size_t  nb12,
                            ধ্রুৱ size_t  nb13,
                            // strides for dst
                            ধ্রুৱ size_t  nb1,
                            ধ্রুৱ size_t  nb2,
                            ধ্রুৱ size_t  nb3,
                            queue_ptr     stream) {
    ধ্রুৱ int64_t total_blocks = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr পূর্ণাঙ্ক block_size   = 256;
    ধ্রুৱ int64_t grid_size    = ceil_div(total_blocks, block_size);

    stream->parallel_for(sycl::nd_range<1>(grid_size * block_size, block_size), [=](sycl::nd_item<1> item_ct1) {
        ধ্রুৱ int64_t i = item_ct1.get_global_linear_id();
        যদি (i >= total_blocks) {
            ঘূৰাই;
        }
        ধ্রুৱ int64_t i_base      = i * qk;
        ধ্রুৱ int64_t i03         = i_base / (ne00 * ne01 * ne02);
        ধ্রুৱ int64_t rem1        = i_base - i03 * (ne00 * ne01 * ne02);
        ধ্রুৱ int64_t i02         = rem1 / (ne00 * ne01);
        ধ্রুৱ int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ধ্রুৱ int64_t i01         = rem2 / ne00;
        ধ্রুৱ int64_t i00         = rem2 - i01 * ne00;
        ধ্রুৱ int64_t i12         = i03 % ne12;
        ধ্রুৱ int64_t i11         = i02 % ne11;
        ধ্রুৱ int64_t i10         = i01;
        ধ্রুৱ size_t  src_offset  = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ধ্রুৱ ৱর্ণ *  src_block   = src0_d + src_offset + i00 * পরিমাণ(TIn);
        ধ্রুৱ size_t  src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ধ্রুৱ int64_t dst_row     = src1_d[src1_offset / পরিমাণ(TIdx)];
        ধ্রুৱ size_t  dst_offset =
            calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * পরিমাণ(blockType);
        ৱর্ণ * dst_block = reinterpret_cast<ৱর্ণ *>(reinterpret_cast<ৱর্ণ *>(dst_d) + dst_offset);
        যদি constexpr (std::is_same_v<TIn, ভিন্ন>) {
            cpyblck(src_block, dst_block);
        } অন্যথা {
            ভিন্ন src_block_f32[qk];
            ধ্রুৱ TIn * src_block_t = reinterpret_cast<ধ্রুৱ TIn *>(src_block);
            বাবে (পূর্ণাঙ্ক j = 0; j < qk; ++j) {
                src_block_f32[j] = (ভিন্ন) src_block_t[j];
            }
            cpyblck(reinterpret_cast<ধ্রুৱ ৱর্ণ *>(src_block_f32), dst_block);
        }
    });
    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(nb00);
    GGML_UNUSED(nb13);
}

template<typename blockType>
using quantize_row_qk_t = শূন্য (*)(ধ্রুৱ ভিন্ন *, blockType *, int64_t);

using quantize_rows_f_t = size_t (*)(ধ্রুৱ ভিন্ন *, শূন্য *, int64_t, int64_t, ধ্রুৱ ভিন্ন *);

template <typename TIn, typename TIdx, typename blockType, পূর্ণাঙ্ক qk, quantize_row_qk_t<blockType> quantize_row>
স্থির শূন্য set_rows_sycl_qk_host(
        ধ্রুৱ ggml_tensor * src0,
        ধ্রুৱ ggml_tensor * src1,
        ggml_tensor * dst,
        ধ্রুৱ int64_t ne00,
        ধ্রুৱ int64_t ne01,
        ধ্রুৱ int64_t ne02,
        ধ্রুৱ int64_t ne03,
        ধ্রুৱ int64_t ne11,
        ধ্রুৱ int64_t ne12,
        ধ্রুৱ size_t nb01,
        ধ্রুৱ size_t nb02,
        ধ্রুৱ size_t nb03,
        ধ্রুৱ size_t nb10,
        ধ্রুৱ size_t nb11,
        ধ্রুৱ size_t nb12,
        ধ্রুৱ size_t nb1,
        ধ্রুৱ size_t nb2,
        ধ্রুৱ size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ধ্রুৱ size_t src0_bytes = ggml_nbytes(src0);
    ধ্রুৱ size_t src1_bytes = ggml_nbytes(src1);

    std::vector<ৱর্ণ> src0_host(src0_bytes);
    std::vector<ৱর্ণ> src1_host(src1_bytes);

    stream->memcpy(src0_host.data(), src0->data, src0_bytes);
    stream->memcpy(src1_host.data(), src1->data, src1_bytes);
    stream->wait();

    std::vector<ভিন্ন> src_row_f32(ne00);
    ধ্রুৱ int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    বাবে (int64_t i03 = 0; i03 < ne03; ++i03) {
        বাবে (int64_t i02 = 0; i02 < ne02; ++i02) {
            বাবে (int64_t i01 = 0; i01 < ne01; ++i01) {
                ধ্রুৱ int64_t i12 = i03 % ne12;
                ধ্রুৱ int64_t i11 = i02 % ne11;
                ধ্রুৱ int64_t i10 = i01;

                ধ্রুৱ size_t src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ধ্রুৱ int64_t dst_row = *(ধ্রুৱ TIdx *) (src1_host.data() + src1_offset);

                ধ্রুৱ size_t src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ধ্রুৱ TIn * src_row = reinterpret_cast<ধ্রুৱ TIn *>(src0_host.data() + src0_row_offset);

                বাবে (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (ভিন্ন) src_row[i00];
                }

                quantize_row(src_row_f32.data(), dst_row_q.data(), ne00);

                ধ্রুৱ size_t dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->memcpy((ৱর্ণ *) dst->data + dst_offset, dst_row_q.data(), nblocks * পরিমাণ(blockType));
                stream->wait();
            }
        }
    }
}

template <typename TIn, typename TIdx, typename blockType, পূর্ণাঙ্ক qk, quantize_rows_f_t quantize_rows>
স্থির শূন্য set_rows_sycl_iq_host(
        ধ্রুৱ ggml_tensor * src0,
        ধ্রুৱ ggml_tensor * src1,
        ggml_tensor * dst,
        ধ্রুৱ int64_t ne00,
        ধ্রুৱ int64_t ne01,
        ধ্রুৱ int64_t ne02,
        ধ্রুৱ int64_t ne03,
        ধ্রুৱ int64_t ne11,
        ধ্রুৱ int64_t ne12,
        ধ্রুৱ size_t nb01,
        ধ্রুৱ size_t nb02,
        ধ্রুৱ size_t nb03,
        ধ্রুৱ size_t nb10,
        ধ্রুৱ size_t nb11,
        ধ্রুৱ size_t nb12,
        ধ্রুৱ size_t nb1,
        ধ্রুৱ size_t nb2,
        ধ্রুৱ size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ধ্রুৱ size_t src0_bytes = ggml_nbytes(src0);
    ধ্রুৱ size_t src1_bytes = ggml_nbytes(src1);

    std::vector<ৱর্ণ> src0_host(src0_bytes);
    std::vector<ৱর্ণ> src1_host(src1_bytes);

    stream->memcpy(src0_host.data(), src0->data, src0_bytes);
    stream->memcpy(src1_host.data(), src1->data, src1_bytes);
    stream->wait();

    std::vector<ভিন্ন> src_row_f32(ne00);
    ধ্রুৱ int64_t nblocks = ne00 / qk;
    std::vector<blockType> dst_row_q(nblocks);

    বাবে (int64_t i03 = 0; i03 < ne03; ++i03) {
        বাবে (int64_t i02 = 0; i02 < ne02; ++i02) {
            বাবে (int64_t i01 = 0; i01 < ne01; ++i01) {
                ধ্রুৱ int64_t i12 = i03 % ne12;
                ধ্রুৱ int64_t i11 = i02 % ne11;
                ধ্রুৱ int64_t i10 = i01;

                ধ্রুৱ size_t src1_offset = calculate_offset<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ধ্রুৱ int64_t dst_row = *(ধ্রুৱ TIdx *) (src1_host.data() + src1_offset);

                ধ্রুৱ size_t src0_row_offset = calculate_offset<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ধ্রুৱ TIn * src_row = reinterpret_cast<ধ্রুৱ TIn *>(src0_host.data() + src0_row_offset);

                বাবে (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (ভিন্ন) src_row[i00];
                }

                quantize_rows(src_row_f32.data(), dst_row_q.data(), 1, ne00, nullptr);

                ধ্রুৱ size_t dst_offset = calculate_offset<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->memcpy((ৱর্ণ *) dst->data + dst_offset, dst_row_q.data(), nblocks * পরিমাণ(blockType));
                stream->wait();
            }
        }
    }
}

template<typename TIn, typename TIdx, typename TOut>
স্থির শূন্য k_set_rows(
        ধ্রুৱ ৱর্ণ * __restrict__ src0, ধ্রুৱ TIdx * __restrict__ src1, ৱর্ণ * __restrict__ dst,
        ধ্রুৱ int64_t ne00, ধ্রুৱ int64_t ne01, ধ্রুৱ int64_t ne02,
        ধ্রুৱ int64_t ne11, ধ্রুৱ int64_t ne12,
        ধ্রুৱ size_t nb01, ধ্রুৱ size_t nb02, ধ্রুৱ size_t nb03,
        ধ্রুৱ size_t nb10, ধ্রুৱ size_t nb11, ধ্রুৱ size_t nb12,
        ধ্রুৱ size_t nb1, ধ্রুৱ size_t nb2, ধ্রুৱ size_t nb3,
        ধ্রুৱ size_t src_type_size, ধ্রুৱ size_t dst_type_size,
        ধ্রুৱ int64_t total_elements,
        ধ্রুৱ sycl::nd_item<1> & item_ct1) {

    ধ্রুৱ int64_t i = item_ct1.get_global_linear_id();
    যদি (i >= total_elements) {
        ঘূৰাই;
    }

    ধ্রুৱ int64_t i03 = i / (ne00 * ne01 * ne02);
    ধ্রুৱ int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ধ্রুৱ int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ধ্রুৱ int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ধ্রুৱ int64_t i12 = i03 % ne12;
    ধ্রুৱ int64_t i11 = i02 % ne11;
    ধ্রুৱ int64_t i10 = i01;

    ধ্রুৱ int64_t dst_row = *(ধ্রুৱ TIdx *)((ধ্রুৱ ৱর্ণ *)src1 + calculate_offset<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ধ্রুৱ ৱর্ণ * src0_row = src0 + calculate_offset<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ধ্রুৱ ৱর্ণ * src_elem = src0_row + i00 * src_type_size;
    ৱর্ণ * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    ৱর্ণ * dst_elem = dst_row_ptr + i00 * dst_type_size;

    convert<TIn, TOut>(src_elem, dst_elem);
}

template<typename TIn, typename TIdx, typename TOut>
স্থির শূন্য set_rows_sycl(
        ধ্রুৱ ৱর্ণ * src0_d, ধ্রুৱ TIdx * src1_d, ৱর্ণ * dst_d,
        ধ্রুৱ int64_t ne00, ধ্রুৱ int64_t ne01, ধ্রুৱ int64_t ne02, ধ্রুৱ int64_t ne03,
        ধ্রুৱ int64_t ne11, ধ্রুৱ int64_t ne12, ধ্রুৱ size_t nb01, ধ্রুৱ size_t nb02, ধ্রুৱ size_t nb03,
        ধ্রুৱ size_t nb10, ধ্রুৱ size_t nb11, ধ্রুৱ size_t nb12,
        ধ্রুৱ size_t nb1, ধ্রুৱ size_t nb2, ধ্রুৱ size_t nb3,
        ধ্রুৱ size_t src_type_size, ধ্রুৱ size_t dst_type_size,
        queue_ptr stream) {

    ধ্রুৱ int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    constexpr পূর্ণাঙ্ক block_size = 64;
    ধ্রুৱ int64_t grid_size = ceil_div(total_elements, block_size);

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
স্থির শূন্য set_rows_sycl(ggml_backend_sycl_context & ctx, ধ্রুৱ ggml_tensor * src0, ধ্রুৱ ggml_tensor * src1, ggml_tensor * dst) {
    ধ্রুৱ ৱর্ণ * src0_d = (ধ্রুৱ ৱর্ণ *)src0->data;
    ধ্রুৱ TIdx * src1_d = (ধ্রুৱ TIdx *)src1->data;

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    ৱিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F32:
            set_rows_sycl<TIn, TIdx, ভিন্ন>(
                src0_d, src1_d, (ৱর্ণ *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                পরিমাণ(TIn), পরিমাণ(ভিন্ন),
                stream
            );
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            set_rows_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (ৱর্ণ *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                পরিমাণ(TIn), পরিমাণ(sycl::half),
                stream
            );
            ভঙ্গ;
#ifdef GGML_SYCL_HAS_BF16
        পক্ষ GGML_TYPE_BF16:
            set_rows_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (ৱর্ণ *)dst->data,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                পরিমাণ(TIn), পরিমাণ(sycl::ext::oneapi::bfloat16),
                stream
            );
            ভঙ্গ;
#endif
        পক্ষ GGML_TYPE_Q8_0:
            set_rows_sycl_q<TIn, TIdx, block_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (block_q8_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q1_0:
            set_rows_sycl_q<TIn, TIdx, block_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (block_q1_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_0:
            set_rows_sycl_q<TIn, TIdx, block_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (block_q2_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_1:
            set_rows_sycl_q<TIn, TIdx, block_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (block_q5_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_0:
            set_rows_sycl_q<TIn, TIdx, block_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (block_q5_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_1:
            set_rows_sycl_q<TIn, TIdx, block_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (block_q4_1 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_0:
            set_rows_sycl_q<TIn, TIdx, block_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (block_q4_0 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ4_NL:
            set_rows_sycl_q<TIn, TIdx, block_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (block_iq4_nl *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_MXFP4:
            set_rows_sycl_q<TIn, TIdx, block_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (block_mxfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_NVFP4:
            set_rows_sycl_q<TIn, TIdx, block_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (block_nvfp4 *) dst->data, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q2_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q2_K, QK_K, quantize_row_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q3_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q3_K, QK_K, quantize_row_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q4_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q4_K, QK_K, quantize_row_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q5_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q5_K, QK_K, quantize_row_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_Q6_K:
            set_rows_sycl_qk_host<TIn, TIdx, block_q6_K, QK_K, quantize_row_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_XXS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_XS:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ2_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_XXS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_xxs, QK_K, quantize_row_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ3_S:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq3_s, QK_K, quantize_row_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ1_S:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ1_M:
            set_rows_sycl_iq_host<TIn, TIdx, block_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        পক্ষ GGML_TYPE_IQ4_XS:
            set_rows_sycl_qk_host<TIn, TIdx, block_iq4_xs, QK_K, quantize_row_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ভঙ্গ;
        স্ৱভাৱ:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            ভঙ্গ;
    }
}

শূন্য ggml_sycl_op_set_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    যদি (src0->type == GGML_TYPE_F16) {
        যদি (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } অন্যথা {
            set_rows_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } অন্যথা {
        যদি (src1->type == GGML_TYPE_I64) {
            set_rows_sycl<ভিন্ন, int64_t>(ctx, src0, src1, dst);
        } অন্যথা {
            set_rows_sycl<ভিন্ন, int32_t>(ctx, src0, src1, dst);
        }
    }
}
