#अन्तर्भाव "set_rows.hpp"
#अन्तर्भाव "cpy.hpp"

#अन्तर्भाव "ggml-quants.h"

#अन्तर्भाव <vector>

नामाकाश utils {
प्रतिरूपम्<प्रकारनाम T>
स्थिर constexpr तार्किक is_arithmetic_v() {
    प्रत्यागम std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#यदि_अन्तः
        ;
}
}

प्रतिरूपम्<प्रकारनाम TIn, प्रकारनाम TOut>
स्थिर अन्तःस्थ std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), शून्य>
convert (ध्रुव वर्ण* src, वर्ण* dst) {
    स्वचालित src_val = *reinterpret_cast<ध्रुव TIn*>(src);
    स्वचालित dst_val = sycl::vec<TIn, 1>(src_val).प्रतिरूपम् convert<TOut, sycl::rounding_बहुलकम्::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
प्रतिरूपम्<>
अन्तःस्थ शून्य convert<sycl::half, sycl::ext::oneapi::bfloat16>(ध्रुव वर्ण* src, वर्ण* dst) {
    ध्रुव भिन्न tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<ध्रुव sycl::half*>(src))
                          .प्रतिरूपम् convert<भिन्न, sycl::rounding_बहुलकम्::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#यदि_अन्तः

प्रतिरूपम् <प्रकारनाम TIn, प्रकारनाम TIdx, प्रकारनाम खण्डम्_Type, पूर्णाङ्क qk, cpy_kernel_t cpyblck>
स्थिर शून्य समुच्चयः_पङ्क्तिकाः_sycl_q(ध्रुव वर्ण * __restrict__ src0_d,
                            ध्रुव TIdx * __restrict__ src1_d,
                            खण्डम्_Type * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            ध्रुव int64_t ne00,
                            ध्रुव int64_t ne01,
                            ध्रुव int64_t ne02,
                            ध्रुव int64_t ne03,
                            ध्रुव int64_t ne10,
                            ध्रुव int64_t ne11,
                            ध्रुव int64_t ne12,
                            ध्रुव int64_t ne13,
                            // strides for src0
                            ध्रुव size_t  nb00,
                            ध्रुव size_t  nb01,
                            ध्रुव size_t  nb02,
                            ध्रुव size_t  nb03,
                            // strides for src1
                            ध्रुव size_t  nb10,
                            ध्रुव size_t  nb11,
                            ध्रुव size_t  nb12,
                            ध्रुव size_t  nb13,
                            // strides for dst
                            ध्रुव size_t  nb1,
                            ध्रुव size_t  nb2,
                            ध्रुव size_t  nb3,
                            queue_ptr     stream) {
    ध्रुव int64_t योग_खण्डानि = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr पूर्णाङ्क खण्डम्_आकार_3   = 256;
    ध्रुव int64_t grid_आकार    = ceil_div(योग_खण्डानि, खण्डम्_आकार_3);

    stream->parallel_for(sycl::nd_range<1>(grid_आकार * खण्डम्_आकार_3, खण्डम्_आकार_3), [=](sycl::nd_item<1> item_ct1) {
        ध्रुव int64_t i = item_ct1.get_global_रैखिकम्_id();
        यदि (i >= योग_खण्डानि) {
            प्रत्यागम;
        }
        ध्रुव int64_t i_आधारः      = i * qk;
        ध्रुव int64_t i03         = i_आधारः / (ne00 * ne01 * ne02);
        ध्रुव int64_t rem1        = i_आधारः - i03 * (ne00 * ne01 * ne02);
        ध्रुव int64_t i02         = rem1 / (ne00 * ne01);
        ध्रुव int64_t rem2        = rem1 - i02 * ne00 * ne01;
        ध्रुव int64_t i01         = rem2 / ne00;
        ध्रुव int64_t i00         = rem2 - i01 * ne00;
        ध्रुव int64_t i12         = i03 % ne12;
        ध्रुव int64_t i11         = i02 % ne11;
        ध्रुव int64_t i10         = i01;
        ध्रुव size_t  src_विस्थापनम्  = calculate_विस्थापनम्<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        ध्रुव वर्ण *  src_खण्डम्   = src0_d + src_विस्थापनम् + i00 * परिमाण(TIn);
        ध्रुव size_t  src1_विस्थापनम् = calculate_विस्थापनम्<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        ध्रुव int64_t dst_row     = src1_d[src1_विस्थापनम् / परिमाण(TIdx)];
        ध्रुव size_t  dst_विस्थापनम् =
            calculate_विस्थापनम्<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 }) + (i00 / qk) * परिमाण(खण्डम्_Type);
        वर्ण * dst_खण्डम् = reinterpret_cast<वर्ण *>(reinterpret_cast<वर्ण *>(dst_d) + dst_विस्थापनम्);
        यदि constexpr (std::is_same_v<TIn, भिन्न>) {
            cpyblck(src_खण्डम्, dst_खण्डम्);
        } अन्यथा {
            भिन्न src_खण्डम्_f32[qk];
            ध्रुव TIn * src_खण्डम्_t = reinterpret_cast<ध्रुव TIn *>(src_खण्डम्);
            प्रति (पूर्णाङ्क j = 0; j < qk; ++j) {
                src_खण्डम्_f32[j] = (भिन्न) src_खण्डम्_t[j];
            }
            cpyblck(reinterpret_cast<ध्रुव वर्ण *>(src_खण्डम्_f32), dst_खण्डम्);
        }
    });
    GGML_अप्रयुक्तम्(ne10);
    GGML_अप्रयुक्तम्(ne13);
    GGML_अप्रयुक्तम्(nb00);
    GGML_अप्रयुक्तम्(nb13);
}

प्रतिरूपम्<प्रकारनाम खण्डम्_Type>
उपयुज्य quantize_row_qk_t = शून्य (*)(ध्रुव भिन्न *, खण्डम्_Type *, int64_t);

उपयुज्य quantize_पङ्क्तिकाः_f_t = size_t (*)(ध्रुव भिन्न *, शून्य *, int64_t, int64_t, ध्रुव भिन्न *);

प्रतिरूपम् <प्रकारनाम TIn, प्रकारनाम TIdx, प्रकारनाम खण्डम्_Type, पूर्णाङ्क qk, quantize_row_qk_t<खण्डम्_Type> quantize_row>
स्थिर शून्य समुच्चयः_पङ्क्तिकाः_sycl_qk_host(
        ध्रुव ggml_tensor * src0,
        ध्रुव ggml_tensor * src1,
        ggml_tensor * dst,
        ध्रुव int64_t ne00,
        ध्रुव int64_t ne01,
        ध्रुव int64_t ne02,
        ध्रुव int64_t ne03,
        ध्रुव int64_t ne11,
        ध्रुव int64_t ne12,
        ध्रुव size_t nb01,
        ध्रुव size_t nb02,
        ध्रुव size_t nb03,
        ध्रुव size_t nb10,
        ध्रुव size_t nb11,
        ध्रुव size_t nb12,
        ध्रुव size_t nb1,
        ध्रुव size_t nb2,
        ध्रुव size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ध्रुव size_t src0_अष्टकानि = ggml_nbytes(src0);
    ध्रुव size_t src1_अष्टकानि = ggml_nbytes(src1);

    std::vector<वर्ण> src0_host(src0_अष्टकानि);
    std::vector<वर्ण> src1_host(src1_अष्टकानि);

    stream->स्मृतिं_प्रतिलिख(src0_host.दत्त(), src0->दत्त, src0_अष्टकानि);
    stream->स्मृतिं_प्रतिलिख(src1_host.दत्त(), src1->दत्त, src1_अष्टकानि);
    stream->प्रतीक्षस्व();

    std::vector<भिन्न> src_row_f32(ne00);
    ध्रुव int64_t nblocks = ne00 / qk;
    std::vector<खण्डम्_Type> dst_row_q(nblocks);

    प्रति (int64_t i03 = 0; i03 < ne03; ++i03) {
        प्रति (int64_t i02 = 0; i02 < ne02; ++i02) {
            प्रति (int64_t i01 = 0; i01 < ne01; ++i01) {
                ध्रुव int64_t i12 = i03 % ne12;
                ध्रुव int64_t i11 = i02 % ne11;
                ध्रुव int64_t i10 = i01;

                ध्रुव size_t src1_विस्थापनम् = calculate_विस्थापनम्<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ध्रुव int64_t dst_row = *(ध्रुव TIdx *) (src1_host.दत्त() + src1_विस्थापनम्);

                ध्रुव size_t src0_row_विस्थापनम् = calculate_विस्थापनम्<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ध्रुव TIn * src_row = reinterpret_cast<ध्रुव TIn *>(src0_host.दत्त() + src0_row_विस्थापनम्);

                प्रति (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (भिन्न) src_row[i00];
                }

                quantize_row(src_row_f32.दत्त(), dst_row_q.दत्त(), ne00);

                ध्रुव size_t dst_विस्थापनम् = calculate_विस्थापनम्<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->स्मृतिं_प्रतिलिख((वर्ण *) dst->दत्त + dst_विस्थापनम्, dst_row_q.दत्त(), nblocks * परिमाण(खण्डम्_Type));
                stream->प्रतीक्षस्व();
            }
        }
    }
}

प्रतिरूपम् <प्रकारनाम TIn, प्रकारनाम TIdx, प्रकारनाम खण्डम्_Type, पूर्णाङ्क qk, quantize_पङ्क्तिकाः_f_t quantize_पङ्क्तिकाः>
स्थिर शून्य समुच्चयः_पङ्क्तिकाः_sycl_iq_host(
        ध्रुव ggml_tensor * src0,
        ध्रुव ggml_tensor * src1,
        ggml_tensor * dst,
        ध्रुव int64_t ne00,
        ध्रुव int64_t ne01,
        ध्रुव int64_t ne02,
        ध्रुव int64_t ne03,
        ध्रुव int64_t ne11,
        ध्रुव int64_t ne12,
        ध्रुव size_t nb01,
        ध्रुव size_t nb02,
        ध्रुव size_t nb03,
        ध्रुव size_t nb10,
        ध्रुव size_t nb11,
        ध्रुव size_t nb12,
        ध्रुव size_t nb1,
        ध्रुव size_t nb2,
        ध्रुव size_t nb3,
        queue_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    ध्रुव size_t src0_अष्टकानि = ggml_nbytes(src0);
    ध्रुव size_t src1_अष्टकानि = ggml_nbytes(src1);

    std::vector<वर्ण> src0_host(src0_अष्टकानि);
    std::vector<वर्ण> src1_host(src1_अष्टकानि);

    stream->स्मृतिं_प्रतिलिख(src0_host.दत्त(), src0->दत्त, src0_अष्टकानि);
    stream->स्मृतिं_प्रतिलिख(src1_host.दत्त(), src1->दत्त, src1_अष्टकानि);
    stream->प्रतीक्षस्व();

    std::vector<भिन्न> src_row_f32(ne00);
    ध्रुव int64_t nblocks = ne00 / qk;
    std::vector<खण्डम्_Type> dst_row_q(nblocks);

    प्रति (int64_t i03 = 0; i03 < ne03; ++i03) {
        प्रति (int64_t i02 = 0; i02 < ne02; ++i02) {
            प्रति (int64_t i01 = 0; i01 < ne01; ++i01) {
                ध्रुव int64_t i12 = i03 % ne12;
                ध्रुव int64_t i11 = i02 % ne11;
                ध्रुव int64_t i10 = i01;

                ध्रुव size_t src1_विस्थापनम् = calculate_विस्थापनम्<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                ध्रुव int64_t dst_row = *(ध्रुव TIdx *) (src1_host.दत्त() + src1_विस्थापनम्);

                ध्रुव size_t src0_row_विस्थापनम् = calculate_विस्थापनम्<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                ध्रुव TIn * src_row = reinterpret_cast<ध्रुव TIn *>(src0_host.दत्त() + src0_row_विस्थापनम्);

                प्रति (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_row_f32[i00] = (भिन्न) src_row[i00];
                }

                quantize_पङ्क्तिकाः(src_row_f32.दत्त(), dst_row_q.दत्त(), 1, ne00, शून्यसूचकः);

                ध्रुव size_t dst_विस्थापनम् = calculate_विस्थापनम्<3>({ nb1, nb2, nb3 }, { dst_row, i02, i03 });
                stream->स्मृतिं_प्रतिलिख((वर्ण *) dst->दत्त + dst_विस्थापनम्, dst_row_q.दत्त(), nblocks * परिमाण(खण्डम्_Type));
                stream->प्रतीक्षस्व();
            }
        }
    }
}

प्रतिरूपम्<प्रकारनाम TIn, प्रकारनाम TIdx, प्रकारनाम TOut>
स्थिर शून्य k_समुच्चयः_पङ्क्तिकाः(
        ध्रुव वर्ण * __restrict__ src0, ध्रुव TIdx * __restrict__ src1, वर्ण * __restrict__ dst,
        ध्रुव int64_t ne00, ध्रुव int64_t ne01, ध्रुव int64_t ne02,
        ध्रुव int64_t ne11, ध्रुव int64_t ne12,
        ध्रुव size_t nb01, ध्रुव size_t nb02, ध्रुव size_t nb03,
        ध्रुव size_t nb10, ध्रुव size_t nb11, ध्रुव size_t nb12,
        ध्रुव size_t nb1, ध्रुव size_t nb2, ध्रुव size_t nb3,
        ध्रुव size_t src_type_आकार, ध्रुव size_t dst_type_आकार,
        ध्रुव int64_t योग_elements,
        ध्रुव sycl::nd_item<1> & item_ct1) {

    ध्रुव int64_t i = item_ct1.get_global_रैखिकम्_id();
    यदि (i >= योग_elements) {
        प्रत्यागम;
    }

    ध्रुव int64_t i03 = i / (ne00 * ne01 * ne02);
    ध्रुव int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    ध्रुव int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    ध्रुव int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    ध्रुव int64_t i12 = i03 % ne12;
    ध्रुव int64_t i11 = i02 % ne11;
    ध्रुव int64_t i10 = i01;

    ध्रुव int64_t dst_row = *(ध्रुव TIdx *)((ध्रुव वर्ण *)src1 + calculate_विस्थापनम्<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    ध्रुव वर्ण * src0_row = src0 + calculate_विस्थापनम्<3>({nb01, nb02, nb03}, {i01, i02, i03});
    ध्रुव वर्ण * src_elem = src0_row + i00 * src_type_आकार;
    वर्ण * dst_row_ptr = dst + dst_row*nb1 + i02*nb2 + i03*nb3;
    वर्ण * dst_elem = dst_row_ptr + i00 * dst_type_आकार;

    convert<TIn, TOut>(src_elem, dst_elem);
}

प्रतिरूपम्<प्रकारनाम TIn, प्रकारनाम TIdx, प्रकारनाम TOut>
स्थिर शून्य समुच्चयः_पङ्क्तिकाः_sycl(
        ध्रुव वर्ण * src0_d, ध्रुव TIdx * src1_d, वर्ण * dst_d,
        ध्रुव int64_t ne00, ध्रुव int64_t ne01, ध्रुव int64_t ne02, ध्रुव int64_t ne03,
        ध्रुव int64_t ne11, ध्रुव int64_t ne12, ध्रुव size_t nb01, ध्रुव size_t nb02, ध्रुव size_t nb03,
        ध्रुव size_t nb10, ध्रुव size_t nb11, ध्रुव size_t nb12,
        ध्रुव size_t nb1, ध्रुव size_t nb2, ध्रुव size_t nb3,
        ध्रुव size_t src_type_आकार, ध्रुव size_t dst_type_आकार,
        queue_ptr stream) {

    ध्रुव int64_t योग_elements = ne00 * ne01 * ne02 * ne03;

    constexpr पूर्णाङ्क खण्डम्_आकार_3 = 64;
    ध्रुव int64_t grid_आकार = ceil_div(योग_elements, खण्डम्_आकार_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_आकार * खण्डम्_आकार_3, खण्डम्_आकार_3),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            k_समुच्चयः_पङ्क्तिकाः<TIn, TIdx, TOut>(
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

प्रतिरूपम्<प्रकारनाम TIn, प्रकारनाम TIdx>
स्थिर शून्य समुच्चयः_पङ्क्तिकाः_sycl(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    ध्रुव वर्ण * src0_d = (ध्रुव वर्ण *)src0->दत्त;
    ध्रुव TIdx * src1_d = (ध्रुव TIdx *)src1->दत्त;

    GGML_TENSOR_द्विमान_OP_LOCALS

    dpct::queue_ptr stream = ctx.stream();
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F32:
            समुच्चयः_पङ्क्तिकाः_sycl<TIn, TIdx, भिन्न>(
                src0_d, src1_d, (वर्ण *)dst->दत्त,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                परिमाण(TIn), परिमाण(भिन्न),
                stream
            );
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            समुच्चयः_पङ्क्तिकाः_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (वर्ण *)dst->दत्त,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                परिमाण(TIn), परिमाण(sycl::half),
                stream
            );
            भङ्ग;
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
        पक्ष GGML_TYPE_BF16:
            समुच्चयः_पङ्क्तिकाः_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (वर्ण *)dst->दत्त,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                परिमाण(TIn), परिमाण(sycl::ext::oneapi::bfloat16),
                stream
            );
            भङ्ग;
#यदि_अन्तः
        पक्ष GGML_TYPE_Q8_0:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (खण्डम्_q8_0 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q1_0:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (खण्डम्_q1_0 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q2_0:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (खण्डम्_q2_0 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q5_1:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (खण्डम्_q5_1 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q5_0:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (खण्डम्_q5_0 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q4_1:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (खण्डम्_q4_1 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q4_0:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (खण्डम्_q4_0 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ4_NL:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (खण्डम्_iq4_nl *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_MXFP4:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (खण्डम्_mxfp4 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_NVFP4:
            समुच्चयः_पङ्क्तिकाः_sycl_q<TIn, TIdx, खण्डम्_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (खण्डम्_nvfp4 *) dst->दत्त, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q2_K:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_q2_K, QK_K, quantize_row_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q3_K:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_q3_K, QK_K, quantize_row_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q4_K:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_q4_K, QK_K, quantize_row_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q5_K:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_q5_K, QK_K, quantize_row_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_Q6_K:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_q6_K, QK_K, quantize_row_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ2_XXS:
            समुच्चयः_पङ्क्तिकाः_sycl_iq_host<TIn, TIdx, खण्डम्_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ2_XS:
            समुच्चयः_पङ्क्तिकाः_sycl_iq_host<TIn, TIdx, खण्डम्_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ2_S:
            समुच्चयः_पङ्क्तिकाः_sycl_iq_host<TIn, TIdx, खण्डम्_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ3_XXS:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_iq3_xxs, QK_K, quantize_row_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ3_S:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_iq3_s, QK_K, quantize_row_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ1_S:
            समुच्चयः_पङ्क्तिकाः_sycl_iq_host<TIn, TIdx, खण्डम्_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ1_M:
            समुच्चयः_पङ्क्तिकाः_sycl_iq_host<TIn, TIdx, खण्डम्_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        पक्ष GGML_TYPE_IQ4_XS:
            समुच्चयः_पङ्क्तिकाः_sycl_qk_host<TIn, TIdx, खण्डम्_iq4_xs, QK_K, quantize_row_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            भङ्ग;
        स्वभाव:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            भङ्ग;
    }
}

शून्य ggml_sycl_op_समुच्चयः_पङ्क्तिकाः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    यदि (src0->type == GGML_TYPE_F16) {
        यदि (src1->type == GGML_TYPE_I64) {
            समुच्चयः_पङ्क्तिकाः_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } अन्यथा {
            समुच्चयः_पङ्क्तिकाः_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } अन्यथा {
        यदि (src1->type == GGML_TYPE_I64) {
            समुच्चयः_पङ्क्तिकाः_sycl<भिन्न, int64_t>(ctx, src0, src1, dst);
        } अन्यथा {
            समुच्चयः_पङ्क्तिकाः_sycl<भिन्न, int32_t>(ctx, src0, src1, dst);
        }
    }
}
