#అన్తర్భావ "set_rows.hpp"
#అన్తర్భావ "cpy.hpp"

#అన్తర్భావ "ggml-quants.h"

#అన్తర్భావ <vector>

నామాకాశ utils {
మూస<రకం_పేరు T>
స్థిర constexpr తార్కిక is_arithmetic_v() {
    తిరిగి std::is_arithmetic_v<T> || std::is_same_v<T, sycl::half>
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        || std::is_same_v<T, sycl::ext::oneapi::bfloat16>
#అయితే_ముగింపు
        ;
}
}

మూస<రకం_పేరు TIn, రకం_పేరు TOut>
స్థిర అంతఃవరుస std::enable_if_t<utils::is_arithmetic_v<TIn>() && utils::is_arithmetic_v<TOut>(), శూన్య>
convert (స్థిరరాశి అక్షరం* src, అక్షరం* dst) {
    స్వయంచాలక src_val = *reinterpret_cast<స్థిరరాశి TIn*>(src);
    స్వయంచాలక dst_val = sycl::vec<TIn, 1>(src_val).మూస convert<TOut, sycl::rounding_బహుళకం::automatic>()[0];
   *reinterpret_cast<TOut*>(dst) = dst_val;
}

#నిర్వచితమైతే GGML_SYCL_HAS_BF16
// sycl::vec::convert does not provide a half -> bfloat16 path, so route through float.
మూస<>
అంతఃవరుస శూన్య convert<sycl::half, sycl::ext::oneapi::bfloat16>(స్థిరరాశి అక్షరం* src, అక్షరం* dst) {
    స్థిరరాశి దశాంశం tmp = sycl::vec<sycl::half, 1>(*reinterpret_cast<స్థిరరాశి sycl::half*>(src))
                          .మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];
    *reinterpret_cast<sycl::ext::oneapi::bfloat16*>(dst) = sycl::ext::oneapi::bfloat16(tmp);
}
#అయితే_ముగింపు

మూస <రకం_పేరు TIn, రకం_పేరు TIdx, రకం_పేరు ఖండం_Type, పూర్ణాంకం qk, cpy_kernel_t cpyblck>
స్థిర శూన్య సమితి_అడ్డువరుసలు_sycl_q(స్థిరరాశి అక్షరం * __restrict__ src0_d,
                            స్థిరరాశి TIdx * __restrict__ src1_d,
                            ఖండం_Type * __restrict__ dst_d,
                            // tensor dimensions src0 and src1
                            స్థిరరాశి int64_t ne00,
                            స్థిరరాశి int64_t ne01,
                            స్థిరరాశి int64_t ne02,
                            స్థిరరాశి int64_t ne03,
                            స్థిరరాశి int64_t ne10,
                            స్థిరరాశి int64_t ne11,
                            స్థిరరాశి int64_t ne12,
                            స్థిరరాశి int64_t ne13,
                            // strides for src0
                            స్థిరరాశి size_t  nb00,
                            స్థిరరాశి size_t  nb01,
                            స్థిరరాశి size_t  nb02,
                            స్థిరరాశి size_t  nb03,
                            // strides for src1
                            స్థిరరాశి size_t  nb10,
                            స్థిరరాశి size_t  nb11,
                            స్థిరరాశి size_t  nb12,
                            స్థిరరాశి size_t  nb13,
                            // strides for dst
                            స్థిరరాశి size_t  nb1,
                            స్థిరరాశి size_t  nb2,
                            స్థిరరాశి size_t  nb3,
                            వరుస_ptr     stream) {
    స్థిరరాశి int64_t మొత్తం_ఖండాలు = (ne00 * ne01 * ne02 * ne03) / qk;
    constexpr పూర్ణాంకం ఖండం_కొలత_3   = 256;
    స్థిరరాశి int64_t grid_కొలత    = ceil_div(మొత్తం_ఖండాలు, ఖండం_కొలత_3);

    stream->parallel_for(sycl::nd_range<1>(grid_కొలత * ఖండం_కొలత_3, ఖండం_కొలత_3), [=](sycl::nd_అంశం<1> అంశం_ct1) {
        స్థిరరాశి int64_t i = అంశం_ct1.get_global_రేఖీయ_id();
        ఒకవేళ (i >= మొత్తం_ఖండాలు) {
            తిరిగి;
        }
        స్థిరరాశి int64_t i_పునాది      = i * qk;
        స్థిరరాశి int64_t i03         = i_పునాది / (ne00 * ne01 * ne02);
        స్థిరరాశి int64_t rem1        = i_పునాది - i03 * (ne00 * ne01 * ne02);
        స్థిరరాశి int64_t i02         = rem1 / (ne00 * ne01);
        స్థిరరాశి int64_t rem2        = rem1 - i02 * ne00 * ne01;
        స్థిరరాశి int64_t i01         = rem2 / ne00;
        స్థిరరాశి int64_t i00         = rem2 - i01 * ne00;
        స్థిరరాశి int64_t i12         = i03 % ne12;
        స్థిరరాశి int64_t i11         = i02 % ne11;
        స్థిరరాశి int64_t i10         = i01;
        స్థిరరాశి size_t  src_స్థానభ్రంశం  = calculate_స్థానభ్రంశం<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
        స్థిరరాశి అక్షరం *  src_ఖండం   = src0_d + src_స్థానభ్రంశం + i00 * పరిమాణం(TIn);
        స్థిరరాశి size_t  src1_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
        స్థిరరాశి int64_t dst_అడ్డువరుస     = src1_d[src1_స్థానభ్రంశం / పరిమాణం(TIdx)];
        స్థిరరాశి size_t  dst_స్థానభ్రంశం =
            calculate_స్థానభ్రంశం<3>({ nb1, nb2, nb3 }, { dst_అడ్డువరుస, i02, i03 }) + (i00 / qk) * పరిమాణం(ఖండం_Type);
        అక్షరం * dst_ఖండం = reinterpret_cast<అక్షరం *>(reinterpret_cast<అక్షరం *>(dst_d) + dst_స్థానభ్రంశం);
        ఒకవేళ constexpr (std::is_same_v<TIn, దశాంశం>) {
            cpyblck(src_ఖండం, dst_ఖండం);
        } లేకపోతే {
            దశాంశం src_ఖండం_f32[qk];
            స్థిరరాశి TIn * src_ఖండం_t = reinterpret_cast<స్థిరరాశి TIn *>(src_ఖండం);
            కొరకు (పూర్ణాంకం j = 0; j < qk; ++j) {
                src_ఖండం_f32[j] = (దశాంశం) src_ఖండం_t[j];
            }
            cpyblck(reinterpret_cast<స్థిరరాశి అక్షరం *>(src_ఖండం_f32), dst_ఖండం);
        }
    });
    GGML_వాడని(ne10);
    GGML_వాడని(ne13);
    GGML_వాడని(nb00);
    GGML_వాడని(nb13);
}

మూస<రకం_పేరు ఖండం_Type>
ఉపయోగించి quantize_అడ్డువరుస_qk_t = శూన్య (*)(స్థిరరాశి దశాంశం *, ఖండం_Type *, int64_t);

ఉపయోగించి quantize_అడ్డువరుసలు_f_t = size_t (*)(స్థిరరాశి దశాంశం *, శూన్య *, int64_t, int64_t, స్థిరరాశి దశాంశం *);

మూస <రకం_పేరు TIn, రకం_పేరు TIdx, రకం_పేరు ఖండం_Type, పూర్ణాంకం qk, quantize_అడ్డువరుస_qk_t<ఖండం_Type> quantize_అడ్డువరుస>
స్థిర శూన్య సమితి_అడ్డువరుసలు_sycl_qk_host(
        స్థిరరాశి ggml_tensor * src0,
        స్థిరరాశి ggml_tensor * src1,
        ggml_tensor * dst,
        స్థిరరాశి int64_t ne00,
        స్థిరరాశి int64_t ne01,
        స్థిరరాశి int64_t ne02,
        స్థిరరాశి int64_t ne03,
        స్థిరరాశి int64_t ne11,
        స్థిరరాశి int64_t ne12,
        స్థిరరాశి size_t nb01,
        స్థిరరాశి size_t nb02,
        స్థిరరాశి size_t nb03,
        స్థిరరాశి size_t nb10,
        స్థిరరాశి size_t nb11,
        స్థిరరాశి size_t nb12,
        స్థిరరాశి size_t nb1,
        స్థిరరాశి size_t nb2,
        స్థిరరాశి size_t nb3,
        వరుస_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    స్థిరరాశి size_t src0_బైట్లు = ggml_nbytes(src0);
    స్థిరరాశి size_t src1_బైట్లు = ggml_nbytes(src1);

    std::vector<అక్షరం> src0_host(src0_బైట్లు);
    std::vector<అక్షరం> src1_host(src1_బైట్లు);

    stream->జ్ఞాపకం_నకలు(src0_host.దత్తాంశం(), src0->దత్తాంశం, src0_బైట్లు);
    stream->జ్ఞాపకం_నకలు(src1_host.దత్తాంశం(), src1->దత్తాంశం, src1_బైట్లు);
    stream->wait();

    std::vector<దశాంశం> src_అడ్డువరుస_f32(ne00);
    స్థిరరాశి int64_t nblocks = ne00 / qk;
    std::vector<ఖండం_Type> dst_అడ్డువరుస_q(nblocks);

    కొరకు (int64_t i03 = 0; i03 < ne03; ++i03) {
        కొరకు (int64_t i02 = 0; i02 < ne02; ++i02) {
            కొరకు (int64_t i01 = 0; i01 < ne01; ++i01) {
                స్థిరరాశి int64_t i12 = i03 % ne12;
                స్థిరరాశి int64_t i11 = i02 % ne11;
                స్థిరరాశి int64_t i10 = i01;

                స్థిరరాశి size_t src1_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                స్థిరరాశి int64_t dst_అడ్డువరుస = *(స్థిరరాశి TIdx *) (src1_host.దత్తాంశం() + src1_స్థానభ్రంశం);

                స్థిరరాశి size_t src0_అడ్డువరుస_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                స్థిరరాశి TIn * src_అడ్డువరుస = reinterpret_cast<స్థిరరాశి TIn *>(src0_host.దత్తాంశం() + src0_అడ్డువరుస_స్థానభ్రంశం);

                కొరకు (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_అడ్డువరుస_f32[i00] = (దశాంశం) src_అడ్డువరుస[i00];
                }

                quantize_అడ్డువరుస(src_అడ్డువరుస_f32.దత్తాంశం(), dst_అడ్డువరుస_q.దత్తాంశం(), ne00);

                స్థిరరాశి size_t dst_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb1, nb2, nb3 }, { dst_అడ్డువరుస, i02, i03 });
                stream->జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం + dst_స్థానభ్రంశం, dst_అడ్డువరుస_q.దత్తాంశం(), nblocks * పరిమాణం(ఖండం_Type));
                stream->wait();
            }
        }
    }
}

మూస <రకం_పేరు TIn, రకం_పేరు TIdx, రకం_పేరు ఖండం_Type, పూర్ణాంకం qk, quantize_అడ్డువరుసలు_f_t quantize_అడ్డువరుసలు>
స్థిర శూన్య సమితి_అడ్డువరుసలు_sycl_iq_host(
        స్థిరరాశి ggml_tensor * src0,
        స్థిరరాశి ggml_tensor * src1,
        ggml_tensor * dst,
        స్థిరరాశి int64_t ne00,
        స్థిరరాశి int64_t ne01,
        స్థిరరాశి int64_t ne02,
        స్థిరరాశి int64_t ne03,
        స్థిరరాశి int64_t ne11,
        స్థిరరాశి int64_t ne12,
        స్థిరరాశి size_t nb01,
        స్థిరరాశి size_t nb02,
        స్థిరరాశి size_t nb03,
        స్థిరరాశి size_t nb10,
        స్థిరరాశి size_t nb11,
        స్థిరరాశి size_t nb12,
        స్థిరరాశి size_t nb1,
        స్థిరరాశి size_t nb2,
        స్థిరరాశి size_t nb3,
        వరుస_ptr stream) {
    GGML_ASSERT(ne00 % qk == 0);

    స్థిరరాశి size_t src0_బైట్లు = ggml_nbytes(src0);
    స్థిరరాశి size_t src1_బైట్లు = ggml_nbytes(src1);

    std::vector<అక్షరం> src0_host(src0_బైట్లు);
    std::vector<అక్షరం> src1_host(src1_బైట్లు);

    stream->జ్ఞాపకం_నకలు(src0_host.దత్తాంశం(), src0->దత్తాంశం, src0_బైట్లు);
    stream->జ్ఞాపకం_నకలు(src1_host.దత్తాంశం(), src1->దత్తాంశం, src1_బైట్లు);
    stream->wait();

    std::vector<దశాంశం> src_అడ్డువరుస_f32(ne00);
    స్థిరరాశి int64_t nblocks = ne00 / qk;
    std::vector<ఖండం_Type> dst_అడ్డువరుస_q(nblocks);

    కొరకు (int64_t i03 = 0; i03 < ne03; ++i03) {
        కొరకు (int64_t i02 = 0; i02 < ne02; ++i02) {
            కొరకు (int64_t i01 = 0; i01 < ne01; ++i01) {
                స్థిరరాశి int64_t i12 = i03 % ne12;
                స్థిరరాశి int64_t i11 = i02 % ne11;
                స్థిరరాశి int64_t i10 = i01;

                స్థిరరాశి size_t src1_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb10, nb11, nb12 }, { i10, i11, i12 });
                స్థిరరాశి int64_t dst_అడ్డువరుస = *(స్థిరరాశి TIdx *) (src1_host.దత్తాంశం() + src1_స్థానభ్రంశం);

                స్థిరరాశి size_t src0_అడ్డువరుస_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb01, nb02, nb03 }, { i01, i02, i03 });
                స్థిరరాశి TIn * src_అడ్డువరుస = reinterpret_cast<స్థిరరాశి TIn *>(src0_host.దత్తాంశం() + src0_అడ్డువరుస_స్థానభ్రంశం);

                కొరకు (int64_t i00 = 0; i00 < ne00; ++i00) {
                    src_అడ్డువరుస_f32[i00] = (దశాంశం) src_అడ్డువరుస[i00];
                }

                quantize_అడ్డువరుసలు(src_అడ్డువరుస_f32.దత్తాంశం(), dst_అడ్డువరుస_q.దత్తాంశం(), 1, ne00, శూన్యసూచిక);

                స్థిరరాశి size_t dst_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({ nb1, nb2, nb3 }, { dst_అడ్డువరుస, i02, i03 });
                stream->జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం + dst_స్థానభ్రంశం, dst_అడ్డువరుస_q.దత్తాంశం(), nblocks * పరిమాణం(ఖండం_Type));
                stream->wait();
            }
        }
    }
}

మూస<రకం_పేరు TIn, రకం_పేరు TIdx, రకం_పేరు TOut>
స్థిర శూన్య k_సమితి_అడ్డువరుసలు(
        స్థిరరాశి అక్షరం * __restrict__ src0, స్థిరరాశి TIdx * __restrict__ src1, అక్షరం * __restrict__ dst,
        స్థిరరాశి int64_t ne00, స్థిరరాశి int64_t ne01, స్థిరరాశి int64_t ne02,
        స్థిరరాశి int64_t ne11, స్థిరరాశి int64_t ne12,
        స్థిరరాశి size_t nb01, స్థిరరాశి size_t nb02, స్థిరరాశి size_t nb03,
        స్థిరరాశి size_t nb10, స్థిరరాశి size_t nb11, స్థిరరాశి size_t nb12,
        స్థిరరాశి size_t nb1, స్థిరరాశి size_t nb2, స్థిరరాశి size_t nb3,
        స్థిరరాశి size_t src_type_కొలత, స్థిరరాశి size_t dst_type_కొలత,
        స్థిరరాశి int64_t మొత్తం_elements,
        స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_global_రేఖీయ_id();
    ఒకవేళ (i >= మొత్తం_elements) {
        తిరిగి;
    }

    స్థిరరాశి int64_t i03 = i / (ne00 * ne01 * ne02);
    స్థిరరాశి int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    స్థిరరాశి int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
    స్థిరరాశి int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;

    స్థిరరాశి int64_t i12 = i03 % ne12;
    స్థిరరాశి int64_t i11 = i02 % ne11;
    స్థిరరాశి int64_t i10 = i01;

    స్థిరరాశి int64_t dst_అడ్డువరుస = *(స్థిరరాశి TIdx *)((స్థిరరాశి అక్షరం *)src1 + calculate_స్థానభ్రంశం<3>({nb10, nb11, nb12}, {i10, i11, i12}));

    స్థిరరాశి అక్షరం * src0_అడ్డువరుస = src0 + calculate_స్థానభ్రంశం<3>({nb01, nb02, nb03}, {i01, i02, i03});
    స్థిరరాశి అక్షరం * src_elem = src0_అడ్డువరుస + i00 * src_type_కొలత;
    అక్షరం * dst_అడ్డువరుస_ptr = dst + dst_అడ్డువరుస*nb1 + i02*nb2 + i03*nb3;
    అక్షరం * dst_elem = dst_అడ్డువరుస_ptr + i00 * dst_type_కొలత;

    convert<TIn, TOut>(src_elem, dst_elem);
}

మూస<రకం_పేరు TIn, రకం_పేరు TIdx, రకం_పేరు TOut>
స్థిర శూన్య సమితి_అడ్డువరుసలు_sycl(
        స్థిరరాశి అక్షరం * src0_d, స్థిరరాశి TIdx * src1_d, అక్షరం * dst_d,
        స్థిరరాశి int64_t ne00, స్థిరరాశి int64_t ne01, స్థిరరాశి int64_t ne02, స్థిరరాశి int64_t ne03,
        స్థిరరాశి int64_t ne11, స్థిరరాశి int64_t ne12, స్థిరరాశి size_t nb01, స్థిరరాశి size_t nb02, స్థిరరాశి size_t nb03,
        స్థిరరాశి size_t nb10, స్థిరరాశి size_t nb11, స్థిరరాశి size_t nb12,
        స్థిరరాశి size_t nb1, స్థిరరాశి size_t nb2, స్థిరరాశి size_t nb3,
        స్థిరరాశి size_t src_type_కొలత, స్థిరరాశి size_t dst_type_కొలత,
        వరుస_ptr stream) {

    స్థిరరాశి int64_t మొత్తం_elements = ne00 * ne01 * ne02 * ne03;

    constexpr పూర్ణాంకం ఖండం_కొలత_3 = 64;
    స్థిరరాశి int64_t grid_కొలత = ceil_div(మొత్తం_elements, ఖండం_కొలత_3);

    stream->parallel_for(
        sycl::nd_range<1>(grid_కొలత * ఖండం_కొలత_3, ఖండం_కొలత_3),
        [=](sycl::nd_అంశం<1> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            k_సమితి_అడ్డువరుసలు<TIn, TIdx, TOut>(
                src0_d, src1_d, dst_d,
                ne00, ne01, ne02,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                src_type_కొలత, dst_type_కొలత,
                మొత్తం_elements,
                అంశం_ct1
            );
        }
    );
}

మూస<రకం_పేరు TIn, రకం_పేరు TIdx>
స్థిర శూన్య సమితి_అడ్డువరుసలు_sycl(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    స్థిరరాశి అక్షరం * src0_d = (స్థిరరాశి అక్షరం *)src0->దత్తాంశం;
    స్థిరరాశి TIdx * src1_d = (స్థిరరాశి TIdx *)src1->దత్తాంశం;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    dpct::వరుస_ptr stream = ctx.stream();
    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F32:
            సమితి_అడ్డువరుసలు_sycl<TIn, TIdx, దశాంశం>(
                src0_d, src1_d, (అక్షరం *)dst->దత్తాంశం,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                పరిమాణం(TIn), పరిమాణం(దశాంశం),
                stream
            );
            ఆపు;
        స్థితి GGML_TYPE_F16:
            dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
            సమితి_అడ్డువరుసలు_sycl<TIn, TIdx, sycl::half>(
                src0_d, src1_d, (అక్షరం *)dst->దత్తాంశం,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                పరిమాణం(TIn), పరిమాణం(sycl::half),
                stream
            );
            ఆపు;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        స్థితి GGML_TYPE_BF16:
            సమితి_అడ్డువరుసలు_sycl<TIn, TIdx, sycl::ext::oneapi::bfloat16>(
                src0_d, src1_d, (అక్షరం *)dst->దత్తాంశం,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                పరిమాణం(TIn), పరిమాణం(sycl::ext::oneapi::bfloat16),
                stream
            );
            ఆపు;
#అయితే_ముగింపు
        స్థితి GGML_TYPE_Q8_0:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q8_0, QK8_0, cpy_blck_f32_q8_0>(
                src0_d, src1_d, (ఖండం_q8_0 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q1_0:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q1_0, QK1_0, cpy_blck_f32_q1_0>(
                src0_d, src1_d, (ఖండం_q1_0 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q2_0:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q2_0, QK2_0, cpy_blck_f32_q2_0>(
                src0_d, src1_d, (ఖండం_q2_0 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q5_1, QK5_1, cpy_blck_f32_q5_1>(
                src0_d, src1_d, (ఖండం_q5_1 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q5_0, QK5_0, cpy_blck_f32_q5_0>(
                src0_d, src1_d, (ఖండం_q5_0 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q4_1, QK4_1, cpy_blck_f32_q4_1>(
                src0_d, src1_d, (ఖండం_q4_1 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_q4_0, QK4_0, cpy_blck_f32_q4_0>(
                src0_d, src1_d, (ఖండం_q4_0 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ4_NL:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_iq4_nl, QK4_NL, cpy_blck_f32_iq4_nl>(
                src0_d, src1_d, (ఖండం_iq4_nl *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_mxfp4, QK_MXFP4, cpy_blck_f32_mxfp4>(
                src0_d, src1_d, (ఖండం_mxfp4 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_NVFP4:
            సమితి_అడ్డువరుసలు_sycl_q<TIn, TIdx, ఖండం_nvfp4, QK_NVFP4, cpy_blck_f32_nvfp4>(
                src0_d, src1_d, (ఖండం_nvfp4 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
                ne10, ne11, ne12, ne13, nb00, nb01,
                nb02, nb03, nb10, nb11, nb12, nb13, nb1, nb2, nb3, stream);
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_q2_K, QK_K, quantize_అడ్డువరుస_q2_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_q3_K, QK_K, quantize_అడ్డువరుస_q3_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_q4_K, QK_K, quantize_అడ్డువరుస_q4_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_q5_K, QK_K, quantize_అడ్డువరుస_q5_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_q6_K, QK_K, quantize_అడ్డువరుస_q6_K_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ2_XXS:
            సమితి_అడ్డువరుసలు_sycl_iq_host<TIn, TIdx, ఖండం_iq2_xxs, QK_K, quantize_iq2_xxs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ2_XS:
            సమితి_అడ్డువరుసలు_sycl_iq_host<TIn, TIdx, ఖండం_iq2_xs, QK_K, quantize_iq2_xs>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ2_S:
            సమితి_అడ్డువరుసలు_sycl_iq_host<TIn, TIdx, ఖండం_iq2_s, QK_K, quantize_iq2_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ3_XXS:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_iq3_xxs, QK_K, quantize_అడ్డువరుస_iq3_xxs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ3_S:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_iq3_s, QK_K, quantize_అడ్డువరుస_iq3_s_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ1_S:
            సమితి_అడ్డువరుసలు_sycl_iq_host<TIn, TIdx, ఖండం_iq1_s, QK_K, quantize_iq1_s>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ1_M:
            సమితి_అడ్డువరుసలు_sycl_iq_host<TIn, TIdx, ఖండం_iq1_m, QK_K, quantize_iq1_m>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        స్థితి GGML_TYPE_IQ4_XS:
            సమితి_అడ్డువరుసలు_sycl_qk_host<TIn, TIdx, ఖండం_iq4_xs, QK_K, quantize_అడ్డువరుస_iq4_xs_ref>(
                src0, src1, dst,
                ne00, ne01, ne02, ne03,
                ne11, ne12,
                nb01, nb02, nb03,
                nb10, nb11, nb12,
                nb1, nb2, nb3,
                stream);
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("Unsupported tensor type: src0 %s src1 %s dst %s", ggml_type_name(dst->src[0]->type),
                ggml_type_name(dst->src[1]->type), ggml_type_name(dst->type));
            ఆపు;
    }
}

శూన్య ggml_sycl_op_సమితి_అడ్డువరుసలు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I64 || dst->src[1]->type == GGML_TYPE_I32);

    // dispatch on the index type (src1) and the source value type (src0)
    ఒకవేళ (src0->type == GGML_TYPE_F16) {
        ఒకవేళ (src1->type == GGML_TYPE_I64) {
            సమితి_అడ్డువరుసలు_sycl<sycl::half, int64_t>(ctx, src0, src1, dst);
        } లేకపోతే {
            సమితి_అడ్డువరుసలు_sycl<sycl::half, int32_t>(ctx, src0, src1, dst);
        }
    } లేకపోతే {
        ఒకవేళ (src1->type == GGML_TYPE_I64) {
            సమితి_అడ్డువరుసలు_sycl<దశాంశం, int64_t>(ctx, src0, src1, dst);
        } లేకపోతే {
            సమితి_అడ్డువరుసలు_sycl<దశాంశం, int32_t>(ctx, src0, src1, dst);
        }
    }
}
