#అన్తర్భావ "ops.h"

#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "binary-ops.h"
#అన్తర్భావ "simd-gemm.h"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "unary-ops.h"
#అన్తర్భావ "vec.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cfloat>
#అన్తర్భావ <cmath>

// ggml_compute_forward_dup

స్థిర శూన్య ggml_compute_forward_dup_same_cont(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    స్థిరరాశి size_t nb0 = ggml_type_కొలత(src0->type);

    స్థిరరాశి పూర్ణాంకం ith = params->ith; // thread index
    స్థిరరాశి పూర్ణాంకం nth = params->nth; // number of threads

    // parallelize by blocks
    స్థిరరాశి పూర్ణాంకం nk = ggml_nelements(src0)/ggml_blck_కొలత(src0->type);
    స్థిరరాశి పూర్ణాంకం dr = (nk + nth - 1) / nth;
    స్థిరరాశి పూర్ణాంకం k0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం k1 = MIN(k0 + dr, nk);

    ఒకవేళ (k0 < k1) {
        జ్ఞాపకం_నకలు(
            ((అక్షరం *)  dst->దత్తాంశం + k0*nb0),
            ((అక్షరం *) src0->దత్తాంశం + k0*nb0),
            (k1 - k0) * nb0);
    }
}

మూస<రకం_పేరు src_t, రకం_పేరు dst_t>
స్థిర శూన్య ggml_compute_forward_dup_flt(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith; // thread index
    స్థిరరాశి పూర్ణాంకం nth = params->nth; // number of threads

    // parallelize by rows
    స్థిరరాశి పూర్ణాంకం nr = ne01;
    // number of rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1) / nth;
    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    ఒకవేళ (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_కొలత(src0->type) && nb0 == ggml_type_కొలత(dst->type)) {
        // copy by rows
        స్థిరరాశి size_t rs = ne00*nb00;
        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                    జ్ఞాపకం_నకలు(
                        ((అక్షరం *)  dst->దత్తాంశం + i01*nb1  + i02*nb2  + i03*nb3),
                        ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        తిరిగి;
    }

    // case: dst tensor is contiguous
    ఒకవేళ (ggml_is_contiguous(dst)) {
        ఒకవేళ (nb00 == పరిమాణం(src_t)) {
            ఒకవేళ constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                స్థిరరాశి size_t rs = ne00 * nb00;
                అక్షరం * dst_ptr = (అక్షరం *) dst->దత్తాంశం;

                కొరకు (పూర్ణాంకం i03 = 0; i03 < ne03; i03++) {
                    కొరకు (పూర్ణాంకం i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        కొరకు (పూర్ణాంకం i01 = ir0; i01 < ir1; i01++) {
                            స్థిరరాశి అక్షరం * src0_ptr = (అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
                            జ్ఞాపకం_నకలు(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } లేకపోతే {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->దత్తాంశం;

                కొరకు (పూర్ణాంకం i03 = 0; i03 < ne03; i03++) {
                    కొరకు (పూర్ణాంకం i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        కొరకు (పూర్ణాంకం i01 = ir0; i01 < ir1; i01++) {
                            స్థిరరాశి src_t * src0_ptr = (src_t *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03);
                            కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; i00++) {
                                దశాంశం tmp = type_conversion_పట్టిక<src_t>::గమ్యం_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_పట్టిక<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } లేకపోతే {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->దత్తాంశం;

            కొరకు (పూర్ణాంకం i03 = 0; i03 < ne03; i03++) {
                కొరకు (పూర్ణాంకం i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    కొరకు (పూర్ణాంకం i01 = ir0; i01 < ir1; i01++) {
                        కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; i00++) {
                            స్థిరరాశి src_t * src0_ptr = (src_t *) ((అక్షరం *) src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            దశాంశం tmp = type_conversion_పట్టిక<src_t>::గమ్యం_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_పట్టిక<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        తిరిగి;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    ఒకవేళ constexpr (std::is_same_v<dst_t, src_t>) {
        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                అయితే (i10 >= ne0) {
                    i10 -= ne0;
                    ఒకవేళ (++i11 == ne1) {
                        i11 = 0;
                        ఒకవేళ (++i12 == ne2) {
                            i12 = 0;
                            ఒకవేళ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        స్థిరరాశి అక్షరం * src0_ptr = ((అక్షరం *) src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              అక్షరం * dst_ptr  = ((అక్షరం *)  dst->దత్తాంశం + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        జ్ఞాపకం_నకలు(dst_ptr, src0_ptr, పరిమాణం(dst_t));

                        ఒకవేళ (++i10 == ne00) {
                            i10 = 0;
                            ఒకవేళ (++i11 == ne01) {
                                i11 = 0;
                                ఒకవేళ (++i12 == ne02) {
                                    i12 = 0;
                                    ఒకవేళ (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                అయితే (i10 >= ne0) {
                    i10 -= ne0;
                    ఒకవేళ (++i11 == ne1) {
                        i11 = 0;
                        ఒకవేళ (++i12 == ne2) {
                            i12 = 0;
                            ఒకవేళ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } లేకపోతే {
        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                అయితే (i10 >= ne0) {
                    i10 -= ne0;
                    ఒకవేళ (++i11 == ne1) {
                        i11 = 0;
                        ఒకవేళ (++i12 == ne2) {
                            i12 = 0;
                            ఒకవేళ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        స్థిరరాశి అక్షరం * src0_ptr = ((అక్షరం *) src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              అక్షరం * dst_ptr  = ((అక్షరం *)  dst->దత్తాంశం + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        దశాంశం tmp = type_conversion_పట్టిక<src_t>::గమ్యం_f32(*(స్థిరరాశి src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_పట్టిక<dst_t>::from_f32(tmp);

                        ఒకవేళ (++i10 == ne0) {
                            i10 = 0;
                            ఒకవేళ (++i11 == ne1) {
                                i11 = 0;
                                ఒకవేళ (++i12 == ne2) {
                                    i12 = 0;
                                    ఒకవేళ (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                అయితే (i10 >= ne0) {
                    i10 -= ne0;
                    ఒకవేళ (++i11 == ne1) {
                        i11 = 0;
                        ఒకవేళ (++i12 == ne2) {
                            i12 = 0;
                            ఒకవేళ (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


మూస<రకం_పేరు src_t>
స్థిర శూన్య ggml_compute_forward_dup_గమ్యం_q(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith; // thread index
    స్థిరరాశి పూర్ణాంకం nth = params->nth; // number of threads

    // parallelize by rows
    స్థిరరాశి పూర్ణాంకం nr = ne01;
    // number of rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1) / nth;
    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    ఒకవేళ (ggml_is_contiguous(dst) &&
            nb00 == పరిమాణం(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t స్థిరరాశి quantize_అడ్డువరుస_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        దశాంశం * src0_f32 = (దశాంశం *) params->wdata + (ne00 + CACHE_పంక్తి_కొలత_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_కొలత(dst->type));
        అక్షరం * dst_ptr = (అక్షరం *) dst->దత్తాంశం;

        కొరకు (పూర్ణాంకం i03 = 0; i03 < ne03; i03++) {
            కొరకు (పూర్ణాంకం i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                కొరకు (పూర్ణాంకం i01 = ir0; i01 < ir1; i01++) {
                    స్థిరరాశి src_t * src0_ptr = (src_t *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03);

                    కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_పట్టిక<src_t>::గమ్యం_f32(src0_ptr[i00]);
                    }

                    quantize_అడ్డువరుస_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } లేకపోతే {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
స్థిర శూన్య ggml_compute_forward_dup_బైట్లు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    ఒకవేళ (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        తిరిగి;
    }

    స్థిరరాశి size_t type_కొలత = ggml_type_కొలత(src0->type);

    స్థిరరాశి పూర్ణాంకం ith = params->ith; // thread index
    స్థిరరాశి పూర్ణాంకం nth = params->nth; // number of threads

    // parallelize by rows
    స్థిరరాశి పూర్ణాంకం nr = ne01;
    // number of rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1) / nth;
    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    ఒకవేళ (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_కొలత && nb0 == type_కొలత) {
        // copy by rows
        స్థిరరాశి size_t rs = ggml_అడ్డువరుస_కొలత(src0->type, ne00);
        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                    జ్ఞాపకం_నకలు(
                        ((అక్షరం *)  dst->దత్తాంశం + i01*nb1  + i02*nb2  + i03*nb3),
                        ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        తిరిగి;
    }

    ఒకవేళ (ggml_is_contiguous(dst)) {
        size_t id = 0;
        అక్షరం * dst_ptr = (అక్షరం *) dst->దత్తాంశం;
        స్థిరరాశి size_t rs = ne00 * type_కొలత;

        ఒకవేళ (nb00 == type_కొలత) {
            // src0 is contiguous on first dimension, copy by rows
            కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
                కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                        స్థిరరాశి అక్షరం * src0_ptr = (అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
                        జ్ఞాపకం_నకలు(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } లేకపోతే {
            //printf("%s: this is not optimal - fix me\n", __func__);

            కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
                కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                        కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                            స్థిరరాశి అక్షరం * src0_ptr = (అక్షరం *) src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            జ్ఞాపకం_నకలు(dst_ptr + id, src0_ptr, type_కొలత);

                            id += type_కొలత;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        తిరిగి;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    స్థిరరాశి int64_t nk00 = ne00 / ggml_blck_కొలత(src0->type);
    స్థిరరాశి int64_t nk0  = ne0  / ggml_blck_కొలత(dst->type);

    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            అయితే (k10 >= nk0) {
                k10 -= nk0;
                ఒకవేళ (++i11 == ne1) {
                    i11 = 0;
                    ఒకవేళ (++i12 == ne2) {
                        i12 = 0;
                        ఒకవేళ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            కొరకు (int64_t i01 = ir0; i01 < ir1; i01++) {
                కొరకు (int64_t k00 = 0; k00 < nk00; k00++) {
                    స్థిరరాశి అక్షరం * src0_ptr = ((అక్షరం *) src0->దత్తాంశం + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          అక్షరం * dst_ptr  = ((అక్షరం *)  dst->దత్తాంశం + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    జ్ఞాపకం_నకలు(dst_ptr, src0_ptr, type_కొలత);

                    ఒకవేళ (++k10 == nk0) {
                        k10 = 0;
                        ఒకవేళ (++i11 == ne1) {
                            i11 = 0;
                            ఒకవేళ (++i12 == ne2) {
                                i12 = 0;
                                ఒకవేళ (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            అయితే (k10 >= nk0) {
                k10 -= nk0;
                ఒకవేళ (++i11 == ne1) {
                    i11 = 0;
                    ఒకవేళ (++i12 == ne2) {
                        i12 = 0;
                        ఒకవేళ (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_dup_from_q(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి ggml_type type = src0->type;
    ggml_గమ్యం_float_t స్థిరరాశి dequantize_అడ్డువరుస_q = ggml_get_type_traits(type)->గమ్యం_float;

    size_t qk = ggml_blck_కొలత(type);
    స్థిరరాశి int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_కొలత(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        స్థిరరాశి int64_t i03 = i/(ne00 * ne01 * ne02);
        స్థిరరాశి int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        స్థిరరాశి int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        స్థిరరాశి int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        స్థిరరాశి int64_t x_స్థానభ్రంశం = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        స్థిరరాశి int64_t i13 = i/(ne10 * ne11 * ne12);
        స్థిరరాశి int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        స్థిరరాశి int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        స్థిరరాశి int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        స్థిరరాశి int64_t dst_స్థానభ్రంశం = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_అడ్డువరుస_q(
                (స్థిరరాశి శూన్య *) ((అక్షరం *) src0->దత్తాంశం + x_స్థానభ్రంశం),
                     (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + dst_స్థానభ్రంశం), qk);
    }
}

శూన్య ggml_compute_forward_dup(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (src0->type == dst->type) {
        ggml_compute_forward_dup_బైట్లు(params, dst);
        తిరిగి;
    }

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            {
                /**/ ఒకవేళ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, దశాంశం      >(params, dst);
                లేకపోతే ggml_compute_forward_dup_గమ్యం_q<ggml_fp16_t>(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                /**/ ఒకవేళ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, దశాంశం      >(params, dst);
                లేకపోతే ggml_compute_forward_dup_గమ్యం_q<ggml_bf16_t>(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                /**/ ఒకవేళ (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<దశాంశం, ggml_fp16_t>(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<దశాంశం, ggml_bf16_t>(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<దశాంశం, దశాంశం      >(params, dst);
                లేకపోతే ఒకవేళ (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<దశాంశం, int32_t    >(params, dst);
                లేకపోతే ggml_compute_forward_dup_గమ్యం_q<దశాంశం>(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ఒకవేళ (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, దశాంశం>(params, dst);
                లేకపోతే GGML_ABORT("not implemented");
            } ఆపు;
        అప్రమేయ:
            {
                ఒకవేళ (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    ఆపు;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

స్థిర శూన్య ggml_compute_forward_కూడు_q_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి ggml_type type = src0->type;
    స్థిరరాశి ggml_type dtype = dst->type;
    ggml_గమ్యం_float_t స్థిరరాశి dequantize_అడ్డువరుస_q = ggml_get_type_traits(type)->గమ్యం_float;
    ggml_from_float_t స్థిరరాశి quantize_అడ్డువరుస_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(type));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    దశాంశం * wdata = (దశాంశం *) params->wdata + (ne00 + CACHE_పంక్తి_కొలత_F32) * ith;

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        స్థిరరాశి పూర్ణాంకం i03 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        స్థిరరాశి పూర్ణాంకం i13 = i03;
        స్థిరరాశి పూర్ణాంకం i12 = i02;
        స్థిరరాశి పూర్ణాంకం i11 = i01;

        స్థిరరాశి పూర్ణాంకం i3 = i03;
        స్థిరరాశి పూర్ణాంకం i2 = i02;
        స్థిరరాశి పూర్ణాంకం i1 = i01;

        శూన్య  * src0_అడ్డువరుస = (శూన్య *) ((అక్షరం *) src0->దత్తాంశం + (i01*nb01 + i02*nb02 + i03*nb03));
        దశాంశం * src1_అడ్డువరుస = (దశాంశం *)((అక్షరం *) src1->దత్తాంశం + (i11*nb11 + i12*nb12 + i13*nb13));
        శూన్య  * dst_అడ్డువరుస  = (శూన్య *) ((అక్షరం *)  dst->దత్తాంశం + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        ప్రతిజ్ఞా(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_అడ్డువరుస_q(src0_అడ్డువరుస, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_అడ్డువరుస);
        // quantize row to dst
        ఒకవేళ (quantize_అడ్డువరుస_q != ఖాళీ) {
            quantize_అడ్డువరుస_q(wdata, dst_అడ్డువరుస, ne00);
        } లేకపోతే {
            జ్ఞాపకం_నకలు(dst_అడ్డువరుస, wdata, ne0*nb0);
        }
    }
}

శూన్య ggml_compute_forward_కూడు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_కూడు_non_quantized(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_కూడు_q_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

స్థిర శూన్య ggml_compute_forward_కూడు_id_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src1->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        స్థిరరాశి పూర్ణాంకం i11 = *(int32_t *) ((అక్షరం *) src2->దత్తాంశం + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_కూడు_f32(ne0,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01),
                (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i11*nb11));
    }
}

శూన్య ggml_compute_forward_కూడు_id(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_కూడు_id_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

స్థిర శూన్య ggml_compute_forward_add1_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

#నిర్వచితమైతే GGML_USE_ACCELERATE
        GGML_వాడని(ggml_vec_add1_f32);

        vDSP_vadd(
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం), 0,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#లేకపోతే
        ggml_vec_add1_f32(ne0,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01),
               *(దశాంశం *) src1->దత్తాంశం);
#అయితే_ముగింపు
    }
}

స్థిర శూన్య ggml_compute_forward_add1_f16_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    స్థిరరాశి దశాంశం v = *(దశాంశం *) src1->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb00 == పరిమాణం(ggml_fp16_t));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01);
        కొరకు (పూర్ణాంకం i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_గమ్యం_FP16(GGML_CPU_FP16_గమ్యం_FP32(src0_ptr[i]) + v);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_add1_f16_f16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    స్థిరరాశి దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(*(ggml_fp16_t *) src1->దత్తాంశం);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb00 == పరిమాణం(ggml_fp16_t));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01);
        కొరకు (పూర్ణాంకం i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_గమ్యం_FP16(GGML_CPU_FP16_గమ్యం_FP32(src0_ptr[i]) + v);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_add1_q_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    స్థిరరాశి దశాంశం v = *(దశాంశం *) src1->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి ggml_type type = src0->type;
    ggml_గమ్యం_float_t స్థిరరాశి dequantize_అడ్డువరుస_q = ggml_get_type_traits(type)->గమ్యం_float;
    ggml_from_float_t స్థిరరాశి quantize_అడ్డువరుస_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_కొలత(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    దశాంశం * wdata = (దశాంశం *) params->wdata + (ne0 + CACHE_పంక్తి_కొలత_F32) * ith;

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

        శూన్య  * src0_అడ్డువరుస = (శూన్య *) ((అక్షరం *) src0->దత్తాంశం + (i1*nb01 + i2*nb02 + i3*nb03));
        శూన్య  * dst_అడ్డువరుస  = (శూన్య *) ((అక్షరం *)  dst->దత్తాంశం + (i1*nb1  + i2*nb2  + i3*nb0 ));

        ప్రతిజ్ఞా(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_అడ్డువరుస_q(src0_అడ్డువరుస, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_అడ్డువరుస_q(wdata, dst_అడ్డువరుస, ne0);
    }
}

స్థిర శూన్య ggml_compute_forward_add1_bf16_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    స్థిరరాశి దశాంశం v = *(దశాంశం *) src1->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == పరిమాణం(ggml_bf16_t));
    GGML_ASSERT(nb00 == పరిమాణం(ggml_bf16_t));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01);
        కొరకు (పూర్ణాంకం i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_గమ్యం_BF16(GGML_BF16_గమ్యం_FP32(src0_ptr[i]) + v);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_add1_bf16_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    స్థిరరాశి దశాంశం v = GGML_BF16_గమ్యం_FP32(*(ggml_bf16_t *) src1->దత్తాంశం);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == పరిమాణం(ggml_bf16_t));
    GGML_ASSERT(nb00 == పరిమాణం(ggml_bf16_t));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne2*ne1);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01);
        కొరకు (పూర్ణాంకం i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_గమ్యం_BF16(GGML_BF16_గమ్యం_FP32(src0_ptr[i]) + v);
        }
    }
}

శూన్య ggml_compute_forward_add1(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ఒకవేళ (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                లేకపోతే ఒకవేళ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                లేకపోతే {
                    GGML_ABORT("fatal error");
                }
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ఒకవేళ (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                లేకపోతే ఒకవేళ (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                లేకపోతే {
                    GGML_ABORT("fatal error");
                }
            } ఆపు;
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q8_1:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

స్థిర శూన్య ggml_compute_forward_acc_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t స్థానభ్రంశం  = ((int32_t *) dst->op_params)[3];
    తార్కిక   inplace = (తార్కిక) ((int32_t *) dst->op_params)[4];

    ఒకవేళ (!inplace) {
        ఒకవేళ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            జ్ఞాపకం_నకలు(
                ((అక్షరం *)  dst->దత్తాంశం),
                ((అక్షరం *) src0->దత్తాంశం),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src1);
    స్థిరరాశి పూర్ణాంకం nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    స్థిరరాశి size_t nb0 = ggml_element_కొలత(src0);

    స్థిరరాశి size_t nb00 = nb0;
    స్థిరరాశి size_t nb01 = nb1;
    స్థిరరాశి size_t nb02 = nb2;
    స్థిరరాశి size_t nb03 = nb3;

    GGML_ASSERT(స్థానభ్రంశం + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(స్థానభ్రంశం + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne12*ne11);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne12*ne11)/ne11;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne12*ne11 - i2*ne11);

#నిర్వచితమైతే GGML_USE_ACCELERATE
        vDSP_vadd(
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01 + స్థానభ్రంశం), 1,
                (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1  + స్థానభ్రంశం), 1, nc);
#లేకపోతే
        ggml_vec_కూడు_f32(nc,
                (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i3*nb3  + i2*nb2  + i1*nb1  + స్థానభ్రంశం),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01 + స్థానభ్రంశం),
                (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i3*nb13 + i2*nb12 + i1*nb11));
#అయితే_ముగింపు
    }
}

శూన్య ggml_compute_forward_acc(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q8_1:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

స్థిర శూన్య ggml_compute_forward_కూడిక_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float కూడిక = 0;

    ఒకవేళ (nb00 == పరిమాణం(దశాంశం)) {
        ggml_float అడ్డువరుస_కూడిక = 0;
        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_కూడిక_f32_ggf(ne00,
                            &అడ్డువరుస_కూడిక,
                            (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03));
                    కూడిక += అడ్డువరుస_కూడిక;
                }
            }
        }
    } లేకపోతే {
        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    స్థిరరాశి అక్షరం * అడ్డువరుస = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        కూడిక += (ggml_float) *(స్థిరరాశి దశాంశం *) (అడ్డువరుస + i00*nb00);
                    }
                }
            }
        }
    }

    ((దశాంశం *) dst->దత్తాంశం)[0] = కూడిక;
}

స్థిర శూన్య ggml_compute_forward_కూడిక_f16(
    స్థిరరాశి ggml_compute_params * params,
          ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(ggml_is_scalar(dst));

    ప్రతిజ్ఞా(src0->nb[0] == పరిమాణం(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    దశాంశం కూడిక = 0;
    దశాంశం అడ్డువరుస_కూడిక = 0;

    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_కూడిక_f16_ggf(ne00,
                    &అడ్డువరుస_కూడిక,
                    (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03));
                కూడిక += అడ్డువరుస_కూడిక;
            }
        }
    }
    ((ggml_fp16_t *) dst->దత్తాంశం)[0] = GGML_CPU_FP32_గమ్యం_FP16(కూడిక);
}

స్థిర శూన్య ggml_compute_forward_కూడిక_bf16(
    స్థిరరాశి ggml_compute_params * params,
          ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(ggml_is_scalar(dst));

    ప్రతిజ్ఞా(src0->nb[0] == పరిమాణం(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    దశాంశం కూడిక = 0;
    దశాంశం అడ్డువరుస_కూడిక = 0;

    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_కూడిక_bf16_ggf(ne00,
                    &అడ్డువరుస_కూడిక,
                    (ggml_bf16_t *) ((అక్షరం *) src0->దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03));
                కూడిక += అడ్డువరుస_కూడిక;
            }
        }
    }
    ((ggml_bf16_t *) dst->దత్తాంశం)[0] = GGML_FP32_గమ్యం_BF16(కూడిక);
}

శూన్య ggml_compute_forward_కూడిక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_కూడిక_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_కూడిక_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_కూడిక_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

స్థిర శూన్య ggml_compute_forward_cumsum_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(dst->nb[0] == పరిమాణం(దశాంశం));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, src0);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        దశాంశం * src_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03);
        దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_అడ్డువరుస, src_అడ్డువరుస);
    }
}

శూన్య ggml_compute_forward_cumsum(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

స్థిర శూన్య ggml_compute_forward_కూడిక_అడ్డువరుసలు_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(dst->nb[0] == పరిమాణం(దశాంశం));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    కొరకు (int64_t i3 = 0; i3 < ne03; i3++) {
        కొరకు (int64_t i2 = 0; i2 < ne02; i2++) {
            కొరకు (int64_t i1 = 0; i1 < ne01; i1++) {
                దశాంశం * src_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i1*nb01 + i2*nb02 + i3*nb03);
                దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i1*nb1  + i2*nb2  + i3*nb3);
                దశాంశం అడ్డువరుస_కూడిక = 0;
                ggml_vec_కూడిక_f32(ne00, &అడ్డువరుస_కూడిక, src_అడ్డువరుస);
                dst_అడ్డువరుస[0] = అడ్డువరుస_కూడిక;
            }
        }
    }
}

శూన్య ggml_compute_forward_కూడిక_అడ్డువరుసలు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_కూడిక_అడ్డువరుసలు_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

స్థిర శూన్య ggml_compute_forward_సగటు_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(src0->nb[0] == పరిమాణం(దశాంశం));

    GGML_TENSOR_UNARY_OP_LOCALS

    ప్రతిజ్ఞా(ne0 == 1);
    ప్రతిజ్ఞా(ne1 == ne01);
    ప్రతిజ్ఞా(ne2 == ne02);
    ప్రతిజ్ఞా(ne3 == ne03);

    GGML_వాడని(ne0);
    GGML_వాడని(ne1);
    GGML_వాడని(ne2);
    GGML_వాడని(ne3);

    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_కూడిక_f32(ne00,
                        (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i01*nb1  + i02*nb2  + i03*nb3),
                        (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03));

                *(దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i01*nb1 + i02*nb2 + i03*nb3) /= (దశాంశం) ne00;
            }
        }
    }
}

శూన్య ggml_compute_forward_సగటు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_సగటు_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

స్థిర శూన్య ggml_compute_forward_argmax_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(src0->nb[0] == పరిమాణం(దశాంశం));
    ప్రతిజ్ఞా(dst->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];

    స్థిరరాశి size_t nb01 = src0->nb[1];
    స్థిరరాశి size_t nb0 = dst->nb[0];

    కొరకు (int64_t i1 = 0; i1 < ne01; i1++) {
        దశాంశం * src = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i1*nb01);
        int32_t * dst_ = (int32_t *) ((అక్షరం *)  dst->దత్తాంశం + i1*nb0);
        పూర్ణాంకం v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

శూన్య ggml_compute_forward_argmax(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

స్థిర శూన్య ggml_compute_forward_లెక్క_equal_i32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    స్థిరరాశి int64_t nr = ggml_nrows(src0);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t కూడిక_thread = 0;

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 =  ir                        / (ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne03)            /       ne01;
        స్థిరరాశి int64_t i01 =  ir - i03*ne03 - i02*ne02;

        స్థిరరాశి అక్షరం * data0 = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i03*nb03 + i02*nb02 + i01*nb01;
        స్థిరరాశి అక్షరం * data1 = (స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i03*nb13 + i02*nb12 + i01*nb11;

        కొరకు (int64_t i00 = 0; i00 < ne00; ++i00) {
            స్థిరరాశి int32_t val0 = *((స్థిరరాశి int32_t *) (data0 + i00*nb00));
            స్థిరరాశి int32_t val1 = *((స్థిరరాశి int32_t *) (data1 + i00*nb10));

            కూడిక_thread += val0 == val1;
        }
    }
    ఒకవేళ (ith != 0) {
        sums[ith] = కూడిక_thread;
    }
    ggml_barrier(params->threadpool);

    ఒకవేళ (ith != 0) {
        తిరిగి;
    }

    కొరకు (పూర్ణాంకం ith_other = 1; ith_other < nth; ++ith_other) {
        కూడిక_thread += sums[ith_other];
    }
    *((int64_t *) dst->దత్తాంశం) = కూడిక_thread;
}

శూన్య ggml_compute_forward_లెక్క_equal(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_I32:
            {
                ggml_compute_forward_లెక్క_equal_i32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

స్థిర శూన్య ggml_compute_forward_repeat_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    స్థిరరాశి పూర్ణాంకం nr0 = (పూర్ణాంకం)(ne0/ne00);
    స్థిరరాశి పూర్ణాంకం nr1 = (పూర్ణాంకం)(ne1/ne01);
    స్థిరరాశి పూర్ణాంకం nr2 = (పూర్ణాంకం)(ne2/ne02);
    స్థిరరాశి పూర్ణాంకం nr3 = (పూర్ణాంకం)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    // TODO: maybe this is not optimal?
    కొరకు                         (పూర్ణాంకం i3 = 0; i3 < nr3;  i3++) {
        కొరకు                     (పూర్ణాంకం k3 = 0; k3 < ne03; k3++) {
            కొరకు                 (పూర్ణాంకం i2 = 0; i2 < nr2;  i2++) {
                కొరకు             (పూర్ణాంకం k2 = 0; k2 < ne02; k2++) {
                    కొరకు         (పూర్ణాంకం i1 = 0; i1 < nr1;  i1++) {
                        కొరకు     (పూర్ణాంకం k1 = 0; k1 < ne01; k1++) {
                            కొరకు (పూర్ణాంకం i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_repeat_f16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    స్థిరరాశి పూర్ణాంకం nr0 = (పూర్ణాంకం)(ne0/ne00);
    స్థిరరాశి పూర్ణాంకం nr1 = (పూర్ణాంకం)(ne1/ne01);
    స్థిరరాశి పూర్ణాంకం nr2 = (పూర్ణాంకం)(ne2/ne02);
    స్థిరరాశి పూర్ణాంకం nr3 = (పూర్ణాంకం)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb00 == పరిమాణం(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    కొరకు                         (పూర్ణాంకం i3 = 0; i3 < nr3;  i3++) {
        కొరకు                     (పూర్ణాంకం k3 = 0; k3 < ne03; k3++) {
            కొరకు                 (పూర్ణాంకం i2 = 0; i2 < nr2;  i2++) {
                కొరకు             (పూర్ణాంకం k2 = 0; k2 < ne02; k2++) {
                    కొరకు         (పూర్ణాంకం i1 = 0; i1 < nr1;  i1++) {
                        కొరకు     (పూర్ణాంకం k1 = 0; k1 < ne01; k1++) {
                            కొరకు (పూర్ణాంకం i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((అక్షరం *)  dst->దత్తాంశం + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                కొరకు (పూర్ణాంకం i = 0; i < ne00; ++i) {
                                    y[i]  = x[i];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_repeat(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } ఆపు;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

స్థిర శూన్య ggml_compute_forward_repeat_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    స్థిరరాశి పూర్ణాంకం nr0 = (పూర్ణాంకం)(ne00/ne0);
    స్థిరరాశి పూర్ణాంకం nr1 = (పూర్ణాంకం)(ne01/ne1);
    స్థిరరాశి పూర్ణాంకం nr2 = (పూర్ణాంకం)(ne02/ne2);
    స్థిరరాశి పూర్ణాంకం nr3 = (పూర్ణాంకం)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    ఒకవేళ (ggml_is_contiguous(dst)) {
        ggml_vec_సమితి_f32(ne0*ne1*ne2*ne3, (దశాంశం *)dst->దత్తాంశం, 0);
    } లేకపోతే {
        కొరకు         (పూర్ణాంకం k3 = 0; k3 < ne3; k3++) {
            కొరకు     (పూర్ణాంకం k2 = 0; k2 < ne2; k2++) {
                కొరకు (పూర్ణాంకం k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_సమితి_f32(ne0,
                        (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    కొరకు                         (పూర్ణాంకం i3 = 0; i3 < nr3; i3++) {
        కొరకు                     (పూర్ణాంకం k3 = 0; k3 < ne3; k3++) {
            కొరకు                 (పూర్ణాంకం i2 = 0; i2 < nr2; i2++) {
                కొరకు             (పూర్ణాంకం k2 = 0; k2 < ne2; k2++) {
                    కొరకు         (పూర్ణాంకం i1 = 0; i1 < nr1; i1++) {
                        కొరకు     (పూర్ణాంకం k1 = 0; k1 < ne1; k1++) {
                            కొరకు (పూర్ణాంకం i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_repeat_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_వెనుక_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
స్థిర శూన్య ggml_compute_forward_concat_any(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src0));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src1));

    int64_t o[4] = {0, 0, 0, 0};

    ఒకవేళ (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_కొలత(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_కొలత(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_కొలత(src0->type);
    } లేకపోతే {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    కొరకు (పూర్ణాంకం i3 = 0; i3 < ne03; i3++) {
        కొరకు (పూర్ణాంకం i2 = ith; i2 < ne02; i2 += nth) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ne01; i1++) {
                స్థిరరాశి అక్షరం * x = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i1*nb01 + i2*nb02 + i3*nb03;
                      అక్షరం * y = (      అక్షరం *) dst->దత్తాంశం  + i1*nb1  + i2*nb2  + i3*nb3;
                జ్ఞాపకం_నకలు(y, x, ggml_అడ్డువరుస_కొలత(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    కొరకు (పూర్ణాంకం i3 = 0; i3 < ne13; i3++) {
        కొరకు (పూర్ణాంకం i2 = ith; i2 < ne12; i2 += nth) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ne11; i1++) {
                స్థిరరాశి అక్షరం * x = (స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i1*nb11         + i2*nb12         + i3*nb13;
                      అక్షరం * y = (      అక్షరం *)  dst->దత్తాంశం + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                జ్ఞాపకం_నకలు(y, x, ggml_అడ్డువరుస_కొలత(src1->type, ne10));
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_concat_i8(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_కొలత(src0->type) == పరిమాణం(int8_t));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    స్థిరరాశి int8_t * x;

    // TODO: smarter multi-theading
    కొరకు (పూర్ణాంకం i3 = 0; i3 < ne3; i3++) {
        కొరకు (పూర్ణాంకం i2 = ith; i2 < ne2; i2 += nth) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ne1; i1++) {
                కొరకు (పూర్ణాంకం i0 = 0; i0 < ne0; i0++) {
                    ఒకవేళ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (స్థిరరాశి int8_t *) ((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } లేకపోతే {
                        x = (స్థిరరాశి int8_t *) ((స్థిరరాశి అక్షరం *)src1->దత్తాంశం + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((అక్షరం *)dst->దత్తాంశం + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_concat_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_కొలత(src0->type) == పరిమాణం(ggml_fp16_t));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    స్థిరరాశి ggml_fp16_t * x;

    // TODO: smarter multi-theading
    కొరకు (పూర్ణాంకం i3 = 0; i3 < ne3; i3++) {
        కొరకు (పూర్ణాంకం i2 = ith; i2 < ne2; i2 += nth) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ne1; i1++) {
                కొరకు (పూర్ణాంకం i0 = 0; i0 < ne0; i0++) {
                    ఒకవేళ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (స్థిరరాశి ggml_fp16_t *) ((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } లేకపోతే {
                        x = (స్థిరరాశి ggml_fp16_t *) ((స్థిరరాశి అక్షరం *)src1->దత్తాంశం + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((అక్షరం *)dst->దత్తాంశం + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_concat_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_కొలత(src0->type) == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    స్థిరరాశి దశాంశం * x;

    // TODO: smarter multi-theading
    కొరకు (పూర్ణాంకం i3 = 0; i3 < ne3; i3++) {
        కొరకు (పూర్ణాంకం i2 = ith; i2 < ne2; i2 += nth) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ne1; i1++) {
                కొరకు (పూర్ణాంకం i0 = 0; i0 < ne0; i0++) {
                    ఒకవేళ (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } లేకపోతే {
                        x = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *)src1->దత్తాంశం + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    దశాంశం * y = (దశాంశం *)((అక్షరం *)dst->దత్తాంశం + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_concat(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
మూస <శూన్య (*vec_op)(స్థిరరాశి పూర్ణాంకం, ggml_bf16_t *, స్థిరరాశి ggml_bf16_t *)>
స్థిర శూన్య ggml_compute_forward_unary_bf16_అడ్డువరుసలు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

స్థిర శూన్య ggml_compute_forward_gelu_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_gelu_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_gelu_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_అడ్డువరుసలు<ggml_vec_gelu_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_gelu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

స్థిర శూన్య ggml_compute_forward_fill_f32(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి దశాంశం c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, dst);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne2*ne1);
        స్థిరరాశి int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        స్థిరరాశి int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        దశాంశం * dst_ptr  = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_సమితి_f32(ne0, dst_ptr, c);
    }
}

స్థిర శూన్య ggml_compute_forward_fill_f16(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_fp16_t c = GGML_CPU_FP32_గమ్యం_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, dst);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne2*ne1);
        స్థిరరాశి int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        స్థిరరాశి int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_సమితి_f16(ne0, dst_ptr, c);
    }
}

శూన్య ggml_compute_forward_fill(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

స్థిర శూన్య ggml_compute_forward_tri_f32(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    స్థిరరాశి ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి స్వయంచాలక [ir0, ir1] = get_thread_range(params, src0);

    తార్కిక (*bipred)(పూర్ణాంకం, పూర్ణాంకం);

    ఎంపిక (ttype) {
        స్థితి GGML_TRI_TYPE_క్రింది:      bipred = [](పూర్ణాంకం i, పూర్ణాంకం r) { తిరిగి i <  r; }; ఆపు;
        స్థితి GGML_TRI_TYPE_క్రింది_DIAG: bipred = [](పూర్ణాంకం i, పూర్ణాంకం r) { తిరిగి i <= r; }; ఆపు;
        స్థితి GGML_TRI_TYPE_పై:      bipred = [](పూర్ణాంకం i, పూర్ణాంకం r) { తిరిగి i >  r; }; ఆపు;
        స్థితి GGML_TRI_TYPE_పై_DIAG: bipred = [](పూర్ణాంకం i, పూర్ణాంకం r) { తిరిగి i >= r; }; ఆపు;
        అప్రమేయ: GGML_ABORT("invalid tri type");
    }

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        స్థిరరాశి దశాంశం * src_ptr = (స్థిరరాశి దశాంశం  *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i03*nb03 + i02*nb02 + i01*nb01);
              దశాంశం * dst_ptr = (      దశాంశం  *) ((      అక్షరం *) dst->దత్తాంశం  + i03*nb3  + i02*nb2  + i01*nb1);

        కొరకు (పూర్ణాంకం i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

శూన్య ggml_compute_forward_tri(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

స్థిర శూన్య ggml_compute_forward_gelu_erf_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_gelu_erf_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_gelu_erf_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_అడ్డువరుసలు<ggml_vec_gelu_erf_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_gelu_erf(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

స్థిర శూన్య ggml_compute_forward_gelu_శీఘ్ర_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_శీఘ్ర_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_gelu_శీఘ్ర_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_శీఘ్ర_f16(nc,
                (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_gelu_శీఘ్ర_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_అడ్డువరుసలు<ggml_vec_gelu_శీఘ్ర_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_gelu_శీఘ్ర(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_శీఘ్ర_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_శీఘ్ర_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_శీఘ్ర_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

స్థిర శూన్య ggml_compute_forward_silu_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_silu_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(ggml_is_contiguous_అడ్డువరుసలు(src0));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne02*ne01);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne02*ne01)/ne01;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_silu_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_అడ్డువరుసలు<ggml_vec_silu_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_silu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

స్థిర శూన్య ggml_compute_forward_leaky_relu_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(ggml_is_contiguous_1(src0));
    ప్రతిజ్ఞా(ggml_is_contiguous_1(dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    స్థిరరాశి పూర్ణాంకం n  = ggml_nrows(src0);
    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];

    దశాంశం negative_వాలు;
    జ్ఞాపకం_నకలు(&negative_వాలు, dst->op_params, పరిమాణం(దశాంశం));

    ప్రతిజ్ఞా(dst->nb[0]  == పరిమాణం(దశాంశం));
    ప్రతిజ్ఞా(src0->nb[0] == పరిమాణం(దశాంశం));

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i*( dst->nb[1])),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i*(src0->nb[1])), negative_వాలు);
    }
}

స్థిర శూన్య ggml_compute_forward_leaky_relu_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    ప్రతిజ్ఞా(ggml_is_contiguous_1(src0));
    ప్రతిజ్ఞా(ggml_is_contiguous_1(dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    స్థిరరాశి పూర్ణాంకం n  = ggml_nrows(src0);
    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];

    దశాంశం negative_వాలు;
    జ్ఞాపకం_నకలు(&negative_వాలు, dst->op_params, పరిమాణం(దశాంశం));

    ప్రతిజ్ఞా(dst->nb[0]  == పరిమాణం(ggml_fp16_t));
    ప్రతిజ్ఞా(src0->nb[0] == పరిమాణం(ggml_fp16_t));

    కొరకు (పూర్ణాంకం i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i*(src0->nb[1])), negative_వాలు);
    }
}

శూన్య ggml_compute_forward_leaky_relu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

స్థిర శూన్య ggml_compute_forward_silu_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * grad = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ప్రతిజ్ఞా(ggml_is_contiguous_1(grad));
    ప్రతిజ్ఞా(ggml_is_contiguous_1(src1));
    ప్రతిజ్ఞా(ggml_is_contiguous_1(dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src1, dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src1, grad));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i1*( dst->nb[1])),
                (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i1*(src1->nb[1])),
                (దశాంశం *) ((అక్షరం *) grad->దత్తాంశం + i1*(grad->nb[1])));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_silu_వెనుక_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * grad = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ప్రతిజ్ఞా(ggml_is_contiguous_1(grad));
    ప్రతిజ్ఞా(ggml_is_contiguous_1(src1));
    ప్రతిజ్ఞా(ggml_is_contiguous_1(dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src1, dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src1, grad));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((అక్షరం *) src1->దత్తాంశం + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((అక్షరం *) grad->దత్తాంశం + i1*(grad->nb[1])));

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

శూన్య ggml_compute_forward_silu_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_వెనుక_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_వెనుక_f16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
మూస <శూన్య (*vec_op)(స్థిరరాశి పూర్ణాంకం, ggml_bf16_t *, స్థిరరాశి ggml_bf16_t *, స్థిరరాశి ggml_bf16_t *)>
స్థిర శూన్య ggml_compute_forward_glu_bf16_అడ్డువరుసలు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        స్థిరరాశి ggml_bf16_t * src0_p = (స్థిరరాశి ggml_bf16_t *) (src0_d + i1*src0_o);
        స్థిరరాశి ggml_bf16_t * src1_p = (స్థిరరాశి ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1]));

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

స్థిర శూన్య ggml_compute_forward_reglu_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1*src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_reglu_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_reglu_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_అడ్డువరుసలు<ggml_vec_reglu_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_reglu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

స్థిర శూన్య ggml_compute_forward_geglu_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1*src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_geglu_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_geglu_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_అడ్డువరుసలు<ggml_vec_geglu_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_geglu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

స్థిర శూన్య ggml_compute_forward_swiglu_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1*src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_swiglu_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_swiglu_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_అడ్డువరుసలు<ggml_vec_swiglu_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_swiglu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

స్థిర శూన్య ggml_compute_forward_swiglu_oai_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి దశాంశం alpha = ggml_get_op_params_f32(dst, 2);
    స్థిరరాశి దశాంశం హద్దు = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1*src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1*src1_o);
        దశాంశం * dst_p  = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1]));

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = std::min(src0_p[k], హద్దు);
            స్థిరరాశి దశాంశం y = std::clamp(src1_p[k], -హద్దు, హద్దు);
            స్థిరరాశి దశాంశం బహిర్గతం_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = బహిర్గతం_glu * (y + 1.f);
        }

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = dst_p[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_swiglu_oai_f16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి దశాంశం alpha = ggml_get_op_params_f32(dst, 2);
    స్థిరరాశి దశాంశం హద్దు = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1]));

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = std::min(GGML_CPU_FP16_గమ్యం_FP32(src0_p[k]), హద్దు);
            స్థిరరాశి దశాంశం y = std::clamp(GGML_CPU_FP16_గమ్యం_FP32(src1_p[k]), -హద్దు, హద్దు);
            స్థిరరాశి దశాంశం బహిర్గతం_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_గమ్యం_FP16(బహిర్గతం_glu * (y + 1.f));
        }

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = GGML_CPU_FP16_గమ్యం_FP32(dst_p[k]);
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_swiglu_oai(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

స్థిర శూన్య ggml_compute_forward_swiglu_clamp_f32(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    అక్షరం *              src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం *              src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t        src0_o = src0->nb[1];
    స్థిరరాశి size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి దశాంశం   హద్దు   = ggml_get_op_params_f32(dst, 3);

    స్థిరరాశి పూర్ణాంకం dr  = (nr + nth - 1) / nth;
    స్థిరరాశి పూర్ణాంకం ir0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1 * src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1 * src1_o);
        దశాంశం * dst_p  = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1 * (dst->nb[1]));

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం gate = std::min(src0_p[k], హద్దు);
            స్థిరరాశి దశాంశం పైకి   = std::clamp(src1_p[k], -హద్దు, హద్దు);
            dst_p[k]         = gate / (1.f + expf(-gate)) * పైకి;
        }

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = dst_p[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు  // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_swiglu_clamp_f16(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0   = dst->src[0];
    స్థిరరాశి ggml_tensor * src1   = dst->src[1];
    అక్షరం *              src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం *              src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t        src0_o = src0->nb[1];
    స్థిరరాశి size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి దశాంశం   హద్దు   = ggml_get_op_params_f32(dst, 3);

    స్థిరరాశి పూర్ణాంకం dr  = (nr + nth - 1) / nth;
    స్థిరరాశి పూర్ణాంకం ir0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1 * (dst->nb[1]));

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం gate = std::min(GGML_FP16_గమ్యం_FP32(src0_p[k]), హద్దు);
            స్థిరరాశి దశాంశం పైకి   = std::clamp(GGML_FP16_గమ్యం_FP32(src1_p[k]), -హద్దు, హద్దు);
            dst_p[k]         = GGML_FP32_గమ్యం_FP16(gate / (1.f + expf(-gate)) * పైకి);
        }

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = GGML_FP16_గమ్యం_FP32(dst_p[k]);
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు  // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_swiglu_clamp(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    ఎంపిక (dst->src[0]->type) {
        స్థితి GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            ఆపు;
        స్థితి GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

స్థిర శూన్య ggml_compute_forward_geglu_erf_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1*src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_geglu_erf_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_geglu_erf_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_అడ్డువరుసలు<ggml_vec_geglu_erf_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_geglu_erf(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

స్థిర శూన్య ggml_compute_forward_geglu_శీఘ్ర_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * src0_p = (దశాంశం *) (src0_d + i1*src0_o);
        దశాంశం * src1_p = (దశాంశం *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_శీఘ్ర_f32(nc, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి దశాంశం x = ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            GGML_వాడని(x);
            ప్రతిజ్ఞా(!isnan(x));
            ప్రతిజ్ఞా(!isinf(x));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_geglu_శీఘ్ర_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    అక్షరం * src0_d = (అక్షరం *) src0->దత్తాంశం;
    అక్షరం * src1_d = (అక్షరం *) (src1 ? src1->దత్తాంశం : src0->దత్తాంశం);
    స్థిరరాశి size_t src0_o = src0->nb[1];
    స్థిరరాశి size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    ఒకవేళ (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    స్థిరరాశి int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        ఒకవేళ (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_శీఘ్ర_f16(nc, (ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*(dst->nb[1])), src0_p, src1_p);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            స్థిరరాశి ggml_fp16_t x = ((ggml_fp16_t *) ((అక్షరం *) dst->దత్తాంశం + i1*( dst->nb[1])))[k];
            స్థిరరాశి దశాంశం v = GGML_FP16_గమ్యం_FP32(x);
            GGML_వాడని(v);
            ప్రతిజ్ఞా(!isnan(v));
            ప్రతిజ్ఞా(!isinf(v));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

స్థిర శూన్య ggml_compute_forward_geglu_శీఘ్ర_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_అడ్డువరుసలు<ggml_vec_geglu_శీఘ్ర_bf16>(params, dst);
}

స్థిర శూన్య ggml_compute_forward_geglu_శీఘ్ర(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_శీఘ్ర_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_శీఘ్ర_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_శీఘ్ర_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

స్థిర శూన్య ggml_compute_forward_norm_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));

    GGML_ASSERT(eps >= 0.0f);

    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                స్థిరరాశి అక్షరం * x = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;
                అక్షరం * y = (అక్షరం *) dst->దత్తాంశం + i01*nb1 + i02*nb2 + i03*nb3;

                ఒకవేళ (nb00 == పరిమాణం(దశాంశం) && nb0 == పరిమాణం(దశాంశం)) {
                    స్థిరరాశి దశాంశం * xf = (స్థిరరాశి దశాంశం *) x;

                    దశాంశం కూడిక = 0.0;
                    ggml_vec_కూడిక_f32(ne00, &కూడిక, xf);
                    దశాంశం సగటు_2 = కూడిక/ne00;

                    దశాంశం * yf = (దశాంశం *) y;
                    దశాంశం variance = 0;

#నిర్వచితమైతే GGML_USE_ACCELERATE
                    సగటు_2 = -సగటు_2;
                    vDSP_vsadd(xf, 1, &సగటు_2, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#లేకపోతే
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, సగటు_2);
#అయితే_ముగింపు //GGML_USE_ACCELERATE

                    స్థిరరాశి దశాంశం scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } లేకపోతే {
                    దశాంశం కూడిక = 0.0;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        కూడిక += *(స్థిరరాశి దశాంశం *) (x + i00*nb00);
                    }
                    స్థిరరాశి దశాంశం సగటు_2 = కూడిక/ne00;

                    దశాంశం variance = 0.0f;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        స్థిరరాశి దశాంశం v = *(స్థిరరాశి దశాంశం *) (x + i00*nb00) - సగటు_2;
                        *(దశాంశం *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    స్థిరరాశి దశాంశం scale = 1.0f/sqrtf(variance + eps);
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(దశాంశం *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_norm(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
గణన ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

మూస <ggml_rms_norm_fuse_op FUSE_OP>
స్థిర శూన్య ggml_compute_forward_rms_norm_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = శూన్యసూచిక) {

    స్థిరరాశి ggml_tensor * src0 = dst_rms_norm->src[0];
    స్థిరరాశి ggml_tensor * src1 = శూన్యసూచిక;
    ggml_tensor       * dst  = dst_rms_norm;

    ఒకవేళ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst_rms_norm->op_params, పరిమాణం(దశాంశం));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                స్థిరరాశి దశాంశం * x = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float కూడిక = 0.0;
                // worth switching to explicit SIMD?
                కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                    కూడిక += (ggml_float)(x[i00] * x[i00]);
                }

                స్థిరరాశి దశాంశం సగటు_2  = కూడిక/ne00;
                స్థిరరాశి దశాంశం scale = 1.0f/sqrtf(సగటు_2 + eps);

                // if you hit this, likely you got an inf somewhere earlier
                ప్రతిజ్ఞా(scale > 0.0f);

                దశాంశం * y = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i01*nb1 + i02*nb2 + i03*nb3);

                ఒకవేళ constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    స్థిరరాశి int64_t i11 = i01 % ne11;
                    స్థిరరాశి int64_t i12 = i02 % ne12;
                    స్థిరరాశి int64_t i13 = i03 % ne13;
                    స్థిరరాశి దశాంశం * w = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i11*nb11 + i12*nb12 + i13*nb13);

                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } లేకపోతే {
                    జ్ఞాపకం_నకలు(y, x, ne00 * పరిమాణం(దశాంశం));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_rms_norm(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
శూన్య ggml_compute_forward_rms_norm_mul_fused(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != శూన్యసూచిక);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    స్థిరరాశి ggml_tensor * src0 = dst_rms_norm->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_compute_forward_rms_norm_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    స్థిరరాశి ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src1->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));

    // TODO: optimize
    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                స్థిరరాశి int64_t i11 = i01;
                స్థిరరాశి int64_t i12 = i02;
                స్థిరరాశి int64_t i13 = i03;

                స్థిరరాశి దశాంశం * dz = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03);
                స్థిరరాశి దశాంశం * x  = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float కూడిక_xx  = 0.0;
                ggml_float కూడిక_xdz = 0.0;

                కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                    కూడిక_xx  += (ggml_float)(x[i00] * x[i00]);
                    కూడిక_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                స్థిరరాశి దశాంశం సగటు_eps = (దశాంశం)(కూడిక_xx)/ne00 + eps;
                స్థిరరాశి దశాంశం కూడిక_eps  = (దశాంశం)(కూడిక_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                స్థిరరాశి దశాంశం rrms     = 1.0f / sqrtf(సగటు_eps);
                //const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

                {
                    // z = rms_norm(x)
                    //
                    // rms_norm(src1) =
                    //     scale(
                    //         src1,
                    //         div(
                    //             1,
                    //             sqrt(
                    //                 add(
                    //                     scale(
                    //                         sum(
                    //                             sqr(
                    //                                 src1)),
                    //                         (1.0/N)),
                    //                     eps))));

                    // postorder:
                    // ## op    args         grad
                    // 00 param src1         grad[#00]
                    // 01 const 1
                    // 02 sqr   (#00)        grad[#02]
                    // 03 sum   (#02)        grad[#03]
                    // 04 const 1/N
                    // 05 scale (#03, #04)   grad[#05]
                    // 06 const eps
                    // 07 add   (#05, #06)   grad[#07]
                    // 08 sqrt  (#07)        grad[#08]
                    // 09 div   (#01,#08)    grad[#09]
                    // 10 scale (#00,#09)    grad[#10]
                    //
                    // backward pass, given grad[#10]
                    // #10: scale
                    // grad[#00] += scale(grad[#10],#09)
                    // grad[#09] += sum(mul(grad[#10],#00))
                    // #09: div
                    // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
                    // #08: sqrt
                    // grad[#07] += mul(grad[#08], div(0.5, #08))
                    // #07: add
                    // grad[#05] += grad[#07]
                    // #05: scale
                    // grad[#03] += scale(grad[#05],#04)
                    // #03: sum
                    // grad[#02] += repeat(grad[#03], #02)
                    // #02:
                    // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
                    //
                    // substitute and simplify:
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#02] = repeat(grad[#03], #02)
                    // grad[#02] = repeat(scale(grad[#05],#04), #02)
                    // grad[#02] = repeat(scale(grad[#07],#04), #02)
                    // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
                    // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N))), 2.0)
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps)))
                    // grad[#00] = scale(grad(#10), #09) + scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum(mul(dz,x)) * div(-1,rms*N*mean_eps))
                    // grad[#00] = scale(dz, rrms) + scale(x, sum_xdz * div(-1,rms*N*mean_eps))
                    // a = b*c + d*e
                    // a = b*c*f/f + d*e*f/f
                    // a = (b*c*f + d*e*f)*(1/f)
                    // a = (b*c*(1/c) + d*e*(1/c))*(1/(1/c))
                    // a = (b + d*e/c)*c
                    // b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps)
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms
                    // a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
                    // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
                    // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
                    // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
                    // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
                    // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
                    // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                    // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                }
                // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
                // post-order:
                // dx := x
                // dx := scale(dx,-mean_xdz/mean_eps)
                // dx := add(dx, dz)
                // dx := scale(dx, rrms)
                దశాంశం * dx = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                స్థిరరాశి దశాంశం scale_x = (దశాంశం) (-కూడిక_xdz) / కూడిక_eps;
                కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_rms_norm_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_వెనుక_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

స్థిర శూన్య ggml_compute_forward_సమూహం_norm_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params + 1, పరిమాణం(దశాంశం));

    పూర్ణాంకం n_channels = src0->ne[2];
    పూర్ణాంకం n_groups = dst->op_params[0];
    పూర్ణాంకం n_channels_చొప్పున_సమూహం = (n_channels + n_groups - 1) / n_groups;
    కొరకు (పూర్ణాంకం i = ith; i < n_groups; i += nth) {
        పూర్ణాంకం start = i * n_channels_చొప్పున_సమూహం;
        పూర్ణాంకం end = start + n_channels_చొప్పున_సమూహం;
        ఒకవేళ (end > n_channels) {
            end = n_channels;
        }
        పూర్ణాంకం అడుగు = end - start;

        కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float కూడిక = 0.0;
            కొరకు (int64_t i02 = start; i02 < end; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    స్థిరరాశి దశాంశం * x = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    కూడిక += sumr;
                }
            }
            స్థిరరాశి దశాంశం సగటు_2 = కూడిక / (ne00 * ne01 * అడుగు);

            ggml_float sum2 = 0.0;
            కొరకు (int64_t i02 = start; i02 < end; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    స్థిరరాశి దశాంశం * x = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    దశాంశం * y = (దశాంశం *)((అక్షరం *) dst->దత్తాంశం + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        దశాంశం v = x[i00] - సగటు_2;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            స్థిరరాశి దశాంశం variance = sum2 / (ne00 * ne01 * అడుగు);
            స్థిరరాశి దశాంశం scale = 1.0f / sqrtf(variance + eps);

            కొరకు (int64_t i02 = start; i02 < end; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    దశాంశం * y = (దశాంశం *)((అక్షరం *) dst->దత్తాంశం + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_సమూహం_norm(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_సమూహం_norm_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

స్థిర శూన్య ggml_compute_forward_l2_norm_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                స్థిరరాశి అక్షరం * x = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float కూడిక = 0.0;
                కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                    స్థిరరాశి దశాంశం xi = *(స్థిరరాశి దశాంశం *) (x + i00*nb00);
                    కూడిక += (ggml_float)(xi * xi);
                }

                స్థిరరాశి దశాంశం scale = 1.0f/fmaxf(sqrtf(కూడిక), eps);

                అక్షరం * y = (అక్షరం *) dst->దత్తాంశం + i01*nb1 + i02*nb2 + i03*nb3;

                ఒకవేళ (nb00 == పరిమాణం(దశాంశం) && nb0 == పరిమాణం(దశాంశం)) {
                    జ్ఞాపకం_నకలు(y, x, ne00 * పరిమాణం(దశాంశం));
                    ggml_vec_scale_f32(ne00, (దశాంశం *) y, scale);
                } లేకపోతే {
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        స్థిరరాశి దశాంశం xi = *(స్థిరరాశి దశాంశం *) (x + i00*nb00);
                        *(దశాంశం *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_l2_norm(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

స్థిర శూన్య ggml_compute_forward_బహిర్గతం_prod_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ఒకవేళ (ith == 0) {
        ggml_vec_సమితి_f32(ne0*ne1*ne2*ne3, (దశాంశం *)dst->దత్తాంశం, 0);
    }
    ggml_barrier(params->threadpool);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    // parallelize by last three dimensions

    // total rows in dst
    స్థిరరాశి int64_t nr = ne1*ne2*ne3;

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    స్థిరరాశి int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    స్థిరరాశి int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    స్థిరరాశి int64_t dps2 = ne2 / ne02;
    స్థిరరాశి int64_t dps3 = ne3 / ne03;

    కొరకు (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        స్థిరరాశి int64_t bir1 = MIN(bir + blck_1, ir1);
        కొరకు (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            స్థిరరాశి int64_t bne01 = MIN(bi01 + blck_0, ne01);
            కొరకు (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                స్థిరరాశి int64_t i3 = ir/(ne2*ne1);
                స్థిరరాశి int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                స్థిరరాశి int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                స్థిరరాశి int64_t i02 = i2 / dps2;
                స్థిరరాశి int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                స్థిరరాశి int64_t i12 = i2;
                స్థిరరాశి int64_t i13 = i3;

#ఒకవేళ GGML_VEC_MAD_UNROLL > 2
                స్థిరరాశి int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                కొరకు (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    స్థిరరాశి int64_t i11 = i01;

                    దశాంశం * s0 = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + (          i01*nb01 + i02*nb02 + i03*nb03));
                    దశాంశం * s1 = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    దశాంశం * d  = (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                కొరకు (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    స్థిరరాశి int64_t i11 = i01;

                    దశాంశం * s0 = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + (          i01*nb01 + i02*nb02 + i03*nb03));
                    దశాంశం * s1 = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    దశాంశం * d  = (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#లేకపోతే
                కొరకు (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    స్థిరరాశి int64_t i11 = i01;

                    దశాంశం * s0 = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + (          i01*nb01 + i02*nb02 + i03*nb03));
                    దశాంశం * s1 = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    దశాంశం * d  = (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#అయితే_ముగింపు
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_బహిర్గతం_prod_q_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి ggml_type type = src0->type;
    ggml_గమ్యం_float_t స్థిరరాశి dequantize_అడ్డువరుస_q = ggml_get_type_traits(type)->గమ్యం_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_కొలత(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    ఒకవేళ (ith == 0) {
        ggml_vec_సమితి_f32(ne0*ne1*ne2*ne3, (దశాంశం *)dst->దత్తాంశం, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    స్థిరరాశి int64_t nr = ne1*ne2*ne3;

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    దశాంశం * wdata = (దశాంశం *) params->wdata + (ne0 + CACHE_పంక్తి_కొలత_F32) * ith;

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        స్థిరరాశి int64_t i3 = ir/(ne2*ne1);
        స్థిరరాశి int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        స్థిరరాశి int64_t i02 = i2;
        స్థిరరాశి int64_t i03 = i3;

        //const int64_t i10 = i1;
        స్థిరరాశి int64_t i12 = i2;
        స్థిరరాశి int64_t i13 = i3;

        కొరకు (int64_t i01 = 0; i01 < ne01; ++i01) {
            స్థిరరాశి int64_t i11 = i01;

            దశాంశం * s0 = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + (          i01*nb01 + i02*nb02 + i03*nb03));
            దశాంశం * s1 = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            దశాంశం * d  = (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_అడ్డువరుస_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_బహిర్గతం_prod_f16_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb0  == పరిమాణం(దశాంశం));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ఒకవేళ (ith == 0) {
        ggml_vec_సమితి_f32(ne0*ne1*ne2*ne3, (దశాంశం *)dst->దత్తాంశం, 0);
    }
    ggml_barrier(params->threadpool);

    స్థిరరాశి int64_t nr = ne1*ne2*ne3;
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    దశాంశం * wdata = (దశాంశం *) params->wdata + (ne0 + CACHE_పంక్తి_కొలత_F32) * ith;

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i3 = ir/(ne2*ne1);
        స్థిరరాశి int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        స్థిరరాశి int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        స్థిరరాశి int64_t i02 = i2;
        స్థిరరాశి int64_t i03 = i3;

        స్థిరరాశి int64_t i12 = i2;
        స్థిరరాశి int64_t i13 = i3;

        దశాంశం * d = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (i1*nb1 + i2*nb2 + i3*nb3));

        కొరకు (int64_t i01 = 0; i01 < ne01; ++i01) {
            స్థిరరాశి int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + (i01*nb01 + i02*nb02 + i03*nb03));
            దశాంశం * s1 = (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_గమ్యం_fp32_అడ్డువరుస(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

శూన్య ggml_compute_forward_బహిర్గతం_prod(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_బహిర్గతం_prod_q_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_బహిర్గతం_prod_f16_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_బహిర్గతం_prod_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

స్థిర శూన్య ggml_compute_forward_scale_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    దశాంశం s; // scale factor
    దశాంశం b; // bias

    జ్ఞాపకం_నకలు(&s, (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&b, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి size_t nb01 = src0->nb[1];

    స్థిరరాశి size_t nb1 = dst->nb[1];

    ఒకవేళ (b == 0.0f) {
        కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
            ఒకవేళ (dst->దత్తాంశం != src0->దత్తాంశం) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                జ్ఞాపకం_నకలు((అక్షరం *)dst->దత్తాంశం + i1*nb1, (అక్షరం *)src0->దత్తాంశం + i1*nb01, nc * పరిమాణం(దశాంశం));
            }
            ggml_vec_scale_f32(nc, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i1*nb1), s);
        }
    } లేకపోతే {
        కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం  + i1*nb1),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i1*nb1),
                s, b);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_scale_bf16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    దశాంశం s; // scale factor
    దశాంశం b; // bias

    జ్ఞాపకం_నకలు(&s, (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&b, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        స్థిరరాశి ggml_bf16_t * x = (స్థిరరాశి ggml_bf16_t *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((అక్షరం *)       dst->దత్తాంశం  + i1*dst->nb[1]);
        కొరకు (పూర్ణాంకం k = 0; k < nc; k++) {
            y[k] = GGML_FP32_గమ్యం_BF16(s*GGML_BF16_గమ్యం_FP32(x[k]) + b);
        }
    }
}

శూన్య ggml_compute_forward_scale(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

స్థిర శూన్య ggml_compute_forward_సమితి_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t స్థానభ్రంశం  = ((int32_t *) dst->op_params)[3];
    తార్కిక   inplace = (తార్కిక) ((int32_t *) dst->op_params)[4];

    ఒకవేళ (!inplace) {
        ఒకవేళ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            జ్ఞాపకం_నకలు(
                ((అక్షరం *)  dst->దత్తాంశం),
                ((అక్షరం *) src0->దత్తాంశం),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src1);
    స్థిరరాశి పూర్ణాంకం nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    స్థిరరాశి size_t nb0 = ggml_element_కొలత(src0);

    స్థిరరాశి పూర్ణాంకం im0 = (ne10 == 0 ? 0 : ne10-1);
    స్థిరరాశి పూర్ణాంకం im1 = (ne11 == 0 ? 0 : ne11-1);
    స్థిరరాశి పూర్ణాంకం im2 = (ne12 == 0 ? 0 : ne12-1);
    స్థిరరాశి పూర్ణాంకం im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(స్థానభ్రంశం + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne12*ne11);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne12*ne11)/ne11;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i3*nb3  + i2*nb2  + i1*nb1  + స్థానభ్రంశం),
                (దశాంశం *) ((అక్షరం *) src1->దత్తాంశం + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

స్థిర శూన్య ggml_compute_forward_సమితి_i32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t స్థానభ్రంశం  = ((int32_t *) dst->op_params)[3];
    తార్కిక   inplace = (తార్కిక) ((int32_t *) dst->op_params)[4];

    ఒకవేళ (!inplace) {
        ఒకవేళ (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            జ్ఞాపకం_నకలు(
                ((అక్షరం *)  dst->దత్తాంశం),
                ((అక్షరం *) src0->దత్తాంశం),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src1);
    స్థిరరాశి పూర్ణాంకం nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    స్థిరరాశి size_t nb0 = ggml_element_కొలత(src0);

    స్థిరరాశి పూర్ణాంకం im0 = (ne10 == 0 ? 0 : ne10-1);
    స్థిరరాశి పూర్ణాంకం im1 = (ne11 == 0 ? 0 : ne11-1);
    స్థిరరాశి పూర్ణాంకం im2 = (ne12 == 0 ? 0 : ne12-1);
    స్థిరరాశి పూర్ణాంకం im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(స్థానభ్రంశం + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == పరిమాణం(int32_t));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        స్థిరరాశి పూర్ణాంకం i3 = ir/(ne12*ne11);
        స్థిరరాశి పూర్ణాంకం i2 = (ir - i3*ne12*ne11)/ne11;
        స్థిరరాశి పూర్ణాంకం i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((అక్షరం *)  dst->దత్తాంశం + i3*nb3  + i2*nb2  + i1*nb1  + స్థానభ్రంశం),
                (int32_t *) ((అక్షరం *) src1->దత్తాంశం + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

శూన్య ggml_compute_forward_సమితి(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_సమితి_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_I32:
            {
                ggml_compute_forward_సమితి_i32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q8_1:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

శూన్య ggml_compute_forward_cpy(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

శూన్య ggml_compute_forward_cont(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

స్థిర శూన్య ggml_compute_forward_get_అడ్డువరుసలు_q(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int64_t nc = ne00;
    స్థిరరాశి int64_t nr = ggml_nelements(src1);

    స్థిరరాశి ggml_type type = src0->type;
    ggml_గమ్యం_float_t స్థిరరాశి dequantize_అడ్డువరుస_q = ggml_get_type_traits(type)->గమ్యం_float;

    ప్రతిజ్ఞా(ne0  == nc);
    ప్రతిజ్ఞా(ne02 == ne11);
    ప్రతిజ్ఞా(nb00 == ggml_type_కొలత(type));
    ప్రతిజ్ఞా(ggml_nrows(dst) == nr);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి int64_t i12 = i/(ne11*ne10);
        స్థిరరాశి int64_t i11 = (i - i12*ne11*ne10)/ne10;
        స్థిరరాశి int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        స్థిరరాశి int64_t i01 = *(int32_t *) ((అక్షరం *) src1->దత్తాంశం + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_అడ్డువరుస_q(
                (స్థిరరాశి శూన్య *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i11*nb02 + i12*nb03),
                     (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

స్థిర శూన్య ggml_compute_forward_get_అడ్డువరుసలు_f16(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int64_t nc = ne00;
    స్థిరరాశి int64_t nr = ggml_nelements(src1);

    ప్రతిజ్ఞా(ne0  == nc);
    ప్రతిజ్ఞా(ne02 == ne11);
    ప్రతిజ్ఞా(nb00 == పరిమాణం(ggml_fp16_t));
    ప్రతిజ్ఞా(ggml_nrows(dst) == nr);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి int64_t i12 = i/(ne11*ne10);
        స్థిరరాశి int64_t i11 = (i - i12*ne11*ne10)/ne10;
        స్థిరరాశి int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        స్థిరరాశి int64_t i01 = *(int32_t *) ((అక్షరం *) src1->దత్తాంశం + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_గమ్యం_fp32(
            (స్థిరరాశి ggml_fp16_t*) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i11*nb02 + i12*nb03),
                       (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

స్థిర శూన్య ggml_compute_forward_get_అడ్డువరుసలు_bf16(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int64_t nc = ne00;
    స్థిరరాశి int64_t nr = ggml_nelements(src1);

    ప్రతిజ్ఞా(ne0  == nc);
    ప్రతిజ్ఞా(ne02 == ne11);
    ప్రతిజ్ఞా(nb00 == పరిమాణం(ggml_bf16_t));
    ప్రతిజ్ఞా(ggml_nrows(dst) == nr);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి int64_t i12 = i/(ne11*ne10);
        స్థిరరాశి int64_t i11 = (i - i12*ne11*ne10)/ne10;
        స్థిరరాశి int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        స్థిరరాశి int64_t i01 = *(int32_t *) ((అక్షరం *) src1->దత్తాంశం + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_గమ్యం_fp32(
            (స్థిరరాశి ggml_bf16_t *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i11*nb02 + i12*nb03),
                        (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

స్థిర శూన్య ggml_compute_forward_get_అడ్డువరుసలు_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int64_t nc = ne00;
    స్థిరరాశి int64_t nr = ggml_nelements(src1);

    ప్రతిజ్ఞా(ne0  == nc);
    ప్రతిజ్ఞా(ne02 == ne11);
    ప్రతిజ్ఞా(nb00 == పరిమాణం(దశాంశం));
    ప్రతిజ్ఞా(ggml_nrows(dst) == nr);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t i = ir0; i < ir1; ++i) {
        స్థిరరాశి int64_t i12 = i/(ne11*ne10);
        స్థిరరాశి int64_t i11 = (i - i12*ne11*ne10)/ne10;
        స్థిరరాశి int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        స్థిరరాశి int64_t i01 = *(int32_t *) ((అక్షరం *) src1->దత్తాంశం + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i10*nb1  + i11*nb2  + i12*nb3),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

శూన్య ggml_compute_forward_get_అడ్డువరుసలు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q8_1:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_q(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_bf16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
        స్థితి GGML_TYPE_I32:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

మూస<రకం_పేరు src_t, రకం_పేరు idx_t>
స్థిర శూన్య ggml_compute_forward_సమితి_అడ్డువరుసలు_impl(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int64_t nc = ne00;
    స్థిరరాశి int64_t nr = ne01;

    ప్రతిజ్ఞా(ne0  == nc);
    ప్రతిజ్ఞా(ne2  == ne02);
    ప్రతిజ్ఞా(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    ప్రతిజ్ఞా(ne02 % ne11 == 0);
    ప్రతిజ్ఞా(ne03 % ne12 == 0);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = std::min(ir0 + dr, nr);

    స్థిరరాశి size_t rs = ggml_అడ్డువరుస_కొలత(src0->type, nc);

    ggml_from_float_t స్థిరరాశి from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    కొరకు (int64_t i03 = 0; i03 < ne03; ++i03) {
        కొరకు (int64_t i02 = 0; i02 < ne02; ++i02) {
            కొరకు (int64_t i = ir0; i < ir1; ++i) {
                స్థిరరాశి int64_t i12 = i03%ne12;
                స్థిరరాశి int64_t i11 = i02%ne11;
                స్థిరరాశి int64_t i10 = i;

                స్థిరరాశి int64_t i1 = *(idx_t *) ((అక్షరం *) src1->దత్తాంశం + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                ఒకవేళ constexpr (std::is_same_v<src_t, దశాంశం>) {
                    from_float(
                            (స్థిరరాశి దశాంశం *) ((అక్షరం *) src0->దత్తాంశం +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((అక్షరం *)  dst->దత్తాంశం + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } లేకపోతే ఒకవేళ constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    ఒకవేళ (dst->type == GGML_TYPE_F16) {
                        జ్ఞాపకం_నకలు(
                                            ((అక్షరం *)  dst->దత్తాంశం + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((అక్షరం *) src0->దత్తాంశం +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } లేకపోతే {
                        దశాంశం * wdata = (దశాంశం *) params->wdata + (nc + CACHE_పంక్తి_కొలత_F32) * ith;
                        ggml_fp16_గమ్యం_fp32_అడ్డువరుస(
                                (స్థిరరాశి ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((అక్షరం *) dst->దత్తాంశం + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } లేకపోతే {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_సమితి_అడ్డువరుసలు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ఒకవేళ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_సమితి_అడ్డువరుసలు_impl<దశాంశం, int64_t>(params, dst);
                } లేకపోతే ఒకవేళ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_సమితి_అడ్డువరుసలు_impl<దశాంశం, int32_t>(params, dst);
                } లేకపోతే {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ఒకవేళ (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_సమితి_అడ్డువరుసలు_impl<ggml_fp16_t, int64_t>(params, dst);
                } లేకపోతే ఒకవేళ (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_సమితి_అడ్డువరుసలు_impl<ggml_fp16_t, int32_t>(params, dst);
                } లేకపోతే {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

స్థిర శూన్య ggml_compute_forward_get_అడ్డువరుసలు_వెనుక_f32_f16(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->దత్తాంశం, 0, ggml_nbytes(dst));

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == పరిమాణం(ggml_fp16_t));

    కొరకు (పూర్ణాంకం i = 0; i < nr; ++i) {
        స్థిరరాశి పూర్ణాంకం r = ((int32_t *) src1->దత్తాంశం)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        కొరకు (పూర్ణాంకం j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i*src0->nb[1]))[j];
            ((దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + r*dst->nb[1]))[j] += GGML_CPU_FP16_గమ్యం_FP32(v);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_get_అడ్డువరుసలు_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->దత్తాంశం, 0, ggml_nbytes(dst));

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    కొరకు (పూర్ణాంకం i = 0; i < nr; ++i) {
        స్థిరరాశి పూర్ణాంకం r = ((int32_t *) src1->దత్తాంశం)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_కూడు_f32(nc,
                (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + r*dst->nb[1]),
                (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + r*dst->nb[1]),
                (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i*src0->nb[1]));
    }
}

శూన్య ggml_compute_forward_get_అడ్డువరుసలు_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_వెనుక_f32_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_get_అడ్డువరుసలు_వెనుక_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// ggml_compute_forward_diag

స్థిర శూన్య ggml_compute_forward_diag_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0  == పరిమాణం(దశాంశం));

    కొరకు (పూర్ణాంకం i3 = 0; i3 < ne3; i3++) {
        కొరకు (పూర్ణాంకం i2 = 0; i2 < ne2; i2++) {
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ne1; i1++) {
                దశాంశం * d = (దశాంశం *)((అక్షరం *)  dst->దత్తాంశం + i3*nb3  + i2*nb2 + i1*nb1);
                దశాంశం * s = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02);
                కొరకు (పూర్ణాంకం i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                కొరకు (పూర్ణాంకం i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_diag(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

స్థిర శూన్య ggml_compute_forward_diag_ముసుగు_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst,
        స్థిరరాశి దశాంశం value) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం  n_past  = ((int32_t *) dst->op_params)[0];
    స్థిరరాశి తార్కిక inplace = src0->దత్తాంశం == dst->దత్తాంశం;

    GGML_ASSERT(n_past >= 0);

    ఒకవేళ (!inplace) {
        ఒకవేళ (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            జ్ఞాపకం_నకలు(
                ((అక్షరం *)  dst->దత్తాంశం),
                ((అక్షరం *) src0->దత్తాంశం),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    స్థిరరాశి పూర్ణాంకం n  = ggml_nrows(src0);
    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = src0->ne[1];
    స్థిరరాశి పూర్ణాంకం nz = n/nr;

    GGML_ASSERT( dst->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    కొరకు (పూర్ణాంకం k = 0; k < nz; k++) {
        కొరకు (పూర్ణాంకం j = ith; j < nr; j += nth) {
            కొరకు (పూర్ణాంకం i = n_past; i < nc; i++) {
                ఒకవేళ (i > n_past + j) {
                    *(దశాంశం *)((అక్షరం *) dst->దత్తాంశం + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_diag_ముసుగు_inf(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_ముసుగు_f32(params, dst, -INFINITY);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

శూన్య ggml_compute_forward_diag_ముసుగు_zero(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_ముసుగు_f32(params, dst, 0);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

స్థిర శూన్య ggml_compute_forward_soft_max_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];

    ప్రతిజ్ఞా(ggml_is_contiguous(dst));
    ప్రతిజ్ఞా(ggml_are_same_shape(src0, dst));

    దశాంశం scale    = 1.0f;
    దశాంశం max_bias = 0.0f;

    జ్ఞాపకం_నకలు(&scale,    (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి int64_t nb11 = src1 ? src1->nb[1] : 1;
    స్థిరరాశి int64_t nb12 = src1 ? src1->nb[2] : 1;
    స్థిరరాశి int64_t nb13 = src1 ? src1->nb[3] : 1;

    స్థిరరాశి int64_t ne12 = src1 ? src1->ne[2] : 1;
    స్థిరరాశి int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    స్థిరరాశి uint32_t n_తల      = ne02;
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floor(log2(n_తల));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    దశాంశం * wp = (దశాంశం *) params->wdata + (ne00 + CACHE_పంక్తి_కొలత_F32) * ith;

    స్థిరరాశి తార్కిక use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    స్థిరరాశి దశాంశం * sk = src2 ? (దశాంశం *)((అక్షరం *) src2->దత్తాంశం) : శూన్యసూచిక;

    కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
        కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
            కొరకు (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                స్థిరరాశి int64_t i11 = i01;
                స్థిరరాశి int64_t i12 = i02%ne12;
                స్థిరరాశి int64_t i13 = i03%ne13;

                // ALiBi
                స్థిరరాశి uint32_t h = i02; //తల
                స్థిరరాశి దశాంశం వాలు = (max_bias > 0.0f) ? h < n_తల_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_తల_log2) + 1) : 1.0f;

                దశాంశం * sp = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i01*nb01 + i02*nb02 + i03*nb03);
                దశాంశం * dp = (దశాంశం *)((అక్షరం *)  dst->దత్తాంశం + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((అక్షరం *) src1->దత్తాంశం + i11*nb11 + i12*nb12 + i13*nb13) : ఖాళీ;
                దశాంశం       * mp_f32 = src1 ? (దశాంశం       *)((అక్షరం *) src1->దత్తాంశం + i11*nb11 + i12*nb12 + i13*nb13) : ఖాళీ;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                ఒకవేళ (mp_f32) {
                    ఒకవేళ (use_f16) {
                        కొరకు (పూర్ణాంకం i = 0; i < ne00; ++i) {
                            wp[i] += వాలు*GGML_CPU_FP16_గమ్యం_FP32(mp_f16[i]);
                        }
                    } లేకపోతే {
                        కొరకు (పూర్ణాంకం i = 0; i < ne00; ++i) {
                            wp[i] += వాలు*mp_f32[i];
                        }
                    }
                }

#నిర్వచితం_కాకపోతే NDEBUG
                కొరకు (పూర్ణాంకం i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    ప్రతిజ్ఞా(!isnan(wp[i]));
                }
#అయితే_ముగింపు // NDEBUG

                దశాంశం max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                ఒకవేళ (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float కూడిక = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                ప్రతిజ్ఞా(కూడిక > 0.0);

                ఒకవేళ (sk) {
                    కూడిక += (ggml_float) expf(sk[i02] - max);
                }

                కూడిక = 1.0/కూడిక;
                ggml_vec_scale_f32(ne00, dp, కూడిక);

#నిర్వచితం_కాకపోతే NDEBUG
                కొరకు (పూర్ణాంకం i = 0; i < ne00; ++i) {
                    ప్రతిజ్ఞా(!isnan(dp[i]));
                    ప్రతిజ్ఞా(!isinf(dp[i]));
                }
#అయితే_ముగింపు // NDEBUG
            }
        }
    }
}

శూన్య ggml_compute_forward_soft_max(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

స్థిర శూన్య ggml_compute_forward_soft_max_ext_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    దశాంశం scale    = 1.0f;
    దశాంశం max_bias = 0.0f;

    జ్ఞాపకం_నకలు(&scale,    (స్థిరరాశి దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];
    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం *dy = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i1*src0->nb[1]);
        దశాంశం *y  = (దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i1*src1->nb[1]);
        దశాంశం *dx = (దశాంశం *)((అక్షరం *) dst->దత్తాంశం  + i1*dst->nb[1]);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ప్రతిజ్ఞా(!isnan(dy[i]));
            ప్రతిజ్ఞా(!isnan(y[i]));
        }
#అయితే_ముగింపు // NDEBUG
        // Jii = yi - yi*yi
        // Jij = -yi*yj
        // J = diag(y)-y.T*y
        // dx = J * dy
        // dxk = sum_i(Jki * dyi)
        // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
        // dxk = sum_i(-yk*yi * dyi) + yk*dyk
        // dxk = -yk * sum_i(yi * dyi) + yk*dyk
        // dxk = -yk * dot(y, dy) + yk*dyk
        // dxk = yk * (- dot(y, dy) + dyk)
        // dxk = yk * (dyk - dot(y, dy))
        //
        // post-order:
        // dot_y_dy := dot(y, dy)
        // dx := dy
        // dx := dx - dot_y_dy
        // dx := dx * y

        // linear runtime, no additional memory
        దశాంశం dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        కొరకు (పూర్ణాంకం i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (పూర్ణాంకం i = 0; i < nc; ++i) {
            ప్రతిజ్ఞా(!isnan(dx[i]));
            ప్రతిజ్ఞా(!isinf(dx[i]));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

శూన్య ggml_compute_forward_soft_max_ext_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_వెనుక_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

స్థిర శూన్య ggml_compute_forward_clamp_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    దశాంశం min;
    దశాంశం max;
    జ్ఞాపకం_నకలు(&min, (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం n  = ggml_nrows(src0);
    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    కొరకు (పూర్ణాంకం j = ith; j < n; j += nth) {
        స్థిరరాశి int64_t i1 = j % ne01;
        స్థిరరాశి int64_t i2 = (j / ne01) % ne02;
        స్థిరరాశి int64_t i3 = j / (ne01*ne02);

        దశాంశం * dst_ptr  = (దశాంశం *) ((అక్షరం *)  dst->దత్తాంశం + i1*nb1  + i2*nb2  + i3*nb3);
        దశాంశం * src0_ptr = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i1*nb01 + i2*nb02 + i3*nb03);

        కొరకు (పూర్ణాంకం i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_clamp_f16(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    దశాంశం min;
    దశాంశం max;
    జ్ఞాపకం_నకలు(&min, (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max, (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం n  = ggml_nrows(src0);
    స్థిరరాశి పూర్ణాంకం nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb00 == పరిమాణం(ggml_fp16_t));

    కొరకు (పూర్ణాంకం j = ith; j < n; j += nth) {
        స్థిరరాశి int64_t i1 = j % ne01;
        స్థిరరాశి int64_t i2 = (j / ne01) % ne02;
        స్థిరరాశి int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((అక్షరం *)  dst->దత్తాంశం + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((అక్షరం *) src0->దత్తాంశం + i1*nb01 + i2*nb02 + i3*nb03);

        కొరకు (పూర్ణాంకం i = 0; i < nc; i++) {
            దశాంశం v = GGML_CPU_FP16_గమ్యం_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_గమ్యం_FP16(MAX(MIN(v, max), min));
        }
    }
}

శూన్య ggml_compute_forward_clamp(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_BF16:
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q2_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q8_1:
        స్థితి GGML_TYPE_MXFP4:
        స్థితి GGML_TYPE_NVFP4:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_TQ1_0:
        స్థితి GGML_TYPE_TQ2_0:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ4_NL:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ3_S:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_Q8_K:
        స్థితి GGML_TYPE_I8:
        స్థితి GGML_TYPE_I16:
        స్థితి GGML_TYPE_I32:
        స్థితి GGML_TYPE_I64:
        స్థితి GGML_TYPE_F64:
        స్థితి GGML_TYPE_లెక్క:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

స్థిర దశాంశం rope_yarn_ramp(స్థిరరాశి దశాంశం దిగువ, స్థిరరాశి దశాంశం ఎగువ, స్థిరరాశి పూర్ణాంకం i0) {
    స్థిరరాశి దశాంశం y = (i0 / 2 - దిగువ) / MAX(0.001f, ఎగువ - దిగువ);
    తిరిగి 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
స్థిర శూన్య rope_yarn(
    దశాంశం theta_extrap, దశాంశం freq_scale, దశాంశం corr_dims[2], int64_t i0, దశాంశం ext_గుణకం, దశాంశం mscale,
    దశాంశం * cos_theta, దశాంశం * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    దశాంశం theta_interp = freq_scale * theta_extrap;
    దశాంశం theta = theta_interp;
    ఒకవేళ (ext_గుణకం != 0.0f) {
        దశాంశం ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_గుణకం;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

స్థిర శూన్య ggml_rope_cache_init(
     దశాంశం theta_పునాది, దశాంశం freq_scale, స్థిరరాశి దశాంశం * freq_factors, దశాంశం corr_dims[2], int64_t ne0, దశాంశం ext_గుణకం, దశాంశం mscale,
     దశాంశం * cache, దశాంశం sin_sign, దశాంశం theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    దశాంశం theta = theta_పునాది;
    కొరకు (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

స్థిర శూన్య ggml_mrope_cache_init(
     దశాంశం theta_పునాది_t, దశాంశం theta_పునాది_h, దశాంశం theta_పునాది_w, దశాంశం theta_పునాది_e, పూర్ణాంకం sections[4], తార్కిక is_imrope, తార్కిక indep_sects,
     దశాంశం freq_scale, స్థిరరాశి దశాంశం * freq_factors, దశాంశం corr_dims[2], int64_t ne0, దశాంశం ext_గుణకం, దశాంశం mscale,
     దశాంశం * cache, దశాంశం sin_sign, దశాంశం theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    దశాంశం theta_t = theta_పునాది_t;
    దశాంశం theta_h = theta_పునాది_h;
    దశాంశం theta_w = theta_పునాది_w;
    దశాంశం theta_e = theta_పునాది_e;  // extra position id for vision encoder
    పూర్ణాంకం sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    పూర్ణాంకం sec_w = sections[1] + sections[0];
    పూర్ణాంకం sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    కొరకు (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        పూర్ణాంకం sector = (i0 / 2) % sect_dims;
        ఒకవేళ (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            ఒకవేళ (sector == 0) {
                theta_t = theta_పునాది_t;
            }
            లేకపోతే ఒకవేళ (sector == sections[0]) {
                theta_h = theta_పునాది_h;;
            }
            లేకపోతే ఒకవేళ (sector == sec_w) {
                theta_w = theta_పునాది_w;
            }
            లేకపోతే ఒకవేళ (sector == sec_e) {
                theta_e = theta_పునాది_e;
            }
        }

        దశాంశం theta = theta_t;
        ఒకవేళ (is_imrope) { // qwen3vl apply interleaved mrope
            ఒకవేళ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } లేకపోతే ఒకవేళ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } లేకపోతే ఒకవేళ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } లేకపోతే {
                theta = theta_e;
            }
        } లేకపోతే {
            ఒకవేళ (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            లేకపోతే ఒకవేళ (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            లేకపోతే ఒకవేళ (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


మూస<రకం_పేరు T>
స్థిర శూన్య rotate_జతలు(స్థిరరాశి int64_t n, స్థిరరాశి int64_t n_స్థానభ్రంశం, స్థిరరాశి దశాంశం * cache, స్థిరరాశి T * src_దత్తాంశం, T * dst_దత్తాంశం, స్థిరరాశి పూర్ణాంకం scale = 2) {
  కొరకు (int64_t i0 = 0; i0 < n; i0 += 2) {
    స్థిరరాశి int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    స్థిరరాశి దశాంశం cos_theta = cache[i0 + 0];
    స్థిరరాశి దశాంశం sin_theta = cache[i0 + 1];

    స్థిరరాశి T * స్థిరరాశి src = src_దత్తాంశం + ic;
    T * dst             = dst_దత్తాంశం + ic;

    స్థిరరాశి దశాంశం x0 = type_conversion_పట్టిక<T>::గమ్యం_f32(src[0]);
    స్థిరరాశి దశాంశం x1 = type_conversion_పట్టిక<T>::గమ్యం_f32(src[n_స్థానభ్రంశం]);

    dst[0]        = type_conversion_పట్టిక<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_స్థానభ్రంశం] = type_conversion_పట్టిక<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

మూస<రకం_పేరు T> //float or ggml_fp16_t
స్థిర శూన్య ggml_compute_forward_rope_flt(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst,
        స్థిరరాశి తార్కిక forward) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    దశాంశం freq_పునాది, freq_scale, ext_గుణకం, attn_గుణకం, beta_fast, beta_slow;
    పూర్ణాంకం sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    స్థిరరాశి పూర్ణాంకం n_dims     = ((int32_t *) dst->op_params)[1];
    స్థిరరాశి పూర్ణాంకం mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    స్థిరరాశి పూర్ణాంకం n_ctx_orig = ((int32_t *) dst->op_params)[4];

    జ్ఞాపకం_నకలు(&freq_పునాది,   (int32_t *) dst->op_params +  5, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&freq_scale,  (int32_t *) dst->op_params +  6, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&ext_గుణకం,  (int32_t *) dst->op_params +  7, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&attn_గుణకం, (int32_t *) dst->op_params +  8, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_fast,   (int32_t *) dst->op_params +  9, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_slow,   (int32_t *) dst->op_params + 10, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&sections,    (int32_t *) dst->op_params + 11, పరిమాణం(పూర్ణాంకం)*4);

    స్థిరరాశి పూర్ణాంకం n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == పరిమాణం(T));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    పూర్ణాంకం ir = 0;

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f/n_dims);

    దశాంశం corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast, beta_slow, corr_dims);

    స్థిరరాశి తార్కిక is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    స్థిరరాశి తార్కిక mrope_వాడిన = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    స్థిరరాశి తార్కిక is_vision = mode == GGML_ROPE_TYPE_VISION;

    ఒకవేళ (mrope_వాడిన) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    ఒకవేళ (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    స్థిరరాశి దశాంశం * freq_factors = ఖాళీ;
    ఒకవేళ (src2 != ఖాళీ) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (స్థిరరాశి దశాంశం *) src2->దత్తాంశం;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    స్థిరరాశి దశాంశం sin_sign = forward ? 1.0f : -1.0f;

    స్థిరరాశి int32_t * pos = (స్థిరరాశి int32_t *) src1->దత్తాంశం;

    int64_t చివరి_i2 = -1;

    కొరకు (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        కొరకు (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            కొరకు (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                ఒకవేళ (ir++ < ir0) కొనసాగు; // skip rows mapped to other threads
                ఒకవేళ (ir   > ir1) ఆపు;

                దశాంశం * cache = (దశాంశం *) params->wdata + (ne0 + CACHE_పంక్తి_కొలత_F32)*ith;
                ఒకవేళ (చివరి_i2 != i2) {
                    ఒకవేళ (!mrope_వాడిన) {
                        స్థిరరాశి int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_గుణకం, attn_గుణకం, cache, sin_sign, theta_scale);
                    }
                    లేకపోతే {
                        స్థిరరాశి int64_t p_t = pos[i2];
                        స్థిరరాశి int64_t p_h = pos[i2 + ne2];
                        స్థిరరాశి int64_t p_w = pos[i2 + ne2 * 2];
                        స్థిరరాశి int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_గుణకం, attn_గుణకం, cache, sin_sign, theta_scale);
                    }

                    చివరి_i2 = i2;
                }

                T * src = (T *)((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_దత్తాంశం  = (T *)((అక్షరం *)  dst->దత్తాంశం + i3*nb3  + i2*nb2  + i1*nb1);

                ఎంపిక (mode) {
                    స్థితి GGML_ROPE_TYPE_సాధారణ:
                        rotate_జతలు<T>(n_dims, 1, cache, src + n_offs, dst_దత్తాంశం + n_offs, 1);
                        ఆపు;
                    స్థితి GGML_ROPE_TYPE_NEOX:
                    స్థితి GGML_ROPE_TYPE_MROPE:
                    స్థితి GGML_ROPE_TYPE_IMROPE:
                        rotate_జతలు<T>(n_dims, n_dims/2, cache, src + n_offs, dst_దత్తాంశం + n_offs);
                        ఆపు;
                    స్థితి GGML_ROPE_TYPE_VISION:
                        rotate_జతలు<T>(ne0, n_dims, cache, src, dst_దత్తాంశం);
                        ఆపు;
                    అప్రమేయ:
                        GGML_ABORT("rope type not supported");
                }

                ఒకవేళ (!is_vision) {
                    // fill the remain channels with data from src tensor
                    కొరకు (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        ఒకవేళ (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            కొనసాగు;
                        }
                        స్థిరరాశి T * స్థిరరాశి src = (T *)((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_దత్తాంశం  = (T *)((అక్షరం *)  dst->దత్తాంశం + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_దత్తాంశం[0] = src[0];
                        dst_దత్తాంశం[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

శూన్య ggml_compute_forward_rope(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, నిజం);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<దశాంశం>(params, dst, నిజం);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

శూన్య ggml_compute_forward_rope_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, అబద్ధం);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<దశాంశం>(params, dst, అబద్ధం);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

స్థిర శూన్య ggml_compute_forward_conv_transpose_1d_f16_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == పరిమాణం(ggml_fp16_t));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    ఒకవేళ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * స్థిరరాశి wdata = (ggml_fp16_t *) params->wdata + 0;

            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    స్థిరరాశి ggml_fp16_t * స్థిరరాశి src = (ggml_fp16_t *)((అక్షరం *) src0->దత్తాంశం + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_దత్తాంశం = wdata + i01*ne00*ne02;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_దత్తాంశం[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * స్థిరరాశి wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_దత్తాంశం = wdata;

            కొరకు (int64_t i11 = 0; i11 < ne11; i11++) {
                స్థిరరాశి దశాంశం * స్థిరరాశి src = (దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i11*nb11);
                కొరకు (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_దత్తాంశం[i10*ne11 + i11] = GGML_CPU_FP32_గమ్యం_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->దత్తాంశం, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t*)(dst->op_params))[0];

    // total rows in dst
    స్థిరరాశి పూర్ణాంకం nr = ne1;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * స్థిరరాశి wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * స్థిరరాశి wdata_src = wdata + nk;

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * dst_దత్తాంశం = (దశాంశం *)((అక్షరం *) dst->దత్తాంశం + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        కొరకు (పూర్ణాంకం i10 = 0; i10 < ne10; i10++) {
            స్థిరరాశి పూర్ణాంకం i1n = i10*ne11;
            కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; i00++) {
                దశాంశం v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_దత్తాంశం[i10*s0 + i00] += v;
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_conv_transpose_1d_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    ఒకవేళ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            దశాంశం * స్థిరరాశి wdata = (దశాంశం *) params->wdata + 0;

            కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                    స్థిరరాశి దశాంశం * స్థిరరాశి src = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i02*nb02 + i01*nb01);
                    దశాంశం * dst_దత్తాంశం = wdata + i01*ne00*ne02;
                    కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_దత్తాంశం[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            దశాంశం * స్థిరరాశి wdata = (దశాంశం *) params->wdata + nk;
            దశాంశం * dst_దత్తాంశం = wdata;

            కొరకు (int64_t i11 = 0; i11 < ne11; i11++) {
                స్థిరరాశి దశాంశం * స్థిరరాశి src = (దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i11*nb11);
                కొరకు (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_దత్తాంశం[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->దత్తాంశం, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t*)(dst->op_params))[0];

    // total rows in dst
    స్థిరరాశి పూర్ణాంకం nr = ne1;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    దశాంశం * స్థిరరాశి wdata     = (దశాంశం *) params->wdata + 0;
    దశాంశం * స్థిరరాశి wdata_src = wdata + nk;

    కొరకు (పూర్ణాంకం i1 = ir0; i1 < ir1; i1++) {
        దశాంశం * dst_దత్తాంశం = (దశాంశం *)((అక్షరం *) dst->దత్తాంశం + i1*nb1);
        దశాంశం * wdata_kernel = wdata + i1*ne02*ne00;
        కొరకు (పూర్ణాంకం i10 = 0; i10 < ne10; i10++) {
            స్థిరరాశి పూర్ణాంకం i1n = i10*ne11;
            కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; i00++) {
                దశాంశం v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_దత్తాంశం[i10*s0 + i00] += v;
            }
        }
    }
}

శూన్య ggml_compute_forward_conv_transpose_1d(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
స్థిర శూన్య ggml_compute_forward_im2col_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి తార్కిక is_2D = ((స్థిరరాశి int32_t *)(dst->op_params))[6] == 1;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t N  = is_2D ? ne13 : ne12;
    స్థిరరాశి int64_t IC = is_2D ? ne12 : ne11;
    స్థిరరాశి int64_t IH = is_2D ? ne11 : 1;
    స్థిరరాశి int64_t IW = ne10;

    స్థిరరాశి int64_t KH = is_2D ? ne01 : 1;
    స్థిరరాశి int64_t KW = ne00;

    స్థిరరాశి int64_t OH = is_2D ? ne2 : 1;
    స్థిరరాశి int64_t OW = ne1;

    పూర్ణాంకం ofs0 = is_2D ? nb13 : nb12;
    పూర్ణాంకం ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        దశాంశం * స్థిరరాశి wdata = (దశాంశం *) dst->దత్తాంశం;

        కొరకు (int64_t లోపల = 0; లోపల < N; లోపల++) {
            కొరకు (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                కొరకు (int64_t iow = 0; iow < OW; iow++) {
                    కొరకు (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        దశాంశం * dst_దత్తాంశం = wdata + (లోపల*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        స్థిరరాశి దశాంశం * స్థిరరాశి src_దత్తాంశం = (దశాంశం *)((అక్షరం *) src1->దత్తాంశం + లోపల*ofs0 + iic*ofs1); // [IH, IW]

                        కొరకు (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            కొరకు (int64_t ikw = 0; ikw < KW; ikw++) {
                                స్థిరరాశి int64_t iiw = iow*s0 + ikw*d0 - p0;
                                స్థిరరాశి int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ఒకవేళ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_దత్తాంశం[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } లేకపోతే {
                                    dst_దత్తాంశం[iic*(KH*KW) + ikh*KW + ikw] = (src_దత్తాంశం[iih*IW + iiw]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// ggml_compute_forward_im2col_f16
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
స్థిర శూన్య ggml_compute_forward_im2col_f16(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి తార్కిక is_2D = ((స్థిరరాశి int32_t *)(dst->op_params))[6] == 1;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t N  = is_2D ? ne13 : ne12;
    స్థిరరాశి int64_t IC = is_2D ? ne12 : ne11;
    స్థిరరాశి int64_t IH = is_2D ? ne11 : 1;
    స్థిరరాశి int64_t IW = ne10;

    స్థిరరాశి int64_t KH = is_2D ? ne01 : 1;
    స్థిరరాశి int64_t KW = ne00;

    స్థిరరాశి int64_t OH = is_2D ? ne2 : 1;
    స్థిరరాశి int64_t OW = ne1;

    పూర్ణాంకం ofs0 = is_2D ? nb13 : nb12;
    పూర్ణాంకం ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * స్థిరరాశి wdata = (ggml_fp16_t *) dst->దత్తాంశం;

        కొరకు (int64_t లోపల = 0; లోపల < N; లోపల++) {
            కొరకు (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                కొరకు (int64_t iow = 0; iow < OW; iow++) {
                    కొరకు (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_దత్తాంశం = wdata + (లోపల*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        స్థిరరాశి దశాంశం * స్థిరరాశి src_దత్తాంశం_f32 = src1->type == GGML_TYPE_F32
                            ? (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + లోపల*ofs0 + iic*ofs1)
                            : శూన్యసూచిక; // [IH, IW]
                        స్థిరరాశి ggml_fp16_t * స్థిరరాశి src_దత్తాంశం_f16 = src1->type == GGML_TYPE_F16
                            ? (స్థిరరాశి ggml_fp16_t *)((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + లోపల*ofs0 + iic*ofs1)
                            : శూన్యసూచిక; // [IH, IW]

                        కొరకు (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            కొరకు (int64_t ikw = 0; ikw < KW; ikw++) {
                                స్థిరరాశి int64_t iiw = iow*s0 + ikw*d0 - p0;
                                స్థిరరాశి int64_t iih = ioh*s1 + ikh*d1 - p1;

                                ఒకవేళ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_దత్తాంశం[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } లేకపోతే {
                                    ఒకవేళ (src_దత్తాంశం_f32 != శూన్యసూచిక) {
                                        dst_దత్తాంశం[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_గమ్యం_FP16(src_దత్తాంశం_f32[iih*IW + iiw]);
                                    } లేకపోతే {
                                        dst_దత్తాంశం[iic*(KH*KW) + ikh*KW + ikw] = src_దత్తాంశం_f16[iih*IW + iiw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_im2col(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {
    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

శూన్య ggml_compute_forward_im2col_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    స్థిరరాశి ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి తార్కిక is_2D = ((స్థిరరాశి int32_t *)(dst->op_params))[6] == 1;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t N  = is_2D ? ne3 : ne2;
    స్థిరరాశి int64_t IC = is_2D ? ne2 : ne1;
    స్థిరరాశి int64_t IH = is_2D ? ne1 : 1;
    స్థిరరాశి int64_t IW = ne0;

    స్థిరరాశి int64_t KH = is_2D ? ne11 : 1;
    స్థిరరాశి int64_t KW = ne10;

    స్థిరరాశి int64_t OH = is_2D ? ne02 : 1;
    స్థిరరాశి int64_t OW = ne01;

    పూర్ణాంకం ofs0 = is_2D ? nb3 : nb2;
    పూర్ణాంకం ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == పరిమాణం(దశాంశం));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        దశాంశం * స్థిరరాశి wdata = (దశాంశం *) dst->దత్తాంశం;

        కొరకు (int64_t లోపల = 0; లోపల < N; లోపల++) {
            కొరకు (int64_t iic = ith; iic < IC; iic += nth) {
                కొరకు (int64_t iih = 0; iih < IH; iih++) {
                    కొరకు (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        దశాంశం grad = 0.0f;
                        కొరకు (int64_t ikh = 0; ikh < KH; ikh++) {
                            కొరకు (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                స్థిరరాశి int64_t tmpw = (iiw + p0 - ikw*d0);
                                ఒకవేళ (tmpw % s0 != 0) {
                                    కొనసాగు;
                                }
                                స్థిరరాశి int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                ఒకవేళ (is_2D) {
                                    స్థిరరాశి int64_t tmph = iih + p1 - ikh*d1;

                                    ఒకవేళ (tmph % s1 != 0) {
                                        కొనసాగు;
                                    }

                                    ioh = tmph / s1;
                                } లేకపోతే {
                                    ioh = 0;
                                }

                                ఒకవేళ (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    కొనసాగు;
                                }

                                స్థిరరాశి దశాంశం * స్థిరరాశి grad_లోపల = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం
                                    + (లోపల*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_లోపల[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        దశాంశం * dst_దత్తాంశం = (దశాంశం *)((అక్షరం *) wdata + (లోపల*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_దత్తాంశం[iih*IW + iiw] = grad;
                    }
                }
            }
        }
    }
}


// ggml_compute_forward_im2col_3d_f16
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N*OD, OH, OW, IC * KD * KH * KW]
స్థిర శూన్య ggml_compute_forward_im2col_3d_f16(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t s2 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t p2 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[6];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[7];
    స్థిరరాశి int32_t d2 = ((స్థిరరాశి int32_t *)(dst->op_params))[8];
    స్థిరరాశి int32_t IC = ((స్థిరరాశి int32_t *)(dst->op_params))[9];


    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t N  = ne13 / IC;
    స్థిరరాశి int64_t ID = ne12;
    స్థిరరాశి int64_t IH = ne11;
    స్థిరరాశి int64_t IW = ne10;

    స్థిరరాశి int64_t OC = ne03 / IC;
    GGML_వాడని(OC);
    స్థిరరాశి int64_t KD = ne02;
    స్థిరరాశి int64_t KH = ne01;
    స్థిరరాశి int64_t KW = ne00;

    స్థిరరాశి int64_t OD = ne3 / N;
    స్థిరరాశి int64_t OH = ne2;
    స్థిరరాశి int64_t OW = ne1;
    స్థిరరాశి int64_t OH_OW = OH*OW;
    స్థిరరాశి int64_t KD_KH_KW = KD*KH*KW;
    స్థిరరాశి int64_t KH_KW = KH*KW;
    స్థిరరాశి int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * స్థిరరాశి wdata = (ggml_fp16_t *) dst->దత్తాంశం;

        కొరకు (int64_t లోపల = 0; లోపల < N; లోపల++) {
            కొరకు (int64_t iod = 0; iod < OD; iod++) {
                కొరకు (int64_t ioh = 0; ioh < OH; ioh++) {
                    కొరకు (int64_t iow = 0; iow < OW; iow++) {
                        కొరకు (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_దత్తాంశం = wdata + (లోపల*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            స్థిరరాశి దశాంశం * స్థిరరాశి src_దత్తాంశం = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *)src1->దత్తాంశం + (లోపల*IC + iic)*nb13); // [ID, IH, IW]

                            కొరకు (int64_t ikd = 0; ikd < KD; ikd++) {
                                కొరకు (int64_t ikh = 0; ikh < KH; ikh++) {
                                    కొరకు (int64_t ikw = 0; ikw < KW; ikw++) {
                                        స్థిరరాశి int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        స్థిరరాశి int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        స్థిరరాశి int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ఒకవేళ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_దత్తాంశం[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } లేకపోతే {
                                            స్థిరరాశి దశాంశం * స్థిరరాశి s = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *)src_దత్తాంశం + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_దత్తాంశం[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_గమ్యం_FP16(*s);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ggml_compute_forward_im2col_3d_f32
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N*OD, OH, OW, IC * KD * KH * KW]
స్థిర శూన్య ggml_compute_forward_im2col_3d_f32(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t s2 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t p2 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[6];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[7];
    స్థిరరాశి int32_t d2 = ((స్థిరరాశి int32_t *)(dst->op_params))[8];
    స్థిరరాశి int32_t IC = ((స్థిరరాశి int32_t *)(dst->op_params))[9];


    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t N  = ne13 / IC;
    స్థిరరాశి int64_t ID = ne12;
    స్థిరరాశి int64_t IH = ne11;
    స్థిరరాశి int64_t IW = ne10;

    స్థిరరాశి int64_t OC = ne03 / IC;
    GGML_వాడని(OC);
    స్థిరరాశి int64_t KD = ne02;
    స్థిరరాశి int64_t KH = ne01;
    స్థిరరాశి int64_t KW = ne00;

    స్థిరరాశి int64_t OD = ne3 / N;
    స్థిరరాశి int64_t OH = ne2;
    స్థిరరాశి int64_t OW = ne1;

    స్థిరరాశి int64_t OH_OW = OH*OW;
    స్థిరరాశి int64_t KD_KH_KW = KD*KH*KW;
    స్థిరరాశి int64_t KH_KW = KH*KW;
    స్థిరరాశి int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        దశాంశం * స్థిరరాశి wdata = (దశాంశం *) dst->దత్తాంశం;

        కొరకు (int64_t లోపల = 0; లోపల < N; లోపల++) {
            కొరకు (int64_t iod = 0; iod < OD; iod++) {
                కొరకు (int64_t ioh = 0; ioh < OH; ioh++) {
                    కొరకు (int64_t iow = 0; iow < OW; iow++) {
                        కొరకు (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            దశాంశం * dst_దత్తాంశం = wdata + (లోపల*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            స్థిరరాశి దశాంశం * స్థిరరాశి src_దత్తాంశం = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *)src1->దత్తాంశం + (లోపల*IC + iic)*nb13); // [ID, IH, IW]

                            కొరకు (int64_t ikd = 0; ikd < KD; ikd++) {
                                కొరకు (int64_t ikh = 0; ikh < KH; ikh++) {
                                    కొరకు (int64_t ikw = 0; ikw < KW; ikw++) {
                                        స్థిరరాశి int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        స్థిరరాశి int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        స్థిరరాశి int64_t iid = iod*s2 + ikd*d2 - p2;

                                        ఒకవేళ (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_దత్తాంశం[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } లేకపోతే {
                                            స్థిరరాశి దశాంశం * స్థిరరాశి s = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *)src_దత్తాంశం + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_దత్తాంశం[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


శూన్య ggml_compute_forward_im2col_3d(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {
    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_call_mul_mat(ggml_type type, స్థిరరాశి ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              శూన్య * a, శూన్య * b, దశాంశం * c) {
    స్థిరరాశి ggml_type_traits * traits = ggml_get_type_traits(type);
    నిర్మాణం ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_కొలత;
    src1.nb[1] = k * traits->type_కొలత;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.దత్తాంశం  = a;

    నిర్మాణం ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_కొలత;
    src0.nb[1] = k * traits->type_కొలత;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.దత్తాంశం  = b;

    నిర్మాణం ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = పరిమాణం(దశాంశం);
    dst.nb[1] = n * పరిమాణం(దశాంశం);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.దత్తాంశం  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

స్థిర అంతఃవరుస int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    తిరిగి (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

మూస <రకం_పేరు elem_t>
స్థిర శూన్య ggml_compute_forward_col2im_1d_impl(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t OC = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];

    స్థిరరాశి int64_t K_OC = src->ne[0];
    స్థిరరాశి int64_t T_లోపల = src->ne[1];
    స్థిరరాశి int64_t K    = K_OC / OC;
    స్థిరరాశి int64_t T_బహిర్గతం = dst->ne[0];

    స్థిరరాశి elem_t * నిలువు_దత్తాంశం = (స్థిరరాశి elem_t *) src->దత్తాంశం;
    elem_t       * dst_దత్తాంశం = (elem_t *) dst->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    స్థిరరాశి int64_t dr = (T_బహిర్గతం + nth - 1) / nth;
    స్థిరరాశి int64_t it0 = dr * ith;
    స్థిరరాశి int64_t it1 = it0 + dr < T_బహిర్గతం ? it0 + dr : T_బహిర్గతం;

    కొరకు (int64_t oc = 0; oc < OC; oc++) {
        కొరకు (int64_t t_బహిర్గతం = it0; t_బహిర్గతం < it1; t_బహిర్గతం++) {
            స్థిరరాశి int64_t t_abs = t_బహిర్గతం + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_లోపల_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            ఒకవేళ (t_లోపల_min < 0) t_లోపల_min = 0;
            int64_t t_లోపల_max = t_abs / s0;
            ఒకవేళ (t_లోపల_max >= T_లోపల) t_లోపల_max = T_లోపల - 1;

            దశాంశం కూడిక = 0.0f;
            కొరకు (int64_t t_లోపల = t_లోపల_min; t_లోపల <= t_లోపల_max; t_లోపల++) {
                int64_t k = t_abs - t_లోపల * s0;
                ఒకవేళ (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    కూడిక += type_conversion_పట్టిక<elem_t>::గమ్యం_f32(నిలువు_దత్తాంశం[(oc * K + k) + t_లోపల * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_దత్తాంశం[t_బహిర్గతం + oc * T_బహిర్గతం] = type_conversion_పట్టిక<elem_t>::from_f32(కూడిక);
        }
    }
}

శూన్య ggml_compute_forward_col2im_1d(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ఎంపిక (dst->src[0]->type) {
        స్థితి GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<దశాంశం>      (params, dst); ఆపు;
        స్థితి GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); ఆపు;
        స్థితి GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); ఆపు;
        అప్రమేయ: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


స్థిర శూన్య ggml_compute_forward_conv_2d_impl(స్థిరరాశి ggml_compute_params * params,
                                              స్థిరరాశి ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              స్థిరరాశి ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    స్థిరరాశి ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    స్థిరరాశి int32_t stride_x   = dst->op_params[0];
    స్థిరరాశి int32_t stride_y   = dst->op_params[1];
    స్థిరరాశి int32_t pad_x      = dst->op_params[2];
    స్థిరరాశి int32_t pad_y      = dst->op_params[3];
    స్థిరరాశి int32_t dilation_x = dst->op_params[4];
    స్థిరరాశి int32_t dilation_y = dst->op_params[5];

    స్థిరరాశి int64_t c_లోపల  = src->ne[2];
    స్థిరరాశి int64_t c_బహిర్గతం = kernel->ne[3];
    GGML_ASSERT(c_లోపల == kernel->ne[2]);

    స్థిరరాశి int64_t src_w = src->ne[0];
    స్థిరరాశి int64_t src_h = src->ne[1];
    స్థిరరాశి int64_t knl_w = kernel->ne[0];
    స్థిరరాశి int64_t knl_h = kernel->ne[1];
    స్థిరరాశి int64_t dst_w = dst->ne[0];
    స్థిరరాశి int64_t dst_h = dst->ne[1];

    స్థిరరాశి దశాంశం * src_దత్తాంశం = (దశాంశం *) src->దత్తాంశం;
    శూన్య  * knl_దత్తాంశం       = kernel->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం       = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి int64_t knl_n           = knl_w * knl_h * c_లోపల;
    స్థిరరాశి int64_t patch_మొత్తం     = dst->ne[3] * dst_w * dst_h;

    స్థిరరాశి int64_t space_చొప్పున_patch   = knl_n * traits->type_కొలత + c_బహిర్గతం * పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t batch_కొలత        = params->wsize / space_చొప్పున_patch;
    స్థిరరాశి int64_t patches_చొప్పున_batch = batch_కొలత > 8 ? (batch_కొలత / 8) * 8 : batch_కొలత;
    స్థిరరాశి int64_t batch_n           = (patch_మొత్తం + patches_చొప్పున_batch - 1) / patches_చొప్పున_batch;

    GGML_ASSERT(patches_చొప్పున_batch > 0 && batch_కొలత >= 1);

    శూన్య * tmp = params->wdata;

    కొరకు (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        స్థిరరాశి int64_t patch_ప్రారంభం_batch = batch_i * patches_చొప్పున_batch;
        స్థిరరాశి int64_t patch_చివర_batch   = std::min(patch_ప్రారంభం_batch + patches_చొప్పున_batch,
                                              patch_మొత్తం);
        స్థిరరాశి int64_t patch_n           = patch_చివర_batch - patch_ప్రారంభం_batch;

        స్థిరరాశి int64_t patch_చొప్పున_thread  = (patch_n + params->nth - 1) / params->nth;
        స్థిరరాశి int64_t patch_ప్రారంభం       = patch_ప్రారంభం_batch + params->ith * patch_చొప్పున_thread;
        స్థిరరాశి int64_t patch_చివర         = std::min(patch_ప్రారంభం + patch_చొప్పున_thread, patch_చివర_batch);

        //im2col for a patch
        కొరకు (int64_t p = patch_ప్రారంభం; p < patch_చివర; ++p) {
            స్థిరరాశి int64_t  batch_n     =  p / (dst_w * dst_h);
            స్థిరరాశి int64_t  src_x       = (p / dst_w) % dst_h;
            స్థిరరాశి int64_t  src_y       =  p % dst_w;

            స్థిరరాశి దశాంశం * src_పునాది = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_దత్తాంశం + batch_n * src->nb[3]);
            అక్షరం *        dst_అడ్డువరుస  = (అక్షరం *) tmp + (p % patches_చొప్పున_batch) * knl_n * traits->type_కొలత;

            కొరకు (int64_t ic = 0; ic < c_లోపల; ++ic) {
                కొరకు (int64_t ky = 0; ky < knl_h; ++ky) {
                    కొరకు (int64_t kx = 0; kx < knl_w; ++kx) {
                        స్థిరరాశి int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        స్థిరరాశి int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        దశాంశం src_val;
                        ఒకవేళ (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } లేకపోతే {
                            స్థిరరాశి దశాంశం * src_ptr = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_పునాది + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        అక్షరం * element_ptr = dst_అడ్డువరుస + dst_idx * traits->type_కొలత;
                        ఒకవేళ (kernel_type == GGML_TYPE_F32) {
                            *(దశాంశం *) element_ptr = src_val;
                        } లేకపోతే ఒకవేళ (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_గమ్యం_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        దశాంశం * gemm_output = (దశాంశం *) ((అక్షరం *) tmp + patches_చొప్పున_batch * knl_n * traits->type_కొలత);

        GGML_ASSERT(gemm_output + patch_n * c_బహిర్గతం <= (దశాంశం*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_బహిర్గతం, knl_n, tmp, knl_దత్తాంశం, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        స్థిరరాశి int64_t క్రమమార్పు_చొప్పున_thread = (patch_n + params->nth - 1) / params->nth;
        స్థిరరాశి int64_t క్రమమార్పు_ప్రారంభం = params->ith * క్రమమార్పు_చొప్పున_thread;
        స్థిరరాశి int64_t క్రమమార్పు_చివర = std::min(క్రమమార్పు_ప్రారంభం + క్రమమార్పు_చొప్పున_thread, patch_n);

        కొరకు (int64_t i = క్రమమార్పు_ప్రారంభం; i < క్రమమార్పు_చివర; ++i) {
            స్థిరరాశి int64_t p       = patch_ప్రారంభం_batch + i;
            స్థిరరాశి int64_t batch_n = p / (dst_w * dst_h);
            స్థిరరాశి int64_t dst_y   = (p / dst_w) % dst_h;
            స్థిరరాశి int64_t dst_x   = p % dst_w;

            కొరకు (int64_t oc = 0; oc < c_బహిర్గతం; ++oc) {
                స్థిరరాశి దశాంశం value = gemm_output[i * c_బహిర్గతం + oc];
                దశాంశం * dst_ptr = (దశాంశం *)((అక్షరం *)dst_దత్తాంశం + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

శూన్య ggml_compute_forward_conv_2d(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

స్థిర శూన్య ggml_compute_forward_conv_3d_impl(స్థిరరాశి ggml_compute_params * params,
                                              స్థిరరాశి ggml_tensor *         kernel,
                                              స్థిరరాశి ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    స్థిరరాశి ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    స్థిరరాశి int32_t s0 = dst->op_params[0];
    స్థిరరాశి int32_t s1 = dst->op_params[1];
    స్థిరరాశి int32_t s2 = dst->op_params[2];
    స్థిరరాశి int32_t p0 = dst->op_params[3];
    స్థిరరాశి int32_t p1 = dst->op_params[4];
    స్థిరరాశి int32_t p2 = dst->op_params[5];
    స్థిరరాశి int32_t d0 = dst->op_params[6];
    స్థిరరాశి int32_t d1 = dst->op_params[7];
    స్థిరరాశి int32_t d2 = dst->op_params[8];
    స్థిరరాశి int32_t c  = dst->op_params[9];
    స్థిరరాశి int32_t n  = dst->op_params[10];
    స్థిరరాశి int32_t oc = dst->op_params[11];

    స్థిరరాశి int64_t src_w = src->ne[0];
    స్థిరరాశి int64_t src_h = src->ne[1];
    స్థిరరాశి int64_t src_d = src->ne[2];
    స్థిరరాశి int64_t knl_w = kernel->ne[0];
    స్థిరరాశి int64_t knl_h = kernel->ne[1];
    స్థిరరాశి int64_t knl_d = kernel->ne[2];
    స్థిరరాశి int64_t dst_w = dst->ne[0];
    స్థిరరాశి int64_t dst_h = dst->ne[1];
    స్థిరరాశి int64_t dst_d = dst->ne[2];

    స్థిరరాశి దశాంశం * src_దత్తాంశం = (దశాంశం *) src->దత్తాంశం;
    శూన్య  * knl_దత్తాంశం       = kernel->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం       = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి int64_t knl_n_చొప్పున_channel = knl_w * knl_h * knl_d;
    స్థిరరాశి int64_t knl_n_మొత్తం       = knl_n_చొప్పున_channel * c;
    స్థిరరాశి int64_t patch_మొత్తం       = n * dst_w * dst_h * dst_d;

    స్థిరరాశి int64_t space_చొప్పున_patch   = knl_n_మొత్తం * traits->type_కొలత + oc * పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t batch_కొలత        = params->wsize / space_చొప్పున_patch;
    స్థిరరాశి int64_t patches_చొప్పున_batch = batch_కొలత > 8 ? (batch_కొలత / 8) * 8 : batch_కొలత;
    స్థిరరాశి int64_t batch_n           = (patch_మొత్తం + patches_చొప్పున_batch - 1) / patches_చొప్పున_batch;

    GGML_ASSERT(patches_చొప్పున_batch > 0 && batch_కొలత >= 1);

    శూన్య * tmp = params->wdata;

    కొరకు (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        స్థిరరాశి int64_t patch_ప్రారంభం_batch = batch_i * patches_చొప్పున_batch;
        స్థిరరాశి int64_t patch_చివర_batch   = std::min(patch_ప్రారంభం_batch + patches_చొప్పున_batch, patch_మొత్తం);
        స్థిరరాశి int64_t patch_n_లోపల_batch  = patch_చివర_batch - patch_ప్రారంభం_batch;

        స్థిరరాశి int64_t patch_చొప్పున_thread  = (patch_n_లోపల_batch + params->nth - 1) / params->nth;
        స్థిరరాశి int64_t patch_ప్రారంభం       = patch_ప్రారంభం_batch + params->ith * patch_చొప్పున_thread;
        స్థిరరాశి int64_t patch_చివర         = std::min(patch_ప్రారంభం + patch_చొప్పున_thread, patch_చివర_batch);

        కొరకు (int64_t p = patch_ప్రారంభం; p < patch_చివర; ++p) {
            స్థిరరాశి int64_t p_లోపల_batch = p % (dst_w * dst_h * dst_d);
            స్థిరరాశి int64_t p_లోపల_లోతు = p_లోపల_batch % (dst_w * dst_h);
            స్థిరరాశి int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            స్థిరరాశి int64_t dst_z      = p_లోపల_batch / (dst_w * dst_h);
            స్థిరరాశి int64_t dst_y      = p_లోపల_లోతు / dst_w;
            స్థిరరాశి int64_t dst_x      = p_లోపల_లోతు % dst_w;

            అక్షరం * dst_అడ్డువరుస = (అక్షరం *) tmp + (p % patches_చొప్పున_batch) * knl_n_మొత్తం * traits->type_కొలత;

            కొరకు (int64_t ic = 0; ic < c; ++ic) {
                కొరకు (int64_t kz = 0; kz < knl_d; ++kz) {
                    కొరకు (int64_t ky = 0; ky < knl_h; ++ky) {
                        కొరకు (int64_t kx = 0; kx < knl_w; ++kx) {
                            స్థిరరాశి int64_t sz = dst_z * s2 + kz * d2 - p2;
                            స్థిరరాశి int64_t sy = dst_y * s1 + ky * d1 - p1;
                            స్థిరరాశి int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_చొప్పున_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            దశాంశం src_val;
                            ఒకవేళ (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } లేకపోతే {
                                స్థిరరాశి int64_t cn_idx = batch_idx * c + ic;
                                స్థిరరాశి దశాంశం * src_ptr = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_దత్తాంశం + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            అక్షరం * element_ptr = dst_అడ్డువరుస + dst_idx * traits->type_కొలత;
                            ఒకవేళ (kernel_type == GGML_TYPE_F32) {
                                *(దశాంశం *)element_ptr = src_val;
                            } లేకపోతే ఒకవేళ (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_గమ్యం_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        దశాంశం * gemm_output = (దశాంశం *) ((అక్షరం *) tmp + patches_చొప్పున_batch * knl_n_మొత్తం * traits->type_కొలత);
        ggml_call_mul_mat(kernel_type, params, patch_n_లోపల_batch, oc, knl_n_మొత్తం, tmp, knl_దత్తాంశం, gemm_output);

        ggml_barrier(params->threadpool);

        స్థిరరాశి int64_t క్రమమార్పు_చొప్పున_thread = (patch_n_లోపల_batch + params->nth - 1) / params->nth;
        స్థిరరాశి int64_t క్రమమార్పు_ప్రారంభం = params->ith * క్రమమార్పు_చొప్పున_thread;
        స్థిరరాశి int64_t క్రమమార్పు_చివర = std::min(క్రమమార్పు_ప్రారంభం + క్రమమార్పు_చొప్పున_thread, patch_n_లోపల_batch);

        కొరకు (int64_t i = క్రమమార్పు_ప్రారంభం; i < క్రమమార్పు_చివర; ++i) {
            స్థిరరాశి int64_t p = patch_ప్రారంభం_batch + i;
            స్థిరరాశి int64_t p_లోపల_batch = p % (dst_w * dst_h * dst_d);
            స్థిరరాశి int64_t p_లోపల_లోతు = p_లోపల_batch % (dst_w * dst_h);
            స్థిరరాశి int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            స్థిరరాశి int64_t dst_z      = p_లోపల_batch / (dst_w * dst_h);
            స్థిరరాశి int64_t dst_y      = p_లోపల_లోతు / dst_w;
            స్థిరరాశి int64_t dst_x      = p_లోపల_లోతు % dst_w;

            కొరకు (int64_t ioc = 0; ioc < oc; ++ioc) {
                స్థిరరాశి దశాంశం value = gemm_output[i * oc + ioc];
                స్థిరరాశి int64_t ocn_idx = batch_idx * oc + ioc;
                దశాంశం * dst_ptr = (దశాంశం *)((అక్షరం *)dst_దత్తాంశం + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

శూన్య ggml_compute_forward_conv_3d(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

మూస <రకం_పేరు kernel_t>
స్థిర శూన్య ggml_compute_forward_conv_transpose_2d_impl(
    స్థిరరాశి ggml_compute_params * params,
          ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_కొలత(src0->type));
    GGML_ASSERT(nb10 == పరిమాణం(దశాంశం));

    ఒకవేళ (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * స్థిరరాశి wdata = (kernel_t *) params->wdata + 0;

            కొరకు (int64_t i03 = 0; i03 < ne03; i03++) {
                కొరకు (int64_t i02 = 0; i02 < ne02; i02++) {
                    స్థిరరాశి kernel_t * స్థిరరాశి src = (kernel_t *)((అక్షరం *) src0->దత్తాంశం + i03*nb03 + i02*nb02);
                    kernel_t * dst_దత్తాంశం = wdata + i02*ne01*ne00*ne03;
                    కొరకు (int64_t i01 = 0; i01 < ne01; i01++) {
                        కొరకు (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_దత్తాంశం[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * స్థిరరాశి wdata = (kernel_t *) params->wdata + nk;
            కొరకు (పూర్ణాంకం i13 = 0; i13 < ne13; i13++) {
                kernel_t * స్థిరరాశి wdata_b = wdata + i13*ne10*ne11*ne12;
                కొరకు (పూర్ణాంకం i12 = 0; i12 < ne12; i12++) {
                    కొరకు (పూర్ణాంకం i11 = 0; i11 < ne11; i11++) {
                        స్థిరరాశి దశాంశం * స్థిరరాశి src = (దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_దత్తాంశం = wdata_b + i11*ne10*ne12;
                        కొరకు (పూర్ణాంకం i10 = 0; i10 < ne10; i10++) {
                            ఒకవేళ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_దత్తాంశం[i10*ne12 + i12] = GGML_CPU_FP32_గమ్యం_FP16(src[i10]);
                            } లేకపోతే {
                                dst_దత్తాంశం[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        memset(dst->దత్తాంశం, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    స్థిరరాశి int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    స్థిరరాశి పూర్ణాంకం np = ne2;

    // patches per thread
    స్థిరరాశి పూర్ణాంకం dp = (np + nth - 1)/nth;

    // patch range for this thread
    స్థిరరాశి పూర్ణాంకం ip0 = dp*ith;
    స్థిరరాశి పూర్ణాంకం ip1 = MIN(ip0 + dp, np);

    kernel_t * స్థిరరాశి wdata = (kernel_t *) params->wdata + 0;
    kernel_t * స్థిరరాశి wdata_src = wdata + nk;

    కొరకు (పూర్ణాంకం i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        కొరకు (పూర్ణాంకం i3 = 0; i3 < ne3; i3++) { // batch
            దశాంశం * dst_దత్తాంశం = (దశాంశం *)((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            కొరకు (పూర్ణాంకం i11 = 0; i11 < ne11; i11++) {
                కొరకు (పూర్ణాంకం i10 = 0; i10 < ne10; i10++) {
                    స్థిరరాశి పూర్ణాంకం i1n = i11*ne10*ne12 + i10*ne12;
                    కొరకు (పూర్ణాంకం i01 = 0; i01 < ne01; i01++) {
                        కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; i00++) {
                            దశాంశం v = 0;
                            ఒకవేళ constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } లేకపోతే {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_దత్తాంశం[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_conv_transpose_2d(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } ఆపు;
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<దశాంశం>(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

నిర్మాణం ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    పూర్ణాంకం stride_x;
    పూర్ణాంకం stride_y;
    పూర్ణాంకం pad_x;
    పూర్ణాంకం pad_y;
    పూర్ణాంకం dilation_x;
    పూర్ణాంకం dilation_y;
};

స్థిర అంతఃవరుస దశాంశం ggml_conv_2d_dw_knl_f32(స్థిరరాశి అక్షరం * దత్తాంశం, int64_t i, ggml_type type) {
    ఒకవేళ (type == GGML_TYPE_F16) {
        తిరిగి GGML_FP16_గమ్యం_FP32(((స్థిరరాశి ggml_fp16_t *)దత్తాంశం)[i]);
    }
    తిరిగి ((స్థిరరాశి దశాంశం *)దత్తాంశం)[i];
}

స్థిర శూన్య ggml_compute_forward_conv_2d_dw_cwhn(
        స్థిరరాశి ggml_compute_params * params,
        స్థిరరాశి ggml_tensor * src,
        స్థిరరాశి ggml_tensor * kernel,
        ggml_tensor * dst,
        స్థిరరాశి ggml_conv_2d_dw_params & p) {

    స్థిరరాశి int64_t c = p.channels;
    స్థిరరాశి అక్షరం * knl_దత్తాంశం = (స్థిరరాశి అక్షరం *)kernel->దత్తాంశం;
    స్థిరరాశి ggml_type knl_type = kernel->type;

    స్థిరరాశి int64_t అడ్డువరుసలు_మొత్తం = p.dst_h * p.batch;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_thread = (అడ్డువరుసలు_మొత్తం + params->nth - 1) / params->nth;
    స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = params->ith * అడ్డువరుసలు_చొప్పున_thread;
    స్థిరరాశి int64_t అడ్డువరుస_చివర = MIN(అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_thread, అడ్డువరుసలు_మొత్తం);

#నిర్వచితమైతే GGML_SIMD
    int64_t c_pkg_చివర = 0;
    int64_t pkg_కొలత = GGML_F32_EPR;
    ఒకవేళ (knl_type == GGML_TYPE_F32) {
    #ఒకవేళ defined(__ARM_FEATURE_SVE)
        pkg_కొలత = svcntw();
    #లేకపోతే
        pkg_కొలత = GGML_F32_EPR;
    #అయితే_ముగింపు
        c_pkg_చివర = (c / pkg_కొలత) * pkg_కొలత;
    }
#లేకపోతే
    స్థిరరాశి int64_t c_pkg_చివర = 0;
#అయితే_ముగింపు

    కొరకు (int64_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; ++అడ్డువరుస) {
        స్థిరరాశి int64_t dst_y = అడ్డువరుస % p.dst_h;
        స్థిరరాశి దశాంశం * src_దత్తాంశం = (స్థిరరాశి దశాంశం *)src->దత్తాంశం + (అడ్డువరుస / p.dst_h) * p.src_w * p.src_h * c;
        కొరకు (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            దశాంశం * dst_దత్తాంశం = (దశాంశం *)dst->దత్తాంశం + (అడ్డువరుస * p.dst_w + dst_x) * c;
            స్థిరరాశి int64_t src_y_పునాది = dst_y * p.stride_y - p.pad_y;
            స్థిరరాశి int64_t src_x_పునాది = dst_x * p.stride_x - p.pad_x;

#నిర్వచితమైతే GGML_SIMD
            కొరకు (int64_t c_i = 0; c_i < c_pkg_చివర; c_i += pkg_కొలత) {
                GGML_F32_VEC కూడిక = GGML_F32_VEC_ZERO;
                కొరకు (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    స్థిరరాశి int64_t src_y = src_y_పునాది + knl_y * p.dilation_y;
                    ఒకవేళ (src_y < 0 || src_y >= p.src_h) {
                        కొనసాగు;
                    }
                    కొరకు (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        స్థిరరాశి int64_t src_x = src_x_పునాది + knl_x * p.dilation_x;
                        ఒకవేళ (src_x < 0 || src_x >= p.src_w) {
                            కొనసాగు;
                        }
                        స్థిరరాశి దశాంశం * kp = (స్థిరరాశి దశాంశం *)knl_దత్తాంశం + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_ఎక్కించు(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_ఎక్కించు(src_దత్తాంశం + (src_y * p.src_w + src_x) * c + c_i);
                        కూడిక = GGML_F32_VEC_FMA(కూడిక, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_దత్తాంశం + c_i, కూడిక);
            }
#అయితే_ముగింపు
            కొరకు (int64_t c_i = c_pkg_చివర; c_i < c; ++c_i) {
                దశాంశం కూడిక = 0.0f;
                కొరకు (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    స్థిరరాశి int64_t src_y = src_y_పునాది + knl_y * p.dilation_y;
                    ఒకవేళ (src_y < 0 || src_y >= p.src_h) {
                        కొనసాగు;
                    }
                    కొరకు (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        స్థిరరాశి int64_t src_x = src_x_పునాది + knl_x * p.dilation_x;
                        ఒకవేళ (src_x < 0 || src_x >= p.src_w) {
                            కొనసాగు;
                        }
                        కూడిక += ggml_conv_2d_dw_knl_f32(knl_దత్తాంశం, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_దత్తాంశం[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_దత్తాంశం[c_i] = కూడిక;
            }
        }
    }
}

స్థిర శూన్య ggml_compute_forward_conv_2d_dw_whcn(
        స్థిరరాశి ggml_compute_params * params,
        స్థిరరాశి ggml_tensor * src,
        స్థిరరాశి ggml_tensor * kernel,
        ggml_tensor * dst,
        స్థిరరాశి ggml_conv_2d_dw_params & p) {

    స్థిరరాశి int64_t n = p.channels * p.batch;
    స్థిరరాశి int64_t చొప్పున_thread = (n + params->nth - 1) / params->nth;
    స్థిరరాశి int64_t start = params->ith * చొప్పున_thread;
    స్థిరరాశి int64_t end = MIN(start + చొప్పున_thread, n);
    స్థిరరాశి అక్షరం * knl_పునాది = (స్థిరరాశి అక్షరం *)kernel->దత్తాంశం;
    స్థిరరాశి ggml_type knl_type = kernel->type;

    కొరకు (int64_t i = start; i < end; ++i) {
        స్థిరరాశి int64_t knl_స్థానభ్రంశం = (i % p.channels) * p.knl_w * p.knl_h;
        స్థిరరాశి దశాంశం * src_దత్తాంశం = (స్థిరరాశి దశాంశం *)src->దత్తాంశం + i * p.src_w * p.src_h;
        దశాంశం * dst_దత్తాంశం = (దశాంశం *)dst->దత్తాంశం + i * p.dst_w * p.dst_h;

        కొరకు (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            కొరకు (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                దశాంశం కూడిక = 0.0f;
                కొరకు (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    స్థిరరాశి int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    ఒకవేళ (src_y < 0 || src_y >= p.src_h) {
                        కొనసాగు;
                    }
                    కొరకు (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        స్థిరరాశి int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        ఒకవేళ (src_x < 0 || src_x >= p.src_w) {
                            కొనసాగు;
                        }
                        కూడిక += ggml_conv_2d_dw_knl_f32(knl_పునాది, knl_స్థానభ్రంశం + knl_y * p.knl_w + knl_x, knl_type)
                             * src_దత్తాంశం[src_y * p.src_w + src_x];
                    }
                }
                dst_దత్తాంశం[dst_y * p.dst_w + dst_x] = కూడిక;
            }
        }
    }
}

శూన్య ggml_compute_forward_conv_2d_dw(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * kernel = dst->src[0];
    స్థిరరాశి ggml_tensor * src = dst->src[1];
    ggml_conv_2d_dw_params p;
    p.channels = src->ne[2];
    p.batch = src->ne[3];
    p.src_w = src->ne[0];
    p.src_h = src->ne[1];
    p.dst_w = dst->ne[0];
    p.dst_h = dst->ne[1];
    p.knl_w = kernel->ne[0];
    p.knl_h = kernel->ne[1];
    p.stride_x = dst->op_params[0];
    p.stride_y = dst->op_params[1];
    p.pad_x = dst->op_params[2];
    p.pad_y = dst->op_params[3];
    p.dilation_x = dst->op_params[4];
    p.dilation_y = dst->op_params[5];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16);
    GGML_ASSERT(kernel->ne[3] == p.channels);
    GGML_ASSERT(dst->ne[3] == p.batch);

    ఒకవేళ (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } లేకపోతే ఒకవేళ (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } లేకపోతే {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
స్థిర శూన్య ggml_compute_forward_pool_1d_ksp(
        స్థిరరాశి ggml_compute_params * params,
        స్థిరరాశి ggml_op_pool op,
        స్థిరరాశి పూర్ణాంకం k,
        స్థిరరాశి పూర్ణాంకం s,
        స్థిరరాశి పూర్ణాంకం p,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src = dst->src[0];

    ప్రతిజ్ఞా(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    స్థిరరాశి int64_t IW = src->ne[0];
    స్థిరరాశి int64_t OW = dst->ne[0];

    స్థిరరాశి int64_t nr = ggml_nrows(src);

    కొరకు (int64_t ir = 0; ir < nr; ++ir) {
        స్థిరరాశి అక్షరం * srow_బైట్లు =            (స్థిరరాశి అక్షరం *) src->దత్తాంశం + ir * src->nb[1];
        దశాంశం      * drow       = (దశాంశం *) ((      అక్షరం *) dst->దత్తాంశం + ir * dst->nb[1]);

        కొరకు (int64_t ow = 0; ow < OW; ++ow) {
            దశాంశం res = 0;
            ఎంపిక (op) {
                స్థితి GGML_OP_POOL_AVG: res = 0.0f;     ఆపు;
                స్థితి GGML_OP_POOL_MAX: res = -FLT_MAX; ఆపు;
                స్థితి GGML_OP_POOL_లెక్క: GGML_ABORT("fatal error");
            }

            పూర్ణాంకం count = 0;
            స్థిరరాశి పూర్ణాంకం పునాది = (పూర్ణాంకం) ow * s - p;

            కొరకు (పూర్ణాంకం ki = 0; ki < k; ++ki) {
                స్థిరరాశి పూర్ణాంకం j = పునాది + ki;
                ఒకవేళ (j < 0 || j >= (పూర్ణాంకం) IW) {
                    కొనసాగు;
                }

                దశాంశం v;
                ఒకవేళ (src->type == GGML_TYPE_F32) {
                    v = ((స్థిరరాశి దశాంశం *) srow_బైట్లు)[j];
                } లేకపోతే {
                    v = GGML_CPU_FP16_గమ్యం_FP32(((స్థిరరాశి ggml_fp16_t *) srow_బైట్లు)[j]);
                }

                ఎంపిక (op) {
                    స్థితి GGML_OP_POOL_AVG: res += v;                ఆపు;
                    స్థితి GGML_OP_POOL_MAX: res =  std::max(v, res); ఆపు;
                    స్థితి GGML_OP_POOL_లెక్క: GGML_ABORT("fatal error");
                }

                ++count;
            }

            ఎంపిక (op) {
                స్థితి GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; ఆపు;
                స్థితి GGML_OP_POOL_MAX:                                           ఆపు;
                స్థితి GGML_OP_POOL_లెక్క: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

శూన్య ggml_compute_forward_pool_1d(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    స్థిరరాశి పూర్ణాంకం k0 = opts[1];
    స్థిరరాశి పూర్ణాంకం s0 = opts[2];
    స్థిరరాశి పూర్ణాంకం p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

శూన్య ggml_compute_forward_pool_2d(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src = dst->src[0];

    ప్రతిజ్ఞా(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    స్థిరరాశి పూర్ణాంకం k0 = opts[1];
    స్థిరరాశి పూర్ణాంకం k1 = opts[2];
    స్థిరరాశి పూర్ణాంకం s0 = opts[3];
    స్థిరరాశి పూర్ణాంకం s1 = opts[4];
    స్థిరరాశి పూర్ణాంకం p0 = opts[5];
    స్థిరరాశి పూర్ణాంకం p1 = opts[6];
    స్థిరరాశి అక్షరం * cdata = (స్థిరరాశి అక్షరం*)src->దత్తాంశం;
    స్థిరరాశి అక్షరం * స్థిరరాశి దత్తాంశం_చివర = cdata + ggml_nbytes(src);

    స్థిరరాశి int64_t px = dst->ne[0];
    స్థిరరాశి int64_t py = dst->ne[1];
    స్థిరరాశి int64_t pa = px * py;

    దశాంశం * dplane = (దశాంశం *)dst->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ka = k0 * k1;
    స్థిరరాశి పూర్ణాంకం offset0 = -p0;
    స్థిరరాశి పూర్ణాంకం offset1 = -p1;

    అయితే (cdata < దత్తాంశం_చివర) {
        కొరకు (పూర్ణాంకం oy = 0; oy < py; ++oy) {
            దశాంశం * స్థిరరాశి drow = dplane + oy * px;
            దశాంశం * స్థిరరాశి బహిర్గతం  = drow;

            కొరకు (పూర్ణాంకం ox = 0; ox < px; ++ox) {
                దశాంశం res = 0;
                ఎంపిక (op) {
                    స్థితి GGML_OP_POOL_AVG: res = 0;        ఆపు;
                    స్థితి GGML_OP_POOL_MAX: res = -FLT_MAX; ఆపు;
                    స్థితి GGML_OP_POOL_లెక్క: GGML_ABORT("fatal error");
                }

                స్థిరరాశి పూర్ణాంకం ix = offset0 + ox * s0;
                స్థిరరాశి పూర్ణాంకం iy = offset1 + oy * s1;

                కొరకు (పూర్ణాంకం ky = 0; ky < k1; ++ky) {
                    ఒకవేళ (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        కొనసాగు;
                    }

                    స్థిరరాశి శూన్య * srow = (స్థిరరాశి శూన్య *)(cdata + src->nb[1] * (iy + ky));
                    కొరకు (పూర్ణాంకం kx = 0; kx < k0; ++kx) {
                        పూర్ణాంకం j = ix + kx;
                        ఒకవేళ (j < 0 || j >= src->ne[0]) {
                            కొనసాగు;
                        }

                        స్థిరరాశి దశాంశం srow_j = (src->type == GGML_TYPE_F32) ? ((స్థిరరాశి దశాంశం*)srow)[j] : GGML_CPU_FP16_గమ్యం_FP32(((స్థిరరాశి ggml_fp16_t*)srow)[j]);
                        ఎంపిక (op) {
                            స్థితి GGML_OP_POOL_AVG: res += srow_j;                ఆపు;
                            స్థితి GGML_OP_POOL_MAX: res =  std::max(srow_j, res); ఆపు;
                            స్థితి GGML_OP_POOL_లెక్క:               GGML_ABORT("fatal error");
                        }
                    }
                }
                ఎంపిక (op) {
                    స్థితి GGML_OP_POOL_AVG:           res /= ka; ఆపు;
                    స్థితి GGML_OP_POOL_MAX:                      ఆపు;
                    స్థితి GGML_OP_POOL_లెక్క: GGML_ABORT("fatal error");
                }

                బహిర్గతం[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

శూన్య ggml_compute_forward_pool_2d_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src  = dst->src[0];
    స్థిరరాశి ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    ప్రతిజ్ఞా(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    ఒకవేళ (params->ith != 0) {
        తిరిగి;
    }

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    స్థిరరాశి పూర్ణాంకం k0 = opts[1];
    స్థిరరాశి పూర్ణాంకం k1 = opts[2];
    స్థిరరాశి పూర్ణాంకం s0 = opts[3];
    స్థిరరాశి పూర్ణాంకం s1 = opts[4];
    స్థిరరాశి పూర్ణాంకం p0 = opts[5];
    స్థిరరాశి పూర్ణాంకం p1 = opts[6];

    అక్షరం       * cdata  = (అక్షరం       *) dst->దత్తాంశం;
    స్థిరరాశి అక్షరం * cdataf = (స్థిరరాశి అక్షరం *) dstf->దత్తాంశం;
    స్థిరరాశి అక్షరం * స్థిరరాశి దత్తాంశం_చివర = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    స్థిరరాశి int64_t px = src->ne[0];
    స్థిరరాశి int64_t py = src->ne[1];
    స్థిరరాశి int64_t pa = px * py;

    స్థిరరాశి దశాంశం * splane = (స్థిరరాశి దశాంశం *) src->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ka = k0 * k1;
    స్థిరరాశి పూర్ణాంకం offset0 = -p0;
    స్థిరరాశి పూర్ణాంకం offset1 = -p1;

    అయితే (cdata < దత్తాంశం_చివర) {
        కొరకు (పూర్ణాంకం oy = 0; oy < py; ++oy) {
            స్థిరరాశి దశాంశం * స్థిరరాశి srow = splane + oy * px;
            కొరకు (పూర్ణాంకం ox = 0; ox < px; ++ox) {
                స్థిరరాశి దశాంశం grad0 = srow[ox];

                స్థిరరాశి పూర్ణాంకం ix = offset0 + ox * s0;
                స్థిరరాశి పూర్ణాంకం iy = offset1 + oy * s1;

                ఒకవేళ (op == GGML_OP_POOL_MAX) {
                    దశాంశం maxval = -FLT_MAX;
                    పూర్ణాంకం kxmax = -1;
                    పూర్ణాంకం kymax = -1;

                    కొరకు (పూర్ణాంకం ky = 0; ky < k1; ++ky) {
                        ఒకవేళ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            కొనసాగు;
                        }
                        స్థిరరాశి శూన్య * drowf = (స్థిరరాశి శూన్య *)(cdataf + dst->nb[1] * (iy + ky));
                        కొరకు (పూర్ణాంకం kx = 0; kx < k0; ++kx) {
                            పూర్ణాంకం j = ix + kx;
                            ఒకవేళ (j < 0 || j >= dst->ne[0]) {
                                కొనసాగు;
                            }

                            స్థిరరాశి దశాంశం val = dst->type == GGML_TYPE_F32 ?
                                ((స్థిరరాశి దశాంశం *) drowf)[j] : GGML_CPU_FP16_గమ్యం_FP32(((స్థిరరాశి ggml_fp16_t *) drowf)[j]);
                            ఒకవేళ (val <= maxval) {
                                కొనసాగు;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    ఒకవేళ (kxmax == -1 || kymax == -1) {
                        కొనసాగు;
                    }

                    శూన్య * drow = (శూన్య *)(cdata + dst->nb[1] * (iy + kymax));
                    స్థిరరాశి పూర్ణాంకం j = ix + kxmax;
                    ఒకవేళ (dst->type == GGML_TYPE_F32) {
                        ((దశాంశం *) drow)[j] += grad0;
                    } లేకపోతే {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_గమ్యం_FP16(grad0 + GGML_CPU_FP16_గమ్యం_FP32(((స్థిరరాశి ggml_fp16_t *) drow)[j]));
                    }
                } లేకపోతే ఒకవేళ (op == GGML_OP_POOL_AVG) {
                    స్థిరరాశి దశాంశం grad = grad0 / ka;

                    కొరకు (పూర్ణాంకం ky = 0; ky < k1; ++ky) {
                        ఒకవేళ (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            కొనసాగు;
                        }
                        శూన్య * drow = (శూన్య *)(cdata + dst->nb[1] * (iy + ky));
                        కొరకు (పూర్ణాంకం kx = 0; kx < k0; ++kx) {
                            పూర్ణాంకం j = ix + kx;
                            ఒకవేళ (j < 0 || j >= dst->ne[0]) {
                                కొనసాగు;
                            }

                            ఒకవేళ (dst->type == GGML_TYPE_F32) {
                                ((దశాంశం *) drow)[j] += grad;
                            } లేకపోతే {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_గమ్యం_FP16(grad);
                            }
                        }
                    }
                } లేకపోతే {
                    GGML_ASSERT(అబద్ధం);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

స్థిర శూన్య ggml_compute_forward_upscale_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం sf0 = (దశాంశం)ne0/src0->ne[0];
    దశాంశం sf1 = (దశాంశం)ne1/src0->ne[1];
    దశాంశం sf2 = (దశాంశం)ne2/src0->ne[2];
    దశాంశం sf3 = (దశాంశం)ne3/src0->ne[3];
    దశాంశం pixel_స్థానభ్రంశం = 0.5f;

    స్థిరరాశి int32_t బహుళకం_flags = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి ggml_scale_బహుళకం mode = (ggml_scale_బహుళకం) (బహుళకం_flags & 0xFF);

    ఒకవేళ (బహుళకం_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_స్థానభ్రంశం = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (దశాంశం)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (దశాంశం)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ఒకవేళ (mode == GGML_SCALE_బహుళకం_సమీప) {
        కొరకు (int64_t i3 = 0; i3 < ne3; i3++) {
            స్థిరరాశి int64_t i03 = i3 / sf3;
            కొరకు (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                స్థిరరాశి int64_t i02 = i2 / sf2;
                కొరకు (int64_t i1 = 0; i1 < ne1; i1++) {
                    స్థిరరాశి int64_t i01 = i1 / sf1;
                    కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
                        స్థిరరాశి int64_t i00 = i0 / sf0;

                        స్థిరరాశి దశాంశం * x = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              దశాంశం * y = (దశాంశం *)((అక్షరం *)  dst->దత్తాంశం +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } లేకపోతే ఒకవేళ (mode == GGML_SCALE_బహుళకం_BILINEAR && (బహుళకం_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        స్వయంచాలక triangle_filter = [](దశాంశం x) -> దశాంశం {
            తిరిగి std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        స్థిరరాశి దశాంశం support1  = std::max(1.0f, 1.0f / sf1);
        స్థిరరాశి దశాంశం invscale1 = 1.0f / support1;
        స్థిరరాశి దశాంశం support0  = std::max(1.0f, 1.0f / sf0);
        స్థిరరాశి దశాంశం invscale0 = 1.0f / support0;

        కొరకు (int64_t i3 = 0; i3 < ne3; i3++) {
            స్థిరరాశి int64_t i03 = i3 / sf3;
            కొరకు (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                స్థిరరాశి int64_t i02 = i2 / sf2;
                కొరకు (int64_t i1 = 0; i1 < ne1; i1++) {
                    స్థిరరాశి దశాంశం y = ((దశాంశం) i1 + pixel_స్థానభ్రంశం) / sf1;
                    కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
                        స్థిరరాశి దశాంశం x = ((దశాంశం) i0 + pixel_స్థానభ్రంశం) / sf0;

                        // the range of source pixels that contribute
                        స్థిరరాశి int64_t x_min = std::max<int64_t>(x - support0 + pixel_స్థానభ్రంశం, 0);
                        స్థిరరాశి int64_t x_max = std::min<int64_t>(x + support0 + pixel_స్థానభ్రంశం, ne00);
                        స్థిరరాశి int64_t y_min = std::max<int64_t>(y - support1 + pixel_స్థానభ్రంశం, 0);
                        స్థిరరాశి int64_t y_max = std::min<int64_t>(y + support1 + pixel_స్థానభ్రంశం, ne01);

                        // bilinear filter with antialiasing
                        దశాంశం val = 0.0f;
                        దశాంశం మొత్తం_బరువు = 0.0f;

                        కొరకు (int64_t sy = y_min; sy < y_max; sy++) {
                            స్థిరరాశి దశాంశం బరువు_y = triangle_filter((sy - y + pixel_స్థానభ్రంశం) * invscale1);

                            కొరకు (int64_t sx = x_min; sx < x_max; sx++) {
                                స్థిరరాశి దశాంశం బరువు_x = triangle_filter((sx - x + pixel_స్థానభ్రంశం) * invscale0);
                                స్థిరరాశి దశాంశం బరువు = బరువు_x * బరువు_y;

                                ఒకవేళ (బరువు <= 0.0f) {
                                    కొనసాగు;
                                }

                                స్థిరరాశి దశాంశం pixel = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * బరువు;
                                మొత్తం_బరువు += బరువు;
                            }
                        }

                        ఒకవేళ (మొత్తం_బరువు > 0.0f) {
                            val /= మొత్తం_బరువు;
                        }

                        దశాంశం * dst_ptr = (దశాంశం *)((అక్షరం *)dst->దత్తాంశం + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } లేకపోతే ఒకవేళ (mode == GGML_SCALE_బహుళకం_BILINEAR) {
        కొరకు (int64_t i3 = 0; i3 < ne3; i3++) {
            స్థిరరాశి int64_t i03 = i3 / sf3;
            కొరకు (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                స్థిరరాశి int64_t i02 = i2 / sf2;
                కొరకు (int64_t i1 = 0; i1 < ne1; i1++) {
                    స్థిరరాశి దశాంశం y = ((దశాంశం)i1 + pixel_స్థానభ్రంశం) / sf1 - pixel_స్థానభ్రంశం;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    దశాంశం dy = y - (దశాంశం)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
                        స్థిరరాశి దశాంశం x = ((దశాంశం)i0 + pixel_స్థానభ్రంశం) / sf0 - pixel_స్థానభ్రంశం;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        దశాంశం dx = x - (దశాంశం)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        స్థిరరాశి దశాంశం a = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        స్థిరరాశి దశాంశం b = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        స్థిరరాశి దశాంశం c = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        స్థిరరాశి దశాంశం d = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        స్థిరరాశి దశాంశం val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        దశాంశం * y_dst = (దశాంశం *)((అక్షరం *)dst->దత్తాంశం + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } లేకపోతే ఒకవేళ (mode == GGML_SCALE_బహుళకం_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        స్థిరరాశి దశాంశం a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        స్వయంచాలక weight1 = [a](దశాంశం x) { తిరిగి ((a + 2) * x - (a + 3)) * x * x + 1; };
        స్వయంచాలక weight2 = [a](దశాంశం x) { తిరిగి ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        స్వయంచాలక bicubic = [=](దశాంశం p0, దశాంశం p1, దశాంశం p2, దశాంశం p3, దశాంశం x) {
            స్థిరరాశి దశాంశం w0 = weight2(x + 1);
            స్థిరరాశి దశాంశం w1 = weight1(x + 0);
            స్థిరరాశి దశాంశం w2 = weight1(1 - x);
            స్థిరరాశి దశాంశం w3 = weight2(2 - x);
            తిరిగి p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        కొరకు (int64_t i3 = 0; i3 < ne3; i3++) {
            స్థిరరాశి int64_t i03 = i3 / sf3;
            కొరకు (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                స్థిరరాశి int64_t i02 = i2 / sf2;
                కొరకు (int64_t i1 = 0; i1 < ne1; i1++) {
                    స్థిరరాశి దశాంశం y = ((దశాంశం)i1 + pixel_స్థానభ్రంశం) / sf1 - pixel_స్థానభ్రంశం;
                    స్థిరరాశి int64_t y0 = (int64_t)floorf(y);
                    స్థిరరాశి దశాంశం dy = y - (దశాంశం)y0;

                    కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
                        స్థిరరాశి దశాంశం x = ((దశాంశం)i0 + pixel_స్థానభ్రంశం) / sf0 - pixel_స్థానభ్రంశం;
                        స్థిరరాశి int64_t x0 = (int64_t)floorf(x);
                        స్థిరరాశి దశాంశం dx = x - (దశాంశం)x0;

                        స్వయంచాలక p = [=](int64_t x_off, int64_t y_off) -> దశాంశం {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            తిరిగి *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src0->దత్తాంశం + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        స్థిరరాశి దశాంశం val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        దశాంశం * y_dst = (దశాంశం *)((అక్షరం *)dst->దత్తాంశం + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } లేకపోతే {
        GGML_ABORT("unsupported upscale mode");
    }
}

శూన్య ggml_compute_forward_upscale(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

మూస<తార్కిక circular_t>
స్థిర శూన్య ggml_compute_forward_pad_f32(
    స్థిరరాశి ggml_compute_params * params,
          ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ప్రతిజ్ఞా(dst->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    దశాంశం * dst_ptr = (దశాంశం *) dst->దత్తాంశం;
    స్థిరరాశి int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    స్థిరరాశి int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    స్థిరరాశి int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    స్థిరరాశి int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    స్థిరరాశి int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    స్థిరరాశి int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    స్థిరరాశి int64_t nrows = ne1 * ne2 * ne3;
    స్థిరరాశి తార్కిక src_అడ్డువరుస_contig = src0->nb[0] == పరిమాణం(దశాంశం);

    కొరకు (int64_t అడ్డువరుస = ith; అడ్డువరుస < nrows; అడ్డువరుస += nth) {
        స్థిరరాశి int64_t i1 = అడ్డువరుస % ne1;
        స్థిరరాశి int64_t i2 = (అడ్డువరుస / ne1) % ne2;
        స్థిరరాశి int64_t i3 = అడ్డువరుస / (ne1 * ne2);

        దశాంశం * dst_అడ్డువరుస = dst_ptr + అడ్డువరుస * ne0;

        ఒకవేళ constexpr (circular_t) {
            // circular padding wraps around on a torus
            స్థిరరాశి int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            స్థిరరాశి int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            స్థిరరాశి int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                స్థిరరాశి int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_అడ్డువరుస[i0] = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } లేకపోతే {
            స్థిరరాశి తార్కిక interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            ఒకవేళ (!interior) {
                memset(dst_అడ్డువరుస, 0, ne0 * పరిమాణం(దశాంశం));
                కొనసాగు;
            }

            స్థిరరాశి అక్షరం * src_అడ్డువరుస = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            ఒకవేళ (src_అడ్డువరుస_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_అడ్డువరుస, 0, lp0 * పరిమాణం(దశాంశం));
                జ్ఞాపకం_నకలు(dst_అడ్డువరుస + lp0, src_అడ్డువరుస, ne00 * పరిమాణం(దశాంశం));
                memset(dst_అడ్డువరుస + lp0 + ne00, 0, rp0 * పరిమాణం(దశాంశం));
            } లేకపోతే {
                కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ఒకవేళ (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_అడ్డువరుస[i0] = *(స్థిరరాశి దశాంశం *)(src_అడ్డువరుస + (i0 - lp0)*nb00);
                    } లేకపోతే {
                        dst_అడ్డువరుస[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


శూన్య ggml_compute_forward_pad(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి తార్కిక circular = (తార్కిక) ggml_get_op_params_i32(dst, 8);
    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ఒకవేళ (circular) {
                    ggml_compute_forward_pad_f32<నిజం>(params, dst);
                } లేకపోతే {
                    ggml_compute_forward_pad_f32<అబద్ధం>(params, dst);
                }
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

శూన్య ggml_compute_forward_pad_reflect_1d(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *) dst->op_params;
    స్థిరరాశి పూర్ణాంకం p0 = opts[0];
    స్థిరరాశి పూర్ణాంకం p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    కొరకు (int64_t i3 = 0; i3 < ne3; i3++) {
        కొరకు (int64_t i2 = 0; i2 < ne2; i2++) {
            కొరకు (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                దశాంశం * ఎడమ  = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                దశాంశం * కుడి = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, ఎడమ, (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + i3*nb03 + i2*nb02 + i1*nb01));

                కొరకు (పూర్ణాంకం i0 = 1; i0 <= p0; i0++) { ఎడమ[-i0] = ఎడమ[i0];   }
                కొరకు (పూర్ణాంకం i0 = 1; i0 <= p1; i0++) { కుడి[i0] = కుడి[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

స్థిర int64_t ggml_wrap_సూచీ(int64_t i, int64_t ne) {
    ఒకవేళ (i < 0) {
        తిరిగి i + ne;
    } లేకపోతే ఒకవేళ (i >= ne) {
        తిరిగి i - ne;
    }
    తిరిగి i;
}

స్థిర శూన్య ggml_compute_forward_roll_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి దశాంశం * src_దత్తాంశం = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం s0 = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి పూర్ణాంకం s1 = ggml_get_op_params_i32(dst, 1);
    స్థిరరాశి పూర్ణాంకం s2 = ggml_get_op_params_i32(dst, 2);
    స్థిరరాశి పూర్ణాంకం s3 = ggml_get_op_params_i32(dst, 3);

    స్థిరరాశి int64_t మొత్తం = ne1 * ne2 * ne3;
    స్థిరరాశి int64_t చొప్పున_thread = (మొత్తం + params->nth) / params->nth;
    స్థిరరాశి int64_t start = params->ith * చొప్పున_thread;
    స్థిరరాశి int64_t end   = std::min(start + చొప్పున_thread, మొత్తం);

    కొరకు (int64_t i = start; i < end; ++i) {
        స్థిరరాశి int64_t i1 = i % ne1;
        స్థిరరాశి int64_t i2 = (i / ne1) % ne2;
        స్థిరరాశి int64_t i3 = i / (ne2 * ne1);
        దశాంశం * dst_అడ్డువరుస = dst_దత్తాంశం + (i3*nb3 + i2*nb2 + i1*nb1) / పరిమాణం(దశాంశం);

        స్థిరరాశి int64_t i01 = ggml_wrap_సూచీ(i1 - s1, ne01);
        స్థిరరాశి int64_t i02 = ggml_wrap_సూచీ(i2 - s2, ne02);
        స్థిరరాశి int64_t i03 = ggml_wrap_సూచీ(i3 - s3, ne03);
        స్థిరరాశి దశాంశం * src_అడ్డువరుస = src_దత్తాంశం + (i03*nb03 + i02*nb02 + i01*nb01) / పరిమాణం(దశాంశం);

        స్థిరరాశి int64_t s = ggml_wrap_సూచీ(-s0, ne00);
        స్థిరరాశి int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_అడ్డువరుస,     src_అడ్డువరుస + s);
        ggml_vec_cpy_f32(s, dst_అడ్డువరుస + n, src_అడ్డువరుస);
    }
}

శూన్య ggml_compute_forward_roll(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

స్థిర శూన్య ggml_compute_forward_arange_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి దశాంశం start = ggml_get_op_params_f32(dst, 0);
    స్థిరరాశి దశాంశం stop  = ggml_get_op_params_f32(dst, 1);
    స్థిరరాశి దశాంశం అడుగు  = ggml_get_op_params_f32(dst, 2);

    స్థిరరాశి int64_t అడుగులు = (int64_t) ceilf((stop - start) / అడుగు);

    GGML_ASSERT(ggml_nelements(dst) == అడుగులు);

    కొరకు (int64_t i = ith; i < అడుగులు; i+= nth) {
        దశాంశం value = start + అడుగు * i;
        ((దశాంశం *)dst->దత్తాంశం)[i] = value;
    }
}

శూన్య ggml_compute_forward_arange(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {
    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_compute_forward_timestep_embedding_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం dim = ggml_get_op_params_i32(dst, 0);
    స్థిరరాశి పూర్ణాంకం max_period = ggml_get_op_params_i32(dst, 1);

    పూర్ణాంకం half = dim / 2;

    కొరకు (int64_t i = 0; i < ne00; i++) {
        దశాంశం * embed_దత్తాంశం = (దశాంశం *)((అక్షరం *)  dst->దత్తాంశం +  i*nb1);
        కొరకు (int64_t j = ith; j < half; j += nth) {
            దశాంశం timestep = ((దశాంశం *)src0->దత్తాంశం)[i];
            దశాంశం freq = (దశాంశం)expf(-logf(max_period) * j / half);
            దశాంశం arg = timestep * freq;
            embed_దత్తాంశం[j] = cosf(arg);
            embed_దత్తాంశం[j + half] = sinf(arg);
        }
        ఒకవేళ (dim % 2 != 0 && ith == 0) {
            embed_దత్తాంశం[2 * half] = 0.f;
        }
    }
}

శూన్య ggml_compute_forward_timestep_embedding(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

మూస<గణన ggml_క్రమీకరించు_క్రమం క్రమం>
నిర్మాణం cmp_argsort {
    స్థిరరాశి దశాంశం * దత్తాంశం;
    తార్కిక క్రియాకారకం()(int32_t a, int32_t b) స్థిరరాశి {
        ఒకవేళ constexpr (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC) {
            తిరిగి దత్తాంశం[a] < దత్తాంశం[b];
        } లేకపోతే {
            తిరిగి దత్తాంశం[a] > దత్తాంశం[b];
        }
    }
};

స్థిర శూన్య ggml_compute_forward_argsort_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t nr = ggml_nrows(src0);

    ggml_క్రమీకరించు_క్రమం క్రమం = (ggml_క్రమీకరించు_క్రమం) ggml_get_op_params_i32(dst, 0);

    కొరకు (int64_t i = ith; i < nr; i += nth) {
        స్థిరరాశి దశాంశం * src_దత్తాంశం = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i*nb01);

        int32_t * dst_దత్తాంశం = (int32_t *)((అక్షరం *) dst->దత్తాంశం + i*nb1);

        కొరకు (int64_t j = 0; j < ne0; j++) {
            dst_దత్తాంశం[j] = j;
        }

        ఎంపిక (క్రమం) {
            స్థితి GGML_క్రమీకరించు_క్రమం_ASC:
                std::sort(dst_దత్తాంశం, dst_దత్తాంశం + ne0, cmp_argsort<GGML_క్రమీకరించు_క్రమం_ASC>{src_దత్తాంశం});
                ఆపు;

            స్థితి GGML_క్రమీకరించు_క్రమం_DESC:
                std::sort(dst_దత్తాంశం, dst_దత్తాంశం + ne0, cmp_argsort<GGML_క్రమీకరించు_క్రమం_DESC>{src_దత్తాంశం});
                ఆపు;

            అప్రమేయ:
                GGML_ABORT("invalid sort order");
        }
    }
}

శూన్య ggml_compute_forward_argsort(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

నిర్మాణం cmp_శిఖరం_k {
    స్థిరరాశి దశాంశం * దత్తాంశం;
    తార్కిక క్రియాకారకం()(int32_t a, int32_t b) స్థిరరాశి {
        తిరిగి దత్తాంశం[a] > దత్తాంశం[b];
    }
};

స్థిర శూన్య ggml_compute_forward_శిఖరం_k_f32(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t nr = ggml_nrows(src0);

    స్థిరరాశి పూర్ణాంకం శిఖరం_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_పంక్తి_కొలత_F32) * ith;

    కొరకు (int64_t i = ith; i < nr; i += nth) {
        స్థిరరాశి దశాంశం * src_దత్తాంశం = (దశాంశం *)((అక్షరం *) src0->దత్తాంశం + i*nb01);

        కొరకు (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + శిఖరం_k, tmp + ne00, cmp_శిఖరం_k{src_దత్తాంశం});

        int32_t * dst_దత్తాంశం = (int32_t *)((అక్షరం *) dst->దత్తాంశం + i*nb1);

        std::copy(tmp, tmp + శిఖరం_k, dst_దత్తాంశం);

        // emphasize that the order is not important
        ఒకవేళ (శిఖరం_k > 1) {
            std::swap(dst_దత్తాంశం[0], dst_దత్తాంశం[1]);
        }
    }
}

శూన్య ggml_compute_forward_శిఖరం_k(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_శిఖరం_k_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst,
        పూర్ణాంకం ir0, పూర్ణాంకం ir1,
        int64_t ic_ప్రారంభం, int64_t ic_చివర,
        దశాంశం * partials, int64_t partial_stride) {

    స్థిరరాశి తార్కిక write_partials = (partials != శూన్యసూచిక);
    స్థిరరాశి ggml_tensor * q     = dst->src[0];
    స్థిరరాశి ggml_tensor * k     = dst->src[1];
    స్థిరరాశి ggml_tensor * v     = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    స్థిరరాశి int64_t DK = nek0;
    స్థిరరాశి int64_t DV = nev0;
    స్థిరరాశి int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_కొలత(q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(k->type));
    GGML_ASSERT(nbv0 == ggml_type_కొలత(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    స్థిరరాశి int64_t rk2 = neq2/nek2;
    స్థిరరాశి int64_t rk3 = neq3/nek3;

    స్థిరరాశి int64_t rv2 = neq2/nev2;
    స్థిరరాశి int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    దశాంశం scale         = 1.0f;
    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;

    జ్ఞాపకం_నకలు(&scale,         (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias,      (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (దశాంశం *) dst->op_params + 2, పరిమాణం(దశాంశం));

    ఒకవేళ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    స్థిరరాశి uint32_t n_తల      = neq2;
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floor(log2(n_తల));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    ggml_type         స్థిరరాశి k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t స్థిరరాశి q_గమ్యం_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    స్థిరరాశి kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_గమ్యం_float_t   స్థిరరాశి v_గమ్యం_float     = ggml_get_type_traits(v->type)->గమ్యం_float;

    GGML_ASSERT((                            q_గమ్యం_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_గమ్యం_float  ) && "fattn: unsupported V-type");

    పూర్ణాంకం ith = params->ith;

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // q indices
        స్థిరరాశి పూర్ణాంకం iq3 = ir/(neq2*neq1);
        స్థిరరాశి పూర్ణాంకం iq2 = (ir - iq3*neq2*neq1)/neq1;
        స్థిరరాశి పూర్ణాంకం iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        స్థిరరాశి uint32_t h = iq2; // head index
        స్థిరరాశి దశాంశం వాలు = (max_bias > 0.0f) ? h < n_తల_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_తల_log2) + 1) : 1.0f;

        దశాంశం S = 0.0f;      //కూడిక
        దశాంశం M = -INFINITY; // maximum KQ value

        దశాంశం       * VKQ32 = (దశాంశం       *) params->wdata + ith*(1*DK + 2*DV + CACHE_పంక్తి_కొలత_F32); // FP32 VKQ accumulator
        దశాంశం       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        ఒకవేళ (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*పరిమాణం(ggml_fp16_t));
        } లేకపోతే {
            memset(VKQ32, 0, DV*పరిమాణం(దశాంశం));
        }

        స్థిరరాశి ggml_fp16_t * mp = ముసుగు ? (ggml_fp16_t *)((అక్షరం *) ముసుగు->దత్తాంశం + iq1*ముసుగు->nb[1] + (iq2%ముసుగు->ne[2])*ముసుగు->nb[2] + (iq3%ముసుగు->ne[3])*ముసుగు->nb[3]) : ఖాళీ;

        // k indices
        స్థిరరాశి పూర్ణాంకం ik3 = iq3 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2 = iq2 / rk2;

        // v indices
        స్థిరరాశి పూర్ణాంకం iv3 = iq3 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2 = iq2 / rv2;

        స్థిరరాశి దశాంశం * pq = (స్థిరరాశి దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_గమ్యం_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        కొరకు (int64_t ic = ic_ప్రారంభం; ic < ic_చివర; ++ic) {
            స్థిరరాశి దశాంశం mv = mp ? వాలు*GGML_CPU_FP16_గమ్యం_FP32(mp[ic]) : 0.0f;
            ఒకవేళ (mv == -INFINITY) {
                కొనసాగు;
            }

            దశాంశం s; // KQ value

            స్థిరరాశి అక్షరం * k_దత్తాంశం = (స్థిరరాశి అక్షరం *) k->దత్తాంశం + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_దత్తాంశం, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            ఒకవేళ (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            స్థిరరాశి దశాంశం Mold = M;

            దశాంశం ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            దశాంశం vs = 1.0f; // post-softmax KQ value, expf(s - M)

            స్థిరరాశి అక్షరం * v_దత్తాంశం = ((స్థిరరాశి అక్షరం *) v->దత్తాంశం + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            ఒకవేళ (v->type == GGML_TYPE_F16) {
                ఒకవేళ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } లేకపోతే {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (స్థిరరాశి ggml_fp16_t *) v_దత్తాంశం, vs);
            } లేకపోతే {
                ఒకవేళ (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } లేకపోతే {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ఒకవేళ (v_గమ్యం_float) {
                    v_గమ్యం_float(v_దత్తాంశం, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } లేకపోతే {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (స్థిరరాశి దశాంశం *) v_దత్తాంశం, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        ఒకవేళ (v->type == GGML_TYPE_F16) {
            కొరకు (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_గమ్యం_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        ఒకవేళ (sinks && ic_ప్రారంభం == 0) {
            స్థిరరాశి దశాంశం s = ((దశాంశం *)((అక్షరం *) sinks->దత్తాంశం))[h];

            దశాంశం ms = 1.0f;
            దశాంశం vs = 1.0f;

            ఒకవేళ (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } లేకపోతే {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        ఒకవేళ (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            దశాంశం * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            జ్ఞాపకం_నకలు(partial + 2, VKQ32, DV * పరిమాణం(దశాంశం));
        } లేకపోతే {
            // V /= S
            స్థిరరాశి దశాంశం S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            స్థిరరాశి పూర్ణాంకం i1 = iq1;
            స్థిరరాశి పూర్ణాంకం i2 = iq2;
            స్థిరరాశి పూర్ణాంకం i3 = iq3;

            // permute(0, 2, 1, 3)
            జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

స్థిర శూన్య ggml_compute_forward_flash_attn_ext_tiled(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst,
        పూర్ణాంకం ir0, పూర్ణాంకం ir1) {
    స్థిరరాశి ggml_tensor * q     = dst->src[0];
    స్థిరరాశి ggml_tensor * k     = dst->src[1];
    స్థిరరాశి ggml_tensor * v     = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు  = dst->src[3];
    స్థిరరాశి ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    స్థిరరాశి int64_t DK = nek0;
    స్థిరరాశి int64_t DV = nev0;
    స్థిరరాశి int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_కొలత(q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(k->type));
    GGML_ASSERT(nbv0 == ggml_type_కొలత(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    స్థిరరాశి ggml_type kv_type = k->type;


    // broadcast factors
    స్థిరరాశి int64_t rk2 = neq2/nek2;
    స్థిరరాశి int64_t rk3 = neq3/nek3;

    స్థిరరాశి int64_t rv2 = neq2/nev2;
    స్థిరరాశి int64_t rv3 = neq3/nev3;

    దశాంశం scale         = 1.0f;
    దశాంశం max_bias      = 0.0f;
    దశాంశం logit_softcap = 0.0f;

    జ్ఞాపకం_నకలు(&scale,         (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias,      (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (దశాంశం *) dst->op_params + 2, పరిమాణం(దశాంశం));

    ఒకవేళ (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    స్థిరరాశి uint32_t n_తల      = neq2;
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floor(log2(n_తల));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);

    పూర్ణాంకం ith = params->ith;

    స్థిర constexpr పూర్ణాంకం Q_TILE_SZ  = ggml_fa_tile_config::Q;
    స్థిర constexpr పూర్ణాంకం KV_TILE_SZ = ggml_fa_tile_config::KV;

    పూర్ణాంకం ir = ir0;
    అయితే (ir < ir1) {
        // q indices for the start of this tile
        స్థిరరాశి పూర్ణాంకం iq3 = ir/(neq2*neq1);
        స్థిరరాశి పూర్ణాంకం iq2 = (ir - iq3*neq2*neq1)/neq1;
        స్థిరరాశి పూర్ణాంకం iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        స్థిరరాశి పూర్ణాంకం tile_అడ్డువరుసలు = MIN(Q_TILE_SZ, MIN((పూర్ణాంకం)(ir1 - ir), (పూర్ణాంకం)(neq1 - iq1)));
        GGML_ASSERT(tile_అడ్డువరుసలు > 0);

        స్థిరరాశి uint32_t h = iq2; // head index
        స్థిరరాశి దశాంశం వాలు = (max_bias > 0.0f) ? h < n_తల_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_తల_log2) + 1) : 1.0f;

        దశాంశం S[Q_TILE_SZ];
        దశాంశం M[Q_TILE_SZ];

        కొరకు (పూర్ణాంకం i = 0 ; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -INFINITY;
        }

        // Per-thread scratch layout:
        // Q_q:    Q_TILE_SZ * DK (converted Q tile — F32 for GEMM, KV type for scalar)
        // KQ:     Q_TILE_SZ * KV_TILE_SZ (attention scores in float)
        // mask:   Q_TILE_SZ * KV_TILE_SZ (mask in float)
        // VKQ32:  Q_TILE_SZ * DV (FP32 output accumulator)
        // V32:    KV_TILE_SZ * DV (F32 buffer for V tile)
        // K_f32:  KV_TILE_SZ * DK (F32 buffer for K tile — GEMM path)
        దశాంశం * పునాది  = (దశాంశం *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_పంక్తి_కొలత_F32);

        శూన్య  * Q_q    = పునాది;
        దశాంశం * KQ     = (దశాంశం *)((అక్షరం *)పునాది + Q_TILE_SZ * DK * పరిమాణం(దశాంశం));
        దశాంశం * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        దశాంశం * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        దశాంశం * V32    = VKQ32 + Q_TILE_SZ * DV;
        దశాంశం * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * పరిమాణం(దశాంశం));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * పరిమాణం(దశాంశం));

        // k indices
        స్థిరరాశి పూర్ణాంకం ik3 = iq3 / rk3;
        స్థిరరాశి పూర్ణాంకం ik2 = iq2 / rk2;

        // v indices
        స్థిరరాశి పూర్ణాంకం iv3 = iq3 / rv3;
        స్థిరరాశి పూర్ణాంకం iv2 = iq2 / rv2;

        {
            దశాంశం * Q_f32 = (దశాంశం *)Q_q;
            కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                స్థిరరాశి దశాంశం * pq = (స్థిరరాశి దశాంశం *) ((అక్షరం *) q->దత్తాంశం + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                జ్ఞాపకం_నకలు(Q_f32 + tq * DK, pq, DK * పరిమాణం(దశాంశం));
            }
            కొరకు (పూర్ణాంకం tq = tile_అడ్డువరుసలు; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * పరిమాణం(దశాంశం));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * పరిమాణం(దశాంశం));
        memset(V32,   0, KV_TILE_SZ * DV * పరిమాణం(దశాంశం));

        కొరకు (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            స్థిరరాశి పూర్ణాంకం kv_tile = (పూర్ణాంకం)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            ఒకవేళ (ముసుగు) {
                తార్కిక can_skip = నిజం;
                కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                    స్థిరరాశి ggml_fp16_t * mp_అడ్డువరుస = (స్థిరరాశి ggml_fp16_t *)((స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం + (iq1 + tq)*ముసుగు->nb[1] + (iq2%ముసుగు->ne[2])*ముసుగు->nb[2] + (iq3%ముసుగు->ne[3])*ముసుగు->nb[3]);
                    కొరకు (పూర్ణాంకం tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = వాలు * GGML_CPU_FP16_గమ్యం_FP32(mp_అడ్డువరుస[ic + tk]);
                        ఒకవేళ (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = అబద్ధం;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    కొరకు (పూర్ణాంకం tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                ఒకవేళ (can_skip) {
                    కొనసాగు;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            కొరకు (పూర్ణాంకం tk = 0; tk < kv_tile; tk++) {
                స్థిరరాశి అక్షరం * k_దత్తాంశం = (స్థిరరాశి అక్షరం *)k->దత్తాంశం + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                ఒకవేళ (kv_type == GGML_TYPE_F16) {
                    స్థిరరాశి ggml_fp16_t * k_f16 = (స్థిరరాశి ggml_fp16_t *)k_దత్తాంశం;
                    కొరకు (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_గమ్యం_FP32(k_f16[dk]);
                    }
                } లేకపోతే {
                    స్థిరరాశి దశాంశం * k_f32_src = (స్థిరరాశి దశాంశం *)k_దత్తాంశం;
                    కొరకు (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * పరిమాణం(దశాంశం));
            simd_gemm(KQ, (స్థిరరాశి దశాంశం *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            ఒకవేళ (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            ఒకవేళ (kv_tile < KV_TILE_SZ) {
                కొరకు (పూర్ణాంకం tq = 0; tq < Q_TILE_SZ; tq++) {
                    కొరకు (పూర్ణాంకం tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            ఒకవేళ (ముసుగు) {
                ggml_vec_కూడు_f32(tile_అడ్డువరుసలు * KV_TILE_SZ, KQ, KQ, mask32);
            }

            తార్కిక skip[Q_TILE_SZ] = {};

            కొరకు (పూర్ణాంకం tq = 0; tq < Q_TILE_SZ; tq++) {
                దశాంశం * kq_అడ్డువరుస = KQ + tq * KV_TILE_SZ;

                దశాంశం tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_అడ్డువరుస);

                ఒకవేళ (tile_max == -INFINITY) {
                    skip[tq] = నిజం;
                    కొనసాగు;
                }

                స్థిరరాశి దశాంశం Mold = M[tq];
                స్థిరరాశి దశాంశం Mnew = fmaxf(Mold, tile_max);

                ఒకవేళ (Mnew > Mold) {
                    స్థిరరాశి దశాంశం ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_అడ్డువరుస, kq_అడ్డువరుస, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            కొరకు (పూర్ణాంకం tk = 0; tk < kv_tile; tk++) {
                స్థిరరాశి అక్షరం * v_దత్తాంశం = (స్థిరరాశి అక్షరం *)v->దత్తాంశం + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                ఒకవేళ (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_గమ్యం_fp32((స్థిరరాశి ggml_fp16_t *)v_దత్తాంశం, V32 + tk * DV, DV);
                } లేకపోతే {
                    జ్ఞాపకం_నకలు(V32 + tk * DV, v_దత్తాంశం, DV * పరిమాణం(దశాంశం));
                }
            }
            కొరకు (పూర్ణాంకం tq = 0; tq < Q_TILE_SZ; tq++) {
                ఒకవేళ (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * పరిమాణం(దశాంశం));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        ఒకవేళ (sinks) {
            స్థిరరాశి దశాంశం s = ((దశాంశం *)((అక్షరం *) sinks->దత్తాంశం))[h];

            కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
                దశాంశం ms = 1.0f;
                దశాంశం vs = 1.0f;

                ఒకవేళ (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } లేకపోతే {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        కొరకు (పూర్ణాంకం tq = 0; tq < tile_అడ్డువరుసలు; tq++) {
            // V /= S
            స్థిరరాశి దశాంశం S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            స్థిరరాశి పూర్ణాంకం i1 = iq1 + tq;
            స్థిరరాశి పూర్ణాంకం i2 = iq2;
            స్థిరరాశి పూర్ణాంకం i3 = iq3;

            // permute(0, 2, 1, 3)
            జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_అడ్డువరుసలు;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
స్థిర శూన్య ggml_flash_attn_ext_reduce_partials(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst,
        స్థిరరాశి int64_t n_chunks,
        స్థిరరాశి int64_t chunk_కొలత) {

    స్థిరరాశి ggml_tensor * q = dst->src[0];
    స్థిరరాశి ggml_tensor * k = dst->src[1];
    స్థిరరాశి ggml_tensor * v = dst->src[2];

    స్థిరరాశి int64_t DK        = k->ne[0];
    స్థిరరాశి int64_t DV        = v->ne[0];
    స్థిరరాశి int64_t nek1      = k->ne[1];
    స్థిరరాశి int64_t n_q_heads = q->ne[2];

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t wdata_చొప్పున_thread = DK + 2*DV + CACHE_పంక్తి_కొలత_F32;
    దశాంశం *       thread_wdata     = (దశాంశం *) params->wdata + ith * wdata_చొప్పున_thread;

    స్థిరరాశి int64_t partials_స్థానభ్రంశం  = nth * (DK + 2*DV + CACHE_పంక్తి_కొలత_F32);
    స్థిరరాశి int64_t partial_కొలత     = 2 + DV;
    స్థిరరాశి దశాంశం * partials_పునాది    = (స్థిరరాశి దశాంశం *) params->wdata + partials_స్థానభ్రంశం;

    // Output layout
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    కొరకు (int64_t q_తల = ith; q_తల < n_q_heads; q_తల += nth) {
        దశాంశం   M_final   = -INFINITY;
        దశాంశం   S_final   = 0.0f;
        దశాంశం * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * పరిమాణం(దశాంశం));

        // Combine partials from all chunks
        కొరకు (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            స్థిరరాశి int64_t ic_ప్రారంభం = chunk_idx * chunk_కొలత;
            ఒకవేళ (ic_ప్రారంభం >= nek1) కొనసాగు;

            స్థిరరాశి దశాంశం * partial   = partials_పునాది + (q_తల * n_chunks + chunk_idx) * partial_కొలత;
            స్థిరరాశి దశాంశం   M_chunk   = partial[0];
            స్థిరరాశి దశాంశం   S_chunk   = partial[1];
            స్థిరరాశి దశాంశం * VKQ_chunk = partial + 2;

            ఒకవేళ (S_chunk == 0.0f) కొనసాగు;

            స్థిరరాశి దశాంశం M_new     = fmaxf(M_final, M_chunk);
            స్థిరరాశి దశాంశం scale_old = expf(M_final - M_new);
            స్థిరరాశి దశాంశం scale_new = expf(M_chunk - M_new);

            కొరకు (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        ఒకవేళ (S_final != 0.0f) {
            స్థిరరాశి దశాంశం S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం + (0*ne2*ne1 + q_తల + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

స్థిర శూన్య ggml_compute_forward_flash_attn_ext_f16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * q     = dst->src[0];
    స్థిరరాశి ggml_tensor * k     = dst->src[1];
    స్థిరరాశి ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    స్థిరరాశి int64_t DK = nek0;
    స్థిరరాశి int64_t DV = nev0;
    స్థిరరాశి int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_కొలత(q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(k->type));
    GGML_ASSERT(nbv0 == ggml_type_కొలత(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    స్థిరరాశి తార్కిక use_ref = params->use_ref;

    స్థిరరాశి తార్కిక kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    స్థిరరాశి తార్కిక use_విభజించు_kv_మార్గం = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    ఒకవేళ (use_విభజించు_kv_మార్గం) {
        స్థిరరాశి int64_t chunk_కొలత = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        స్థిరరాశి int64_t partial_కొలత  = 2 + DV;
        దశాంశం *       partials_పునాది = (దశాంశం *) params->wdata + nth * (DK + 2*DV + CACHE_పంక్తి_కొలత_F32);

        స్థిరరాశి int64_t ic_ప్రారంభం = ith * chunk_కొలత;
        స్థిరరాశి int64_t ic_చివర   = std::min(ic_ప్రారంభం + chunk_కొలత, nek1);

        స్థిరరాశి int64_t partial_stride = nth * partial_కొలత;
        దశాంశం *       chunk_partials = partials_పునాది + ith * partial_కొలత;

        ఒకవేళ (ic_ప్రారంభం < nek1) {
            కొరకు (int64_t q_తల = 0; q_తల < neq2; q_తల++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_తల, q_తల + 1, ic_ప్రారంభం, ic_చివర,
                    chunk_partials, partial_stride);
            }
        } లేకపోతే {
            కొరకు (int64_t q_తల = 0; q_తల < neq2; q_తల++) {
                దశాంశం * q_partials = chunk_partials + q_తల * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_కొలత);
    } లేకపోతే {

        // total rows in q
        స్థిరరాశి int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        స్థిరరాశి తార్కిక disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        పూర్ణాంకం nth_scaled = nth * 4;
        int64_t chunk_కొలత = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_కొలత - 1) / chunk_కొలత;

        ఒకవేళ (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        ఒకవేళ (ith == 0) {
            ggml_threadpool_chunk_సమితి(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        స్థిరరాశి int64_t dr = (nr + nchunk - 1) / nchunk;

        స్థిర constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        తార్కిక use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#ఒకవేళ defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#ఒకవేళ defined(__ARM_FEATURE_SVE)
        స్థిరరాశి int64_t f32_epr = svcntw();
#లేకపోతే
        స్థిరరాశి int64_t f32_epr = GGML_F32_EPR;
#అయితే_ముగింపు
        use_tiled &= (DV % f32_epr == 0);
#అయితే_ముగింపు
        పూర్ణాంకం current_chunk = ith;

        అయితే (current_chunk < nchunk) {
            స్థిరరాశి int64_t ir0 = dr * current_chunk;
            స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

            ఒకవేళ (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } లేకపోతే {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, శూన్యసూచిక, 0);
            }

            current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
        }
    }
}

శూన్య ggml_compute_forward_flash_attn_ext(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ఎంపిక (dst->op_params[3]) {
        స్థితి GGML_PREC_DEFAULT:
        స్థితి GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

స్థిర శూన్య ggml_compute_forward_flash_attn_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
        స్థిరరాశి తార్కిక masked,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * q = dst->src[0];
    స్థిరరాశి ggml_tensor * k = dst->src[1];
    స్థిరరాశి ggml_tensor * v = dst->src[2];
    స్థిరరాశి ggml_tensor * d = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ned, d,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbd, d,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t D = neq0;
    స్థిరరాశి int64_t N = neq1;
    స్థిరరాశి int64_t P = nek1 - N;
    స్థిరరాశి int64_t M = P + N;

    స్థిరరాశి పూర్ణాంకం Mup  = ggml_పైకి(M, GGML_SOFT_MAX_UNROLL);
    స్థిరరాశి పూర్ణాంకం mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nbk0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nbv0 == పరిమాణం(దశాంశం));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ఒకవేళ (ith == 0) {
        memset(dst->దత్తాంశం, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    స్థిరరాశి int64_t elem_q = ggml_nelements(q);
    స్థిరరాశి int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_కొలత(result_type) == 1);
    స్థిరరాశి size_t tsize = ggml_type_కొలత(result_type);

    స్థిరరాశి size_t offs_q = 0;
    స్థిరరాశి size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    స్థిరరాశి size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    శూన్య * grad_q = (అక్షరం *) dst->దత్తాంశం;
    శూన్య * grad_k = (అక్షరం *) dst->దత్తాంశం + offs_k;
    శూన్య * grad_v = (అక్షరం *) dst->దత్తాంశం + offs_v;

    స్థిరరాశి size_t nbgq1 = nb0*neq0;
    స్థిరరాశి size_t nbgq2 = nb0*neq0*neq1;
    స్థిరరాశి size_t nbgq3 = nb0*neq0*neq1*neq2;

    స్థిరరాశి size_t nbgk1 = nb0*nek0;
    స్థిరరాశి size_t nbgk2 = nb0*nek0*nek1;
    స్థిరరాశి size_t nbgk3 = nb0*nek0*nek1*neq2;

    స్థిరరాశి size_t nbgv1 = nb0*nev0;
    స్థిరరాశి size_t nbgv2 = nb0*nev0*nev1;
    స్థిరరాశి size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    స్థిరరాశి పూర్ణాంకం nr = nek2*nek3;

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి దశాంశం scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    పూర్ణాంకం nrep = neq2/nek2;

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        // q indices
        స్థిరరాశి పూర్ణాంకం ik3 = ir/(nek2);
        స్థిరరాశి పూర్ణాంకం ik2 = ir - ik3*nek2;

        స్థిరరాశి పూర్ణాంకం iq3 = ik3;
        స్థిరరాశి పూర్ణాంకం id3 = ik3;
        స్థిరరాశి పూర్ణాంకం iv3 = ik3;
        స్థిరరాశి పూర్ణాంకం iv2 = ik2;

        కొరకు (పూర్ణాంకం irep = 0; irep < nrep; ++irep) {
            స్థిరరాశి పూర్ణాంకం iq2 = ik2 + irep*nek2;
            స్థిరరాశి పూర్ణాంకం id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            కొరకు (పూర్ణాంకం iq1 = 0; iq1 < neq1; ++iq1) {
                స్థిరరాశి పూర్ణాంకం id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                దశాంశం * S  = (దశాంశం *) params->wdata + ith*2*(mxDM + CACHE_పంక్తి_కొలత_F32) + 0*(mxDM+CACHE_పంక్తి_కొలత_F32);
                దశాంశం * SM = (దశాంశం *) params->wdata + ith*2*(mxDM + CACHE_పంక్తి_కొలత_F32) + 1*(mxDM+CACHE_పంక్తి_కొలత_F32);

                కొరకు (పూర్ణాంకం i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                స్థిరరాశి int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                కొరకు (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    స్థిరరాశి పూర్ణాంకం ik1 = ic;

                    // S indices
                    స్థిరరాశి పూర్ణాంకం i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (దశాంశం *) ((అక్షరం *) k->దత్తాంశం + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                కొరకు (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    దశాంశం max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float కూడిక = 0.0;
                    {
#నిర్వచితమైతే GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_కూడిక_f32(Mup, &కూడిక, SM);
#లేకపోతే
                        కూడిక = ggml_vec_soft_max_f32(Mup, SM, S, max);
#అయితే_ముగింపు
                    }

                    ప్రతిజ్ఞా(కూడిక > 0.0);

                    కూడిక = 1.0/కూడిక;
                    ggml_vec_scale_f32(masked_begin, SM, కూడిక);

                }

                // step-by-step explanation
                {
                    // forward-process                    shape      grads from backward process
                    // parallel_for ik2,ik3:
                    //  for irep:
                    //   iq2 = ik2 + irep*nek2
                    //   k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,ik2,ik3]  += grad[kcur]
                    //   q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
                    //   v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iv2,iv3]  += grad[vcur]
                    //   for iq1:
                    //    kcur   = k[:D,:M,ik2,ik3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
                    //    qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
                    //    vcur   = v[:M,:D,iv2,iv3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
                    //    S0     = -Inf                   [D,1,1,1]
                    //   ~S1[i]  = dot(kcur[:D,i], qcur)
                    //    S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
                    //    S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
                    //    S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    //    S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
                    //   ~S5[i]  = dot(vcur[:,i], S4)
                    //    S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,id1,id2,id3]
                    //   ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
                    //    dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,id1,id2,id3]
                    // dst                               backward-/ grad[dst]                 = d
                    //
                    // output gradients with their dependencies:
                    //
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S4]   = grad[S5] @ vcur
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[vcur] = grad[S5].T @ S4
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // in post-order:
                    //
                    // S1         = qcur @ kcur.T
                    // S2         = S1 * scale
                    // S3         = diag_mask_inf(S2, P)
                    // S4         = softmax(S3)
                    // grad[S4]   = d[:D,id1,id2,id3] @ vcur
                    // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
                    // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
                    // grad[qcur] = grad[S1]   @ kcur
                    // grad[kcur] = grad[S1].T @ qcur
                    // grad[vcur] = d[:D,id1,id2,id3].T @ S4
                    //
                    // using less variables (SM=S4):
                    //
                    // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
                    // SM            = softmax(S)
                    // S             = d[:D,iq1,iq2,iq3] @ vcur
                    // dot_SM_gradSM = dot(SM, S)
                    // S             = SM * (S - dot(SM, S))
                    // S             = diag_mask_zero(S, P) * scale
                    //
                    // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
                    // grad[k][:D,:M,ik2,ik3]  += S.T @ qcur
                    // grad[v][:M,:D,iv2,iv3]  += d[:D,id1,id2,id3].T @ SM
                }

                // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // S = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
                // for ic:
                //   S[:M] += vcur[:M,ic,iv2,iv3] * d[ic,id1,id2,id3]
                // exclude known future zero S[..] values from operation
                ggml_vec_సమితి_f32(masked_begin, S, 0);
                కొరకు (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (దశాంశం *) ((అక్షరం *) v->దత్తాంశం + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(దశాంశం *) ((అక్షరం *) d->దత్తాంశం + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                దశాంశం dot_SM_gradSM = 0;
                ggml_vec_dot_f32 (masked_begin, &dot_SM_gradSM, 0, SM, 0, S, 0, 1);
                ggml_vec_acc1_f32(M, S, -dot_SM_gradSM);
                ggml_vec_mul_f32 (masked_begin, S, S, SM);

                // S = diag_mask_zero(S, P) * scale
                // already done by above ggml_vec_set_f32

                // exclude known zero S[..] values from operation
                ggml_vec_scale_f32(masked_begin, S, scale);

                // S    shape [M,1]
                // SM   shape [M,1]
                // kcur shape [D,M]
                // qcur shape [D,1]
                // vcur shape [M,D]

                // grad[q][:D,iq1,iq2,iq3] += S @ kcur
                // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
                // for ic:
                //  grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic,ik2,ik3]
                // exclude known zero S[..] values from loop
                కొరకు (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (దశాంశం *) ((అక్షరం *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (దశాంశం *) ((అక్షరం *) k->దత్తాంశం + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                కొరకు (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (దశాంశం *) ((అక్షరం *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                కొరకు (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (దశాంశం *) ((అక్షరం *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(దశాంశం *) ((అక్షరం *) d->దత్తాంశం + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_flash_attn_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        స్థిరరాశి తార్కిక masked,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * q = dst->src[0];

    ఎంపిక (q->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_వెనుక_f32(params, masked, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

స్థిర శూన్య ggml_compute_forward_ssm_conv_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0]; // conv_x
    స్థిరరాశి ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nc  = src1->ne[0]; // d_conv
    స్థిరరాశి పూర్ణాంకం ncs = src0->ne[0]; // d_conv - 1 + n_t
    స్థిరరాశి పూర్ణాంకం nr  = src0->ne[1]; // d_inner
    స్థిరరాశి పూర్ణాంకం n_t =  dst->ne[1]; // tokens per sequence
    స్థిరరాశి పూర్ణాంకం n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src1->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);
    స్థిరరాశి పూర్ణాంకం ir  = ir1 - ir0;

    కొరకు (పూర్ణాంకం i3 = 0; i3 < n_s; ++i3) {
        కొరకు (పూర్ణాంకం i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            స్థిరరాశి దశాంశం * s = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            స్థిరరాశి దశాంశం * c = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + ir0*(src1->nb[1])); // {d_conv, d_inner}
            దశాంశం * x = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            కొరకు (పూర్ణాంకం i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                దశాంశం sumf = 0.0f;

                // d_conv
                కొరకు (పూర్ణాంకం i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

శూన్య ggml_compute_forward_ssm_conv(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ఎంపిక (dst->src[0]->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

స్థిర శూన్య ggml_compute_forward_ssm_పరిశీలన_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    స్థిరరాశి ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    స్థిరరాశి ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    స్థిరరాశి ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    స్థిరరాశి ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    స్థిరరాశి ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    స్థిరరాశి ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t nc = src0->ne[0]; // d_state
    స్థిరరాశి int64_t nr = src0->ne[1]; // dim
    స్థిరరాశి int64_t nh = src1->ne[1]; // n_head
    స్థిరరాశి int64_t ng = src4->ne[1];
    స్థిరరాశి int64_t nt = src1->ne[2]; // number of tokens per sequence
    స్థిరరాశి int64_t ns = src1->ne[3]; // number of sequences in the batch
    స్థిరరాశి int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    స్థిరరాశి int64_t s_off = ggml_nelements(src1) * ggml_element_కొలత(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src1->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src2->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src3->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src4->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src5->nb[0] == పరిమాణం(దశాంశం));
    GGML_ASSERT(src6->nb[0] == పరిమాణం(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    స్థిరరాశి పూర్ణాంకం dh = (nh + nth - 1)/nth;

    // head range for this thread
    స్థిరరాశి పూర్ణాంకం ih0 = dh*ith;
    స్థిరరాశి పూర్ణాంకం ih1 = MIN(ih0 + dh, nh);

    స్థిరరాశి int32_t * ids = (స్థిరరాశి int32_t *) src6->దత్తాంశం;

    కొరకు (పూర్ణాంకం i3 = 0; i3 < ns; ++i3) {
        స్థిరరాశి దశాంశం * s0 = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              దశాంశం * s  = (      దశాంశం *) ((      అక్షరం *) dst->దత్తాంశం  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        కొరకు (పూర్ణాంకం i2 = 0; i2 < nt; ++i2) {
            స్థిరరాశి దశాంశం * x  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            స్థిరరాశి దశాంశం * dt = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src2->దత్తాంశం + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            స్థిరరాశి దశాంశం * A  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src3->దత్తాంశం); // {d_state, nh} or {1, nh}
            స్థిరరాశి దశాంశం * B  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src4->దత్తాంశం + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            స్థిరరాశి దశాంశం * C  = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src5->దత్తాంశం + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  దశాంశం * y  = (      దశాంశం *) ((      అక్షరం *) dst->దత్తాంశం + i2*(nh*nr*పరిమాణం(దశాంశం)) + i3*(nt*nh*nr*పరిమాణం(దశాంశం))); // {dim, nh, nt, ns}

            ఒకవేళ (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                కొరకు (పూర్ణాంకం h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    స్థిరరాశి దశాంశం dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    స్థిరరాశి దశాంశం dA = expf(dt_soft_plus * A[h]);
                    స్థిరరాశి పూర్ణాంకం g = h / (nh / ng); // repeat_interleave

                    // dim
                    కొరకు (పూర్ణాంకం i1 = 0; i1 < nr; ++i1) {
                        స్థిరరాశి పూర్ణాంకం ii = i1 + h*nr;
                        స్థిరరాశి దశాంశం x_dt = x[ii] * dt_soft_plus;
                        దశాంశం sumf = 0.0f;
#ఒకవేళ defined(GGML_SIMD)
    #ఒకవేళ defined(__ARM_FEATURE_SVE)
                        స్థిరరాశి పూర్ణాంకం ggml_f32_epr = svcntw();
                        స్థిరరాశి పూర్ణాంకం ggml_f32_అడుగు = 1 * ggml_f32_epr;

                        స్థిరరాశి పూర్ణాంకం np = (nc & ~(ggml_f32_అడుగు - 1));

                        GGML_F32_VEC కూడిక = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        కొరకు (పూర్ణాంకం i = 0; i < np; i += ggml_f32_అడుగు) {
                            // TODO: maybe unroll more?
                            కొరకు (పూర్ణాంకం j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_ఎక్కించు(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_ఎక్కించు(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_ఎక్కించు(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_కూడు(t0, t1);

                                కూడిక = GGML_F32_VEC_FMA(కూడిక, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(కూడిక);
    #లేదావేళ defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        స్థిరరాశి పూర్ణాంకం np = 0;
    #లేకపోతే
                        స్థిరరాశి పూర్ణాంకం np = (nc & ~(GGML_F32_అడుగు - 1));

                        GGML_F32_VEC కూడిక[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        కొరకు (పూర్ణాంకం i = 0; i < np; i += GGML_F32_అడుగు) {
                            కొరకు (పూర్ణాంకం j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_ఎక్కించు(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_ఎక్కించు(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_ఎక్కించు(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_కూడు(ax[j], ay[j]);

                                కూడిక[j] = GGML_F32_VEC_FMA(కూడిక[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, కూడిక);
    #అయితే_ముగింపు
#లేకపోతే
                        స్థిరరాశి పూర్ణాంకం np = 0;
#అయితే_ముగింపు
                        // d_state
                        కొరకు (పూర్ణాంకం i0 = np; i0 < nc; ++i0) {
                            స్థిరరాశి పూర్ణాంకం i = i0 + ii*nc;
                            స్థిరరాశి పూర్ణాంకం ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            స్థిరరాశి దశాంశం state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } లేకపోతే {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                కొరకు (పూర్ణాంకం h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    స్థిరరాశి దశాంశం dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    స్థిరరాశి పూర్ణాంకం g = h / (nh / ng); // repeat_interleave

                    // dim
                    కొరకు (పూర్ణాంకం i1 = 0; i1 < nr; ++i1) {
                        స్థిరరాశి పూర్ణాంకం ii = i1 + h*nr;
                        స్థిరరాశి దశాంశం x_dt = x[ii] * dt_soft_plus;
#ఒకవేళ defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_సదిశ = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        కొరకు (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_ఎక్కించు(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_ఎక్కించు(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_ఎక్కించు(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_ఎక్కించు(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_సదిశ = GGML_F32_VEC_కూడు(GGML_F32_VEC_MUL(vs0, vC), r1_సదిశ);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_సదిశ);
#లేకపోతే
                        దశాంశం sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        కొరకు (పూర్ణాంకం i0 = 0; i0 < nc; ++i0) {
                            స్థిరరాశి పూర్ణాంకం i = i0 + ii*nc;
                            స్థిరరాశి పూర్ణాంకం ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            స్థిరరాశి దశాంశం state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#అయితే_ముగింపు
                    }
                }
            }
            స్థిరరాశి int64_t స్థానం = nt - 1 - i2;
            ఒకవేళ (K > 1 && స్థానం > 0 && స్థానం < K) {
                దశాంశం * s_snapshot = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + s_off + (స్థానం*ns + i3)*(src0->nb[3]));
                కొరకు (పూర్ణాంకం h = ih0; h < ih1; ++h) {
                    జ్ఞాపకం_నకలు((అక్షరం *) s_snapshot + h*src0->nb[2], (అక్షరం *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

శూన్య ggml_compute_forward_ssm_పరిశీలన(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    ఎంపిక (dst->src[0]->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_పరిశీలన_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

స్థిర శూన్య ggml_compute_forward_win_part_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_వాడని(params);

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    స్థిరరాశి int32_t nep0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t nep1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t w    = ((స్థిరరాశి int32_t *)(dst->op_params))[2];

    ప్రతిజ్ఞా(ne00 == ne0);
    ప్రతిజ్ఞా(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    కొరకు (పూర్ణాంకం py = 0; py < nep1; ++py) {
        కొరకు (పూర్ణాంకం px = 0; px < nep0; ++px) {
            స్థిరరాశి int64_t i3 = py*nep0 + px;
            కొరకు (int64_t i2 = 0; i2 < ne2; ++i2) {
                కొరకు (int64_t i1 = 0; i1 < ne1; ++i1) {
                    కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                        స్థిరరాశి int64_t i02 = py*w + i2;
                        స్థిరరాశి int64_t i01 = px*w + i1;
                        స్థిరరాశి int64_t i00 = i0;

                        స్థిరరాశి int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        స్థిరరాశి int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        ఒకవేళ (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((దశాంశం *) dst->దత్తాంశం)[i] = 0.0f;
                        } లేకపోతే {
                            ((దశాంశం *) dst->దత్తాంశం)[i] = ((దశాంశం *) src0->దత్తాంశం)[j];
                        }
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_win_part(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

స్థిర శూన్య ggml_compute_forward_win_unpart_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_వాడని(params);

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    స్థిరరాశి int32_t w = ((స్థిరరాశి int32_t *)(dst->op_params))[0];

    // padding
    స్థిరరాశి పూర్ణాంకం px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    స్థిరరాశి పూర్ణాంకం npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    ప్రతిజ్ఞా(ne0 == ne00);

    // TODO: optimize / multi-thread
    కొరకు (int64_t i2 = 0; i2 < ne2; ++i2) {
        కొరకు (int64_t i1 = 0; i1 < ne1; ++i1) {
            కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                స్థిరరాశి పూర్ణాంకం ip2 = i2/w;
                స్థిరరాశి పూర్ణాంకం ip1 = i1/w;

                స్థిరరాశి int64_t i02 = i2%w;
                స్థిరరాశి int64_t i01 = i1%w;
                స్థిరరాశి int64_t i00 = i0;

                స్థిరరాశి int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                స్థిరరాశి int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((దశాంశం *) dst->దత్తాంశం)[j] = ((దశాంశం *) src0->దత్తాంశం)[i];
            }
        }
    }
}

శూన్య ggml_compute_forward_win_unpart(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

శూన్య ggml_compute_forward_unary(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_unary_op op = ggml_get_unary_op(dst);

    ఎంపిక (op) {
        స్థితి GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_అడుగు:
            {
                ggml_compute_forward_అడుగు(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
            {
                ggml_compute_forward_gelu_శీఘ్ర(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_విడత:
            {
                ggml_compute_forward_విడత(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } ఆపు;
        స్థితి GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

శూన్య ggml_compute_forward_glu(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_glu_op op = ggml_get_glu_op(dst);

    ఎంపిక (op) {
        స్థితి GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } ఆపు;
        స్థితి GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } ఆపు;
        స్థితి GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } ఆపు;
        స్థితి GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } ఆపు;
        స్థితి GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } ఆపు;
        స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
            {
                ggml_compute_forward_geglu_శీఘ్ర(params, dst);
            } ఆపు;
        స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

స్థిర శూన్య ggml_compute_forward_get_rel_pos_f16(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_వాడని(params);

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    స్థిరరాశి int64_t w = ne1;

    ggml_fp16_t * src0_దత్తాంశం = (ggml_fp16_t *) src0->దత్తాంశం;
    ggml_fp16_t * dst_దత్తాంశం  = (ggml_fp16_t *) dst->దత్తాంశం;

    కొరకు (int64_t i2 = 0; i2 < ne2; ++i2) {
        కొరకు (int64_t i1 = 0; i1 < ne1; ++i1) {
            స్థిరరాశి int64_t pos = (w - i1 - 1) + i2;
            కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_దత్తాంశం[i2*ne1*ne0 + i1*ne0 + i0] = src0_దత్తాంశం[pos*ne00 + i0];
            }
        }
    }
}

శూన్య ggml_compute_forward_get_rel_pos(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

స్థిర శూన్య ggml_compute_forward_కూడు_rel_pos_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];

    స్థిరరాశి తార్కిక inplace = (తార్కిక) ((int32_t *) dst->op_params)[0];
    ఒకవేళ (!inplace) {
        ఒకవేళ (params->ith == 0) {
            జ్ఞాపకం_నకలు((అక్షరం *) dst->దత్తాంశం, (అక్షరం *) src0->దత్తాంశం, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    దశాంశం * src1_దత్తాంశం = (దశాంశం *) src1->దత్తాంశం;
    దశాంశం * src2_దత్తాంశం = (దశాంశం *) src2->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి int64_t ne10 = src1->ne[0];
    స్థిరరాశి int64_t ne11 = src1->ne[1];
    స్థిరరాశి int64_t ne12 = src1->ne[2];
    స్థిరరాశి int64_t ne13 = src1->ne[3];

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // total patches in dst
    స్థిరరాశి పూర్ణాంకం np = ne13;

    // patches per thread
    స్థిరరాశి పూర్ణాంకం dp = (np + nth - 1)/nth;

    // patch range for this thread
    స్థిరరాశి పూర్ణాంకం ip0 = dp*ith;
    స్థిరరాశి పూర్ణాంకం ip1 = MIN(ip0 + dp, np);

    కొరకు (int64_t i13 = ip0; i13 < ip1; ++i13) {
        కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
            కొరకు (int64_t i11 = 0; i11 < ne11; ++i11) {
                స్థిరరాశి int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                కొరకు (int64_t i10 = 0; i10 < ne10; ++i10) {
                    స్థిరరాశి int64_t jp0  = jp1 + i10;
                    స్థిరరాశి దశాంశం src1_e = src1_దత్తాంశం[jp0];
                    స్థిరరాశి దశాంశం src2_e = src2_దత్తాంశం[jp0];

                    స్థిరరాశి int64_t jdh = jp0 * ne10;
                    స్థిరరాశి int64_t jdw = jdh - (ne10 - 1) * i10;

                    కొరకు (int64_t j = 0; j < ne10; ++j) {
                        dst_దత్తాంశం[jdh + j     ] += src2_e;
                        dst_దత్తాంశం[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

శూన్య ggml_compute_forward_కూడు_rel_pos(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_కూడు_rel_pos_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

స్థిర శూన్య ggml_compute_forward_rwkv_wkv6_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి int64_t T = dst->src[1]->ne[2];
    స్థిరరాశి int64_t C = dst->ne[0];
    స్థిరరాశి int64_t HEADS = dst->src[1]->ne[1];
    స్థిరరాశి int64_t n_seqs = dst->src[5]->ne[1];
    స్థిరరాశి int64_t తల_కొలత = C / HEADS;

    దశాంశం * dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;
    దశాంశం * state = ((దశాంశం *) dst->దత్తాంశం) + C * T;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం h_ప్రారంభం =  (HEADS * (ith    )) / nth;
    స్థిరరాశి పూర్ణాంకం h_చివర   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    దశాంశం * k =          (దశాంశం *) dst->src[0]->దత్తాంశం;
    దశాంశం * v =          (దశాంశం *) dst->src[1]->దత్తాంశం;
    దశాంశం * r =          (దశాంశం *) dst->src[2]->దత్తాంశం;
    దశాంశం * సమయం_faaaa = (దశాంశం *) dst->src[3]->దత్తాంశం;
    దశాంశం * సమయం_decay = (దశాంశం *) dst->src[4]->దత్తాంశం;

    size_t t_stride = HEADS * తల_కొలత; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = తల_కొలత * తల_కొలత;

    ఒకవేళ (ith == 0) {
        memset(dst_దత్తాంశం, 0, T * C * పరిమాణం(దశాంశం));
    }
    ggml_barrier(params->threadpool);


    #ఒకవేళ defined(__AVX__) && !defined(__AVX512F__)
        #నిర్వచించు GGML_F32X GGML_F32x8
        #నిర్వచించు GGML_F32X_SET1 GGML_F32x8_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32x8_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32x8_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32x8_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32x8_FMA
        #నిర్వచించు WKV_సదిశ_కొలత 8
    #లేదావేళ defined(__AVX512F__)
        #నిర్వచించు GGML_F32X GGML_F32x16
        #నిర్వచించు GGML_F32X_SET1 GGML_F32x16_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32x16_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32x16_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32x16_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32x16_FMA
        #నిర్వచించు WKV_సదిశ_కొలత 16
    #లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #నిర్వచించు GGML_F32X GGML_F32xt
        #నిర్వచించు GGML_F32X_SET1 GGML_F32xt_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32xt_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32xt_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32xt_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32xt_FMA
        #నిర్వచించు WKV_సదిశ_కొలత 8
    #లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
        #నిర్వచించు GGML_F32X GGML_F32x4
        #నిర్వచించు GGML_F32X_SET1 GGML_F32x4_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32x4_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32x4_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32x4_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32x4_FMA
        #నిర్వచించు WKV_సదిశ_కొలత 4
    #అయితే_ముగింపు

    #నిర్వచితమైతే WKV_సదిశ_కొలత
        పూర్ణాంకం wkv_సదిశ_కొలత;
        #ఒకవేళ defined(__ARM_FEATURE_SVE)
            wkv_సదిశ_కొలత = svcntw();
        #లేకపోతే
            wkv_సదిశ_కొలత = WKV_సదిశ_కొలత;
        #అయితే_ముగింపు
        స్థిరరాశి int64_t vec_లెక్క = తల_కొలత / wkv_సదిశ_కొలత;

        కొరకు (int64_t t = 0; t < T; t++) {
            size_t t_స్థానభ్రంశం = t * t_stride;
            size_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
            దశాంశం * state_cur = state + state_స్థానభ్రంశం;
            దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[5]->దత్తాంశం + state_స్థానభ్రంశం;

            కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                size_t h_స్థానభ్రంశం = h * h_stride;
                size_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                size_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                కొరకు (int64_t i = 0; i < తల_కొలత; i++) {
                    size_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + i;
                    size_t h_i_స్థానభ్రంశం = h_స్థానభ్రంశం + i;
                    size_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + i * h_stride;

                    దశాంశం k_val = k[t_h_i_స్థానభ్రంశం];
                    దశాంశం r_val = r[t_h_i_స్థానభ్రంశం];
                    దశాంశం సమయం_faaaa_val = సమయం_faaaa[h_i_స్థానభ్రంశం];
                    దశాంశం సమయం_decay_val = సమయం_decay[t_h_i_స్థానభ్రంశం];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X సమయం_faaaa_vec = GGML_F32X_SET1(సమయం_faaaa_val);
                    GGML_F32X సమయం_decay_vec = GGML_F32X_SET1(సమయం_decay_val);

                    కొరకు (int64_t j = 0; j < vec_లెక్క; j++) {
                        size_t పునాది_j = j * wkv_సదిశ_కొలత;
                        size_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + పునాది_j;
                        size_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + పునాది_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ఎక్కించు(&v[t_h_j_స్థానభ్రంశం]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ఎక్కించు(&state_prev[h_2d_i_j_స్థానభ్రంశం]);
                        GGML_F32X dst_vec = GGML_F32_X_ఎక్కించు(&dst_దత్తాంశం[t_h_j_స్థానభ్రంశం]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X తాత్కాలిక_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, సమయం_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, తాత్కాలిక_vec, r_vec);
                        GGML_F32X_STORE(&dst_దత్తాంశం[t_h_j_స్థానభ్రంశం], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, సమయం_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_స్థానభ్రంశం], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    కొరకు (int64_t j = vec_లెక్క * wkv_సదిశ_కొలత; j < తల_కొలత; j++) {
                        size_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                        size_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;
                        దశాంశం v_val = v[t_h_j_స్థానభ్రంశం];
                        దశాంశం kv_val = v_val * k_val;
                        దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                        దశాంశం తాత్కాలిక_val = kv_val * సమయం_faaaa_val + prev_state_val;
                        dst_దత్తాంశం[t_h_j_స్థానభ్రంశం] += తాత్కాలిక_val * r_val;
                        state_cur[h_2d_i_j_స్థానభ్రంశం] = prev_state_val * సమయం_decay_val + kv_val;
                    }
                }
            }
        }

    #లేకపోతే
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        కొరకు (int64_t t = 0; t < T; t++) {
            size_t t_స్థానభ్రంశం = t * t_stride;
            size_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
            దశాంశం * state_cur = state + state_స్థానభ్రంశం;
            దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[5]->దత్తాంశం + state_స్థానభ్రంశం;

            కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                size_t h_స్థానభ్రంశం = h * h_stride;
                size_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                size_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                కొరకు (int64_t i = 0; i < తల_కొలత; i++) {
                    size_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + i;
                    size_t h_i_స్థానభ్రంశం = h_స్థానభ్రంశం + i;
                    size_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + i * h_stride;

                    దశాంశం k_val = k[t_h_i_స్థానభ్రంశం];
                    దశాంశం r_val = r[t_h_i_స్థానభ్రంశం];
                    దశాంశం సమయం_faaaa_val = సమయం_faaaa[h_i_స్థానభ్రంశం];
                    // RWKV v6: different time_decay for each token.
                    దశాంశం సమయం_decay_val = సమయం_decay[t_h_i_స్థానభ్రంశం];

                    కొరకు (int64_t j = 0; j < తల_కొలత; j++) {
                        size_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                        size_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;

                        దశాంశం v_val = v[t_h_j_స్థానభ్రంశం];
                        దశాంశం kv_val = v_val * k_val;
                        దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                        దశాంశం తాత్కాలిక_val = kv_val * సమయం_faaaa_val + prev_state_val;
                        dst_దత్తాంశం[t_h_j_స్థానభ్రంశం] += తాత్కాలిక_val * r_val;
                        state_cur[h_2d_i_j_స్థానభ్రంశం] = prev_state_val * సమయం_decay_val + kv_val;
                    }
                }
            }
        }
    #అయితే_ముగింపు
}


శూన్య ggml_compute_forward_rwkv_wkv6(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

స్థిర శూన్య ggml_compute_forward_gla_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి int64_t T = dst->src[1]->ne[2];
    స్థిరరాశి int64_t C = dst->ne[0];
    స్థిరరాశి int64_t HEADS = dst->src[1]->ne[1];
    స్థిరరాశి int64_t n_seqs = dst->src[4]->ne[1];
    స్థిరరాశి int64_t తల_కొలత = C / HEADS;
    స్థిరరాశి దశాంశం scale = ggml_get_op_params_f32(dst, 0);

    దశాంశం * dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;
    దశాంశం * state = ((దశాంశం *) dst->దత్తాంశం) + C * T;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం h_ప్రారంభం =  (HEADS * (ith    )) / nth;
    స్థిరరాశి పూర్ణాంకం h_చివర   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    దశాంశం * k = (దశాంశం *) dst->src[0]->దత్తాంశం;
    దశాంశం * v = (దశాంశం *) dst->src[1]->దత్తాంశం;
    దశాంశం * q = (దశాంశం *) dst->src[2]->దత్తాంశం;
    దశాంశం * g = (దశాంశం *) dst->src[3]->దత్తాంశం;

    size_t t_stride = HEADS * తల_కొలత; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = తల_కొలత * తల_కొలత;

    ఒకవేళ (ith == 0) {
        memset(dst_దత్తాంశం, 0, T * C * పరిమాణం(దశాంశం));
    }
    ggml_barrier(params->threadpool);


    #ఒకవేళ defined(__AVX__) && !defined(__AVX512F__)
        #నిర్వచించు GGML_F32X GGML_F32x8
        #నిర్వచించు GGML_F32X_SET1 GGML_F32x8_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32x8_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32x8_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32x8_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32x8_FMA
        #నిర్వచించు GLA_సదిశ_కొలత 8
    #లేదావేళ defined(__AVX512F__)
        #నిర్వచించు GGML_F32X GGML_F32x16
        #నిర్వచించు GGML_F32X_SET1 GGML_F32x16_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32x16_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32x16_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32x16_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32x16_FMA
        #నిర్వచించు GLA_సదిశ_కొలత 16
    #లేదావేళ defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #నిర్వచించు GGML_F32X GGML_F32xt
        #నిర్వచించు GGML_F32X_SET1 GGML_F32xt_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32xt_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32xt_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32xt_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32xt_FMA
        #నిర్వచించు GLA_సదిశ_కొలత 8
    #లేదావేళ defined(__ARM_NEON) && defined(__aarch64__)
        #నిర్వచించు GGML_F32X GGML_F32x4
        #నిర్వచించు GGML_F32X_SET1 GGML_F32x4_SET1
        #నిర్వచించు GGML_F32_X_ఎక్కించు GGML_F32x4_ఎక్కించు
        #నిర్వచించు GGML_F32X_STORE GGML_F32x4_STORE
        #నిర్వచించు GGML_F32X_MUL GGML_F32x4_MUL
        #నిర్వచించు GGML_F32X_FMA GGML_F32x4_FMA
        #నిర్వచించు GLA_సదిశ_కొలత 4
    #అయితే_ముగింపు

    #నిర్వచితమైతే GLA_సదిశ_కొలత
        పూర్ణాంకం gla_సదిశ_కొలత;
        #ఒకవేళ defined(__ARM_FEATURE_SVE)
            gla_సదిశ_కొలత = svcntw();
        #లేకపోతే
            gla_సదిశ_కొలత = GLA_సదిశ_కొలత;
        #అయితే_ముగింపు
        స్థిరరాశి int64_t vec_లెక్క = తల_కొలత / gla_సదిశ_కొలత;

        కొరకు (int64_t t = 0; t < T; t++) {
            size_t t_స్థానభ్రంశం = t * t_stride;
            size_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
            దశాంశం * state_cur = state + state_స్థానభ్రంశం;
            దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[4]->దత్తాంశం + state_స్థానభ్రంశం;

            కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                size_t h_స్థానభ్రంశం = h * h_stride;
                size_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                size_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                కొరకు (int64_t i = 0; i < తల_కొలత; i++) {
                    size_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + i;
                    size_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + i * h_stride;

                    దశాంశం k_val = k[t_h_i_స్థానభ్రంశం];
                    దశాంశం q_val = q[t_h_i_స్థానభ్రంశం] * scale;
                    దశాంశం g_val = g[t_h_i_స్థానభ్రంశం];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    కొరకు (int64_t j = 0; j < vec_లెక్క; j++) {
                        size_t పునాది_j = j * gla_సదిశ_కొలత;
                        size_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + పునాది_j;
                        size_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + పునాది_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_ఎక్కించు(&v[t_h_j_స్థానభ్రంశం]);
                        GGML_F32X prev_state_vec = GGML_F32_X_ఎక్కించు(&state_prev[h_2d_i_j_స్థానభ్రంశం]);
                        GGML_F32X dst_vec = GGML_F32_X_ఎక్కించు(&dst_దత్తాంశం[t_h_j_స్థానభ్రంశం]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X తాత్కాలిక_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, తాత్కాలిక_vec, q_vec);
                        GGML_F32X_STORE(&dst_దత్తాంశం[t_h_j_స్థానభ్రంశం], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_స్థానభ్రంశం], తాత్కాలిక_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    కొరకు (int64_t j = vec_లెక్క * gla_సదిశ_కొలత; j < తల_కొలత; j++) {
                        size_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                        size_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;
                        దశాంశం v_val = v[t_h_j_స్థానభ్రంశం];
                        దశాంశం kv_val = v_val * k_val;
                        దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                        దశాంశం తాత్కాలిక_val = kv_val + prev_state_val * g_val;
                        dst_దత్తాంశం[t_h_j_స్థానభ్రంశం] += తాత్కాలిక_val * q_val;
                        state_cur[h_2d_i_j_స్థానభ్రంశం] = తాత్కాలిక_val;
                    }
                }
            }
        }

    #లేకపోతే
        కొరకు (int64_t t = 0; t < T; t++) {
            size_t t_స్థానభ్రంశం = t * t_stride;
            size_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
            దశాంశం * state_cur = state + state_స్థానభ్రంశం;
            దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[4]->దత్తాంశం + state_స్థానభ్రంశం;

            కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                size_t h_స్థానభ్రంశం = h * h_stride;
                size_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                size_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                కొరకు (int64_t i = 0; i < తల_కొలత; i++) {
                    size_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + i;
                    size_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + i * h_stride;

                    దశాంశం k_val = k[t_h_i_స్థానభ్రంశం];
                    దశాంశం q_val = q[t_h_i_స్థానభ్రంశం] * scale;
                    దశాంశం g_val = g[t_h_i_స్థానభ్రంశం];

                    కొరకు (int64_t j = 0; j < తల_కొలత; j++) {
                        size_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                        size_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;

                        దశాంశం v_val = v[t_h_j_స్థానభ్రంశం];
                        దశాంశం kv_val = v_val * k_val;
                        దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                        దశాంశం తాత్కాలిక_val = prev_state_val * g_val + kv_val;
                        dst_దత్తాంశం[t_h_j_స్థానభ్రంశం] += తాత్కాలిక_val * q_val;
                        state_cur[h_2d_i_j_స్థానభ్రంశం] = తాత్కాలిక_val;
                    }
                }
            }
        }
    #అయితే_ముగింపు
}


శూన్య ggml_compute_forward_gla(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_compute_forward_solve_tri_f32(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) {
    స్థిరరాశి నిర్మాణం ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    స్థిరరాశి నిర్మాణం ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t k = ne10;   // number of RHS columns
    స్థిరరాశి int64_t n = ne11;   // A is n×n
    స్థిరరాశి int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి దశాంశం * A = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;  // [n, n, B1, B2]
    స్థిరరాశి దశాంశం * B = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;  // [n, k, B1, B2]
          దశాంశం * X = (      దశాంశం *) dst->దత్తాంశం;   // [n, k, B1, B2]

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*k);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*k)/k;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*k - i02*k);

        స్థిరరాశి దశాంశం * A_batch = A + i02 * nb02 / పరిమాణం(దశాంశం) + i03 * nb03 / పరిమాణం(దశాంశం);
        స్థిరరాశి దశాంశం * B_batch = B + i02 * nb12 / పరిమాణం(దశాంశం) + i03 * nb13 / పరిమాణం(దశాంశం);

        దశాంశం * X_batch = X + i02 * nb2 / పరిమాణం(దశాంశం) + i03 * nb3 / పరిమాణం(దశాంశం);

        కొరకు (int64_t i00 = 0; i00 < n; ++i00) {
            దశాంశం కూడిక = 0.0f;
            కొరకు (int64_t t = 0; t < i00; ++t) {
                కూడిక += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            స్థిరరాశి దశాంశం diag = A_batch[i00 * n + i00];
            ప్రతిజ్ఞా(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - కూడిక) / diag;
        }
    }
}

శూన్య ggml_compute_forward_solve_tri(స్థిరరాశి నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } లేకపోతే {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
స్థిర శూన్య ggml_compute_forward_gated_delta_net_one_chunk(
    స్థిరరాశి ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    స్థిరరాశి int64_t S_v      = src_v->ne[0];
    స్థిరరాశి int64_t H        = src_v->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = src_v->ne[2];
    స్థిరరాశి int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src_q));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src_k));
    GGML_ASSERT(ggml_is_contiguous_అడ్డువరుసలు(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    GGML_ASSERT(src_g->ne[0] == 1 || src_g->ne[0] == S_v);
    GGML_ASSERT(src_beta->ne[0] == 1);

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t,  nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(int64_t, neg, src_g, ne);
    GGML_TENSOR_LOCALS(size_t,  nbg, src_g, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    స్థిరరాశి తార్కిక kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    స్థిరరాశి int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    స్థిరరాశి int64_t state_seq_stride = src_state->nb[3] / పరిమాణం(దశాంశం);

    స్థిరరాశి int64_t చొప్పున_thread = S_v + (K > 1 ? S_v * S_v : 0);
    స్థిరరాశి పూర్ణాంకం ith = params->ith;

    దశాంశం * delta       = (దశాంశం *)params->wdata + ith * చొప్పున_thread + CACHE_పంక్తి_కొలత_F32;
    దశాంశం * state_work  = K > 1 ? (delta + S_v) : శూన్యసూచిక;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    స్థిరరాశి int64_t attn_score_elems    = S_v * H * n_టోకెన్లు * n_seqs;
    స్థిరరాశి int64_t state_కొలత_చొప్పున_snap = S_v * S_v * H * n_seqs;
    దశాంశం * attn_బహిర్గతం_పునాది  = (దశాంశం *)dst->దత్తాంశం;
    దశాంశం * state_బహిర్గతం_పునాది = (దశాంశం *)dst->దత్తాంశం + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    స్థిరరాశి దశాంశం * state_లోపల_పునాది = (స్థిరరాశి దశాంశం *)src_state->దత్తాంశం;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    స్థిరరాశి int64_t rq3 = nev3 / neq3;
    స్థిరరాశి int64_t rk3 = nev3 / nek3;

    స్థిరరాశి దశాంశం scale = 1.0f / sqrtf((దశాంశం) S_v);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t iv1 = ir % H; // head_index
        స్థిరరాశి int64_t iv3 = ir / H; // sequence

        స్థిరరాశి int64_t iq1 = iv1 % neq1;
        స్థిరరాశి int64_t ik1 = iv1 % nek1;

        స్థిరరాశి int64_t iq3 = iv3 / rq3;
        స్థిరరాశి int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        దశాంశం * s_బహిర్గతం = (K > 1)
            ? state_work
            : state_బహిర్గతం_పునాది + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        స్థిరరాశి దశాంశం * s_లోపల = state_లోపల_పునాది + iv3 * state_seq_stride + iv1 * S_v * S_v;
        జ్ఞాపకం_నకలు(s_బహిర్గతం, s_లోపల, S_v * S_v * పరిమాణం(దశాంశం));

        // attn output pointer for first token of this (head, seq)
        దశాంశం * attn_దత్తాంశం = attn_బహిర్గతం_పునాది + (iv3 * n_టోకెన్లు * H + iv1) * S_v;

        కొరకు (int64_t t = 0; t < n_టోకెన్లు; t++) {
            స్థిరరాశి దశాంశం * q_d = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_q->దత్తాంశం + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            స్థిరరాశి దశాంశం * k_d = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_k->దత్తాంశం + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            స్థిరరాశి దశాంశం * v_d = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_v->దత్తాంశం + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            స్థిరరాశి దశాంశం beta_val = *(స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_beta->దత్తాంశం + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            స్థిరరాశి దశాంశం * g_d    =  (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)src_g->దత్తాంశం    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            ఒకవేళ (kda) {
                // precompute exp(g) into delta scratch (reused below)
                కొరకు (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                కొరకు (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_బహిర్గతం[j * S_v], &s_బహిర్గతం[j * S_v], delta);
                }
            } లేకపోతే {
                ggml_vec_scale_f32(S_v * S_v, s_బహిర్గతం, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            కొరకు (int64_t j = 0; j < S_v; ++j) {
                దశాంశం కూడిక = 0.0f;
                ggml_vec_dot_f32(S_v, &కూడిక, 0, &s_బహిర్గతం[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - కూడిక) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            కొరకు (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_బహిర్గతం[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            కొరకు (int64_t j = 0; j < S_v; ++j) {
                దశాంశం కూడిక = 0.0f;
                ggml_vec_dot_f32(S_v, &కూడిక, 0, &s_బహిర్గతం[j * S_v], 0, q_d, 0, 1);
                attn_దత్తాంశం[j] = కూడిక * scale;
            }

            attn_దత్తాంశం += S_v * H; // advance to next token

            ఒకవేళ (K > 1) {
                స్థిరరాశి int64_t లక్ష్యం_స్థానం = n_టోకెన్లు - 1 - t;
                ఒకవేళ (లక్ష్యం_స్థానం >= 0 && లక్ష్యం_స్థానం < K) {
                    దశాంశం * curr_state_o = state_బహిర్గతం_పునాది + లక్ష్యం_స్థానం * state_కొలత_చొప్పున_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    జ్ఞాపకం_నకలు(curr_state_o, s_బహిర్గతం, S_v * S_v * పరిమాణం(దశాంశం));
                }
            }
        }
    }
}


స్థిర శూన్య ggml_compute_forward_gated_delta_net_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    స్థిరరాశి తార్కిక disable_chunking = ggml_is_numa();

    పూర్ణాంకం nth = params->nth;
    పూర్ణాంకం ith = params->ith;

    // 4x chunks per thread
    పూర్ణాంకం nth_scaled = nth * 4;
    int64_t chunk_కొలత = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_కొలత - 1) / chunk_కొలత;

    ఒకవేళ (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    ఒకవేళ (ith == 0) {
      ggml_threadpool_chunk_సమితి(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    స్థిరరాశి int64_t dr = (nr + nchunk - 1) / nchunk;

    పూర్ణాంకం current_chunk = ith;

    అయితే (current_chunk < nchunk) {
        స్థిరరాశి int64_t ir0 = dr * current_chunk;
        స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
    }
}

శూన్య ggml_compute_forward_gated_delta_net(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

స్థిర శూన్య ggml_dsv4_hc_comb_norm_నిలువుల_సంఖ్య(దశాంశం * comb, దశాంశం eps) {
    constexpr int64_t hc = 4;

    కొరకు (int64_t idst = 0; idst < hc; ++idst) {
        దశాంశం కూడిక = eps;
        కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
            కూడిక += comb[idst + hc*isrc];
        }

        స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
        కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_కూడిక;
        }
    }
}

స్థిర శూన్య ggml_dsv4_hc_comb_norm_అడ్డువరుసలు(దశాంశం * comb, దశాంశం eps) {
    constexpr int64_t hc = 4;

    కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
        దశాంశం కూడిక = eps;
        కొరకు (int64_t idst = 0; idst < hc; ++idst) {
            కూడిక += comb[idst + hc*isrc];
        }

        స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
        కొరకు (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_కూడిక;
        }
    }
}

స్థిర శూన్య ggml_compute_forward_dsv4_hc_comb_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * mixes = dst->src[0];
    స్థిరరాశి ggml_tensor * scale = dst->src[1];
    స్థిరరాశి ggml_tensor * పునాది  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(పునాది->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_స్థానభ్రంశం = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    స్థిరరాశి int64_t n_టోకెన్లు = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_టోకెన్లు);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(పునాది->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, పునాది,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    స్థిరరాశి దశాంశం eps = ggml_get_op_params_f32(dst, 0);
    స్థిరరాశి int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t dr  = (n_టోకెన్లు + nth - 1) / nth;
    స్థిరరాశి int64_t it0 = dr * ith;
    స్థిరరాశి int64_t it1 = MIN(it0 + dr, n_టోకెన్లు);

    స్థిరరాశి దశాంశం scale_comb = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) scale->దత్తాంశం + 2*nbs0);

    కొరకు (int64_t it = it0; it < it1; ++it) {
        దశాంశం comb[hc*hc];

        కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
            దశాంశం max = -INFINITY;
            కొరకు (int64_t idst = 0; idst < hc; ++idst) {
                స్థిరరాశి int64_t idx = idst + hc*isrc;
                స్థిరరాశి దశాంశం xv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) mixes->దత్తాంశం + (comb_స్థానభ్రంశం + idx)*nbm0 + it*nbm1);
                స్థిరరాశి దశాంశం bv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) పునాది->దత్తాంశం  + (comb_స్థానభ్రంశం + idx)*nbb0);
                స్థిరరాశి దశాంశం v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            దశాంశం కూడిక = 0.0f;
            కొరకు (int64_t idst = 0; idst < hc; ++idst) {
                స్థిరరాశి int64_t idx = idst + hc*isrc;
                స్థిరరాశి దశాంశం v = expf(comb[idx] - max);
                comb[idx] = v;
                కూడిక += v;
            }

            స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
            కొరకు (int64_t idst = 0; idst < hc; ++idst) {
                స్థిరరాశి int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_కూడిక + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_నిలువుల_సంఖ్య(comb, eps);
        కొరకు (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_అడ్డువరుసలు(comb, eps);
            ggml_dsv4_hc_comb_norm_నిలువుల_సంఖ్య(comb, eps);
        }

        కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
            కొరకు (int64_t idst = 0; idst < hc; ++idst) {
                స్థిరరాశి int64_t idx = idst + hc*isrc;
                *(దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

శూన్య ggml_compute_forward_dsv4_hc_comb(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

స్థిర శూన్య ggml_compute_forward_dsv4_hc_pre_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * x       = dst->src[0];
    స్థిరరాశి ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t n_embd   = x->ne[0];
    స్థిరరాశి int64_t hc       = x->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = x->ne[2];

    స్థిరరాశి దశాంశం scale = ggml_get_op_params_f32(dst, 0);
    స్థిరరాశి తార్కిక  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_టోకెన్లు);
    ఒకవేళ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_టోకెన్లు);
    } లేకపోతే {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_టోకెన్లు);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t nr  = n_embd * n_టోకెన్లు;
    స్థిరరాశి int64_t dr  = (nr + nth - 1) / nth;
    స్థిరరాశి int64_t ir0 = dr * ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i0 = ir % n_embd;
        స్థిరరాశి int64_t it = ir / n_embd;

        దశాంశం కూడిక = 0.0f;
        కొరకు (int64_t ih = 0; ih < hc; ++ih) {
            స్థిరరాశి దశాంశం xv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) x->దత్తాంశం + i0*nbx0 + ih*nbx1 + it*nbx2);
            దశాంశం wv;
            ఒకవేళ (gated) {
                స్థిరరాశి దశాంశం gv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) weights->దత్తాంశం + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } లేకపోతే {
                wv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) weights->దత్తాంశం + ih*nbw0 + it*nbw1);
            }
            కూడిక += xv * wv;
        }

        *(దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i0*nbd0 + it*nbd1) = scale * కూడిక;
    }
}

శూన్య ggml_compute_forward_dsv4_hc_pre(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

స్థిర శూన్య ggml_compute_forward_dsv4_hc_post_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * x        = dst->src[0];
    స్థిరరాశి ggml_tensor * residual = dst->src[1];
    స్థిరరాశి ggml_tensor * post     = dst->src[2];
    స్థిరరాశి ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t n_embd   = x->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు = x->ne[1];
    స్థిరరాశి int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_టోకెన్లు);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_టోకెన్లు);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_టోకెన్లు);

    // comb == NULL: identity mixing, each stream keeps its own residual
    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    ఒకవేళ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_టోకెన్లు);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t nr  = n_embd * hc * n_టోకెన్లు;
    స్థిరరాశి int64_t dr  = (nr + nth - 1) / nth;
    స్థిరరాశి int64_t ir0 = dr * ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i0     = ir % n_embd;
        స్థిరరాశి int64_t idst   = (ir / n_embd) % hc;
        స్థిరరాశి int64_t it     = ir / (n_embd * hc);

        స్థిరరాశి దశాంశం xv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) x->దత్తాంశం    + i0*nbx0 + it*nbx1);
        స్థిరరాశి దశాంశం pv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) post->దత్తాంశం + idst*nbp0 + it*nbp1);

        దశాంశం కూడిక = xv * pv;
        ఒకవేళ (comb) {
            కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
                స్థిరరాశి దశాంశం rv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) residual->దత్తాంశం + i0*nbr0 + isrc*nbr1 + it*nbr2);
                స్థిరరాశి దశాంశం cv = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) comb->దత్తాంశం     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                కూడిక += rv * cv;
            }
        } లేకపోతే {
            కూడిక += *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) residual->దత్తాంశం + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i0*nbd0 + idst*nbd1 + it*nbd2) = కూడిక;
    }
}

శూన్య ggml_compute_forward_dsv4_hc_post(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

శూన్య ggml_compute_forward_శిఖరం_k_ముసుగు(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * ముసుగు   = dst->src[0];
    స్థిరరాశి ggml_tensor * శిఖరం_k  = dst->src[1];
    స్థిరరాశి ggml_tensor * score  = dst->src[2];
    స్థిరరాశి ggml_tensor * ఖండాలు = dst->src[3];
    స్థిరరాశి ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == ముసుగు->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_కొలత(dst->type));

    స్థిరరాశి int64_t n_kv    = ముసుగు->ne[0];
    స్థిరరాశి int64_t ne1     = dst->ne[1];
    స్థిరరాశి int64_t ne2     = dst->ne[2];
    స్థిరరాశి int64_t n_అడ్డువరుసలు  = ggml_nrows(dst);
    స్థిరరాశి int64_t n_శిఖరం   = శిఖరం_k->ne[0];
    స్థిరరాశి int64_t n_blk   = ఖండాలు ? ఖండాలు->ne[0] : 1;
    స్థిరరాశి int64_t n_pick  = ఖండాలు ? ఖండాలు->ne[1] : score ? score->ne[0] : n_kv;
    స్థిరరాశి int64_t n_extra = extra ? extra->ne[0] : 0;

    స్థిరరాశి size_t ts = ggml_type_కొలత(dst->type);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    కొరకు (int64_t ir = ith; ir < n_అడ్డువరుసలు; ir += nth) {
        స్థిరరాశి int64_t i1 = ir % ne1;
        స్థిరరాశి int64_t i2 = (ir / ne1) % ne2;
        స్థిరరాశి int64_t i3 = ir / (ne1*ne2);

        స్థిరరాశి అక్షరం * ముసుగు_అడ్డువరుస = (స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం
            + (i1 % ముసుగు->ne[1])*ముసుగు->nb[1] + (i2 % ముసుగు->ne[2])*ముసుగు->nb[2] + (i3 % ముసుగు->ne[3])*ముసుగు->nb[3];
        అక్షరం * dst_అడ్డువరుస = (అక్షరం *) dst->దత్తాంశం + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        ఒకవేళ (dst->type == GGML_TYPE_F16) {
            స్థిరరాశి ggml_fp16_t ninf = GGML_CPU_FP32_గమ్యం_FP16(-INFINITY);
            కొరకు (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_అడ్డువరుస)[c] = ninf;
            }
        } లేకపోతే {
            కొరకు (int64_t c = 0; c < n_kv; ++c) {
                ((దశాంశం *) dst_అడ్డువరుస)[c] = -INFINITY;
            }
        }

        స్వయంచాలక keep = [&](int32_t c) {
            ఒకవేళ (c >= 0 && c < n_kv) {
                జ్ఞాపకం_నకలు(dst_అడ్డువరుస + c*ts, ముసుగు_అడ్డువరుస + c*ts, ts);
            }
        };

        స్థిరరాశి అక్షరం * శిఖరం_k_అడ్డువరుస = (స్థిరరాశి అక్షరం *) శిఖరం_k->దత్తాంశం + i1*శిఖరం_k->nb[1] + i2*శిఖరం_k->nb[2] + i3*శిఖరం_k->nb[3];

        కొరకు (int64_t j = 0; j < n_శిఖరం; ++j) {
            స్థిరరాశి int32_t p = *(స్థిరరాశి int32_t *) (శిఖరం_k_అడ్డువరుస + j*శిఖరం_k->nb[0]);
            ఒకవేళ (p < 0 || p >= n_pick) {
                కొనసాగు;
            }
            ఒకవేళ (score) {
                స్థిరరాశి దశాంశం s = *(స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) score->దత్తాంశం
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                ఒకవేళ (!(s > -INFINITY)) {
                    కొనసాగు;
                }
            }
            ఒకవేళ (!ఖండాలు) {
                keep(p);
                కొనసాగు;
            }
            కొరకు (int64_t k = 0; k < n_blk; ++k) {
                keep(*(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) ఖండాలు->దత్తాంశం + k*ఖండాలు->nb[0] + p*ఖండాలు->nb[1]));
            }
        }

        ఒకవేళ (extra) {
            స్థిరరాశి అక్షరం * extra_అడ్డువరుస = (స్థిరరాశి అక్షరం *) extra->దత్తాంశం + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            కొరకు (int64_t j = 0; j < n_extra; ++j) {
                keep(*(స్థిరరాశి int32_t *) (extra_అడ్డువరుస + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

స్థిర శూన్య ggml_compute_forward_rwkv_wkv7_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {
    స్థిరరాశి int64_t T = dst->src[1]->ne[2];
    స్థిరరాశి int64_t C = dst->ne[0];
    స్థిరరాశి int64_t HEADS = dst->src[1]->ne[1];
    స్థిరరాశి int64_t n_seqs = dst->src[6]->ne[1];
    స్థిరరాశి int64_t తల_కొలత = C / HEADS;

    దశాంశం * dst_దత్తాంశం = (దశాంశం *) dst->దత్తాంశం;
    దశాంశం * state = ((దశాంశం *) dst->దత్తాంశం) + C * T;

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం h_ప్రారంభం =  (HEADS * (ith    )) / nth;
    స్థిరరాశి పూర్ణాంకం h_చివర   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    దశాంశం * r = (దశాంశం *) dst->src[0]->దత్తాంశం;
    దశాంశం * w = (దశాంశం *) dst->src[1]->దత్తాంశం;
    దశాంశం * k = (దశాంశం *) dst->src[2]->దత్తాంశం;
    దశాంశం * v = (దశాంశం *) dst->src[3]->దత్తాంశం;
    దశాంశం * a = (దశాంశం *) dst->src[4]->దత్తాంశం;
    దశాంశం * b = (దశాంశం *) dst->src[5]->దత్తాంశం;

    int64_t t_stride = HEADS * తల_కొలత; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = తల_కొలత * తల_కొలత;

    #ఒకవేళ defined(GGML_SIMD)
        #ఒకవేళ defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            కొరకు (int64_t t = 0; t < T; t++) {
                int64_t t_స్థానభ్రంశం = t * t_stride;
                int64_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
                దశాంశం * state_cur = state + state_స్థానభ్రంశం;
                దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[6]->దత్తాంశం + state_స్థానభ్రంశం;

                కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                    int64_t h_స్థానభ్రంశం = h * h_stride;
                    int64_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                    int64_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                    కొరకు (int64_t i = 0; i < తల_కొలత; i++) {
                        int64_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + i;
                        int64_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + i * h_stride;

                        దశాంశం v_val = v[t_h_i_స్థానభ్రంశం];

                        దశాంశం sa = 0, ఫలితం = 0;
                        కొరకు (int64_t j = 0; j < తల_కొలత; j++) {
                            sa += a[t_h_స్థానభ్రంశం + j] * state_prev[h_2d_i_స్థానభ్రంశం + j];
                        }

                        కొరకు (int64_t j = 0; j < తల_కొలత; j++) {
                            int64_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                            int64_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;

                            దశాంశం r_val = r[t_h_j_స్థానభ్రంశం];
                            దశాంశం w_val = w[t_h_j_స్థానభ్రంశం];
                            దశాంశం k_val = k[t_h_j_స్థానభ్రంశం];
                            దశాంశం b_val = b[t_h_j_స్థానభ్రంశం];
                            దశాంశం kv_val = v_val * k_val;
                            దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                            state_cur[h_2d_i_j_స్థానభ్రంశం] = prev_state_val * w_val + kv_val + sa * b_val;
                            ఫలితం += state_cur[h_2d_i_j_స్థానభ్రంశం] * r_val;
                        }
                        dst_దత్తాంశం[t_h_i_స్థానభ్రంశం] = ఫలితం;
                    }
                }
            }
        #లేకపోతే
            కొరకు (int64_t t = 0; t < T; t++) {
                int64_t t_స్థానభ్రంశం = t * t_stride;
                int64_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
                దశాంశం * state_cur = state + state_స్థానభ్రంశం;
                దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[6]->దత్తాంశం + state_స్థానభ్రంశం;

                కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                    int64_t h_స్థానభ్రంశం = h * h_stride;
                    int64_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                    int64_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                    కొరకు (int64_t ii = 0; ii < తల_కొలత; ii++) {
                        int64_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + ii;
                        int64_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_స్థానభ్రంశం]);

                        దశాంశం sa = 0;
                        {
                            GGML_F32_VEC కూడిక[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            కొరకు (int64_t j = 0; j < తల_కొలత; j += GGML_F32_అడుగు) {
                                కొరకు (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_ఎక్కించు(&a[t_h_స్థానభ్రంశం + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_ఎక్కించు(&state_prev[h_2d_i_స్థానభ్రంశం + j + kk * GGML_F32_EPR]);
                                    కూడిక[kk] = GGML_F32_VEC_FMA(కూడిక[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, కూడిక);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC ఫలితం_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        కొరకు (; j < తల_కొలత; j += GGML_F32_అడుగు) {
                            కొరకు (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_ఎక్కించు(&r[t_h_j_స్థానభ్రంశం]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_ఎక్కించు(&w[t_h_j_స్థానభ్రంశం]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_ఎక్కించు(&k[t_h_j_స్థానభ్రంశం]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_ఎక్కించు(&b[t_h_j_స్థానభ్రంశం]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_ఎక్కించు(&state_prev[h_2d_i_j_స్థానభ్రంశం]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_స్థానభ్రంశం], state_vec);

                                ఫలితం_vec[kk] = GGML_F32_VEC_FMA(ఫలితం_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_దత్తాంశం[t_h_i_స్థానభ్రంశం], ఫలితం_vec);

                        // There shouldn't be left-overs though.
                        కొరకు (; j < తల_కొలత; j++) {
                            int64_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                            int64_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;

                            దశాంశం r_val = r[t_h_j_స్థానభ్రంశం];
                            దశాంశం w_val = w[t_h_j_స్థానభ్రంశం];
                            దశాంశం k_val = k[t_h_j_స్థానభ్రంశం];
                            దశాంశం b_val = b[t_h_j_స్థానభ్రంశం];
                            దశాంశం kv_val = v[t_h_i_స్థానభ్రంశం] * k_val;

                            దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                            state_cur[h_2d_i_j_స్థానభ్రంశం] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_దత్తాంశం[t_h_i_స్థానభ్రంశం] += state_cur[h_2d_i_j_స్థానభ్రంశం] * r_val;
                        }
                    }
                }
            }
        #అయితే_ముగింపు
    #లేకపోతే
        కొరకు (int64_t t = 0; t < T; t++) {
            int64_t t_స్థానభ్రంశం = t * t_stride;
            int64_t state_స్థానభ్రంశం = తల_కొలత * C * (t / (T / n_seqs));
            దశాంశం * state_cur = state + state_స్థానభ్రంశం;
            దశాంశం * state_prev = t % (T / n_seqs) ? state_cur : (దశాంశం*)dst->src[6]->దత్తాంశం + state_స్థానభ్రంశం;

            కొరకు (int64_t h = h_ప్రారంభం; h < h_చివర; h++) {
                int64_t h_స్థానభ్రంశం = h * h_stride;
                int64_t t_h_స్థానభ్రంశం = t_స్థానభ్రంశం + h_స్థానభ్రంశం;
                int64_t h_2d_స్థానభ్రంశం = h * h_stride_2d;

                కొరకు (int64_t i = 0; i < తల_కొలత; i++) {
                    int64_t t_h_i_స్థానభ్రంశం = t_h_స్థానభ్రంశం + i;
                    int64_t h_2d_i_స్థానభ్రంశం = h_2d_స్థానభ్రంశం + i * h_stride;

                    దశాంశం v_val = v[t_h_i_స్థానభ్రంశం];

                    దశాంశం sa = 0, ఫలితం = 0;
                    కొరకు (int64_t j = 0; j < తల_కొలత; j++) {
                        sa += a[t_h_స్థానభ్రంశం + j] * state_prev[h_2d_i_స్థానభ్రంశం + j];
                    }

                    కొరకు (int64_t j = 0; j < తల_కొలత; j++) {
                        int64_t t_h_j_స్థానభ్రంశం = t_h_స్థానభ్రంశం + j;
                        int64_t h_2d_i_j_స్థానభ్రంశం = h_2d_i_స్థానభ్రంశం + j;

                        దశాంశం r_val = r[t_h_j_స్థానభ్రంశం];
                        దశాంశం w_val = w[t_h_j_స్థానభ్రంశం];
                        దశాంశం k_val = k[t_h_j_స్థానభ్రంశం];
                        దశాంశం b_val = b[t_h_j_స్థానభ్రంశం];
                        దశాంశం kv_val = v_val * k_val;
                        దశాంశం prev_state_val = state_prev[h_2d_i_j_స్థానభ్రంశం];
                        state_cur[h_2d_i_j_స్థానభ్రంశం] = prev_state_val * w_val + kv_val + sa * b_val;
                        ఫలితం += state_cur[h_2d_i_j_స్థానభ్రంశం] * r_val;
                    }
                    dst_దత్తాంశం[t_h_i_స్థానభ్రంశం] = ఫలితం;
                }
            }
        }
    #అయితే_ముగింపు
}


శూన్య ggml_compute_forward_rwkv_wkv7(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

శూన్య ggml_compute_forward_map_custom1(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * a = dst->src[0];

    నిర్మాణం ggml_map_custom1_op_params p;
    జ్ఞాపకం_నకలు(&p, dst->op_params, పరిమాణం(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

శూన్య ggml_compute_forward_map_custom2(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * a = dst->src[0];
    స్థిరరాశి ggml_tensor * b = dst->src[1];

    నిర్మాణం ggml_map_custom2_op_params p;
    జ్ఞాపకం_నకలు(&p, dst->op_params, పరిమాణం(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

శూన్య ggml_compute_forward_map_custom3(
        స్థిరరాశి ggml_compute_params * params,
              ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * a = dst->src[0];
    స్థిరరాశి ggml_tensor * b = dst->src[1];
    స్థిరరాశి ggml_tensor * c = dst->src[2];

    నిర్మాణం ggml_map_custom3_op_params p;
    జ్ఞాపకం_నకలు(&p, dst->op_params, పరిమాణం(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

శూన్య ggml_compute_forward_custom(
    స్థిరరాశి నిర్మాణం ggml_compute_params * params,
          నిర్మాణం ggml_tensor * dst) {

    నిర్మాణం ggml_custom_op_params p;
    జ్ఞాపకం_నకలు(&p, dst->op_params, పరిమాణం(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

స్థిర శూన్య ggml_compute_forward_cross_entropy_loss_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_కొలత(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_కొలత(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    స్థిరరాశి int64_t nc = src0->ne[0];
    స్థిరరాశి int64_t nr = ggml_nrows(src0);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    దశాంశం * sums =  (దశాంశం *) params->wdata;
    దశాంశం * st   = ((దశాంశం *) params->wdata) + nth + ith*nc;
    దశాంశం కూడిక_thread = 0.0f;

    GGML_ASSERT(params->wsize >= పరిమాణం(దశాంశం) * (nth + nth * nc));

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    కొరకు (int64_t i1 = ir0; i1 < ir1; ++i1) {
        స్థిరరాశి దశాంశం * s0 = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i1*src0->nb[1]);
        స్థిరరాశి దశాంశం * s1 = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i1*src1->nb[1]);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ప్రతిజ్ఞా(!isnan(s0[i]));
            ప్రతిజ్ఞా(!isnan(s1[i]));
        }
#అయితే_ముగింపు // NDEBUG

        దశాంశం max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        స్థిరరాశి ggml_float కూడిక_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        ప్రతిజ్ఞా(కూడిక_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -కూడిక_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        దశాంశం కూడిక_st = 0.0f;
        ggml_vec_కూడిక_f32(nc, &కూడిక_st, st);
        కూడిక_thread += కూడిక_st;

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (int64_t i = 0; i < nc; ++i) {
            ప్రతిజ్ఞా(!isnan(st[i]));
            ప్రతిజ్ఞా(!isinf(st[i]));
        }
#అయితే_ముగింపు // NDEBUG
    }
    sums[ith] = కూడిక_thread;
    ggml_barrier(params->threadpool);

    ఒకవేళ (ith == 0) {
        దశాంశం * dp = (దశాంశం *) dst->దత్తాంశం;
        ggml_vec_కూడిక_f32(nth, dp, sums);
        dp[0] *= -1.0f / (దశాంశం) nr;
    }
}

శూన్య ggml_compute_forward_cross_entropy_loss(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

స్థిర శూన్య ggml_compute_forward_cross_entropy_loss_వెనుక_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    స్థిరరాశి ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    స్థిరరాశి ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    స్థిరరాశి int64_t ith = params->ith;
    స్థిరరాశి int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    స్థిరరాశి int64_t nc = src0f->ne[0];
    స్థిరరాశి int64_t nr = ggml_nrows(src0f);

    // rows per thread
    స్థిరరాశి int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి int64_t ir0 = dr*ith;
    స్థిరరాశి int64_t ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి దశాంశం d_by_nr = ((స్థిరరాశి దశాంశం *) grad->దత్తాంశం)[0] / (దశాంశం) nr;

    కొరకు (int64_t i1 = ir0; i1 < ir1; i1++) {
        దశాంశం       * ds0 = (దశాంశం       *)((అక్షరం       *) dst->దత్తాంశం   + i1*dst->nb[1]);
        స్థిరరాశి దశాంశం * s0  = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) src0f->దత్తాంశం + i1*src0f->nb[1]);
        స్థిరరాశి దశాంశం * s1  = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *) src1f->దత్తాంశం + i1*src1f->nb[1]);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ప్రతిజ్ఞా(!isnan(s0[i]));
            ప్రతిజ్ఞా(!isnan(s1[i]));
        }
#అయితే_ముగింపు // NDEBUG

        // soft_max
        దశాంశం max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        స్థిరరాశి ggml_float కూడిక = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        ప్రతిజ్ఞా(కూడిక > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/కూడిక);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#నిర్వచితం_కాకపోతే NDEBUG
        కొరకు (int64_t i = 0; i < nc; ++i) {
            ప్రతిజ్ఞా(!isnan(ds0[i]));
            ప్రతిజ్ఞా(!isinf(ds0[i]));
        }
#అయితే_ముగింపు // NDEBUG
    }
}

శూన్య ggml_compute_forward_cross_entropy_loss_వెనుక(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_వెనుక_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_compute_forward_opt_అడుగు_adamw_f32(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0         = dst->src[0];
    స్థిరరాశి ggml_tensor * src0_grad    = dst->src[1];
    స్థిరరాశి ggml_tensor * src0_grad_m  = dst->src[2];
    స్థిరరాశి ggml_tensor * src0_grad_v  = dst->src[3];
    స్థిరరాశి ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    స్థిరరాశి దశాంశం * adamw_params_ptr = ggml_get_దత్తాంశం_f32(adamw_params);

    స్థిరరాశి దశాంశం alpha  = adamw_params_ptr[0];
    స్థిరరాశి దశాంశం beta1  = adamw_params_ptr[1];
    స్థిరరాశి దశాంశం beta2  = adamw_params_ptr[2];
    స్థిరరాశి దశాంశం eps    = adamw_params_ptr[3];
    స్థిరరాశి దశాంశం wd     = adamw_params_ptr[4];
    స్థిరరాశి దశాంశం beta1h = adamw_params_ptr[5];
    స్థిరరాశి దశాంశం beta2h = adamw_params_ptr[6];
    స్థిరరాశి దశాంశం keep   = 1.f - alpha * wd;
    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir/(ne02*ne01);
        స్థిరరాశి int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        స్థిరరాశి int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        స్థిరరాశి size_t స్థానభ్రంశం = i03*nb03 + i02*nb02 + i01*nb01;

        దశాంశం       * w = (దశాంశం       *) ((అక్షరం       *) src0->దత్తాంశం        + స్థానభ్రంశం); //బరువు
        స్థిరరాశి దశాంశం * g = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_grad->దత్తాంశం   + స్థానభ్రంశం); // grad
        దశాంశం       * m = (దశాంశం       *) ((అక్షరం       *) src0_grad_m->దత్తాంశం + స్థానభ్రంశం);
        దశాంశం       * v = (దశాంశం       *) ((అక్షరం       *) src0_grad_v->దత్తాంశం + స్థానభ్రంశం);

        కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            స్థిరరాశి దశాంశం mh =       m[i00]*beta1h;
            స్థిరరాశి దశాంశం vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

శూన్య ggml_compute_forward_opt_అడుగు_adamw(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_అడుగు_adamw_f32(params, dst);
            } ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error");
            }
    }
}

స్థిర శూన్య ggml_compute_forward_opt_అడుగు_sgd_f32(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0       = dst->src[0];
    స్థిరరాశి ggml_tensor * src0_grad  = dst->src[1];
    స్థిరరాశి ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి పూర్ణాంకం nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == పరిమాణం(దశాంశం));

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1) / nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr * ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    స్థిరరాశి దశాంశం * sgd_params_ptr   = ggml_get_దత్తాంశం_f32(sgd_params);
    స్థిరరాశి దశాంశం   alpha            = sgd_params_ptr[0];
    స్థిరరాశి దశాంశం   keep             = 1.f - alpha * sgd_params_ptr[1];

    కొరకు (పూర్ణాంకం ir = ir0; ir < ir1; ++ir) {
        స్థిరరాశి int64_t i03 = ir / (ne02 * ne01);
        స్థిరరాశి int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        స్థిరరాశి int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        స్థిరరాశి size_t స్థానభ్రంశం = i03 * nb03 + i02 * nb02 + i01 * nb01;

        దశాంశం *       w = (దశాంశం *) ((అక్షరం *) src0->దత్తాంశం + స్థానభ్రంశం);                   //బరువు
        స్థిరరాశి దశాంశం * g = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_grad->దత్తాంశం + స్థానభ్రంశం);  // grad

        కొరకు (పూర్ణాంకం i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

శూన్య ggml_compute_forward_opt_అడుగు_sgd(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_అడుగు_sgd_f32(params, dst);
            }
            ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

స్థిర అంతఃవరుస దశాంశం ggml_fwht_ఎక్కించు(స్థిరరాశి దశాంశం value) {
    తిరిగి value;
}

స్థిర అంతఃవరుస దశాంశం ggml_fwht_ఎక్కించు(స్థిరరాశి ggml_fp16_t value) {
    తిరిగి ggml_fp16_గమ్యం_fp32(value);
}

మూస<రకం_పేరు src_t>
స్థిర శూన్య ggml_compute_forward_fwht_impl(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, దశాంశం> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    స్థిరరాశి int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    స్థిరరాశి int64_t nr = ne11 * ne12 * ne13;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_thread = (nr + nth - 1) / nth;
    స్థిరరాశి int64_t ప్రారంభం_అడ్డువరుస = ith * అడ్డువరుసలు_చొప్పున_thread;
    స్థిరరాశి int64_t చివర_అడ్డువరుస = MIN(ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread, nr);

    స్థిరరాశి దశాంశం scale = 1.0f / sqrtf((దశాంశం)n);

#ఒకవేళ defined(GGML_SIMD)
    స్థిరరాశి GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#అయితే_ముగింపు

    కొరకు (int64_t r = ప్రారంభం_అడ్డువరుస; r < చివర_అడ్డువరుస; r++) {
        స్థిరరాశి int64_t i13 = r / (ne11 * ne12);
        స్థిరరాశి int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        స్థిరరాశి int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        స్థిరరాశి src_t * src_అడ్డువరుస = (స్థిరరాశి src_t *) ((స్థిరరాశి అక్షరం *) src1->దత్తాంశం + i11 * nb11 + i12 * nb12 + i13 * nb13);
        దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + i11 * nb1 + i12 * nb2 + i13 * nb3);

        కొరకు (int64_t j = 0; j < n; j++) {
            dst_అడ్డువరుస[j] = ggml_fwht_ఎక్కించు(src_అడ్డువరుస[j]) * scale;
        }

        // Scalar passes
#ఒకవేళ defined(GGML_SIMD)
#ఒకవేళ defined(__ARM_FEATURE_SVE)
        స్థిరరాశి పూర్ణాంకం అడుగు = svcntw();
#లేకపోతే
        స్థిరరాశి పూర్ణాంకం అడుగు = GGML_F32_EPR;
#అయితే_ముగింపు
#లేకపోతే
        స్థిరరాశి పూర్ణాంకం అడుగు = n;
#అయితే_ముగింపు
        కొరకు (int64_t len = 1; len < అడుగు && len < n; len <<= 1) {
            కొరకు (int64_t i = 0; i < n; i += 2 * len) {
                కొరకు (int64_t j = 0; j < len; j++) {
                    దశాంశం u = dst_అడ్డువరుస[i + j];
                    దశాంశం v = dst_అడ్డువరుస[i + len + j];
                    dst_అడ్డువరుస[i + j] = u + v;
                    dst_అడ్డువరుస[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#ఒకవేళ defined(GGML_SIMD)
        కొరకు (int64_t len = అడుగు; len < n; len <<= 1) {
            కొరకు (int64_t i = 0; i < n; i += 2 * len) {
                కొరకు (int64_t j = 0; j < len; j += అడుగు) {
                    GGML_F32_VEC u = GGML_F32_VEC_ఎక్కించు(dst_అడ్డువరుస + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_ఎక్కించు(dst_అడ్డువరుస + i + len + j);

                    GGML_F32_VEC_STORE(dst_అడ్డువరుస + i + j,       GGML_F32_VEC_కూడు(u, v));
                    GGML_F32_VEC_STORE(dst_అడ్డువరుస + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#అయితే_ముగింపు
    }
}

శూన్య ggml_compute_forward_fwht(స్థిరరాశి ggml_compute_params * params, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    ఎంపిక (src1->type) {
        స్థితి GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<దశాంశం>(params, dst);
            }
            ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            ఆపు;
        అప్రమేయ:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

శూన్య ggml_compute_forward_lightning_indexer(
        స్థిరరాశి ggml_compute_params * params,
        ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * q = dst->src[0];
    స్థిరరాశి ggml_tensor * k = dst->src[1];
    స్థిరరాశి ggml_tensor * w = dst->src[2]; // weights
    స్థిరరాశి ggml_tensor * m = dst->src[3]; //ముసుగు

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, కొత్త,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_కొలత(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_కొలత(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_కొలత(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_కొలత(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_కొలత(  m->type));

    స్థిరరాశి పూర్ణాంకం n_embd    = q->ne[0];
    స్థిరరాశి పూర్ణాంకం n_తల    = q->ne[1];
    స్థిరరాశి పూర్ణాంకం n_టోకెన్లు  = q->ne[2];
    స్థిరరాశి పూర్ణాంకం n_ప్రవాహం  = q->ne[3];
    స్థిరరాశి పూర్ణాంకం n_kv      = k->ne[2];

    ggml_గమ్యం_float_t స్థిరరాశి k_గమ్యం_float = ggml_get_type_traits(k->type)->గమ్యం_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_గమ్యం_float) && "lightning indexer: unsupported K-type");

    స్థిరరాశి పూర్ణాంకం nr  = n_kv;
    స్థిరరాశి పూర్ణాంకం ith = params->ith;
    స్థిరరాశి పూర్ణాంకం nth = params->nth;

    // (temporary) buffer for K converted to float
    దశాంశం * k_అడ్డువరుస_f32 = (దశాంశం *) params->wdata + ith*(1*n_embd + CACHE_పంక్తి_కొలత_F32);

    // rows per thread
    స్థిరరాశి పూర్ణాంకం dr = (nr + nth - 1)/nth;

    // row range for this thread
    స్థిరరాశి పూర్ణాంకం ir0 = dr*ith;
    స్థిరరాశి పూర్ణాంకం ir1 = MIN(ir0 + dr, nr);

    కొరకు (పూర్ణాంకం s = 0; s < n_ప్రవాహం; ++s) {
        కొరకు (పూర్ణాంకం t = 0; t < n_టోకెన్లు; ++t) {
            స్థిరరాశి దశాంశం       *   w_అడ్డువరుస =       (దశాంశం *) ((అక్షరం *)   w->దత్తాంశం + t*nbw1 +        s*nbw3);
            స్థిరరాశి ggml_fp16_t *   m_అడ్డువరుస = (ggml_fp16_t *) ((అక్షరం *)   m->దత్తాంశం + t*nbm1 + (s%nem3)*nbm3);
            దశాంశం             * dst_అడ్డువరుస =       (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + t*nb1  +        s*nb3 );
            కొరకు (పూర్ణాంకం ik = ir0; ik < ir1; ++ik) {
                అక్షరం * k_అడ్డువరుస = (అక్షరం *) k->దత్తాంశం + ik*nbk2 + s*nbk3;
                ఒకవేళ (k_గమ్యం_float) {
                    k_గమ్యం_float(k_అడ్డువరుస, k_అడ్డువరుస_f32, n_embd);
                } లేకపోతే {
                    k_అడ్డువరుస_f32 = (దశాంశం *) k_అడ్డువరుస;
                }
                దశాంశం score = 0.0f;
                కొరకు (పూర్ణాంకం h = 0; h < n_తల; ++h) {
                    // dot product of q and k for head h
                    దశాంశం qk = 0.0f;
                    స్థిరరాశి దశాంశం * q_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) q->దత్తాంశం + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_అడ్డువరుస, 0, k_అడ్డువరుస_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_అడ్డువరుస[h];
                }
                // apply mask
                dst_అడ్డువరుస[ik] = score + GGML_CPU_FP16_గమ్యం_FP32(m_అడ్డువరుస[ik]);
            }
        }
    }
}
