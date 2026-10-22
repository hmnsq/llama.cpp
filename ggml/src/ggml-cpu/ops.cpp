#अन्तर्भाव "ops.h"

#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "binary-ops.h"
#अन्तर्भाव "simd-gemm.h"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "unary-ops.h"
#अन्तर्भाव "vec.h"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cfloat>
#अन्तर्भाव <cmath>

// ggml_compute_forward_dup

स्थिर शून्य ggml_compute_forward_dup_same_cont(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    ध्रुव size_t nb0 = ggml_type_आकार(src0->type);

    ध्रुव पूर्णाङ्क ith = params->ith; // thread index
    ध्रुव पूर्णाङ्क nth = params->nth; // number of threads

    // parallelize by blocks
    ध्रुव पूर्णाङ्क nk = ggml_nelements(src0)/ggml_blck_आकार(src0->type);
    ध्रुव पूर्णाङ्क dr = (nk + nth - 1) / nth;
    ध्रुव पूर्णाङ्क k0 = dr * ith;
    ध्रुव पूर्णाङ्क k1 = MIN(k0 + dr, nk);

    यदि (k0 < k1) {
        स्मृतिं_प्रतिलिख(
            ((वर्ण *)  dst->दत्त + k0*nb0),
            ((वर्ण *) src0->दत्त + k0*nb0),
            (k1 - k0) * nb0);
    }
}

प्रतिरूपम्<प्रकारनाम src_t, प्रकारनाम dst_t>
स्थिर शून्य ggml_compute_forward_dup_flt(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith; // thread index
    ध्रुव पूर्णाङ्क nth = params->nth; // number of threads

    // parallelize by rows
    ध्रुव पूर्णाङ्क nr = ne01;
    // number of rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1) / nth;
    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    यदि (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_आकार(src0->type) && nb0 == ggml_type_आकार(dst->type)) {
        // copy by rows
        ध्रुव size_t rs = ne00*nb00;
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                    स्मृतिं_प्रतिलिख(
                        ((वर्ण *)  dst->दत्त + i01*nb1  + i02*nb2  + i03*nb3),
                        ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        प्रत्यागम;
    }

    // case: dst tensor is contiguous
    यदि (ggml_is_contiguous(dst)) {
        यदि (nb00 == परिमाण(src_t)) {
            यदि constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                ध्रुव size_t rs = ne00 * nb00;
                वर्ण * dst_ptr = (वर्ण *) dst->दत्त;

                प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
                    प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                            ध्रुव वर्ण * src0_ptr = (वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03;
                            स्मृतिं_प्रतिलिख(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } अन्यथा {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->दत्त;

                प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
                    प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                            ध्रुव src_t * src0_ptr = (src_t *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03);
                            प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                                भिन्न tmp = type_conversion_सारणी<src_t>::गन्तव्यम्_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_सारणी<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } अन्यथा {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->दत्त;

            प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
                प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                        प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                            ध्रुव src_t * src0_ptr = (src_t *) ((वर्ण *) src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            भिन्न tmp = type_conversion_सारणी<src_t>::गन्तव्यम्_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_सारणी<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        प्रत्यागम;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    यदि constexpr (std::is_same_v<dst_t, src_t>) {
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                यावत् (i10 >= ne0) {
                    i10 -= ne0;
                    यदि (++i11 == ne1) {
                        i11 = 0;
                        यदि (++i12 == ne2) {
                            i12 = 0;
                            यदि (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        ध्रुव वर्ण * src0_ptr = ((वर्ण *) src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              वर्ण * dst_ptr  = ((वर्ण *)  dst->दत्त + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        स्मृतिं_प्रतिलिख(dst_ptr, src0_ptr, परिमाण(dst_t));

                        यदि (++i10 == ne00) {
                            i10 = 0;
                            यदि (++i11 == ne01) {
                                i11 = 0;
                                यदि (++i12 == ne02) {
                                    i12 = 0;
                                    यदि (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                यावत् (i10 >= ne0) {
                    i10 -= ne0;
                    यदि (++i11 == ne1) {
                        i11 = 0;
                        यदि (++i12 == ne2) {
                            i12 = 0;
                            यदि (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } अन्यथा {
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                यावत् (i10 >= ne0) {
                    i10 -= ne0;
                    यदि (++i11 == ne1) {
                        i11 = 0;
                        यदि (++i12 == ne2) {
                            i12 = 0;
                            यदि (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        ध्रुव वर्ण * src0_ptr = ((वर्ण *) src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              वर्ण * dst_ptr  = ((वर्ण *)  dst->दत्त + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        भिन्न tmp = type_conversion_सारणी<src_t>::गन्तव्यम्_f32(*(ध्रुव src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_सारणी<dst_t>::from_f32(tmp);

                        यदि (++i10 == ne0) {
                            i10 = 0;
                            यदि (++i11 == ne1) {
                                i11 = 0;
                                यदि (++i12 == ne2) {
                                    i12 = 0;
                                    यदि (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                यावत् (i10 >= ne0) {
                    i10 -= ne0;
                    यदि (++i11 == ne1) {
                        i11 = 0;
                        यदि (++i12 == ne2) {
                            i12 = 0;
                            यदि (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


प्रतिरूपम्<प्रकारनाम src_t>
स्थिर शून्य ggml_compute_forward_dup_गन्तव्यम्_q(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith; // thread index
    ध्रुव पूर्णाङ्क nth = params->nth; // number of threads

    // parallelize by rows
    ध्रुव पूर्णाङ्क nr = ne01;
    // number of rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1) / nth;
    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    यदि (ggml_is_contiguous(dst) &&
            nb00 == परिमाण(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t ध्रुव quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        भिन्न * src0_f32 = (भिन्न *) params->wdata + (ne00 + CACHE_पङ्क्तिः_आकार_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_आकार(dst->type));
        वर्ण * dst_ptr = (वर्ण *) dst->दत्त;

        प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
            प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                    ध्रुव src_t * src0_ptr = (src_t *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03);

                    प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_सारणी<src_t>::गन्तव्यम्_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } अन्यथा {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
स्थिर शून्य ggml_compute_forward_dup_अष्टकानि(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    यदि (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        प्रत्यागम;
    }

    ध्रुव size_t type_आकार = ggml_type_आकार(src0->type);

    ध्रुव पूर्णाङ्क ith = params->ith; // thread index
    ध्रुव पूर्णाङ्क nth = params->nth; // number of threads

    // parallelize by rows
    ध्रुव पूर्णाङ्क nr = ne01;
    // number of rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1) / nth;
    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    यदि (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_आकार && nb0 == type_आकार) {
        // copy by rows
        ध्रुव size_t rs = ggml_row_आकार(src0->type, ne00);
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                    स्मृतिं_प्रतिलिख(
                        ((वर्ण *)  dst->दत्त + i01*nb1  + i02*nb2  + i03*nb3),
                        ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        प्रत्यागम;
    }

    यदि (ggml_is_contiguous(dst)) {
        size_t id = 0;
        वर्ण * dst_ptr = (वर्ण *) dst->दत्त;
        ध्रुव size_t rs = ne00 * type_आकार;

        यदि (nb00 == type_आकार) {
            // src0 is contiguous on first dimension, copy by rows
            प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
                प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ध्रुव वर्ण * src0_ptr = (वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03;
                        स्मृतिं_प्रतिलिख(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } अन्यथा {
            //printf("%s: this is not optimal - fix me\n", __func__);

            प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
                प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                        प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                            ध्रुव वर्ण * src0_ptr = (वर्ण *) src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            स्मृतिं_प्रतिलिख(dst_ptr + id, src0_ptr, type_आकार);

                            id += type_आकार;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        प्रत्यागम;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    ध्रुव int64_t nk00 = ne00 / ggml_blck_आकार(src0->type);
    ध्रुव int64_t nk0  = ne0  / ggml_blck_आकार(dst->type);

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            यावत् (k10 >= nk0) {
                k10 -= nk0;
                यदि (++i11 == ne1) {
                    i11 = 0;
                    यदि (++i12 == ne2) {
                        i12 = 0;
                        यदि (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                प्रति (int64_t k00 = 0; k00 < nk00; k00++) {
                    ध्रुव वर्ण * src0_ptr = ((वर्ण *) src0->दत्त + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          वर्ण * dst_ptr  = ((वर्ण *)  dst->दत्त + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    स्मृतिं_प्रतिलिख(dst_ptr, src0_ptr, type_आकार);

                    यदि (++k10 == nk0) {
                        k10 = 0;
                        यदि (++i11 == ne1) {
                            i11 = 0;
                            यदि (++i12 == ne2) {
                                i12 = 0;
                                यदि (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            यावत् (k10 >= nk0) {
                k10 -= nk0;
                यदि (++i11 == ne1) {
                    i11 = 0;
                    यदि (++i12 == ne2) {
                        i12 = 0;
                        यदि (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_dup_from_q(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव ggml_type type = src0->type;
    ggml_गन्तव्यम्_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->गन्तव्यम्_float;

    size_t qk = ggml_blck_आकार(type);
    ध्रुव int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_आकार(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        ध्रुव int64_t i03 = i/(ne00 * ne01 * ne02);
        ध्रुव int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        ध्रुव int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        ध्रुव int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        ध्रुव int64_t x_विस्थापनम् = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ध्रुव int64_t i13 = i/(ne10 * ne11 * ne12);
        ध्रुव int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ध्रुव int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ध्रुव int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ध्रुव int64_t dst_विस्थापनम् = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (ध्रुव शून्य *) ((वर्ण *) src0->दत्त + x_विस्थापनम्),
                     (भिन्न *) ((वर्ण *)  dst->दत्त + dst_विस्थापनम्), qk);
    }
}

शून्य ggml_compute_forward_dup(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (src0->type == dst->type) {
        ggml_compute_forward_dup_अष्टकानि(params, dst);
        प्रत्यागम;
    }

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                /**/ यदि (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, भिन्न      >(params, dst);
                अन्यथा ggml_compute_forward_dup_गन्तव्यम्_q<ggml_fp16_t>(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                /**/ यदि (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, भिन्न      >(params, dst);
                अन्यथा ggml_compute_forward_dup_गन्तव्यम्_q<ggml_bf16_t>(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                /**/ यदि (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<भिन्न, ggml_fp16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<भिन्न, ggml_bf16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<भिन्न, भिन्न      >(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<भिन्न, int32_t    >(params, dst);
                अन्यथा ggml_compute_forward_dup_गन्तव्यम्_q<भिन्न>(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                यदि (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, भिन्न>(params, dst);
                अन्यथा GGML_ABORT("not implemented");
            } भङ्ग;
        स्वभाव:
            {
                यदि (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    भङ्ग;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

स्थिर शून्य ggml_compute_forward_सङ्कलय_q_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव ggml_type type = src0->type;
    ध्रुव ggml_type dtype = dst->type;
    ggml_गन्तव्यम्_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->गन्तव्यम्_float;
    ggml_from_float_t ध्रुव quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    भिन्न * wdata = (भिन्न *) params->wdata + (ne00 + CACHE_पङ्क्तिः_आकार_F32) * ith;

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ध्रुव पूर्णाङ्क i03 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        ध्रुव पूर्णाङ्क i13 = i03;
        ध्रुव पूर्णाङ्क i12 = i02;
        ध्रुव पूर्णाङ्क i11 = i01;

        ध्रुव पूर्णाङ्क i3 = i03;
        ध्रुव पूर्णाङ्क i2 = i02;
        ध्रुव पूर्णाङ्क i1 = i01;

        शून्य  * src0_row = (शून्य *) ((वर्ण *) src0->दत्त + (i01*nb01 + i02*nb02 + i03*nb03));
        भिन्न * src1_row = (भिन्न *)((वर्ण *) src1->दत्त + (i11*nb11 + i12*nb12 + i13*nb13));
        शून्य  * dst_row  = (शून्य *) ((वर्ण *)  dst->दत्त + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        प्रतिज्ञा(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        यदि (quantize_row_q != रिक्त) {
            quantize_row_q(wdata, dst_row, ne00);
        } अन्यथा {
            स्मृतिं_प्रतिलिख(dst_row, wdata, ne0*nb0);
        }
    }
}

शून्य ggml_compute_forward_सङ्कलय(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_सङ्कलय_non_quantized(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_सङ्कलय_q_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

स्थिर शून्य ggml_compute_forward_सङ्कलय_id_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src1->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        ध्रुव पूर्णाङ्क i11 = *(int32_t *) ((वर्ण *) src2->दत्त + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_सङ्कलय_f32(ne0,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01),
                (भिन्न *) ((वर्ण *) src1->दत्त + i11*nb11));
    }
}

शून्य ggml_compute_forward_सङ्कलय_id(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_सङ्कलय_id_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

स्थिर शून्य ggml_compute_forward_add1_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

#यदि_परिभाषितम् GGML_USE_ACCELERATE
        GGML_अप्रयुक्तम्(ggml_vec_add1_f32);

        vDSP_vadd(
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (भिन्न *) ((वर्ण *) src1->दत्त), 0,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#अन्यथा
        ggml_vec_add1_f32(ne0,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01),
               *(भिन्न *) src1->दत्त);
#यदि_अन्तः
    }
}

स्थिर शून्य ggml_compute_forward_add1_f16_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ध्रुव भिन्न v = *(भिन्न *) src1->दत्त;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb00 == परिमाण(ggml_fp16_t));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(GGML_CPU_FP16_गन्तव्यम्_FP32(src0_ptr[i]) + v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_add1_f16_f16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ध्रुव भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(*(ggml_fp16_t *) src1->दत्त);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb00 == परिमाण(ggml_fp16_t));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(GGML_CPU_FP16_गन्तव्यम्_FP32(src0_ptr[i]) + v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_add1_q_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ध्रुव भिन्न v = *(भिन्न *) src1->दत्त;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव ggml_type type = src0->type;
    ggml_गन्तव्यम्_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->गन्तव्यम्_float;
    ggml_from_float_t ध्रुव quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_आकार(type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(dst->type == src0->type);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    भिन्न * wdata = (भिन्न *) params->wdata + (ne0 + CACHE_पङ्क्तिः_आकार_F32) * ith;

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        शून्य  * src0_row = (शून्य *) ((वर्ण *) src0->दत्त + (i1*nb01 + i2*nb02 + i3*nb03));
        शून्य  * dst_row  = (शून्य *) ((वर्ण *)  dst->दत्त + (i1*nb1  + i2*nb2  + i3*nb0 ));

        प्रतिज्ञा(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

स्थिर शून्य ggml_compute_forward_add1_bf16_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ध्रुव भिन्न v = *(भिन्न *) src1->दत्त;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == परिमाण(ggml_bf16_t));
    GGML_ASSERT(nb00 == परिमाण(ggml_bf16_t));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_गन्तव्यम्_BF16(GGML_BF16_गन्तव्यम्_FP32(src0_ptr[i]) + v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_add1_bf16_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    ध्रुव भिन्न v = GGML_BF16_गन्तव्यम्_FP32(*(ggml_bf16_t *) src1->दत्त);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == परिमाण(ggml_bf16_t));
    GGML_ASSERT(nb00 == परिमाण(ggml_bf16_t));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_गन्तव्यम्_BF16(GGML_BF16_गन्तव्यम्_FP32(src0_ptr[i]) + v);
        }
    }
}

शून्य ggml_compute_forward_add1(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                यदि (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                अन्यथा यदि (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                अन्यथा {
                    GGML_ABORT("fatal error");
                }
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                यदि (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                अन्यथा यदि (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                अन्यथा {
                    GGML_ABORT("fatal error");
                }
            } भङ्ग;
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q8_1:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

स्थिर शून्य ggml_compute_forward_acc_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t विस्थापनम्  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    यदि (!inplace) {
        यदि (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स्मृतिं_प्रतिलिख(
                ((वर्ण *)  dst->दत्त),
                ((वर्ण *) src0->दत्त),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr = ggml_nrows(src1);
    ध्रुव पूर्णाङ्क nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    ध्रुव size_t nb0 = ggml_element_आकार(src0);

    ध्रुव size_t nb00 = nb0;
    ध्रुव size_t nb01 = nb1;
    ध्रुव size_t nb02 = nb2;
    ध्रुव size_t nb03 = nb3;

    GGML_ASSERT(विस्थापनम् + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(विस्थापनम् + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne12*ne11);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne12*ne11)/ne11;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne12*ne11 - i2*ne11);

#यदि_परिभाषितम् GGML_USE_ACCELERATE
        vDSP_vadd(
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01 + विस्थापनम्), 1,
                (भिन्न *) ((वर्ण *) src1->दत्त + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1  + विस्थापनम्), 1, nc);
#अन्यथा
        ggml_vec_सङ्कलय_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->दत्त + i3*nb3  + i2*nb2  + i1*nb1  + विस्थापनम्),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01 + विस्थापनम्),
                (भिन्न *) ((वर्ण *) src1->दत्त + i3*nb13 + i2*nb12 + i1*nb11));
#यदि_अन्तः
    }
}

शून्य ggml_compute_forward_acc(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q8_1:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

स्थिर शून्य ggml_compute_forward_योग_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float योग_2 = 0;

    यदि (nb00 == परिमाण(भिन्न)) {
        ggml_float row_योग = 0;
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_योग_f32_ggf(ne00,
                            &row_योग,
                            (भिन्न *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03));
                    योग_2 += row_योग;
                }
            }
        }
    } अन्यथा {
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव वर्ण * row = (ध्रुव वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        योग_2 += (ggml_float) *(ध्रुव भिन्न *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((भिन्न *) dst->दत्त)[0] = योग_2;
}

स्थिर शून्य ggml_compute_forward_योग_f16(
    ध्रुव ggml_compute_params * params,
          ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    प्रतिज्ञा(src0->nb[0] == परिमाण(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    भिन्न योग_2 = 0;
    भिन्न row_योग = 0;

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_योग_f16_ggf(ne00,
                    &row_योग,
                    (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03));
                योग_2 += row_योग;
            }
        }
    }
    ((ggml_fp16_t *) dst->दत्त)[0] = GGML_CPU_FP32_गन्तव्यम्_FP16(योग_2);
}

स्थिर शून्य ggml_compute_forward_योग_bf16(
    ध्रुव ggml_compute_params * params,
          ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    प्रतिज्ञा(src0->nb[0] == परिमाण(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    भिन्न योग_2 = 0;
    भिन्न row_योग = 0;

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_योग_bf16_ggf(ne00,
                    &row_योग,
                    (ggml_bf16_t *) ((वर्ण *) src0->दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03));
                योग_2 += row_योग;
            }
        }
    }
    ((ggml_bf16_t *) dst->दत्त)[0] = GGML_FP32_गन्तव्यम्_BF16(योग_2);
}

शून्य ggml_compute_forward_योग(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_योग_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_योग_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_योग_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

स्थिर शून्य ggml_compute_forward_cumsum_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(dst->nb[0] == परिमाण(भिन्न));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    ध्रुव स्वचालित [ir0, ir1] = get_thread_range(params, src0);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        भिन्न * src_row = (भिन्न *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03);
        भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst->दत्त  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

शून्य ggml_compute_forward_cumsum(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

स्थिर शून्य ggml_compute_forward_योग_पङ्क्तिकाः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(dst->nb[0] == परिमाण(भिन्न));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    प्रति (int64_t i3 = 0; i3 < ne03; i3++) {
        प्रति (int64_t i2 = 0; i2 < ne02; i2++) {
            प्रति (int64_t i1 = 0; i1 < ne01; i1++) {
                भिन्न * src_row = (भिन्न *) ((वर्ण *) src0->दत्त + i1*nb01 + i2*nb02 + i3*nb03);
                भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst->दत्त  + i1*nb1  + i2*nb2  + i3*nb3);
                भिन्न row_योग = 0;
                ggml_vec_योग_f32(ne00, &row_योग, src_row);
                dst_row[0] = row_योग;
            }
        }
    }
}

शून्य ggml_compute_forward_योग_पङ्क्तिकाः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_योग_पङ्क्तिकाः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

स्थिर शून्य ggml_compute_forward_माध्यम्_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(src0->nb[0] == परिमाण(भिन्न));

    GGML_TENSOR_UNARY_OP_LOCALS

    प्रतिज्ञा(ne0 == 1);
    प्रतिज्ञा(ne1 == ne01);
    प्रतिज्ञा(ne2 == ne02);
    प्रतिज्ञा(ne3 == ne03);

    GGML_अप्रयुक्तम्(ne0);
    GGML_अप्रयुक्तम्(ne1);
    GGML_अप्रयुक्तम्(ne2);
    GGML_अप्रयुक्तम्(ne3);

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_योग_f32(ne00,
                        (भिन्न *) ((वर्ण *)  dst->दत्त + i01*nb1  + i02*nb2  + i03*nb3),
                        (भिन्न *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03));

                *(भिन्न *) ((वर्ण *) dst->दत्त + i01*nb1 + i02*nb2 + i03*nb3) /= (भिन्न) ne00;
            }
        }
    }
}

शून्य ggml_compute_forward_माध्यम्(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_माध्यम्_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

स्थिर शून्य ggml_compute_forward_argmax_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(src0->nb[0] == परिमाण(भिन्न));
    प्रतिज्ञा(dst->nb[0] == परिमाण(भिन्न));

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];

    ध्रुव size_t nb01 = src0->nb[1];
    ध्रुव size_t nb0 = dst->nb[0];

    प्रति (int64_t i1 = 0; i1 < ne01; i1++) {
        भिन्न * src = (भिन्न *) ((वर्ण *) src0->दत्त + i1*nb01);
        int32_t * dst_ = (int32_t *) ((वर्ण *)  dst->दत्त + i1*nb0);
        पूर्णाङ्क v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

शून्य ggml_compute_forward_argmax(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

स्थिर शून्य ggml_compute_forward_गणनम्_equal_i32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    ध्रुव int64_t nr = ggml_nrows(src0);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t योग_thread = 0;

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 =  ir                        / (ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne03)            /       ne01;
        ध्रुव int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ध्रुव वर्ण * data0 = (ध्रुव वर्ण *) src0->दत्त + i03*nb03 + i02*nb02 + i01*nb01;
        ध्रुव वर्ण * data1 = (ध्रुव वर्ण *) src1->दत्त + i03*nb13 + i02*nb12 + i01*nb11;

        प्रति (int64_t i00 = 0; i00 < ne00; ++i00) {
            ध्रुव int32_t val0 = *((ध्रुव int32_t *) (data0 + i00*nb00));
            ध्रुव int32_t val1 = *((ध्रुव int32_t *) (data1 + i00*nb10));

            योग_thread += val0 == val1;
        }
    }
    यदि (ith != 0) {
        sums[ith] = योग_thread;
    }
    ggml_barrier(params->threadpool);

    यदि (ith != 0) {
        प्रत्यागम;
    }

    प्रति (पूर्णाङ्क ith_other = 1; ith_other < nth; ++ith_other) {
        योग_thread += sums[ith_other];
    }
    *((int64_t *) dst->दत्त) = योग_thread;
}

शून्य ggml_compute_forward_गणनम्_equal(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_गणनम्_equal_i32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

स्थिर शून्य ggml_compute_forward_repeat_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ध्रुव पूर्णाङ्क nr0 = (पूर्णाङ्क)(ne0/ne00);
    ध्रुव पूर्णाङ्क nr1 = (पूर्णाङ्क)(ne1/ne01);
    ध्रुव पूर्णाङ्क nr2 = (पूर्णाङ्क)(ne2/ne02);
    ध्रुव पूर्णाङ्क nr3 = (पूर्णाङ्क)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == परिमाण(भिन्न));
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    // TODO: maybe this is not optimal?
    प्रति                         (पूर्णाङ्क i3 = 0; i3 < nr3;  i3++) {
        प्रति                     (पूर्णाङ्क k3 = 0; k3 < ne03; k3++) {
            प्रति                 (पूर्णाङ्क i2 = 0; i2 < nr2;  i2++) {
                प्रति             (पूर्णाङ्क k2 = 0; k2 < ne02; k2++) {
                    प्रति         (पूर्णाङ्क i1 = 0; i1 < nr1;  i1++) {
                        प्रति     (पूर्णाङ्क k1 = 0; k1 < ne01; k1++) {
                            प्रति (पूर्णाङ्क i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (भिन्न *) ((वर्ण *)  dst->दत्त + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (भिन्न *) ((वर्ण *) src0->दत्त + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_repeat_f16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ध्रुव पूर्णाङ्क nr0 = (पूर्णाङ्क)(ne0/ne00);
    ध्रुव पूर्णाङ्क nr1 = (पूर्णाङ्क)(ne1/ne01);
    ध्रुव पूर्णाङ्क nr2 = (पूर्णाङ्क)(ne2/ne02);
    ध्रुव पूर्णाङ्क nr3 = (पूर्णाङ्क)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb00 == परिमाण(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    प्रति                         (पूर्णाङ्क i3 = 0; i3 < nr3;  i3++) {
        प्रति                     (पूर्णाङ्क k3 = 0; k3 < ne03; k3++) {
            प्रति                 (पूर्णाङ्क i2 = 0; i2 < nr2;  i2++) {
                प्रति             (पूर्णाङ्क k2 = 0; k2 < ne02; k2++) {
                    प्रति         (पूर्णाङ्क i1 = 0; i1 < nr1;  i1++) {
                        प्रति     (पूर्णाङ्क k1 = 0; k1 < ne01; k1++) {
                            प्रति (पूर्णाङ्क i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((वर्ण *)  dst->दत्त + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((वर्ण *) src0->दत्त + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
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

शून्य ggml_compute_forward_repeat(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } भङ्ग;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

स्थिर शून्य ggml_compute_forward_repeat_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    ध्रुव पूर्णाङ्क nr0 = (पूर्णाङ्क)(ne00/ne0);
    ध्रुव पूर्णाङ्क nr1 = (पूर्णाङ्क)(ne01/ne1);
    ध्रुव पूर्णाङ्क nr2 = (पूर्णाङ्क)(ne02/ne2);
    ध्रुव पूर्णाङ्क nr3 = (पूर्णाङ्क)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == परिमाण(भिन्न));
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    यदि (ggml_is_contiguous(dst)) {
        ggml_vec_समुच्चयः_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->दत्त, 0);
    } अन्यथा {
        प्रति         (पूर्णाङ्क k3 = 0; k3 < ne3; k3++) {
            प्रति     (पूर्णाङ्क k2 = 0; k2 < ne2; k2++) {
                प्रति (पूर्णाङ्क k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_समुच्चयः_f32(ne0,
                        (भिन्न *) ((वर्ण *) dst->दत्त + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    प्रति                         (पूर्णाङ्क i3 = 0; i3 < nr3; i3++) {
        प्रति                     (पूर्णाङ्क k3 = 0; k3 < ne3; k3++) {
            प्रति                 (पूर्णाङ्क i2 = 0; i2 < nr2; i2++) {
                प्रति             (पूर्णाङ्क k2 = 0; k2 < ne2; k2++) {
                    प्रति         (पूर्णाङ्क i1 = 0; i1 < nr1; i1++) {
                        प्रति     (पूर्णाङ्क k1 = 0; k1 < ne1; k1++) {
                            प्रति (पूर्णाङ्क i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (भिन्न *) ((वर्ण *)  dst->दत्त + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (भिन्न *) ((वर्ण *) src0->दत्त + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_repeat_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_पृष्ठभागः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
स्थिर शून्य ggml_compute_forward_concat_any(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src0));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src1));

    int64_t o[4] = {0, 0, 0, 0};

    यदि (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_आकार(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_आकार(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_आकार(src0->type);
    } अन्यथा {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne03; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne02; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne01; i1++) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src0->दत्त + i1*nb01 + i2*nb02 + i3*nb03;
                      वर्ण * y = (      वर्ण *) dst->दत्त  + i1*nb1  + i2*nb2  + i3*nb3;
                स्मृतिं_प्रतिलिख(y, x, ggml_row_आकार(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne13; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne12; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne11; i1++) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src1->दत्त + i1*nb11         + i2*nb12         + i3*nb13;
                      वर्ण * y = (      वर्ण *)  dst->दत्त + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                स्मृतिं_प्रतिलिख(y, x, ggml_row_आकार(src1->type, ne10));
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_i8(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_आकार(src0->type) == परिमाण(int8_t));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ध्रुव int8_t * x;

    // TODO: smarter multi-theading
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne3; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne2; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne1; i1++) {
                प्रति (पूर्णाङ्क i0 = 0; i0 < ne0; i0++) {
                    यदि (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ध्रुव int8_t *) ((ध्रुव वर्ण *)src0->दत्त + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (ध्रुव int8_t *) ((ध्रुव वर्ण *)src1->दत्त + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((वर्ण *)dst->दत्त + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_आकार(src0->type) == परिमाण(ggml_fp16_t));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ध्रुव ggml_fp16_t * x;

    // TODO: smarter multi-theading
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne3; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne2; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne1; i1++) {
                प्रति (पूर्णाङ्क i0 = 0; i0 < ne0; i0++) {
                    यदि (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ध्रुव ggml_fp16_t *) ((ध्रुव वर्ण *)src0->दत्त + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (ध्रुव ggml_fp16_t *) ((ध्रुव वर्ण *)src1->दत्त + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((वर्ण *)dst->दत्त + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_आकार(src0->type) == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    ध्रुव भिन्न * x;

    // TODO: smarter multi-theading
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne3; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne2; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne1; i1++) {
                प्रति (पूर्णाङ्क i0 = 0; i0 < ne0; i0++) {
                    यदि (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src0->दत्त + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src1->दत्त + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    भिन्न * y = (भिन्न *)((वर्ण *)dst->दत्त + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_concat(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
प्रतिरूपम् <शून्य (*vec_op)(ध्रुव पूर्णाङ्क, ggml_bf16_t *, ध्रुव ggml_bf16_t *)>
स्थिर शून्य ggml_compute_forward_unary_bf16_पङ्क्तिकाः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

स्थिर शून्य ggml_compute_forward_gelu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_पङ्क्तिकाः<ggml_vec_gelu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

स्थिर शून्य ggml_compute_forward_fill_f32(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव भिन्न c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ध्रुव स्वचालित [ir0, ir1] = get_thread_range(params, dst);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne2*ne1);
        ध्रुव int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ध्रुव int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        भिन्न * dst_ptr  = (भिन्न *) ((वर्ण *) dst->दत्त + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_समुच्चयः_f32(ne0, dst_ptr, c);
    }
}

स्थिर शून्य ggml_compute_forward_fill_f16(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_fp16_t c = GGML_CPU_FP32_गन्तव्यम्_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ध्रुव स्वचालित [ir0, ir1] = get_thread_range(params, dst);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne2*ne1);
        ध्रुव int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ध्रुव int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_समुच्चयः_f16(ne0, dst_ptr, c);
    }
}

शून्य ggml_compute_forward_fill(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

स्थिर शून्य ggml_compute_forward_tri_f32(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    ध्रुव ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव स्वचालित [ir0, ir1] = get_thread_range(params, src0);

    तार्किक (*bipred)(पूर्णाङ्क, पूर्णाङ्क);

    विकल्प (ttype) {
        पक्ष GGML_TRI_TYPE_अधस्तनम्:      bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i <  r; }; भङ्ग;
        पक्ष GGML_TRI_TYPE_अधस्तनम्_DIAG: bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i <= r; }; भङ्ग;
        पक्ष GGML_TRI_TYPE_ऊर्ध्वस्थम्:      bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i >  r; }; भङ्ग;
        पक्ष GGML_TRI_TYPE_ऊर्ध्वस्थम्_DIAG: bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i >= r; }; भङ्ग;
        स्वभाव: GGML_ABORT("invalid tri type");
    }

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न  *) ((ध्रुव वर्ण *) src0->दत्त + i03*nb03 + i02*nb02 + i01*nb01);
              भिन्न * dst_ptr = (      भिन्न  *) ((      वर्ण *) dst->दत्त  + i03*nb3  + i02*nb2  + i01*nb1);

        प्रति (पूर्णाङ्क i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

शून्य ggml_compute_forward_tri(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

स्थिर शून्य ggml_compute_forward_gelu_erf_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_erf_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_erf_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_पङ्क्तिकाः<ggml_vec_gelu_erf_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu_erf(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

स्थिर शून्य ggml_compute_forward_gelu_शीघ्र_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_शीघ्र_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_शीघ्र_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_शीघ्र_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_शीघ्र_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_पङ्क्तिकाः<ggml_vec_gelu_शीघ्र_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu_शीघ्र(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_शीघ्र_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_शीघ्र_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_शीघ्र_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

स्थिर शून्य ggml_compute_forward_silu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_पङ्क्तिकाः(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव पूर्णाङ्क i3 = ir/(ne02*ne01);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne02*ne01)/ne01;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_पङ्क्तिकाः<ggml_vec_silu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_silu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

स्थिर शून्य ggml_compute_forward_leaky_relu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(ggml_is_contiguous_1(src0));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    ध्रुव पूर्णाङ्क n  = ggml_nrows(src0);
    ध्रुव पूर्णाङ्क nc = src0->ne[0];

    भिन्न negative_नतिः;
    स्मृतिं_प्रतिलिख(&negative_नतिः, dst->op_params, परिमाण(भिन्न));

    प्रतिज्ञा(dst->nb[0]  == परिमाण(भिन्न));
    प्रतिज्ञा(src0->nb[0] == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i*( dst->nb[1])),
                (भिन्न *) ((वर्ण *) src0->दत्त + i*(src0->nb[1])), negative_नतिः);
    }
}

स्थिर शून्य ggml_compute_forward_leaky_relu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(ggml_is_contiguous_1(src0));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    ध्रुव पूर्णाङ्क n  = ggml_nrows(src0);
    ध्रुव पूर्णाङ्क nc = src0->ne[0];

    भिन्न negative_नतिः;
    स्मृतिं_प्रतिलिख(&negative_नतिः, dst->op_params, परिमाण(भिन्न));

    प्रतिज्ञा(dst->nb[0]  == परिमाण(ggml_fp16_t));
    प्रतिज्ञा(src0->nb[0] == परिमाण(ggml_fp16_t));

    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i*(src0->nb[1])), negative_नतिः);
    }
}

शून्य ggml_compute_forward_leaky_relu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

स्थिर शून्य ggml_compute_forward_silu_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * grad = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    प्रतिज्ञा(ggml_is_contiguous_1(grad));
    प्रतिज्ञा(ggml_is_contiguous_1(src1));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, grad));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i1*( dst->nb[1])),
                (भिन्न *) ((वर्ण *) src1->दत्त + i1*(src1->nb[1])),
                (भिन्न *) ((वर्ण *) grad->दत्त + i1*(grad->nb[1])));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_पृष्ठभागः_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * grad = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    प्रतिज्ञा(ggml_is_contiguous_1(grad));
    प्रतिज्ञा(ggml_is_contiguous_1(src1));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, grad));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->दत्त  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((वर्ण *) src1->दत्त + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((वर्ण *) grad->दत्त + i1*(grad->nb[1])));

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

शून्य ggml_compute_forward_silu_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_पृष्ठभागः_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_पृष्ठभागः_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
प्रतिरूपम् <शून्य (*vec_op)(ध्रुव पूर्णाङ्क, ggml_bf16_t *, ध्रुव ggml_bf16_t *, ध्रुव ggml_bf16_t *)>
स्थिर शून्य ggml_compute_forward_glu_bf16_पङ्क्तिकाः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ध्रुव ggml_bf16_t * src0_p = (ध्रुव ggml_bf16_t *) (src0_d + i1*src0_o);
        ध्रुव ggml_bf16_t * src1_p = (ध्रुव ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

स्थिर शून्य ggml_compute_forward_reglu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (भिन्न *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_reglu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_reglu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_पङ्क्तिकाः<ggml_vec_reglu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_reglu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

स्थिर शून्य ggml_compute_forward_geglu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (भिन्न *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_पङ्क्तिकाः<ggml_vec_geglu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

स्थिर शून्य ggml_compute_forward_swiglu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (भिन्न *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_पङ्क्तिकाः<ggml_vec_swiglu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_swiglu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

स्थिर शून्य ggml_compute_forward_swiglu_oai_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ध्रुव भिन्न alpha = ggml_get_op_params_f32(dst, 2);
    ध्रुव भिन्न सीमा = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);
        भिन्न * dst_p  = (भिन्न *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = std::min(src0_p[k], सीमा);
            ध्रुव भिन्न y = std::clamp(src1_p[k], -सीमा, सीमा);
            ध्रुव भिन्न बहिर्गमनम्_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = बहिर्गमनम्_glu * (y + 1.f);
        }

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = dst_p[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_oai_f16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ध्रुव भिन्न alpha = ggml_get_op_params_f32(dst, 2);
    ध्रुव भिन्न सीमा = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = std::min(GGML_CPU_FP16_गन्तव्यम्_FP32(src0_p[k]), सीमा);
            ध्रुव भिन्न y = std::clamp(GGML_CPU_FP16_गन्तव्यम्_FP32(src1_p[k]), -सीमा, सीमा);
            ध्रुव भिन्न बहिर्गमनम्_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_गन्तव्यम्_FP16(बहिर्गमनम्_glu * (y + 1.f));
        }

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = GGML_CPU_FP16_गन्तव्यम्_FP32(dst_p[k]);
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_oai(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

स्थिर शून्य ggml_compute_forward_swiglu_clamp_f32(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    वर्ण *              src0_d = (वर्ण *) src0->दत्त;
    वर्ण *              src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t        src0_o = src0->nb[1];
    ध्रुव size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ध्रुव भिन्न   सीमा   = ggml_get_op_params_f32(dst, 3);

    ध्रुव पूर्णाङ्क dr  = (nr + nth - 1) / nth;
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1 * src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1 * src1_o);
        भिन्न * dst_p  = (भिन्न *) ((वर्ण *) dst->दत्त + i1 * (dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न gate = std::min(src0_p[k], सीमा);
            ध्रुव भिन्न ऊर्ध्वम्   = std::clamp(src1_p[k], -सीमा, सीमा);
            dst_p[k]         = gate / (1.f + expf(-gate)) * ऊर्ध्वम्;
        }

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = dst_p[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः  // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_clamp_f16(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    वर्ण *              src0_d = (वर्ण *) src0->दत्त;
    वर्ण *              src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t        src0_o = src0->nb[1];
    ध्रुव size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);
    ध्रुव भिन्न   सीमा   = ggml_get_op_params_f32(dst, 3);

    ध्रुव पूर्णाङ्क dr  = (nr + nth - 1) / nth;
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1 * (dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न gate = std::min(GGML_FP16_गन्तव्यम्_FP32(src0_p[k]), सीमा);
            ध्रुव भिन्न ऊर्ध्वम्   = std::clamp(GGML_FP16_गन्तव्यम्_FP32(src1_p[k]), -सीमा, सीमा);
            dst_p[k]         = GGML_FP32_गन्तव्यम्_FP16(gate / (1.f + expf(-gate)) * ऊर्ध्वम्);
        }

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = GGML_FP16_गन्तव्यम्_FP32(dst_p[k]);
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः  // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_clamp(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    विकल्प (dst->src[0]->type) {
        पक्ष GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            भङ्ग;
        स्वभाव:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

स्थिर शून्य ggml_compute_forward_geglu_erf_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (भिन्न *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_erf_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_erf_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_पङ्क्तिकाः<ggml_vec_geglu_erf_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu_erf(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

स्थिर शून्य ggml_compute_forward_geglu_शीघ्र_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_शीघ्र_f32(nc, (भिन्न *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            GGML_अप्रयुक्तम्(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_शीघ्र_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->दत्त;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->दत्त : src0->दत्त);
    ध्रुव size_t src0_o = src0->nb[1];
    ध्रुव size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    यदि (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    ध्रुव int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_शीघ्र_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*(dst->nb[1])), src0_p, src1_p);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->दत्त + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_गन्तव्यम्_FP32(x);
            GGML_अप्रयुक्तम्(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#यदि_अन्तः // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_शीघ्र_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_पङ्क्तिकाः<ggml_vec_geglu_शीघ्र_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu_शीघ्र(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_शीघ्र_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_शीघ्र_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_शीघ्र_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

स्थिर शून्य ggml_compute_forward_norm_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भिन्न eps;
    स्मृतिं_प्रतिलिख(&eps, dst->op_params, परिमाण(भिन्न));

    GGML_ASSERT(eps >= 0.0f);

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03;
                वर्ण * y = (वर्ण *) dst->दत्त + i01*nb1 + i02*nb2 + i03*nb3;

                यदि (nb00 == परिमाण(भिन्न) && nb0 == परिमाण(भिन्न)) {
                    ध्रुव भिन्न * xf = (ध्रुव भिन्न *) x;

                    भिन्न योग_2 = 0.0;
                    ggml_vec_योग_f32(ne00, &योग_2, xf);
                    भिन्न माध्यम्_2 = योग_2/ne00;

                    भिन्न * yf = (भिन्न *) y;
                    भिन्न variance = 0;

#यदि_परिभाषितम् GGML_USE_ACCELERATE
                    माध्यम्_2 = -माध्यम्_2;
                    vDSP_vsadd(xf, 1, &माध्यम्_2, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#अन्यथा
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, माध्यम्_2);
#यदि_अन्तः //GGML_USE_ACCELERATE

                    ध्रुव भिन्न scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } अन्यथा {
                    भिन्न योग_2 = 0.0;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        योग_2 += *(ध्रुव भिन्न *) (x + i00*nb00);
                    }
                    ध्रुव भिन्न माध्यम्_2 = योग_2/ne00;

                    भिन्न variance = 0.0f;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        ध्रुव भिन्न v = *(ध्रुव भिन्न *) (x + i00*nb00) - माध्यम्_2;
                        *(भिन्न *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    ध्रुव भिन्न scale = 1.0f/sqrtf(variance + eps);
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(भिन्न *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_norm(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
गणना ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

प्रतिरूपम् <ggml_rms_norm_fuse_op FUSE_OP>
स्थिर शून्य ggml_compute_forward_rms_norm_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = शून्यसूचकः) {

    ध्रुव ggml_tensor * src0 = dst_rms_norm->src[0];
    ध्रुव ggml_tensor * src1 = शून्यसूचकः;
    ggml_tensor       * dst  = dst_rms_norm;

    यदि constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    भिन्न eps;
    स्मृतिं_प्रतिलिख(&eps, dst_rms_norm->op_params, परिमाण(भिन्न));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव भिन्न * x = (भिन्न *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float योग_2 = 0.0;
                // worth switching to explicit SIMD?
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    योग_2 += (ggml_float)(x[i00] * x[i00]);
                }

                ध्रुव भिन्न माध्यम्_2  = योग_2/ne00;
                ध्रुव भिन्न scale = 1.0f/sqrtf(माध्यम्_2 + eps);

                // if you hit this, likely you got an inf somewhere earlier
                प्रतिज्ञा(scale > 0.0f);

                भिन्न * y = (भिन्न *) ((वर्ण *) dst->दत्त + i01*nb1 + i02*nb2 + i03*nb3);

                यदि constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ध्रुव int64_t i11 = i01 % ne11;
                    ध्रुव int64_t i12 = i02 % ne12;
                    ध्रुव int64_t i13 = i03 % ne13;
                    ध्रुव भिन्न * w = (भिन्न *) ((वर्ण *) src1->दत्त + i11*nb11 + i12*nb12 + i13*nb13);

                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } अन्यथा {
                    स्मृतिं_प्रतिलिख(y, x, ne00 * परिमाण(भिन्न));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

शून्य ggml_compute_forward_rms_norm(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
शून्य ggml_compute_forward_rms_norm_mul_fused(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != शून्यसूचकः);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    ध्रुव ggml_tensor * src0 = dst_rms_norm->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_rms_norm_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    ध्रुव ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src1->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    भिन्न eps;
    स्मृतिं_प्रतिलिख(&eps, dst->op_params, परिमाण(भिन्न));

    // TODO: optimize
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ध्रुव int64_t i11 = i01;
                ध्रुव int64_t i12 = i02;
                ध्रुव int64_t i13 = i03;

                ध्रुव भिन्न * dz = (भिन्न *) ((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03);
                ध्रुव भिन्न * x  = (भिन्न *) ((वर्ण *) src1->दत्त + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float योग_xx  = 0.0;
                ggml_float योग_xdz = 0.0;

                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    योग_xx  += (ggml_float)(x[i00] * x[i00]);
                    योग_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ध्रुव भिन्न माध्यम्_eps = (भिन्न)(योग_xx)/ne00 + eps;
                ध्रुव भिन्न योग_eps  = (भिन्न)(योग_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ध्रुव भिन्न rrms     = 1.0f / sqrtf(माध्यम्_eps);
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
                भिन्न * dx = (भिन्न *) ((वर्ण *) dst->दत्त + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ध्रुव भिन्न scale_x = (भिन्न) (-योग_xdz) / योग_eps;
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_rms_norm_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_पृष्ठभागः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

स्थिर शून्य ggml_compute_forward_गणः_norm_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    भिन्न eps;
    स्मृतिं_प्रतिलिख(&eps, dst->op_params + 1, परिमाण(भिन्न));

    पूर्णाङ्क n_channels = src0->ne[2];
    पूर्णाङ्क n_groups = dst->op_params[0];
    पूर्णाङ्क n_channels_प्रत्येकम्_गणः = (n_channels + n_groups - 1) / n_groups;
    प्रति (पूर्णाङ्क i = ith; i < n_groups; i += nth) {
        पूर्णाङ्क start = i * n_channels_प्रत्येकम्_गणः;
        पूर्णाङ्क end = start + n_channels_प्रत्येकम्_गणः;
        यदि (end > n_channels) {
            end = n_channels;
        }
        पूर्णाङ्क चरणः = end - start;

        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float योग_2 = 0.0;
            प्रति (int64_t i02 = start; i02 < end; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव भिन्न * x = (भिन्न *)((वर्ण *) src0->दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    योग_2 += sumr;
                }
            }
            ध्रुव भिन्न माध्यम्_2 = योग_2 / (ne00 * ne01 * चरणः);

            ggml_float sum2 = 0.0;
            प्रति (int64_t i02 = start; i02 < end; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव भिन्न * x = (भिन्न *)((वर्ण *) src0->दत्त + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    भिन्न * y = (भिन्न *)((वर्ण *) dst->दत्त + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        भिन्न v = x[i00] - माध्यम्_2;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ध्रुव भिन्न variance = sum2 / (ne00 * ne01 * चरणः);
            ध्रुव भिन्न scale = 1.0f / sqrtf(variance + eps);

            प्रति (int64_t i02 = start; i02 < end; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    भिन्न * y = (भिन्न *)((वर्ण *) dst->दत्त + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

शून्य ggml_compute_forward_गणः_norm(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_गणः_norm_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

स्थिर शून्य ggml_compute_forward_l2_norm_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भिन्न eps;
    स्मृतिं_प्रतिलिख(&eps, dst->op_params, परिमाण(भिन्न));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float योग_2 = 0.0;
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    ध्रुव भिन्न xi = *(ध्रुव भिन्न *) (x + i00*nb00);
                    योग_2 += (ggml_float)(xi * xi);
                }

                ध्रुव भिन्न scale = 1.0f/fmaxf(sqrtf(योग_2), eps);

                वर्ण * y = (वर्ण *) dst->दत्त + i01*nb1 + i02*nb2 + i03*nb3;

                यदि (nb00 == परिमाण(भिन्न) && nb0 == परिमाण(भिन्न)) {
                    स्मृतिं_प्रतिलिख(y, x, ne00 * परिमाण(भिन्न));
                    ggml_vec_scale_f32(ne00, (भिन्न *) y, scale);
                } अन्यथा {
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        ध्रुव भिन्न xi = *(ध्रुव भिन्न *) (x + i00*nb00);
                        *(भिन्न *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_l2_norm(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

स्थिर शून्य ggml_compute_forward_बहिर्गमनम्_prod_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    यदि (ith == 0) {
        ggml_vec_समुच्चयः_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->दत्त, 0);
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
    ध्रुव int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    ध्रुव int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    ध्रुव int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    ध्रुव int64_t dps2 = ne2 / ne02;
    ध्रुव int64_t dps3 = ne3 / ne03;

    प्रति (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        ध्रुव int64_t bir1 = MIN(bir + blck_1, ir1);
        प्रति (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            ध्रुव int64_t bne01 = MIN(bi01 + blck_0, ne01);
            प्रति (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                ध्रुव int64_t i3 = ir/(ne2*ne1);
                ध्रुव int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                ध्रुव int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                ध्रुव int64_t i02 = i2 / dps2;
                ध्रुव int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                ध्रुव int64_t i12 = i2;
                ध्रुव int64_t i13 = i3;

#यदि GGML_VEC_MAD_UNROLL > 2
                ध्रुव int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                प्रति (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    ध्रुव int64_t i11 = i01;

                    भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->दत्त + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->दत्त + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->दत्त + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                प्रति (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ध्रुव int64_t i11 = i01;

                    भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->दत्त + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->दत्त + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->दत्त + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#अन्यथा
                प्रति (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ध्रुव int64_t i11 = i01;

                    भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->दत्त + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->दत्त + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->दत्त + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#यदि_अन्तः
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_बहिर्गमनम्_prod_q_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव ggml_type type = src0->type;
    ggml_गन्तव्यम्_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->गन्तव्यम्_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_आकार(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    यदि (ith == 0) {
        ggml_vec_समुच्चयः_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->दत्त, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    ध्रुव int64_t nr = ne1*ne2*ne3;

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    भिन्न * wdata = (भिन्न *) params->wdata + (ne0 + CACHE_पङ्क्तिः_आकार_F32) * ith;

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        ध्रुव int64_t i3 = ir/(ne2*ne1);
        ध्रुव int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ध्रुव int64_t i02 = i2;
        ध्रुव int64_t i03 = i3;

        //const int64_t i10 = i1;
        ध्रुव int64_t i12 = i2;
        ध्रुव int64_t i13 = i3;

        प्रति (int64_t i01 = 0; i01 < ne01; ++i01) {
            ध्रुव int64_t i11 = i01;

            भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->दत्त + (          i01*nb01 + i02*nb02 + i03*nb03));
            भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->दत्त + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->दत्त + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

स्थिर शून्य ggml_compute_forward_बहिर्गमनम्_prod_f16_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb0  == परिमाण(भिन्न));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    यदि (ith == 0) {
        ggml_vec_समुच्चयः_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->दत्त, 0);
    }
    ggml_barrier(params->threadpool);

    ध्रुव int64_t nr = ne1*ne2*ne3;
    ध्रुव int64_t dr = (nr + nth - 1)/nth;
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    भिन्न * wdata = (भिन्न *) params->wdata + (ne0 + CACHE_पङ्क्तिः_आकार_F32) * ith;

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i3 = ir/(ne2*ne1);
        ध्रुव int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ध्रुव int64_t i02 = i2;
        ध्रुव int64_t i03 = i3;

        ध्रुव int64_t i12 = i2;
        ध्रुव int64_t i13 = i3;

        भिन्न * d = (भिन्न *) ((वर्ण *) dst->दत्त + (i1*nb1 + i2*nb2 + i3*nb3));

        प्रति (int64_t i01 = 0; i01 < ne01; ++i01) {
            ध्रुव int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((वर्ण *) src0->दत्त + (i01*nb01 + i02*nb02 + i03*nb03));
            भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->दत्त + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_गन्तव्यम्_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

शून्य ggml_compute_forward_बहिर्गमनम्_prod(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_बहिर्गमनम्_prod_q_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_बहिर्गमनम्_prod_f16_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_बहिर्गमनम्_prod_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

स्थिर शून्य ggml_compute_forward_scale_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    भिन्न s; // scale factor
    भिन्न b; // bias

    स्मृतिं_प्रतिलिख(&s, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&b, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    ध्रुव size_t nb01 = src0->nb[1];

    ध्रुव size_t nb1 = dst->nb[1];

    यदि (b == 0.0f) {
        प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
            यदि (dst->दत्त != src0->दत्त) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                स्मृतिं_प्रतिलिख((वर्ण *)dst->दत्त + i1*nb1, (वर्ण *)src0->दत्त + i1*nb01, nc * परिमाण(भिन्न));
            }
            ggml_vec_scale_f32(nc, (भिन्न *) ((वर्ण *) dst->दत्त + i1*nb1), s);
        }
    } अन्यथा {
        प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (भिन्न *) ((वर्ण *) dst->दत्त  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->दत्त + i1*nb1),
                s, b);
        }
    }
}

स्थिर शून्य ggml_compute_forward_scale_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    भिन्न s; // scale factor
    भिन्न b; // bias

    स्मृतिं_प्रतिलिख(&s, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&b, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ध्रुव ggml_bf16_t * x = (ध्रुव ggml_bf16_t *) ((ध्रुव वर्ण *) src0->दत्त + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((वर्ण *)       dst->दत्त  + i1*dst->nb[1]);
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            y[k] = GGML_FP32_गन्तव्यम्_BF16(s*GGML_BF16_गन्तव्यम्_FP32(x[k]) + b);
        }
    }
}

शून्य ggml_compute_forward_scale(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

स्थिर शून्य ggml_compute_forward_समुच्चयः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t विस्थापनम्  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    यदि (!inplace) {
        यदि (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स्मृतिं_प्रतिलिख(
                ((वर्ण *)  dst->दत्त),
                ((वर्ण *) src0->दत्त),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr = ggml_nrows(src1);
    ध्रुव पूर्णाङ्क nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ध्रुव size_t nb0 = ggml_element_आकार(src0);

    ध्रुव पूर्णाङ्क im0 = (ne10 == 0 ? 0 : ne10-1);
    ध्रुव पूर्णाङ्क im1 = (ne11 == 0 ? 0 : ne11-1);
    ध्रुव पूर्णाङ्क im2 = (ne12 == 0 ? 0 : ne12-1);
    ध्रुव पूर्णाङ्क im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(विस्थापनम् + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne12*ne11);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne12*ne11)/ne11;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->दत्त + i3*nb3  + i2*nb2  + i1*nb1  + विस्थापनम्),
                (भिन्न *) ((वर्ण *) src1->दत्त + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

स्थिर शून्य ggml_compute_forward_समुच्चयः_i32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t विस्थापनम्  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    यदि (!inplace) {
        यदि (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स्मृतिं_प्रतिलिख(
                ((वर्ण *)  dst->दत्त),
                ((वर्ण *) src0->दत्त),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr = ggml_nrows(src1);
    ध्रुव पूर्णाङ्क nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    ध्रुव size_t nb0 = ggml_element_आकार(src0);

    ध्रुव पूर्णाङ्क im0 = (ne10 == 0 ? 0 : ne10-1);
    ध्रुव पूर्णाङ्क im1 = (ne11 == 0 ? 0 : ne11-1);
    ध्रुव पूर्णाङ्क im2 = (ne12 == 0 ? 0 : ne12-1);
    ध्रुव पूर्णाङ्क im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(विस्थापनम् + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == परिमाण(int32_t));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne12*ne11);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne12*ne11)/ne11;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((वर्ण *)  dst->दत्त + i3*nb3  + i2*nb2  + i1*nb1  + विस्थापनम्),
                (int32_t *) ((वर्ण *) src1->दत्त + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

शून्य ggml_compute_forward_समुच्चयः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_समुच्चयः_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_समुच्चयः_i32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q8_1:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

शून्य ggml_compute_forward_cpy(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

शून्य ggml_compute_forward_cont(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

स्थिर शून्य ggml_compute_forward_get_पङ्क्तिकाः_q(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int64_t nc = ne00;
    ध्रुव int64_t nr = ggml_nelements(src1);

    ध्रुव ggml_type type = src0->type;
    ggml_गन्तव्यम्_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->गन्तव्यम्_float;

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == ggml_type_आकार(type));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t i = ir0; i < ir1; ++i) {
        ध्रुव int64_t i12 = i/(ne11*ne10);
        ध्रुव int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ध्रुव int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->दत्त + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (ध्रुव शून्य *) ((वर्ण *) src0->दत्त + i01*nb01 + i11*nb02 + i12*nb03),
                     (भिन्न *) ((वर्ण *)  dst->दत्त + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_पङ्क्तिकाः_f16(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int64_t nc = ne00;
    ध्रुव int64_t nr = ggml_nelements(src1);

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == परिमाण(ggml_fp16_t));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t i = ir0; i < ir1; ++i) {
        ध्रुव int64_t i12 = i/(ne11*ne10);
        ध्रुव int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ध्रुव int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->दत्त + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_गन्तव्यम्_fp32(
            (ध्रुव ggml_fp16_t*) ((वर्ण *) src0->दत्त + i01*nb01 + i11*nb02 + i12*nb03),
                       (भिन्न *) ((वर्ण *)  dst->दत्त + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_पङ्क्तिकाः_bf16(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int64_t nc = ne00;
    ध्रुव int64_t nr = ggml_nelements(src1);

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == परिमाण(ggml_bf16_t));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t i = ir0; i < ir1; ++i) {
        ध्रुव int64_t i12 = i/(ne11*ne10);
        ध्रुव int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ध्रुव int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->दत्त + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_गन्तव्यम्_fp32(
            (ध्रुव ggml_bf16_t *) ((वर्ण *) src0->दत्त + i01*nb01 + i11*nb02 + i12*nb03),
                        (भिन्न *) ((वर्ण *)  dst->दत्त + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_पङ्क्तिकाः_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int64_t nc = ne00;
    ध्रुव int64_t nr = ggml_nelements(src1);

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == परिमाण(भिन्न));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t i = ir0; i < ir1; ++i) {
        ध्रुव int64_t i12 = i/(ne11*ne10);
        ध्रुव int64_t i11 = (i - i12*ne11*ne10)/ne10;
        ध्रुव int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->दत्त + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->दत्त + i10*nb1  + i11*nb2  + i12*nb3),
                (भिन्न *) ((वर्ण *) src0->दत्त + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

शून्य ggml_compute_forward_get_पङ्क्तिकाः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q8_1:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_q(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_bf16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
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

प्रतिरूपम्<प्रकारनाम src_t, प्रकारनाम idx_t>
स्थिर शून्य ggml_compute_forward_समुच्चयः_पङ्क्तिकाः_impl(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int64_t nc = ne00;
    ध्रुव int64_t nr = ne01;

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne2  == ne02);
    प्रतिज्ञा(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    प्रतिज्ञा(ne02 % ne11 == 0);
    प्रतिज्ञा(ne03 % ne12 == 0);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = std::min(ir0 + dr, nr);

    ध्रुव size_t rs = ggml_row_आकार(src0->type, nc);

    ggml_from_float_t ध्रुव from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    प्रति (int64_t i03 = 0; i03 < ne03; ++i03) {
        प्रति (int64_t i02 = 0; i02 < ne02; ++i02) {
            प्रति (int64_t i = ir0; i < ir1; ++i) {
                ध्रुव int64_t i12 = i03%ne12;
                ध्रुव int64_t i11 = i02%ne11;
                ध्रुव int64_t i10 = i;

                ध्रुव int64_t i1 = *(idx_t *) ((वर्ण *) src1->दत्त + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                यदि constexpr (std::is_same_v<src_t, भिन्न>) {
                    from_float(
                            (ध्रुव भिन्न *) ((वर्ण *) src0->दत्त +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((वर्ण *)  dst->दत्त + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } अन्यथा यदि constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    यदि (dst->type == GGML_TYPE_F16) {
                        स्मृतिं_प्रतिलिख(
                                            ((वर्ण *)  dst->दत्त + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((वर्ण *) src0->दत्त +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } अन्यथा {
                        भिन्न * wdata = (भिन्न *) params->wdata + (nc + CACHE_पङ्क्तिः_आकार_F32) * ith;
                        ggml_fp16_गन्तव्यम्_fp32_row(
                                (ध्रुव ggml_fp16_t *) ((वर्ण *) src0->दत्त + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((वर्ण *) dst->दत्त + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } अन्यथा {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_समुच्चयः_पङ्क्तिकाः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                यदि (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_समुच्चयः_पङ्क्तिकाः_impl<भिन्न, int64_t>(params, dst);
                } अन्यथा यदि (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_समुच्चयः_पङ्क्तिकाः_impl<भिन्न, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                यदि (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_समुच्चयः_पङ्क्तिकाः_impl<ggml_fp16_t, int64_t>(params, dst);
                } अन्यथा यदि (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_समुच्चयः_पङ्क्तिकाः_impl<ggml_fp16_t, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

स्थिर शून्य ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः_f32_f16(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    स्मृतिं_पूरय(dst->दत्त, 0, ggml_nbytes(dst));

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == परिमाण(ggml_fp16_t));

    प्रति (पूर्णाङ्क i = 0; i < nr; ++i) {
        ध्रुव पूर्णाङ्क r = ((int32_t *) src1->दत्त)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        प्रति (पूर्णाङ्क j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((वर्ण *) src0->दत्त + i*src0->nb[1]))[j];
            ((भिन्न *) ((वर्ण *) dst->दत्त + r*dst->nb[1]))[j] += GGML_CPU_FP16_गन्तव्यम्_FP32(v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    स्मृतिं_पूरय(dst->दत्त, 0, ggml_nbytes(dst));

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क i = 0; i < nr; ++i) {
        ध्रुव पूर्णाङ्क r = ((int32_t *) src1->दत्त)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_सङ्कलय_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->दत्त + r*dst->nb[1]),
                (भिन्न *) ((वर्ण *)  dst->दत्त + r*dst->nb[1]),
                (भिन्न *) ((वर्ण *) src0->दत्त + i*src0->nb[1]));
    }
}

शून्य ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः_f32_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_get_पङ्क्तिकाः_पृष्ठभागः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
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

स्थिर शून्य ggml_compute_forward_diag_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == परिमाण(भिन्न));
    GGML_ASSERT(nb0  == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क i3 = 0; i3 < ne3; i3++) {
        प्रति (पूर्णाङ्क i2 = 0; i2 < ne2; i2++) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne1; i1++) {
                भिन्न * d = (भिन्न *)((वर्ण *)  dst->दत्त + i3*nb3  + i2*nb2 + i1*nb1);
                भिन्न * s = (भिन्न *)((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02);
                प्रति (पूर्णाङ्क i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                प्रति (पूर्णाङ्क i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_diag(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

स्थिर शून्य ggml_compute_forward_diag_आवरणम्_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        ध्रुव भिन्न value) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क  n_past  = ((int32_t *) dst->op_params)[0];
    ध्रुव तार्किक inplace = src0->दत्त == dst->दत्त;

    GGML_ASSERT(n_past >= 0);

    यदि (!inplace) {
        यदि (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            स्मृतिं_प्रतिलिख(
                ((वर्ण *)  dst->दत्त),
                ((वर्ण *) src0->दत्त),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    ध्रुव पूर्णाङ्क n  = ggml_nrows(src0);
    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = src0->ne[1];
    ध्रुव पूर्णाङ्क nz = n/nr;

    GGML_ASSERT( dst->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क k = 0; k < nz; k++) {
        प्रति (पूर्णाङ्क j = ith; j < nr; j += nth) {
            प्रति (पूर्णाङ्क i = n_past; i < nc; i++) {
                यदि (i > n_past + j) {
                    *(भिन्न *)((वर्ण *) dst->दत्त + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_diag_आवरणम्_inf(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_आवरणम्_f32(params, dst, -INFINITY);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_compute_forward_diag_आवरणम्_zero(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_आवरणम्_f32(params, dst, 0);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

स्थिर शून्य ggml_compute_forward_soft_max_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    प्रतिज्ञा(ggml_is_contiguous(dst));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,    (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव int64_t nb11 = src1 ? src1->nb[1] : 1;
    ध्रुव int64_t nb12 = src1 ? src1->nb[2] : 1;
    ध्रुव int64_t nb13 = src1 ? src1->nb[3] : 1;

    ध्रुव int64_t ne12 = src1 ? src1->ne[2] : 1;
    ध्रुव int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    ध्रुव uint32_t n_शीर्ष      = ne02;
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floor(log2(n_शीर्ष));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    भिन्न * wp = (भिन्न *) params->wdata + (ne00 + CACHE_पङ्क्तिः_आकार_F32) * ith;

    ध्रुव तार्किक use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ध्रुव भिन्न * sk = src2 ? (भिन्न *)((वर्ण *) src2->दत्त) : शून्यसूचकः;

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव int64_t i11 = i01;
                ध्रुव int64_t i12 = i02%ne12;
                ध्रुव int64_t i13 = i03%ne13;

                // ALiBi
                ध्रुव uint32_t h = i02; //शीर्ष
                ध्रुव भिन्न नतिः = (max_bias > 0.0f) ? h < n_शीर्ष_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_शीर्ष_log2) + 1) : 1.0f;

                भिन्न * sp = (भिन्न *)((वर्ण *) src0->दत्त + i01*nb01 + i02*nb02 + i03*nb03);
                भिन्न * dp = (भिन्न *)((वर्ण *)  dst->दत्त + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((वर्ण *) src1->दत्त + i11*nb11 + i12*nb12 + i13*nb13) : रिक्त;
                भिन्न       * mp_f32 = src1 ? (भिन्न       *)((वर्ण *) src1->दत्त + i11*nb11 + i12*nb12 + i13*nb13) : रिक्त;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                यदि (mp_f32) {
                    यदि (use_f16) {
                        प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                            wp[i] += नतिः*GGML_CPU_FP16_गन्तव्यम्_FP32(mp_f16[i]);
                        }
                    } अन्यथा {
                        प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                            wp[i] += नतिः*mp_f32[i];
                        }
                    }
                }

#यदि_अपरिभाषितम् NDEBUG
                प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    प्रतिज्ञा(!isnan(wp[i]));
                }
#यदि_अन्तः // NDEBUG

                भिन्न max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                यदि (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float योग_2 = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                प्रतिज्ञा(योग_2 > 0.0);

                यदि (sk) {
                    योग_2 += (ggml_float) expf(sk[i02] - max);
                }

                योग_2 = 1.0/योग_2;
                ggml_vec_scale_f32(ne00, dp, योग_2);

#यदि_अपरिभाषितम् NDEBUG
                प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                    प्रतिज्ञा(!isnan(dp[i]));
                    प्रतिज्ञा(!isinf(dp[i]));
                }
#यदि_अन्तः // NDEBUG
            }
        }
    }
}

शून्य ggml_compute_forward_soft_max(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

स्थिर शून्य ggml_compute_forward_soft_max_ext_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,    (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न *dy = (भिन्न *)((वर्ण *) src0->दत्त + i1*src0->nb[1]);
        भिन्न *y  = (भिन्न *)((वर्ण *) src1->दत्त + i1*src1->nb[1]);
        भिन्न *dx = (भिन्न *)((वर्ण *) dst->दत्त  + i1*dst->nb[1]);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(dy[i]));
            प्रतिज्ञा(!isnan(y[i]));
        }
#यदि_अन्तः // NDEBUG
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
        भिन्न dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        प्रति (पूर्णाङ्क i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#यदि_अपरिभाषितम् NDEBUG
        प्रति (पूर्णाङ्क i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(dx[i]));
            प्रतिज्ञा(!isinf(dx[i]));
        }
#यदि_अन्तः // NDEBUG
    }
}

शून्य ggml_compute_forward_soft_max_ext_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_पृष्ठभागः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

स्थिर शून्य ggml_compute_forward_clamp_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    भिन्न min;
    भिन्न max;
    स्मृतिं_प्रतिलिख(&min, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क n  = ggml_nrows(src0);
    ध्रुव पूर्णाङ्क nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क j = ith; j < n; j += nth) {
        ध्रुव int64_t i1 = j % ne01;
        ध्रुव int64_t i2 = (j / ne01) % ne02;
        ध्रुव int64_t i3 = j / (ne01*ne02);

        भिन्न * dst_ptr  = (भिन्न *) ((वर्ण *)  dst->दत्त + i1*nb1  + i2*nb2  + i3*nb3);
        भिन्न * src0_ptr = (भिन्न *) ((वर्ण *) src0->दत्त + i1*nb01 + i2*nb02 + i3*nb03);

        प्रति (पूर्णाङ्क i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

स्थिर शून्य ggml_compute_forward_clamp_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    भिन्न min;
    भिन्न max;
    स्मृतिं_प्रतिलिख(&min, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क n  = ggml_nrows(src0);
    ध्रुव पूर्णाङ्क nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb00 == परिमाण(ggml_fp16_t));

    प्रति (पूर्णाङ्क j = ith; j < n; j += nth) {
        ध्रुव int64_t i1 = j % ne01;
        ध्रुव int64_t i2 = (j / ne01) % ne02;
        ध्रुव int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *)  dst->दत्त + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((वर्ण *) src0->दत्त + i1*nb01 + i2*nb02 + i3*nb03);

        प्रति (पूर्णाङ्क i = 0; i < nc; i++) {
            भिन्न v = GGML_CPU_FP16_गन्तव्यम्_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_गन्तव्यम्_FP16(MAX(MIN(v, max), min));
        }
    }
}

शून्य ggml_compute_forward_clamp(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q2_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q8_1:
        पक्ष GGML_TYPE_MXFP4:
        पक्ष GGML_TYPE_NVFP4:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_TQ1_0:
        पक्ष GGML_TYPE_TQ2_0:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ4_NL:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ3_S:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_Q8_K:
        पक्ष GGML_TYPE_I8:
        पक्ष GGML_TYPE_I16:
        पक्ष GGML_TYPE_I32:
        पक्ष GGML_TYPE_I64:
        पक्ष GGML_TYPE_F64:
        पक्ष GGML_TYPE_गणनम्:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

स्थिर भिन्न rope_yarn_ramp(ध्रुव भिन्न नीचः, ध्रुव भिन्न उच्चः, ध्रुव पूर्णाङ्क i0) {
    ध्रुव भिन्न y = (i0 / 2 - नीचः) / MAX(0.001f, उच्चः - नीचः);
    प्रत्यागम 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
स्थिर शून्य rope_yarn(
    भिन्न theta_extrap, भिन्न freq_scale, भिन्न corr_dims[2], int64_t i0, भिन्न ext_गुणकः, भिन्न mscale,
    भिन्न * cos_theta, भिन्न * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    भिन्न theta_interp = freq_scale * theta_extrap;
    भिन्न theta = theta_interp;
    यदि (ext_गुणकः != 0.0f) {
        भिन्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_गुणकः;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

स्थिर शून्य ggml_rope_cache_init(
     भिन्न theta_आधारः, भिन्न freq_scale, ध्रुव भिन्न * freq_factors, भिन्न corr_dims[2], int64_t ne0, भिन्न ext_गुणकः, भिन्न mscale,
     भिन्न * cache, भिन्न sin_sign, भिन्न theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    भिन्न theta = theta_आधारः;
    प्रति (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ध्रुव भिन्न ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

स्थिर शून्य ggml_mrope_cache_init(
     भिन्न theta_आधारः_t, भिन्न theta_आधारः_h, भिन्न theta_आधारः_w, भिन्न theta_आधारः_e, पूर्णाङ्क sections[4], तार्किक is_imrope, तार्किक indep_sects,
     भिन्न freq_scale, ध्रुव भिन्न * freq_factors, भिन्न corr_dims[2], int64_t ne0, भिन्न ext_गुणकः, भिन्न mscale,
     भिन्न * cache, भिन्न sin_sign, भिन्न theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    भिन्न theta_t = theta_आधारः_t;
    भिन्न theta_h = theta_आधारः_h;
    भिन्न theta_w = theta_आधारः_w;
    भिन्न theta_e = theta_आधारः_e;  // extra position id for vision encoder
    पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    पूर्णाङ्क sec_w = sections[1] + sections[0];
    पूर्णाङ्क sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    प्रति (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ध्रुव भिन्न ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        पूर्णाङ्क sector = (i0 / 2) % sect_dims;
        यदि (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            यदि (sector == 0) {
                theta_t = theta_आधारः_t;
            }
            अन्यथा यदि (sector == sections[0]) {
                theta_h = theta_आधारः_h;;
            }
            अन्यथा यदि (sector == sec_w) {
                theta_w = theta_आधारः_w;
            }
            अन्यथा यदि (sector == sec_e) {
                theta_e = theta_आधारः_e;
            }
        }

        भिन्न theta = theta_t;
        यदि (is_imrope) { // qwen3vl apply interleaved mrope
            यदि (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } अन्यथा {
                theta = theta_e;
            }
        } अन्यथा {
            यदि (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            अन्यथा यदि (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            अन्यथा यदि (sector >= sec_w + sections[2]) {
                theta = theta_e;
            }
        }

        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


प्रतिरूपम्<प्रकारनाम T>
स्थिर शून्य rotate_युग्मानि(ध्रुव int64_t n, ध्रुव int64_t n_विस्थापनम्, ध्रुव भिन्न * cache, ध्रुव T * src_दत्त, T * dst_दत्त, ध्रुव पूर्णाङ्क scale = 2) {
  प्रति (int64_t i0 = 0; i0 < n; i0 += 2) {
    ध्रुव int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ध्रुव भिन्न cos_theta = cache[i0 + 0];
    ध्रुव भिन्न sin_theta = cache[i0 + 1];

    ध्रुव T * ध्रुव src = src_दत्त + ic;
    T * dst             = dst_दत्त + ic;

    ध्रुव भिन्न x0 = type_conversion_सारणी<T>::गन्तव्यम्_f32(src[0]);
    ध्रुव भिन्न x1 = type_conversion_सारणी<T>::गन्तव्यम्_f32(src[n_विस्थापनम्]);

    dst[0]        = type_conversion_सारणी<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_विस्थापनम्] = type_conversion_सारणी<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

प्रतिरूपम्<प्रकारनाम T> //float or ggml_fp16_t
स्थिर शून्य ggml_compute_forward_rope_flt(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        ध्रुव तार्किक forward) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    भिन्न freq_आधारः, freq_scale, ext_गुणकः, attn_गुणकः, beta_fast, beta_slow;
    पूर्णाङ्क sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ध्रुव पूर्णाङ्क n_dims     = ((int32_t *) dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ध्रुव पूर्णाङ्क n_ctx_orig = ((int32_t *) dst->op_params)[4];

    स्मृतिं_प्रतिलिख(&freq_आधारः,   (int32_t *) dst->op_params +  5, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&freq_scale,  (int32_t *) dst->op_params +  6, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&ext_गुणकः,  (int32_t *) dst->op_params +  7, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&attn_गुणकः, (int32_t *) dst->op_params +  8, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_fast,   (int32_t *) dst->op_params +  9, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_slow,   (int32_t *) dst->op_params + 10, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&sections,    (int32_t *) dst->op_params + 11, परिमाण(पूर्णाङ्क)*4);

    ध्रुव पूर्णाङ्क n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == परिमाण(T));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    पूर्णाङ्क ir = 0;

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f/n_dims);

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast, beta_slow, corr_dims);

    ध्रुव तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ध्रुव तार्किक mrope_प्रयुक्तम् = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ध्रुव तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

    यदि (mrope_प्रयुक्तम्) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    यदि (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    ध्रुव भिन्न * freq_factors = रिक्त;
    यदि (src2 != रिक्त) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (ध्रुव भिन्न *) src2->दत्त;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ध्रुव भिन्न sin_sign = forward ? 1.0f : -1.0f;

    ध्रुव int32_t * pos = (ध्रुव int32_t *) src1->दत्त;

    int64_t चरमः_i2 = -1;

    प्रति (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        प्रति (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            प्रति (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                यदि (ir++ < ir0) अनुवर्त; // skip rows mapped to other threads
                यदि (ir   > ir1) भङ्ग;

                भिन्न * cache = (भिन्न *) params->wdata + (ne0 + CACHE_पङ्क्तिः_आकार_F32)*ith;
                यदि (चरमः_i2 != i2) {
                    यदि (!mrope_प्रयुक्तम्) {
                        ध्रुव int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_गुणकः, attn_गुणकः, cache, sin_sign, theta_scale);
                    }
                    अन्यथा {
                        ध्रुव int64_t p_t = pos[i2];
                        ध्रुव int64_t p_h = pos[i2 + ne2];
                        ध्रुव int64_t p_w = pos[i2 + ne2 * 2];
                        ध्रुव int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_गुणकः, attn_गुणकः, cache, sin_sign, theta_scale);
                    }

                    चरमः_i2 = i2;
                }

                T * src = (T *)((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_दत्त  = (T *)((वर्ण *)  dst->दत्त + i3*nb3  + i2*nb2  + i1*nb1);

                विकल्प (mode) {
                    पक्ष GGML_ROPE_TYPE_प्रसामान्यम्:
                        rotate_युग्मानि<T>(n_dims, 1, cache, src + n_offs, dst_दत्त + n_offs, 1);
                        भङ्ग;
                    पक्ष GGML_ROPE_TYPE_NEOX:
                    पक्ष GGML_ROPE_TYPE_MROPE:
                    पक्ष GGML_ROPE_TYPE_IMROPE:
                        rotate_युग्मानि<T>(n_dims, n_dims/2, cache, src + n_offs, dst_दत्त + n_offs);
                        भङ्ग;
                    पक्ष GGML_ROPE_TYPE_VISION:
                        rotate_युग्मानि<T>(ne0, n_dims, cache, src, dst_दत्त);
                        भङ्ग;
                    स्वभाव:
                        GGML_ABORT("rope type not supported");
                }

                यदि (!is_vision) {
                    // fill the remain channels with data from src tensor
                    प्रति (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        यदि (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            अनुवर्त;
                        }
                        ध्रुव T * ध्रुव src = (T *)((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_दत्त  = (T *)((वर्ण *)  dst->दत्त + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_दत्त[0] = src[0];
                        dst_दत्त[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

शून्य ggml_compute_forward_rope(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, सत्य);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<भिन्न>(params, dst, सत्य);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

शून्य ggml_compute_forward_rope_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, असत्य);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<भिन्न>(params, dst, असत्य);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

स्थिर शून्य ggml_compute_forward_conv_transpose_1d_f16_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == परिमाण(ggml_fp16_t));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    यदि (ith == 0) {
        स्मृतिं_पूरय(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) params->wdata + 0;

            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव ggml_fp16_t * ध्रुव src = (ggml_fp16_t *)((वर्ण *) src0->दत्त + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_दत्त = wdata + i01*ne00*ne02;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_दत्त[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_दत्त = wdata;

            प्रति (int64_t i11 = 0; i11 < ne11; i11++) {
                ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src1->दत्त + i11*nb11);
                प्रति (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_दत्त[i10*ne11 + i11] = GGML_CPU_FP32_गन्तव्यम्_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        स्मृतिं_पूरय(dst->दत्त, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ध्रुव int32_t s0 = ((ध्रुव int32_t*)(dst->op_params))[0];

    // total rows in dst
    ध्रुव पूर्णाङ्क nr = ne1;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * ध्रुव wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * ध्रुव wdata_src = wdata + nk;

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * dst_दत्त = (भिन्न *)((वर्ण *) dst->दत्त + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
            ध्रुव पूर्णाङ्क i1n = i10*ne11;
            प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                भिन्न v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_दत्त[i10*s0 + i00] += v;
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_conv_transpose_1d_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == परिमाण(भिन्न));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    यदि (ith == 0) {
        स्मृतिं_पूरय(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            भिन्न * ध्रुव wdata = (भिन्न *) params->wdata + 0;

            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src0->दत्त + i02*nb02 + i01*nb01);
                    भिन्न * dst_दत्त = wdata + i01*ne00*ne02;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_दत्त[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            भिन्न * ध्रुव wdata = (भिन्न *) params->wdata + nk;
            भिन्न * dst_दत्त = wdata;

            प्रति (int64_t i11 = 0; i11 < ne11; i11++) {
                ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src1->दत्त + i11*nb11);
                प्रति (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_दत्त[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        स्मृतिं_पूरय(dst->दत्त, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ध्रुव int32_t s0 = ((ध्रुव int32_t*)(dst->op_params))[0];

    // total rows in dst
    ध्रुव पूर्णाङ्क nr = ne1;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    भिन्न * ध्रुव wdata     = (भिन्न *) params->wdata + 0;
    भिन्न * ध्रुव wdata_src = wdata + nk;

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * dst_दत्त = (भिन्न *)((वर्ण *) dst->दत्त + i1*nb1);
        भिन्न * wdata_kernel = wdata + i1*ne02*ne00;
        प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
            ध्रुव पूर्णाङ्क i1n = i10*ne11;
            प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                भिन्न v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_दत्त[i10*s0 + i00] += v;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_transpose_1d(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
स्थिर शून्य ggml_compute_forward_im2col_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t s1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t p1 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t d0 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t d1 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव तार्किक is_2D = ((ध्रुव int32_t *)(dst->op_params))[6] == 1;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t N  = is_2D ? ne13 : ne12;
    ध्रुव int64_t IC = is_2D ? ne12 : ne11;
    ध्रुव int64_t IH = is_2D ? ne11 : 1;
    ध्रुव int64_t IW = ne10;

    ध्रुव int64_t KH = is_2D ? ne01 : 1;
    ध्रुव int64_t KW = ne00;

    ध्रुव int64_t OH = is_2D ? ne2 : 1;
    ध्रुव int64_t OW = ne1;

    पूर्णाङ्क ofs0 = is_2D ? nb13 : nb12;
    पूर्णाङ्क ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        भिन्न * ध्रुव wdata = (भिन्न *) dst->दत्त;

        प्रति (int64_t अन्तर् = 0; अन्तर् < N; अन्तर्++) {
            प्रति (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                प्रति (int64_t iow = 0; iow < OW; iow++) {
                    प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        भिन्न * dst_दत्त = wdata + (अन्तर्*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ध्रुव भिन्न * ध्रुव src_दत्त = (भिन्न *)((वर्ण *) src1->दत्त + अन्तर्*ofs0 + iic*ofs1); // [IH, IW]

                        प्रति (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;

                                यदि (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_दत्त[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    dst_दत्त[iic*(KH*KW) + ikh*KW + ikw] = (src_दत्त[iih*IW + iiw]);
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
स्थिर शून्य ggml_compute_forward_im2col_f16(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t s1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t p1 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t d0 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t d1 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव तार्किक is_2D = ((ध्रुव int32_t *)(dst->op_params))[6] == 1;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t N  = is_2D ? ne13 : ne12;
    ध्रुव int64_t IC = is_2D ? ne12 : ne11;
    ध्रुव int64_t IH = is_2D ? ne11 : 1;
    ध्रुव int64_t IW = ne10;

    ध्रुव int64_t KH = is_2D ? ne01 : 1;
    ध्रुव int64_t KW = ne00;

    ध्रुव int64_t OH = is_2D ? ne2 : 1;
    ध्रुव int64_t OW = ne1;

    पूर्णाङ्क ofs0 = is_2D ? nb13 : nb12;
    पूर्णाङ्क ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) dst->दत्त;

        प्रति (int64_t अन्तर् = 0; अन्तर् < N; अन्तर्++) {
            प्रति (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                प्रति (int64_t iow = 0; iow < OW; iow++) {
                    प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_दत्त = wdata + (अन्तर्*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ध्रुव भिन्न * ध्रुव src_दत्त_f32 = src1->type == GGML_TYPE_F32
                            ? (ध्रुव भिन्न *)((ध्रुव वर्ण *) src1->दत्त + अन्तर्*ofs0 + iic*ofs1)
                            : शून्यसूचकः; // [IH, IW]
                        ध्रुव ggml_fp16_t * ध्रुव src_दत्त_f16 = src1->type == GGML_TYPE_F16
                            ? (ध्रुव ggml_fp16_t *)((ध्रुव वर्ण *) src1->दत्त + अन्तर्*ofs0 + iic*ofs1)
                            : शून्यसूचकः; // [IH, IW]

                        प्रति (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;

                                यदि (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_दत्त[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    यदि (src_दत्त_f32 != शून्यसूचकः) {
                                        dst_दत्त[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_गन्तव्यम्_FP16(src_दत्त_f32[iih*IW + iiw]);
                                    } अन्यथा {
                                        dst_दत्त[iic*(KH*KW) + ikh*KW + ikw] = src_दत्त_f16[iih*IW + iiw];
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

शून्य ggml_compute_forward_im2col(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

शून्य ggml_compute_forward_im2col_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    ध्रुव ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t s1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t p1 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t d0 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t d1 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव तार्किक is_2D = ((ध्रुव int32_t *)(dst->op_params))[6] == 1;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t N  = is_2D ? ne3 : ne2;
    ध्रुव int64_t IC = is_2D ? ne2 : ne1;
    ध्रुव int64_t IH = is_2D ? ne1 : 1;
    ध्रुव int64_t IW = ne0;

    ध्रुव int64_t KH = is_2D ? ne11 : 1;
    ध्रुव int64_t KW = ne10;

    ध्रुव int64_t OH = is_2D ? ne02 : 1;
    ध्रुव int64_t OW = ne01;

    पूर्णाङ्क ofs0 = is_2D ? nb3 : nb2;
    पूर्णाङ्क ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == परिमाण(भिन्न));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        भिन्न * ध्रुव wdata = (भिन्न *) dst->दत्त;

        प्रति (int64_t अन्तर् = 0; अन्तर् < N; अन्तर्++) {
            प्रति (int64_t iic = ith; iic < IC; iic += nth) {
                प्रति (int64_t iih = 0; iih < IH; iih++) {
                    प्रति (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        भिन्न grad = 0.0f;
                        प्रति (int64_t ikh = 0; ikh < KH; ikh++) {
                            प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                ध्रुव int64_t tmpw = (iiw + p0 - ikw*d0);
                                यदि (tmpw % s0 != 0) {
                                    अनुवर्त;
                                }
                                ध्रुव int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                यदि (is_2D) {
                                    ध्रुव int64_t tmph = iih + p1 - ikh*d1;

                                    यदि (tmph % s1 != 0) {
                                        अनुवर्त;
                                    }

                                    ioh = tmph / s1;
                                } अन्यथा {
                                    ioh = 0;
                                }

                                यदि (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    अनुवर्त;
                                }

                                ध्रुव भिन्न * ध्रुव grad_अन्तर् = (ध्रुव भिन्न *) src0->दत्त
                                    + (अन्तर्*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_अन्तर्[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        भिन्न * dst_दत्त = (भिन्न *)((वर्ण *) wdata + (अन्तर्*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_दत्त[iih*IW + iiw] = grad;
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
स्थिर शून्य ggml_compute_forward_im2col_3d_f16(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t s1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t s2 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t p1 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t p2 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव int32_t d0 = ((ध्रुव int32_t *)(dst->op_params))[6];
    ध्रुव int32_t d1 = ((ध्रुव int32_t *)(dst->op_params))[7];
    ध्रुव int32_t d2 = ((ध्रुव int32_t *)(dst->op_params))[8];
    ध्रुव int32_t IC = ((ध्रुव int32_t *)(dst->op_params))[9];


    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t N  = ne13 / IC;
    ध्रुव int64_t ID = ne12;
    ध्रुव int64_t IH = ne11;
    ध्रुव int64_t IW = ne10;

    ध्रुव int64_t OC = ne03 / IC;
    GGML_अप्रयुक्तम्(OC);
    ध्रुव int64_t KD = ne02;
    ध्रुव int64_t KH = ne01;
    ध्रुव int64_t KW = ne00;

    ध्रुव int64_t OD = ne3 / N;
    ध्रुव int64_t OH = ne2;
    ध्रुव int64_t OW = ne1;
    ध्रुव int64_t OH_OW = OH*OW;
    ध्रुव int64_t KD_KH_KW = KD*KH*KW;
    ध्रुव int64_t KH_KW = KH*KW;
    ध्रुव int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) dst->दत्त;

        प्रति (int64_t अन्तर् = 0; अन्तर् < N; अन्तर्++) {
            प्रति (int64_t iod = 0; iod < OD; iod++) {
                प्रति (int64_t ioh = 0; ioh < OH; ioh++) {
                    प्रति (int64_t iow = 0; iow < OW; iow++) {
                        प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_दत्त = wdata + (अन्तर्*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ध्रुव भिन्न * ध्रुव src_दत्त = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src1->दत्त + (अन्तर्*IC + iic)*nb13); // [ID, IH, IW]

                            प्रति (int64_t ikd = 0; ikd < KD; ikd++) {
                                प्रति (int64_t ikh = 0; ikh < KH; ikh++) {
                                    प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ध्रुव int64_t iid = iod*s2 + ikd*d2 - p2;

                                        यदि (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_दत्त[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            ध्रुव भिन्न * ध्रुव s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src_दत्त + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_दत्त[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_गन्तव्यम्_FP16(*s);
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
स्थिर शून्य ggml_compute_forward_im2col_3d_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t s1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t s2 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t p1 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t p2 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव int32_t d0 = ((ध्रुव int32_t *)(dst->op_params))[6];
    ध्रुव int32_t d1 = ((ध्रुव int32_t *)(dst->op_params))[7];
    ध्रुव int32_t d2 = ((ध्रुव int32_t *)(dst->op_params))[8];
    ध्रुव int32_t IC = ((ध्रुव int32_t *)(dst->op_params))[9];


    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t N  = ne13 / IC;
    ध्रुव int64_t ID = ne12;
    ध्रुव int64_t IH = ne11;
    ध्रुव int64_t IW = ne10;

    ध्रुव int64_t OC = ne03 / IC;
    GGML_अप्रयुक्तम्(OC);
    ध्रुव int64_t KD = ne02;
    ध्रुव int64_t KH = ne01;
    ध्रुव int64_t KW = ne00;

    ध्रुव int64_t OD = ne3 / N;
    ध्रुव int64_t OH = ne2;
    ध्रुव int64_t OW = ne1;

    ध्रुव int64_t OH_OW = OH*OW;
    ध्रुव int64_t KD_KH_KW = KD*KH*KW;
    ध्रुव int64_t KH_KW = KH*KW;
    ध्रुव int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        भिन्न * ध्रुव wdata = (भिन्न *) dst->दत्त;

        प्रति (int64_t अन्तर् = 0; अन्तर् < N; अन्तर्++) {
            प्रति (int64_t iod = 0; iod < OD; iod++) {
                प्रति (int64_t ioh = 0; ioh < OH; ioh++) {
                    प्रति (int64_t iow = 0; iow < OW; iow++) {
                        प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            भिन्न * dst_दत्त = wdata + (अन्तर्*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ध्रुव भिन्न * ध्रुव src_दत्त = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src1->दत्त + (अन्तर्*IC + iic)*nb13); // [ID, IH, IW]

                            प्रति (int64_t ikd = 0; ikd < KD; ikd++) {
                                प्रति (int64_t ikh = 0; ikh < KH; ikh++) {
                                    प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ध्रुव int64_t iid = iod*s2 + ikd*d2 - p2;

                                        यदि (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_दत्त[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            ध्रुव भिन्न * ध्रुव s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src_दत्त + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_दत्त[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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


शून्य ggml_compute_forward_im2col_3d(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_call_mul_mat(ggml_type type, ध्रुव ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              शून्य * a, शून्य * b, भिन्न * c) {
    ध्रुव ggml_type_traits * traits = ggml_get_type_traits(type);
    संरचना ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_आकार;
    src1.nb[1] = k * traits->type_आकार;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.दत्त  = a;

    संरचना ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_आकार;
    src0.nb[1] = k * traits->type_आकार;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.दत्त  = b;

    संरचना ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = परिमाण(भिन्न);
    dst.nb[1] = n * परिमाण(भिन्न);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.दत्त  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

स्थिर अन्तःस्थ int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    प्रत्यागम (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

प्रतिरूपम् <प्रकारनाम elem_t>
स्थिर शून्य ggml_compute_forward_col2im_1d_impl(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ध्रुव int32_t s0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t OC = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t p0 = ((ध्रुव int32_t *)(dst->op_params))[2];

    ध्रुव int64_t K_OC = src->ne[0];
    ध्रुव int64_t T_अन्तर् = src->ne[1];
    ध्रुव int64_t K    = K_OC / OC;
    ध्रुव int64_t T_बहिर्गमनम् = dst->ne[0];

    ध्रुव elem_t * col_दत्त = (ध्रुव elem_t *) src->दत्त;
    elem_t       * dst_दत्त = (elem_t *) dst->दत्त;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ध्रुव int64_t dr = (T_बहिर्गमनम् + nth - 1) / nth;
    ध्रुव int64_t it0 = dr * ith;
    ध्रुव int64_t it1 = it0 + dr < T_बहिर्गमनम् ? it0 + dr : T_बहिर्गमनम्;

    प्रति (int64_t oc = 0; oc < OC; oc++) {
        प्रति (int64_t t_बहिर्गमनम् = it0; t_बहिर्गमनम् < it1; t_बहिर्गमनम्++) {
            ध्रुव int64_t t_abs = t_बहिर्गमनम् + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_अन्तर्_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            यदि (t_अन्तर्_min < 0) t_अन्तर्_min = 0;
            int64_t t_अन्तर्_max = t_abs / s0;
            यदि (t_अन्तर्_max >= T_अन्तर्) t_अन्तर्_max = T_अन्तर् - 1;

            भिन्न योग_2 = 0.0f;
            प्रति (int64_t t_अन्तर् = t_अन्तर्_min; t_अन्तर् <= t_अन्तर्_max; t_अन्तर्++) {
                int64_t k = t_abs - t_अन्तर् * s0;
                यदि (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    योग_2 += type_conversion_सारणी<elem_t>::गन्तव्यम्_f32(col_दत्त[(oc * K + k) + t_अन्तर् * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_दत्त[t_बहिर्गमनम् + oc * T_बहिर्गमनम्] = type_conversion_सारणी<elem_t>::from_f32(योग_2);
        }
    }
}

शून्य ggml_compute_forward_col2im_1d(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    विकल्प (dst->src[0]->type) {
        पक्ष GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<भिन्न>      (params, dst); भङ्ग;
        पक्ष GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); भङ्ग;
        पक्ष GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); भङ्ग;
        स्वभाव: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


स्थिर शून्य ggml_compute_forward_conv_2d_impl(ध्रुव ggml_compute_params * params,
                                              ध्रुव ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              ध्रुव ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ध्रुव ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ध्रुव int32_t stride_x   = dst->op_params[0];
    ध्रुव int32_t stride_y   = dst->op_params[1];
    ध्रुव int32_t pad_x      = dst->op_params[2];
    ध्रुव int32_t pad_y      = dst->op_params[3];
    ध्रुव int32_t dilation_x = dst->op_params[4];
    ध्रुव int32_t dilation_y = dst->op_params[5];

    ध्रुव int64_t c_अन्तर्  = src->ne[2];
    ध्रुव int64_t c_बहिर्गमनम् = kernel->ne[3];
    GGML_ASSERT(c_अन्तर् == kernel->ne[2]);

    ध्रुव int64_t src_w = src->ne[0];
    ध्रुव int64_t src_h = src->ne[1];
    ध्रुव int64_t knl_w = kernel->ne[0];
    ध्रुव int64_t knl_h = kernel->ne[1];
    ध्रुव int64_t dst_w = dst->ne[0];
    ध्रुव int64_t dst_h = dst->ne[1];

    ध्रुव भिन्न * src_दत्त = (भिन्न *) src->दत्त;
    शून्य  * knl_दत्त       = kernel->दत्त;
    भिन्न * dst_दत्त       = (भिन्न *) dst->दत्त;

    ध्रुव int64_t knl_n           = knl_w * knl_h * c_अन्तर्;
    ध्रुव int64_t patch_योग     = dst->ne[3] * dst_w * dst_h;

    ध्रुव int64_t space_प्रत्येकम्_patch   = knl_n * traits->type_आकार + c_बहिर्गमनम् * परिमाण(भिन्न);
    ध्रुव int64_t batch_आकार        = params->wsize / space_प्रत्येकम्_patch;
    ध्रुव int64_t patches_प्रत्येकम्_batch = batch_आकार > 8 ? (batch_आकार / 8) * 8 : batch_आकार;
    ध्रुव int64_t batch_n           = (patch_योग + patches_प्रत्येकम्_batch - 1) / patches_प्रत्येकम्_batch;

    GGML_ASSERT(patches_प्रत्येकम्_batch > 0 && batch_आकार >= 1);

    शून्य * tmp = params->wdata;

    प्रति (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ध्रुव int64_t patch_start_batch = batch_i * patches_प्रत्येकम्_batch;
        ध्रुव int64_t patch_अन्तः_batch   = std::min(patch_start_batch + patches_प्रत्येकम्_batch,
                                              patch_योग);
        ध्रुव int64_t patch_n           = patch_अन्तः_batch - patch_start_batch;

        ध्रुव int64_t patch_प्रत्येकम्_thread  = (patch_n + params->nth - 1) / params->nth;
        ध्रुव int64_t patch_start       = patch_start_batch + params->ith * patch_प्रत्येकम्_thread;
        ध्रुव int64_t patch_अन्तः         = std::min(patch_start + patch_प्रत्येकम्_thread, patch_अन्तः_batch);

        //im2col for a patch
        प्रति (int64_t p = patch_start; p < patch_अन्तः; ++p) {
            ध्रुव int64_t  batch_n     =  p / (dst_w * dst_h);
            ध्रुव int64_t  src_x       = (p / dst_w) % dst_h;
            ध्रुव int64_t  src_y       =  p % dst_w;

            ध्रुव भिन्न * src_आधारः = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_दत्त + batch_n * src->nb[3]);
            वर्ण *        dst_row  = (वर्ण *) tmp + (p % patches_प्रत्येकम्_batch) * knl_n * traits->type_आकार;

            प्रति (int64_t ic = 0; ic < c_अन्तर्; ++ic) {
                प्रति (int64_t ky = 0; ky < knl_h; ++ky) {
                    प्रति (int64_t kx = 0; kx < knl_w; ++kx) {
                        ध्रुव int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ध्रुव int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        भिन्न src_val;
                        यदि (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } अन्यथा {
                            ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_आधारः + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        वर्ण * element_ptr = dst_row + dst_idx * traits->type_आकार;
                        यदि (kernel_type == GGML_TYPE_F32) {
                            *(भिन्न *) element_ptr = src_val;
                        } अन्यथा यदि (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_गन्तव्यम्_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        भिन्न * gemm_output = (भिन्न *) ((वर्ण *) tmp + patches_प्रत्येकम्_batch * knl_n * traits->type_आकार);

        GGML_ASSERT(gemm_output + patch_n * c_बहिर्गमनम् <= (भिन्न*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_बहिर्गमनम्, knl_n, tmp, knl_दत्त, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ध्रुव int64_t permute_प्रत्येकम्_thread = (patch_n + params->nth - 1) / params->nth;
        ध्रुव int64_t permute_start = params->ith * permute_प्रत्येकम्_thread;
        ध्रुव int64_t permute_अन्तः = std::min(permute_start + permute_प्रत्येकम्_thread, patch_n);

        प्रति (int64_t i = permute_start; i < permute_अन्तः; ++i) {
            ध्रुव int64_t p       = patch_start_batch + i;
            ध्रुव int64_t batch_n = p / (dst_w * dst_h);
            ध्रुव int64_t dst_y   = (p / dst_w) % dst_h;
            ध्रुव int64_t dst_x   = p % dst_w;

            प्रति (int64_t oc = 0; oc < c_बहिर्गमनम्; ++oc) {
                ध्रुव भिन्न value = gemm_output[i * c_बहिर्गमनम् + oc];
                भिन्न * dst_ptr = (भिन्न *)((वर्ण *)dst_दत्त + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_2d(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

स्थिर शून्य ggml_compute_forward_conv_3d_impl(ध्रुव ggml_compute_params * params,
                                              ध्रुव ggml_tensor *         kernel,
                                              ध्रुव ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    ध्रुव ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    ध्रुव int32_t s0 = dst->op_params[0];
    ध्रुव int32_t s1 = dst->op_params[1];
    ध्रुव int32_t s2 = dst->op_params[2];
    ध्रुव int32_t p0 = dst->op_params[3];
    ध्रुव int32_t p1 = dst->op_params[4];
    ध्रुव int32_t p2 = dst->op_params[5];
    ध्रुव int32_t d0 = dst->op_params[6];
    ध्रुव int32_t d1 = dst->op_params[7];
    ध्रुव int32_t d2 = dst->op_params[8];
    ध्रुव int32_t c  = dst->op_params[9];
    ध्रुव int32_t n  = dst->op_params[10];
    ध्रुव int32_t oc = dst->op_params[11];

    ध्रुव int64_t src_w = src->ne[0];
    ध्रुव int64_t src_h = src->ne[1];
    ध्रुव int64_t src_d = src->ne[2];
    ध्रुव int64_t knl_w = kernel->ne[0];
    ध्रुव int64_t knl_h = kernel->ne[1];
    ध्रुव int64_t knl_d = kernel->ne[2];
    ध्रुव int64_t dst_w = dst->ne[0];
    ध्रुव int64_t dst_h = dst->ne[1];
    ध्रुव int64_t dst_d = dst->ne[2];

    ध्रुव भिन्न * src_दत्त = (भिन्न *) src->दत्त;
    शून्य  * knl_दत्त       = kernel->दत्त;
    भिन्न * dst_दत्त       = (भिन्न *) dst->दत्त;

    ध्रुव int64_t knl_n_प्रत्येकम्_channel = knl_w * knl_h * knl_d;
    ध्रुव int64_t knl_n_योग       = knl_n_प्रत्येकम्_channel * c;
    ध्रुव int64_t patch_योग       = n * dst_w * dst_h * dst_d;

    ध्रुव int64_t space_प्रत्येकम्_patch   = knl_n_योग * traits->type_आकार + oc * परिमाण(भिन्न);
    ध्रुव int64_t batch_आकार        = params->wsize / space_प्रत्येकम्_patch;
    ध्रुव int64_t patches_प्रत्येकम्_batch = batch_आकार > 8 ? (batch_आकार / 8) * 8 : batch_आकार;
    ध्रुव int64_t batch_n           = (patch_योग + patches_प्रत्येकम्_batch - 1) / patches_प्रत्येकम्_batch;

    GGML_ASSERT(patches_प्रत्येकम्_batch > 0 && batch_आकार >= 1);

    शून्य * tmp = params->wdata;

    प्रति (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ध्रुव int64_t patch_start_batch = batch_i * patches_प्रत्येकम्_batch;
        ध्रुव int64_t patch_अन्तः_batch   = std::min(patch_start_batch + patches_प्रत्येकम्_batch, patch_योग);
        ध्रुव int64_t patch_n_अन्तर्_batch  = patch_अन्तः_batch - patch_start_batch;

        ध्रुव int64_t patch_प्रत्येकम्_thread  = (patch_n_अन्तर्_batch + params->nth - 1) / params->nth;
        ध्रुव int64_t patch_start       = patch_start_batch + params->ith * patch_प्रत्येकम्_thread;
        ध्रुव int64_t patch_अन्तः         = std::min(patch_start + patch_प्रत्येकम्_thread, patch_अन्तः_batch);

        प्रति (int64_t p = patch_start; p < patch_अन्तः; ++p) {
            ध्रुव int64_t p_अन्तर्_batch = p % (dst_w * dst_h * dst_d);
            ध्रुव int64_t p_अन्तर्_गभीरता = p_अन्तर्_batch % (dst_w * dst_h);
            ध्रुव int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ध्रुव int64_t dst_z      = p_अन्तर्_batch / (dst_w * dst_h);
            ध्रुव int64_t dst_y      = p_अन्तर्_गभीरता / dst_w;
            ध्रुव int64_t dst_x      = p_अन्तर्_गभीरता % dst_w;

            वर्ण * dst_row = (वर्ण *) tmp + (p % patches_प्रत्येकम्_batch) * knl_n_योग * traits->type_आकार;

            प्रति (int64_t ic = 0; ic < c; ++ic) {
                प्रति (int64_t kz = 0; kz < knl_d; ++kz) {
                    प्रति (int64_t ky = 0; ky < knl_h; ++ky) {
                        प्रति (int64_t kx = 0; kx < knl_w; ++kx) {
                            ध्रुव int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ध्रुव int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ध्रुव int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_प्रत्येकम्_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            भिन्न src_val;
                            यदि (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } अन्यथा {
                                ध्रुव int64_t cn_idx = batch_idx * c + ic;
                                ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_दत्त + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            वर्ण * element_ptr = dst_row + dst_idx * traits->type_आकार;
                            यदि (kernel_type == GGML_TYPE_F32) {
                                *(भिन्न *)element_ptr = src_val;
                            } अन्यथा यदि (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_गन्तव्यम्_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        भिन्न * gemm_output = (भिन्न *) ((वर्ण *) tmp + patches_प्रत्येकम्_batch * knl_n_योग * traits->type_आकार);
        ggml_call_mul_mat(kernel_type, params, patch_n_अन्तर्_batch, oc, knl_n_योग, tmp, knl_दत्त, gemm_output);

        ggml_barrier(params->threadpool);

        ध्रुव int64_t permute_प्रत्येकम्_thread = (patch_n_अन्तर्_batch + params->nth - 1) / params->nth;
        ध्रुव int64_t permute_start = params->ith * permute_प्रत्येकम्_thread;
        ध्रुव int64_t permute_अन्तः = std::min(permute_start + permute_प्रत्येकम्_thread, patch_n_अन्तर्_batch);

        प्रति (int64_t i = permute_start; i < permute_अन्तः; ++i) {
            ध्रुव int64_t p = patch_start_batch + i;
            ध्रुव int64_t p_अन्तर्_batch = p % (dst_w * dst_h * dst_d);
            ध्रुव int64_t p_अन्तर्_गभीरता = p_अन्तर्_batch % (dst_w * dst_h);
            ध्रुव int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ध्रुव int64_t dst_z      = p_अन्तर्_batch / (dst_w * dst_h);
            ध्रुव int64_t dst_y      = p_अन्तर्_गभीरता / dst_w;
            ध्रुव int64_t dst_x      = p_अन्तर्_गभीरता % dst_w;

            प्रति (int64_t ioc = 0; ioc < oc; ++ioc) {
                ध्रुव भिन्न value = gemm_output[i * oc + ioc];
                ध्रुव int64_t ocn_idx = batch_idx * oc + ioc;
                भिन्न * dst_ptr = (भिन्न *)((वर्ण *)dst_दत्त + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_3d(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

प्रतिरूपम् <प्रकारनाम kernel_t>
स्थिर शून्य ggml_compute_forward_conv_transpose_2d_impl(
    ध्रुव ggml_compute_params * params,
          ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    यदि (ith == 0) {
        स्मृतिं_पूरय(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ध्रुव wdata = (kernel_t *) params->wdata + 0;

            प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
                प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                    ध्रुव kernel_t * ध्रुव src = (kernel_t *)((वर्ण *) src0->दत्त + i03*nb03 + i02*nb02);
                    kernel_t * dst_दत्त = wdata + i02*ne01*ne00*ne03;
                    प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                        प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_दत्त[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * ध्रुव wdata = (kernel_t *) params->wdata + nk;
            प्रति (पूर्णाङ्क i13 = 0; i13 < ne13; i13++) {
                kernel_t * ध्रुव wdata_b = wdata + i13*ne10*ne11*ne12;
                प्रति (पूर्णाङ्क i12 = 0; i12 < ne12; i12++) {
                    प्रति (पूर्णाङ्क i11 = 0; i11 < ne11; i11++) {
                        ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src1->दत्त + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_दत्त = wdata_b + i11*ne10*ne12;
                        प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
                            यदि constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_दत्त[i10*ne12 + i12] = GGML_CPU_FP32_गन्तव्यम्_FP16(src[i10]);
                            } अन्यथा {
                                dst_दत्त[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        स्मृतिं_पूरय(dst->दत्त, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    ध्रुव int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    ध्रुव पूर्णाङ्क np = ne2;

    // patches per thread
    ध्रुव पूर्णाङ्क dp = (np + nth - 1)/nth;

    // patch range for this thread
    ध्रुव पूर्णाङ्क ip0 = dp*ith;
    ध्रुव पूर्णाङ्क ip1 = MIN(ip0 + dp, np);

    kernel_t * ध्रुव wdata = (kernel_t *) params->wdata + 0;
    kernel_t * ध्रुव wdata_src = wdata + nk;

    प्रति (पूर्णाङ्क i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        प्रति (पूर्णाङ्क i3 = 0; i3 < ne3; i3++) { // batch
            भिन्न * dst_दत्त = (भिन्न *)((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            प्रति (पूर्णाङ्क i11 = 0; i11 < ne11; i11++) {
                प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
                    ध्रुव पूर्णाङ्क i1n = i11*ne10*ne12 + i10*ne12;
                    प्रति (पूर्णाङ्क i01 = 0; i01 < ne01; i01++) {
                        प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                            भिन्न v = 0;
                            यदि constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } अन्यथा {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_दत्त[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_conv_transpose_2d(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<भिन्न>(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

संरचना ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    पूर्णाङ्क stride_x;
    पूर्णाङ्क stride_y;
    पूर्णाङ्क pad_x;
    पूर्णाङ्क pad_y;
    पूर्णाङ्क dilation_x;
    पूर्णाङ्क dilation_y;
};

स्थिर अन्तःस्थ भिन्न ggml_conv_2d_dw_knl_f32(ध्रुव वर्ण * दत्त, int64_t i, ggml_type type) {
    यदि (type == GGML_TYPE_F16) {
        प्रत्यागम GGML_FP16_गन्तव्यम्_FP32(((ध्रुव ggml_fp16_t *)दत्त)[i]);
    }
    प्रत्यागम ((ध्रुव भिन्न *)दत्त)[i];
}

स्थिर शून्य ggml_compute_forward_conv_2d_dw_cwhn(
        ध्रुव ggml_compute_params * params,
        ध्रुव ggml_tensor * src,
        ध्रुव ggml_tensor * kernel,
        ggml_tensor * dst,
        ध्रुव ggml_conv_2d_dw_params & p) {

    ध्रुव int64_t c = p.channels;
    ध्रुव वर्ण * knl_दत्त = (ध्रुव वर्ण *)kernel->दत्त;
    ध्रुव ggml_type knl_type = kernel->type;

    ध्रुव int64_t पङ्क्तिकाः_योग = p.dst_h * p.batch;
    ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_thread = (पङ्क्तिकाः_योग + params->nth - 1) / params->nth;
    ध्रुव int64_t row_start = params->ith * पङ्क्तिकाः_प्रत्येकम्_thread;
    ध्रुव int64_t row_अन्तः = MIN(row_start + पङ्क्तिकाः_प्रत्येकम्_thread, पङ्क्तिकाः_योग);

#यदि_परिभाषितम् GGML_SIMD
    int64_t c_pkg_अन्तः = 0;
    int64_t pkg_आकार = GGML_F32_EPR;
    यदि (knl_type == GGML_TYPE_F32) {
    #यदि defined(__ARM_FEATURE_SVE)
        pkg_आकार = svcntw();
    #अन्यथा
        pkg_आकार = GGML_F32_EPR;
    #यदि_अन्तः
        c_pkg_अन्तः = (c / pkg_आकार) * pkg_आकार;
    }
#अन्यथा
    ध्रुव int64_t c_pkg_अन्तः = 0;
#यदि_अन्तः

    प्रति (int64_t row = row_start; row < row_अन्तः; ++row) {
        ध्रुव int64_t dst_y = row % p.dst_h;
        ध्रुव भिन्न * src_दत्त = (ध्रुव भिन्न *)src->दत्त + (row / p.dst_h) * p.src_w * p.src_h * c;
        प्रति (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            भिन्न * dst_दत्त = (भिन्न *)dst->दत्त + (row * p.dst_w + dst_x) * c;
            ध्रुव int64_t src_y_आधारः = dst_y * p.stride_y - p.pad_y;
            ध्रुव int64_t src_x_आधारः = dst_x * p.stride_x - p.pad_x;

#यदि_परिभाषितम् GGML_SIMD
            प्रति (int64_t c_i = 0; c_i < c_pkg_अन्तः; c_i += pkg_आकार) {
                GGML_F32_VEC योग_2 = GGML_F32_VEC_ZERO;
                प्रति (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ध्रुव int64_t src_y = src_y_आधारः + knl_y * p.dilation_y;
                    यदि (src_y < 0 || src_y >= p.src_h) {
                        अनुवर्त;
                    }
                    प्रति (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ध्रुव int64_t src_x = src_x_आधारः + knl_x * p.dilation_x;
                        यदि (src_x < 0 || src_x >= p.src_w) {
                            अनुवर्त;
                        }
                        ध्रुव भिन्न * kp = (ध्रुव भिन्न *)knl_दत्त + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_आरोपय(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_आरोपय(src_दत्त + (src_y * p.src_w + src_x) * c + c_i);
                        योग_2 = GGML_F32_VEC_FMA(योग_2, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_दत्त + c_i, योग_2);
            }
#यदि_अन्तः
            प्रति (int64_t c_i = c_pkg_अन्तः; c_i < c; ++c_i) {
                भिन्न योग_2 = 0.0f;
                प्रति (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ध्रुव int64_t src_y = src_y_आधारः + knl_y * p.dilation_y;
                    यदि (src_y < 0 || src_y >= p.src_h) {
                        अनुवर्त;
                    }
                    प्रति (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ध्रुव int64_t src_x = src_x_आधारः + knl_x * p.dilation_x;
                        यदि (src_x < 0 || src_x >= p.src_w) {
                            अनुवर्त;
                        }
                        योग_2 += ggml_conv_2d_dw_knl_f32(knl_दत्त, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_दत्त[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_दत्त[c_i] = योग_2;
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_conv_2d_dw_whcn(
        ध्रुव ggml_compute_params * params,
        ध्रुव ggml_tensor * src,
        ध्रुव ggml_tensor * kernel,
        ggml_tensor * dst,
        ध्रुव ggml_conv_2d_dw_params & p) {

    ध्रुव int64_t n = p.channels * p.batch;
    ध्रुव int64_t प्रत्येकम्_thread = (n + params->nth - 1) / params->nth;
    ध्रुव int64_t start = params->ith * प्रत्येकम्_thread;
    ध्रुव int64_t end = MIN(start + प्रत्येकम्_thread, n);
    ध्रुव वर्ण * knl_आधारः = (ध्रुव वर्ण *)kernel->दत्त;
    ध्रुव ggml_type knl_type = kernel->type;

    प्रति (int64_t i = start; i < end; ++i) {
        ध्रुव int64_t knl_विस्थापनम् = (i % p.channels) * p.knl_w * p.knl_h;
        ध्रुव भिन्न * src_दत्त = (ध्रुव भिन्न *)src->दत्त + i * p.src_w * p.src_h;
        भिन्न * dst_दत्त = (भिन्न *)dst->दत्त + i * p.dst_w * p.dst_h;

        प्रति (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            प्रति (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                भिन्न योग_2 = 0.0f;
                प्रति (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ध्रुव int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    यदि (src_y < 0 || src_y >= p.src_h) {
                        अनुवर्त;
                    }
                    प्रति (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ध्रुव int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        यदि (src_x < 0 || src_x >= p.src_w) {
                            अनुवर्त;
                        }
                        योग_2 += ggml_conv_2d_dw_knl_f32(knl_आधारः, knl_विस्थापनम् + knl_y * p.knl_w + knl_x, knl_type)
                             * src_दत्त[src_y * p.src_w + src_x];
                    }
                }
                dst_दत्त[dst_y * p.dst_w + dst_x] = योग_2;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_2d_dw(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * kernel = dst->src[0];
    ध्रुव ggml_tensor * src = dst->src[1];
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

    यदि (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } अन्यथा यदि (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } अन्यथा {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
स्थिर शून्य ggml_compute_forward_pool_1d_ksp(
        ध्रुव ggml_compute_params * params,
        ध्रुव ggml_op_pool op,
        ध्रुव पूर्णाङ्क k,
        ध्रुव पूर्णाङ्क s,
        ध्रुव पूर्णाङ्क p,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src = dst->src[0];

    प्रतिज्ञा(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    ध्रुव int64_t IW = src->ne[0];
    ध्रुव int64_t OW = dst->ne[0];

    ध्रुव int64_t nr = ggml_nrows(src);

    प्रति (int64_t ir = 0; ir < nr; ++ir) {
        ध्रुव वर्ण * srow_अष्टकानि =            (ध्रुव वर्ण *) src->दत्त + ir * src->nb[1];
        भिन्न      * drow       = (भिन्न *) ((      वर्ण *) dst->दत्त + ir * dst->nb[1]);

        प्रति (int64_t ow = 0; ow < OW; ++ow) {
            भिन्न res = 0;
            विकल्प (op) {
                पक्ष GGML_OP_POOL_AVG: res = 0.0f;     भङ्ग;
                पक्ष GGML_OP_POOL_MAX: res = -FLT_MAX; भङ्ग;
                पक्ष GGML_OP_POOL_गणनम्: GGML_ABORT("fatal error");
            }

            पूर्णाङ्क count = 0;
            ध्रुव पूर्णाङ्क आधारः = (पूर्णाङ्क) ow * s - p;

            प्रति (पूर्णाङ्क ki = 0; ki < k; ++ki) {
                ध्रुव पूर्णाङ्क j = आधारः + ki;
                यदि (j < 0 || j >= (पूर्णाङ्क) IW) {
                    अनुवर्त;
                }

                भिन्न v;
                यदि (src->type == GGML_TYPE_F32) {
                    v = ((ध्रुव भिन्न *) srow_अष्टकानि)[j];
                } अन्यथा {
                    v = GGML_CPU_FP16_गन्तव्यम्_FP32(((ध्रुव ggml_fp16_t *) srow_अष्टकानि)[j]);
                }

                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG: res += v;                भङ्ग;
                    पक्ष GGML_OP_POOL_MAX: res =  std::max(v, res); भङ्ग;
                    पक्ष GGML_OP_POOL_गणनम्: GGML_ABORT("fatal error");
                }

                ++count;
            }

            विकल्प (op) {
                पक्ष GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; भङ्ग;
                पक्ष GGML_OP_POOL_MAX:                                           भङ्ग;
                पक्ष GGML_OP_POOL_गणनम्: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

शून्य ggml_compute_forward_pool_1d(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव int32_t * opts = (ध्रुव int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ध्रुव पूर्णाङ्क k0 = opts[1];
    ध्रुव पूर्णाङ्क s0 = opts[2];
    ध्रुव पूर्णाङ्क p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

शून्य ggml_compute_forward_pool_2d(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src = dst->src[0];

    प्रतिज्ञा(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    ध्रुव int32_t * opts = (ध्रुव int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ध्रुव पूर्णाङ्क k0 = opts[1];
    ध्रुव पूर्णाङ्क k1 = opts[2];
    ध्रुव पूर्णाङ्क s0 = opts[3];
    ध्रुव पूर्णाङ्क s1 = opts[4];
    ध्रुव पूर्णाङ्क p0 = opts[5];
    ध्रुव पूर्णाङ्क p1 = opts[6];
    ध्रुव वर्ण * cdata = (ध्रुव वर्ण*)src->दत्त;
    ध्रुव वर्ण * ध्रुव दत्त_अन्तः = cdata + ggml_nbytes(src);

    ध्रुव int64_t px = dst->ne[0];
    ध्रुव int64_t py = dst->ne[1];
    ध्रुव int64_t pa = px * py;

    भिन्न * dplane = (भिन्न *)dst->दत्त;

    ध्रुव पूर्णाङ्क ka = k0 * k1;
    ध्रुव पूर्णाङ्क offset0 = -p0;
    ध्रुव पूर्णाङ्क offset1 = -p1;

    यावत् (cdata < दत्त_अन्तः) {
        प्रति (पूर्णाङ्क oy = 0; oy < py; ++oy) {
            भिन्न * ध्रुव drow = dplane + oy * px;
            भिन्न * ध्रुव बहिर्गमनम्  = drow;

            प्रति (पूर्णाङ्क ox = 0; ox < px; ++ox) {
                भिन्न res = 0;
                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG: res = 0;        भङ्ग;
                    पक्ष GGML_OP_POOL_MAX: res = -FLT_MAX; भङ्ग;
                    पक्ष GGML_OP_POOL_गणनम्: GGML_ABORT("fatal error");
                }

                ध्रुव पूर्णाङ्क ix = offset0 + ox * s0;
                ध्रुव पूर्णाङ्क iy = offset1 + oy * s1;

                प्रति (पूर्णाङ्क ky = 0; ky < k1; ++ky) {
                    यदि (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        अनुवर्त;
                    }

                    ध्रुव शून्य * srow = (ध्रुव शून्य *)(cdata + src->nb[1] * (iy + ky));
                    प्रति (पूर्णाङ्क kx = 0; kx < k0; ++kx) {
                        पूर्णाङ्क j = ix + kx;
                        यदि (j < 0 || j >= src->ne[0]) {
                            अनुवर्त;
                        }

                        ध्रुव भिन्न srow_j = (src->type == GGML_TYPE_F32) ? ((ध्रुव भिन्न*)srow)[j] : GGML_CPU_FP16_गन्तव्यम्_FP32(((ध्रुव ggml_fp16_t*)srow)[j]);
                        विकल्प (op) {
                            पक्ष GGML_OP_POOL_AVG: res += srow_j;                भङ्ग;
                            पक्ष GGML_OP_POOL_MAX: res =  std::max(srow_j, res); भङ्ग;
                            पक्ष GGML_OP_POOL_गणनम्:               GGML_ABORT("fatal error");
                        }
                    }
                }
                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG:           res /= ka; भङ्ग;
                    पक्ष GGML_OP_POOL_MAX:                      भङ्ग;
                    पक्ष GGML_OP_POOL_गणनम्: GGML_ABORT("fatal error");
                }

                बहिर्गमनम्[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

शून्य ggml_compute_forward_pool_2d_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src  = dst->src[0];
    ध्रुव ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    प्रतिज्ञा(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    ध्रुव int32_t * opts = (ध्रुव int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    ध्रुव पूर्णाङ्क k0 = opts[1];
    ध्रुव पूर्णाङ्क k1 = opts[2];
    ध्रुव पूर्णाङ्क s0 = opts[3];
    ध्रुव पूर्णाङ्क s1 = opts[4];
    ध्रुव पूर्णाङ्क p0 = opts[5];
    ध्रुव पूर्णाङ्क p1 = opts[6];

    वर्ण       * cdata  = (वर्ण       *) dst->दत्त;
    ध्रुव वर्ण * cdataf = (ध्रुव वर्ण *) dstf->दत्त;
    ध्रुव वर्ण * ध्रुव दत्त_अन्तः = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    स्मृतिं_पूरय(cdata, 0, ggml_nbytes(dst));

    ध्रुव int64_t px = src->ne[0];
    ध्रुव int64_t py = src->ne[1];
    ध्रुव int64_t pa = px * py;

    ध्रुव भिन्न * splane = (ध्रुव भिन्न *) src->दत्त;

    ध्रुव पूर्णाङ्क ka = k0 * k1;
    ध्रुव पूर्णाङ्क offset0 = -p0;
    ध्रुव पूर्णाङ्क offset1 = -p1;

    यावत् (cdata < दत्त_अन्तः) {
        प्रति (पूर्णाङ्क oy = 0; oy < py; ++oy) {
            ध्रुव भिन्न * ध्रुव srow = splane + oy * px;
            प्रति (पूर्णाङ्क ox = 0; ox < px; ++ox) {
                ध्रुव भिन्न grad0 = srow[ox];

                ध्रुव पूर्णाङ्क ix = offset0 + ox * s0;
                ध्रुव पूर्णाङ्क iy = offset1 + oy * s1;

                यदि (op == GGML_OP_POOL_MAX) {
                    भिन्न maxval = -FLT_MAX;
                    पूर्णाङ्क kxmax = -1;
                    पूर्णाङ्क kymax = -1;

                    प्रति (पूर्णाङ्क ky = 0; ky < k1; ++ky) {
                        यदि (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            अनुवर्त;
                        }
                        ध्रुव शून्य * drowf = (ध्रुव शून्य *)(cdataf + dst->nb[1] * (iy + ky));
                        प्रति (पूर्णाङ्क kx = 0; kx < k0; ++kx) {
                            पूर्णाङ्क j = ix + kx;
                            यदि (j < 0 || j >= dst->ne[0]) {
                                अनुवर्त;
                            }

                            ध्रुव भिन्न val = dst->type == GGML_TYPE_F32 ?
                                ((ध्रुव भिन्न *) drowf)[j] : GGML_CPU_FP16_गन्तव्यम्_FP32(((ध्रुव ggml_fp16_t *) drowf)[j]);
                            यदि (val <= maxval) {
                                अनुवर्त;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    यदि (kxmax == -1 || kymax == -1) {
                        अनुवर्त;
                    }

                    शून्य * drow = (शून्य *)(cdata + dst->nb[1] * (iy + kymax));
                    ध्रुव पूर्णाङ्क j = ix + kxmax;
                    यदि (dst->type == GGML_TYPE_F32) {
                        ((भिन्न *) drow)[j] += grad0;
                    } अन्यथा {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_गन्तव्यम्_FP16(grad0 + GGML_CPU_FP16_गन्तव्यम्_FP32(((ध्रुव ggml_fp16_t *) drow)[j]));
                    }
                } अन्यथा यदि (op == GGML_OP_POOL_AVG) {
                    ध्रुव भिन्न grad = grad0 / ka;

                    प्रति (पूर्णाङ्क ky = 0; ky < k1; ++ky) {
                        यदि (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            अनुवर्त;
                        }
                        शून्य * drow = (शून्य *)(cdata + dst->nb[1] * (iy + ky));
                        प्रति (पूर्णाङ्क kx = 0; kx < k0; ++kx) {
                            पूर्णाङ्क j = ix + kx;
                            यदि (j < 0 || j >= dst->ne[0]) {
                                अनुवर्त;
                            }

                            यदि (dst->type == GGML_TYPE_F32) {
                                ((भिन्न *) drow)[j] += grad;
                            } अन्यथा {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_गन्तव्यम्_FP16(grad);
                            }
                        }
                    }
                } अन्यथा {
                    GGML_ASSERT(असत्य);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

स्थिर शून्य ggml_compute_forward_upscale_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भिन्न sf0 = (भिन्न)ne0/src0->ne[0];
    भिन्न sf1 = (भिन्न)ne1/src0->ne[1];
    भिन्न sf2 = (भिन्न)ne2/src0->ne[2];
    भिन्न sf3 = (भिन्न)ne3/src0->ne[3];
    भिन्न pixel_विस्थापनम् = 0.5f;

    ध्रुव int32_t बहुलकम्_flags = ggml_get_op_params_i32(dst, 0);
    ध्रुव ggml_scale_बहुलकम् mode = (ggml_scale_बहुलकम्) (बहुलकम्_flags & 0xFF);

    यदि (बहुलकम्_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_विस्थापनम् = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (भिन्न)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (भिन्न)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    यदि (mode == GGML_SCALE_बहुलकम्_निकटतमः) {
        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            ध्रुव int64_t i03 = i3 / sf3;
            प्रति (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ध्रुव int64_t i02 = i2 / sf2;
                प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
                    ध्रुव int64_t i01 = i1 / sf1;
                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव int64_t i00 = i0 / sf0;

                        ध्रुव भिन्न * x = (भिन्न *)((वर्ण *) src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              भिन्न * y = (भिन्न *)((वर्ण *)  dst->दत्त +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } अन्यथा यदि (mode == GGML_SCALE_बहुलकम्_BILINEAR && (बहुलकम्_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        स्वचालित triangle_filter = [](भिन्न x) -> भिन्न {
            प्रत्यागम std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        ध्रुव भिन्न support1  = std::max(1.0f, 1.0f / sf1);
        ध्रुव भिन्न invscale1 = 1.0f / support1;
        ध्रुव भिन्न support0  = std::max(1.0f, 1.0f / sf0);
        ध्रुव भिन्न invscale0 = 1.0f / support0;

        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            ध्रुव int64_t i03 = i3 / sf3;
            प्रति (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ध्रुव int64_t i02 = i2 / sf2;
                प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
                    ध्रुव भिन्न y = ((भिन्न) i1 + pixel_विस्थापनम्) / sf1;
                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव भिन्न x = ((भिन्न) i0 + pixel_विस्थापनम्) / sf0;

                        // the range of source pixels that contribute
                        ध्रुव int64_t x_min = std::max<int64_t>(x - support0 + pixel_विस्थापनम्, 0);
                        ध्रुव int64_t x_max = std::min<int64_t>(x + support0 + pixel_विस्थापनम्, ne00);
                        ध्रुव int64_t y_min = std::max<int64_t>(y - support1 + pixel_विस्थापनम्, 0);
                        ध्रुव int64_t y_max = std::min<int64_t>(y + support1 + pixel_विस्थापनम्, ne01);

                        // bilinear filter with antialiasing
                        भिन्न val = 0.0f;
                        भिन्न योग_weight = 0.0f;

                        प्रति (int64_t sy = y_min; sy < y_max; sy++) {
                            ध्रुव भिन्न weight_y = triangle_filter((sy - y + pixel_विस्थापनम्) * invscale1);

                            प्रति (int64_t sx = x_min; sx < x_max; sx++) {
                                ध्रुव भिन्न weight_x = triangle_filter((sx - x + pixel_विस्थापनम्) * invscale0);
                                ध्रुव भिन्न weight = weight_x * weight_y;

                                यदि (weight <= 0.0f) {
                                    अनुवर्त;
                                }

                                ध्रुव भिन्न pixel = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->दत्त + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                योग_weight += weight;
                            }
                        }

                        यदि (योग_weight > 0.0f) {
                            val /= योग_weight;
                        }

                        भिन्न * dst_ptr = (भिन्न *)((वर्ण *)dst->दत्त + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } अन्यथा यदि (mode == GGML_SCALE_बहुलकम्_BILINEAR) {
        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            ध्रुव int64_t i03 = i3 / sf3;
            प्रति (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ध्रुव int64_t i02 = i2 / sf2;
                प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
                    ध्रुव भिन्न y = ((भिन्न)i1 + pixel_विस्थापनम्) / sf1 - pixel_विस्थापनम्;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    भिन्न dy = y - (भिन्न)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव भिन्न x = ((भिन्न)i0 + pixel_विस्थापनम्) / sf0 - pixel_विस्थापनम्;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        भिन्न dx = x - (भिन्न)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ध्रुव भिन्न a = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->दत्त + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ध्रुव भिन्न b = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->दत्त + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ध्रुव भिन्न c = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->दत्त + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ध्रुव भिन्न d = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->दत्त + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ध्रुव भिन्न val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        भिन्न * y_dst = (भिन्न *)((वर्ण *)dst->दत्त + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } अन्यथा यदि (mode == GGML_SCALE_बहुलकम्_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ध्रुव भिन्न a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        स्वचालित weight1 = [a](भिन्न x) { प्रत्यागम ((a + 2) * x - (a + 3)) * x * x + 1; };
        स्वचालित weight2 = [a](भिन्न x) { प्रत्यागम ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        स्वचालित bicubic = [=](भिन्न p0, भिन्न p1, भिन्न p2, भिन्न p3, भिन्न x) {
            ध्रुव भिन्न w0 = weight2(x + 1);
            ध्रुव भिन्न w1 = weight1(x + 0);
            ध्रुव भिन्न w2 = weight1(1 - x);
            ध्रुव भिन्न w3 = weight2(2 - x);
            प्रत्यागम p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            ध्रुव int64_t i03 = i3 / sf3;
            प्रति (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ध्रुव int64_t i02 = i2 / sf2;
                प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
                    ध्रुव भिन्न y = ((भिन्न)i1 + pixel_विस्थापनम्) / sf1 - pixel_विस्थापनम्;
                    ध्रुव int64_t y0 = (int64_t)floorf(y);
                    ध्रुव भिन्न dy = y - (भिन्न)y0;

                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव भिन्न x = ((भिन्न)i0 + pixel_विस्थापनम्) / sf0 - pixel_विस्थापनम्;
                        ध्रुव int64_t x0 = (int64_t)floorf(x);
                        ध्रुव भिन्न dx = x - (भिन्न)x0;

                        स्वचालित p = [=](int64_t x_off, int64_t y_off) -> भिन्न {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            प्रत्यागम *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->दत्त + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ध्रुव भिन्न val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        भिन्न * y_dst = (भिन्न *)((वर्ण *)dst->दत्त + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } अन्यथा {
        GGML_ABORT("unsupported upscale mode");
    }
}

शून्य ggml_compute_forward_upscale(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

प्रतिरूपम्<तार्किक circular_t>
स्थिर शून्य ggml_compute_forward_pad_f32(
    ध्रुव ggml_compute_params * params,
          ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(dst->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भिन्न * dst_ptr = (भिन्न *) dst->दत्त;
    ध्रुव int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    ध्रुव int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    ध्रुव int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    ध्रुव int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    ध्रुव int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    ध्रुव int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    ध्रुव int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    ध्रुव int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    ध्रुव int64_t nrows = ne1 * ne2 * ne3;
    ध्रुव तार्किक src_row_contig = src0->nb[0] == परिमाण(भिन्न);

    प्रति (int64_t row = ith; row < nrows; row += nth) {
        ध्रुव int64_t i1 = row % ne1;
        ध्रुव int64_t i2 = (row / ne1) % ne2;
        ध्रुव int64_t i3 = row / (ne1 * ne2);

        भिन्न * dst_row = dst_ptr + row * ne0;

        यदि constexpr (circular_t) {
            // circular padding wraps around on a torus
            ध्रुव int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            ध्रुव int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            ध्रुव int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                ध्रुव int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(ध्रुव भिन्न *)((ध्रुव वर्ण *) src0->दत्त + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } अन्यथा {
            ध्रुव तार्किक interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            यदि (!interior) {
                स्मृतिं_पूरय(dst_row, 0, ne0 * परिमाण(भिन्न));
                अनुवर्त;
            }

            ध्रुव वर्ण * src_row = (ध्रुव वर्ण *) src0->दत्त + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            यदि (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                स्मृतिं_पूरय(dst_row, 0, lp0 * परिमाण(भिन्न));
                स्मृतिं_प्रतिलिख(dst_row + lp0, src_row, ne00 * परिमाण(भिन्न));
                स्मृतिं_पूरय(dst_row + lp0 + ne00, 0, rp0 * परिमाण(भिन्न));
            } अन्यथा {
                प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                    यदि (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(ध्रुव भिन्न *)(src_row + (i0 - lp0)*nb00);
                    } अन्यथा {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


शून्य ggml_compute_forward_pad(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव तार्किक circular = (तार्किक) ggml_get_op_params_i32(dst, 8);
    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                यदि (circular) {
                    ggml_compute_forward_pad_f32<सत्य>(params, dst);
                } अन्यथा {
                    ggml_compute_forward_pad_f32<असत्य>(params, dst);
                }
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

शून्य ggml_compute_forward_pad_reflect_1d(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int32_t * opts = (ध्रुव int32_t *) dst->op_params;
    ध्रुव पूर्णाङ्क p0 = opts[0];
    ध्रुव पूर्णाङ्क p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
        प्रति (int64_t i2 = 0; i2 < ne2; i2++) {
            प्रति (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                भिन्न * वामतः  = (भिन्न *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                भिन्न * दक्षिणतः = (भिन्न *) ((वर्ण *) dst->दत्त + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, वामतः, (भिन्न *) ((वर्ण *) src0->दत्त + i3*nb03 + i2*nb02 + i1*nb01));

                प्रति (पूर्णाङ्क i0 = 1; i0 <= p0; i0++) { वामतः[-i0] = वामतः[i0];   }
                प्रति (पूर्णाङ्क i0 = 1; i0 <= p1; i0++) { दक्षिणतः[i0] = दक्षिणतः[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

स्थिर int64_t ggml_wrap_सूचकांक(int64_t i, int64_t ne) {
    यदि (i < 0) {
        प्रत्यागम i + ne;
    } अन्यथा यदि (i >= ne) {
        प्रत्यागम i - ne;
    }
    प्रत्यागम i;
}

स्थिर शून्य ggml_compute_forward_roll_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव भिन्न * src_दत्त = (ध्रुव भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त = (भिन्न *) dst->दत्त;

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव पूर्णाङ्क s0 = ggml_get_op_params_i32(dst, 0);
    ध्रुव पूर्णाङ्क s1 = ggml_get_op_params_i32(dst, 1);
    ध्रुव पूर्णाङ्क s2 = ggml_get_op_params_i32(dst, 2);
    ध्रुव पूर्णाङ्क s3 = ggml_get_op_params_i32(dst, 3);

    ध्रुव int64_t योग = ne1 * ne2 * ne3;
    ध्रुव int64_t प्रत्येकम्_thread = (योग + params->nth) / params->nth;
    ध्रुव int64_t start = params->ith * प्रत्येकम्_thread;
    ध्रुव int64_t end   = std::min(start + प्रत्येकम्_thread, योग);

    प्रति (int64_t i = start; i < end; ++i) {
        ध्रुव int64_t i1 = i % ne1;
        ध्रुव int64_t i2 = (i / ne1) % ne2;
        ध्रुव int64_t i3 = i / (ne2 * ne1);
        भिन्न * dst_row = dst_दत्त + (i3*nb3 + i2*nb2 + i1*nb1) / परिमाण(भिन्न);

        ध्रुव int64_t i01 = ggml_wrap_सूचकांक(i1 - s1, ne01);
        ध्रुव int64_t i02 = ggml_wrap_सूचकांक(i2 - s2, ne02);
        ध्रुव int64_t i03 = ggml_wrap_सूचकांक(i3 - s3, ne03);
        ध्रुव भिन्न * src_row = src_दत्त + (i03*nb03 + i02*nb02 + i01*nb01) / परिमाण(भिन्न);

        ध्रुव int64_t s = ggml_wrap_सूचकांक(-s0, ne00);
        ध्रुव int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

शून्य ggml_compute_forward_roll(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

स्थिर शून्य ggml_compute_forward_arange_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव भिन्न start = ggml_get_op_params_f32(dst, 0);
    ध्रुव भिन्न stop  = ggml_get_op_params_f32(dst, 1);
    ध्रुव भिन्न चरणः  = ggml_get_op_params_f32(dst, 2);

    ध्रुव int64_t चरणाः = (int64_t) ceilf((stop - start) / चरणः);

    GGML_ASSERT(ggml_nelements(dst) == चरणाः);

    प्रति (int64_t i = ith; i < चरणाः; i+= nth) {
        भिन्न value = start + चरणः * i;
        ((भिन्न *)dst->दत्त)[i] = value;
    }
}

शून्य ggml_compute_forward_arange(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {
    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_timestep_embedding_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव पूर्णाङ्क dim = ggml_get_op_params_i32(dst, 0);
    ध्रुव पूर्णाङ्क max_period = ggml_get_op_params_i32(dst, 1);

    पूर्णाङ्क half = dim / 2;

    प्रति (int64_t i = 0; i < ne00; i++) {
        भिन्न * embed_दत्त = (भिन्न *)((वर्ण *)  dst->दत्त +  i*nb1);
        प्रति (int64_t j = ith; j < half; j += nth) {
            भिन्न timestep = ((भिन्न *)src0->दत्त)[i];
            भिन्न freq = (भिन्न)expf(-logf(max_period) * j / half);
            भिन्न arg = timestep * freq;
            embed_दत्त[j] = cosf(arg);
            embed_दत्त[j + half] = sinf(arg);
        }
        यदि (dim % 2 != 0 && ith == 0) {
            embed_दत्त[2 * half] = 0.f;
        }
    }
}

शून्य ggml_compute_forward_timestep_embedding(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

प्रतिरूपम्<गणना ggml_क्रमय_क्रमः क्रमः>
संरचना cmp_argsort {
    ध्रुव भिन्न * दत्त;
    तार्किक संक्रियकः()(int32_t a, int32_t b) ध्रुव {
        यदि constexpr (क्रमः == GGML_क्रमय_क्रमः_ASC) {
            प्रत्यागम दत्त[a] < दत्त[b];
        } अन्यथा {
            प्रत्यागम दत्त[a] > दत्त[b];
        }
    }
};

स्थिर शून्य ggml_compute_forward_argsort_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t nr = ggml_nrows(src0);

    ggml_क्रमय_क्रमः क्रमः = (ggml_क्रमय_क्रमः) ggml_get_op_params_i32(dst, 0);

    प्रति (int64_t i = ith; i < nr; i += nth) {
        ध्रुव भिन्न * src_दत्त = (भिन्न *)((वर्ण *) src0->दत्त + i*nb01);

        int32_t * dst_दत्त = (int32_t *)((वर्ण *) dst->दत्त + i*nb1);

        प्रति (int64_t j = 0; j < ne0; j++) {
            dst_दत्त[j] = j;
        }

        विकल्प (क्रमः) {
            पक्ष GGML_क्रमय_क्रमः_ASC:
                std::sort(dst_दत्त, dst_दत्त + ne0, cmp_argsort<GGML_क्रमय_क्रमः_ASC>{src_दत्त});
                भङ्ग;

            पक्ष GGML_क्रमय_क्रमः_DESC:
                std::sort(dst_दत्त, dst_दत्त + ne0, cmp_argsort<GGML_क्रमय_क्रमः_DESC>{src_दत्त});
                भङ्ग;

            स्वभाव:
                GGML_ABORT("invalid sort order");
        }
    }
}

शून्य ggml_compute_forward_argsort(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

संरचना cmp_शिखरम्_k {
    ध्रुव भिन्न * दत्त;
    तार्किक संक्रियकः()(int32_t a, int32_t b) ध्रुव {
        प्रत्यागम दत्त[a] > दत्त[b];
    }
};

स्थिर शून्य ggml_compute_forward_शिखरम्_k_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t nr = ggml_nrows(src0);

    ध्रुव पूर्णाङ्क शिखरम्_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_पङ्क्तिः_आकार_F32) * ith;

    प्रति (int64_t i = ith; i < nr; i += nth) {
        ध्रुव भिन्न * src_दत्त = (भिन्न *)((वर्ण *) src0->दत्त + i*nb01);

        प्रति (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + शिखरम्_k, tmp + ne00, cmp_शिखरम्_k{src_दत्त});

        int32_t * dst_दत्त = (int32_t *)((वर्ण *) dst->दत्त + i*nb1);

        std::copy(tmp, tmp + शिखरम्_k, dst_दत्त);

        // emphasize that the order is not important
        यदि (शिखरम्_k > 1) {
            std::swap(dst_दत्त[0], dst_दत्त[1]);
        }
    }
}

शून्य ggml_compute_forward_शिखरम्_k(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_शिखरम्_k_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        पूर्णाङ्क ir0, पूर्णाङ्क ir1,
        int64_t ic_start, int64_t ic_अन्तः,
        भिन्न * partials, int64_t partial_stride) {

    ध्रुव तार्किक write_partials = (partials != शून्यसूचकः);
    ध्रुव ggml_tensor * q     = dst->src[0];
    ध्रुव ggml_tensor * k     = dst->src[1];
    ध्रुव ggml_tensor * v     = dst->src[2];
    ध्रुव ggml_tensor * आवरणम्  = dst->src[3];
    ध्रुव ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ध्रुव int64_t DK = nek0;
    ध्रुव int64_t DV = nev0;
    ध्रुव int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbv0 == ggml_type_आकार(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    ध्रुव int64_t rk2 = neq2/nek2;
    ध्रुव int64_t rk3 = neq3/nek3;

    ध्रुव int64_t rv2 = neq2/nev2;
    ध्रुव int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,         (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias,      (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&logit_softcap, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ध्रुव uint32_t n_शीर्ष      = neq2;
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floor(log2(n_शीर्ष));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    ggml_type         ध्रुव k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ध्रुव q_गन्तव्यम्_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ध्रुव kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_गन्तव्यम्_float_t   ध्रुव v_गन्तव्यम्_float     = ggml_get_type_traits(v->type)->गन्तव्यम्_float;

    GGML_ASSERT((                            q_गन्तव्यम्_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_गन्तव्यम्_float  ) && "fattn: unsupported V-type");

    पूर्णाङ्क ith = params->ith;

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // q indices
        ध्रुव पूर्णाङ्क iq3 = ir/(neq2*neq1);
        ध्रुव पूर्णाङ्क iq2 = (ir - iq3*neq2*neq1)/neq1;
        ध्रुव पूर्णाङ्क iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ध्रुव uint32_t h = iq2; // head index
        ध्रुव भिन्न नतिः = (max_bias > 0.0f) ? h < n_शीर्ष_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_शीर्ष_log2) + 1) : 1.0f;

        भिन्न S = 0.0f;      //योग
        भिन्न M = -INFINITY; // maximum KQ value

        भिन्न       * VKQ32 = (भिन्न       *) params->wdata + ith*(1*DK + 2*DV + CACHE_पङ्क्तिः_आकार_F32); // FP32 VKQ accumulator
        भिन्न       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        यदि (v->type == GGML_TYPE_F16) {
            स्मृतिं_पूरय(VKQ16, 0, DV*परिमाण(ggml_fp16_t));
        } अन्यथा {
            स्मृतिं_पूरय(VKQ32, 0, DV*परिमाण(भिन्न));
        }

        ध्रुव ggml_fp16_t * mp = आवरणम् ? (ggml_fp16_t *)((वर्ण *) आवरणम्->दत्त + iq1*आवरणम्->nb[1] + (iq2%आवरणम्->ne[2])*आवरणम्->nb[2] + (iq3%आवरणम्->ne[3])*आवरणम्->nb[3]) : रिक्त;

        // k indices
        ध्रुव पूर्णाङ्क ik3 = iq3 / rk3;
        ध्रुव पूर्णाङ्क ik2 = iq2 / rk2;

        // v indices
        ध्रुव पूर्णाङ्क iv3 = iq3 / rv3;
        ध्रुव पूर्णाङ्क iv2 = iq2 / rv2;

        ध्रुव भिन्न * pq = (ध्रुव भिन्न *) ((वर्ण *) q->दत्त + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_गन्तव्यम्_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        प्रति (int64_t ic = ic_start; ic < ic_अन्तः; ++ic) {
            ध्रुव भिन्न mv = mp ? नतिः*GGML_CPU_FP16_गन्तव्यम्_FP32(mp[ic]) : 0.0f;
            यदि (mv == -INFINITY) {
                अनुवर्त;
            }

            भिन्न s; // KQ value

            ध्रुव वर्ण * k_दत्त = (ध्रुव वर्ण *) k->दत्त + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_दत्त, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            यदि (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ध्रुव भिन्न Mold = M;

            भिन्न ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            भिन्न vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ध्रुव वर्ण * v_दत्त = ((ध्रुव वर्ण *) v->दत्त + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            यदि (v->type == GGML_TYPE_F16) {
                यदि (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } अन्यथा {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (ध्रुव ggml_fp16_t *) v_दत्त, vs);
            } अन्यथा {
                यदि (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } अन्यथा {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                यदि (v_गन्तव्यम्_float) {
                    v_गन्तव्यम्_float(v_दत्त, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } अन्यथा {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ध्रुव भिन्न *) v_दत्त, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        यदि (v->type == GGML_TYPE_F16) {
            प्रति (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_गन्तव्यम्_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        यदि (sinks && ic_start == 0) {
            ध्रुव भिन्न s = ((भिन्न *)((वर्ण *) sinks->दत्त))[h];

            भिन्न ms = 1.0f;
            भिन्न vs = 1.0f;

            यदि (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } अन्यथा {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        यदि (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            भिन्न * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            स्मृतिं_प्रतिलिख(partial + 2, VKQ32, DV * परिमाण(भिन्न));
        } अन्यथा {
            // V /= S
            ध्रुव भिन्न S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ध्रुव पूर्णाङ्क i1 = iq1;
            ध्रुव पूर्णाङ्क i2 = iq2;
            ध्रुव पूर्णाङ्क i3 = iq3;

            // permute(0, 2, 1, 3)
            स्मृतिं_प्रतिलिख((वर्ण *) dst->दत्त + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

स्थिर शून्य ggml_compute_forward_flash_attn_ext_tiled(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        पूर्णाङ्क ir0, पूर्णाङ्क ir1) {
    ध्रुव ggml_tensor * q     = dst->src[0];
    ध्रुव ggml_tensor * k     = dst->src[1];
    ध्रुव ggml_tensor * v     = dst->src[2];
    ध्रुव ggml_tensor * आवरणम्  = dst->src[3];
    ध्रुव ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ध्रुव int64_t DK = nek0;
    ध्रुव int64_t DV = nev0;
    ध्रुव int64_t N  = neq1;

    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbv0 == ggml_type_आकार(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    ध्रुव ggml_type kv_type = k->type;


    // broadcast factors
    ध्रुव int64_t rk2 = neq2/nek2;
    ध्रुव int64_t rk3 = neq3/nek3;

    ध्रुव int64_t rv2 = neq2/nev2;
    ध्रुव int64_t rv3 = neq3/nev3;

    भिन्न scale         = 1.0f;
    भिन्न max_bias      = 0.0f;
    भिन्न logit_softcap = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,         (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias,      (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&logit_softcap, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ध्रुव uint32_t n_शीर्ष      = neq2;
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floor(log2(n_शीर्ष));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    पूर्णाङ्क ith = params->ith;

    स्थिर constexpr पूर्णाङ्क Q_TILE_SZ  = ggml_fa_tile_config::Q;
    स्थिर constexpr पूर्णाङ्क KV_TILE_SZ = ggml_fa_tile_config::KV;

    पूर्णाङ्क ir = ir0;
    यावत् (ir < ir1) {
        // q indices for the start of this tile
        ध्रुव पूर्णाङ्क iq3 = ir/(neq2*neq1);
        ध्रुव पूर्णाङ्क iq2 = (ir - iq3*neq2*neq1)/neq1;
        ध्रुव पूर्णाङ्क iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        ध्रुव पूर्णाङ्क tile_पङ्क्तिकाः = MIN(Q_TILE_SZ, MIN((पूर्णाङ्क)(ir1 - ir), (पूर्णाङ्क)(neq1 - iq1)));
        GGML_ASSERT(tile_पङ्क्तिकाः > 0);

        ध्रुव uint32_t h = iq2; // head index
        ध्रुव भिन्न नतिः = (max_bias > 0.0f) ? h < n_शीर्ष_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_शीर्ष_log2) + 1) : 1.0f;

        भिन्न S[Q_TILE_SZ];
        भिन्न M[Q_TILE_SZ];

        प्रति (पूर्णाङ्क i = 0 ; i < Q_TILE_SZ; ++i) {
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
        भिन्न * आधारः  = (भिन्न *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_पङ्क्तिः_आकार_F32);

        शून्य  * Q_q    = आधारः;
        भिन्न * KQ     = (भिन्न *)((वर्ण *)आधारः + Q_TILE_SZ * DK * परिमाण(भिन्न));
        भिन्न * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        भिन्न * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        भिन्न * V32    = VKQ32 + Q_TILE_SZ * DV;
        भिन्न * K_f32  = V32 + KV_TILE_SZ * DV;

        स्मृतिं_पूरय(VKQ32, 0, Q_TILE_SZ * DV * परिमाण(भिन्न));
        स्मृतिं_पूरय(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * परिमाण(भिन्न));

        // k indices
        ध्रुव पूर्णाङ्क ik3 = iq3 / rk3;
        ध्रुव पूर्णाङ्क ik2 = iq2 / rk2;

        // v indices
        ध्रुव पूर्णाङ्क iv3 = iq3 / rv3;
        ध्रुव पूर्णाङ्क iv2 = iq2 / rv2;

        {
            भिन्न * Q_f32 = (भिन्न *)Q_q;
            प्रति (पूर्णाङ्क tq = 0; tq < tile_पङ्क्तिकाः; tq++) {
                ध्रुव भिन्न * pq = (ध्रुव भिन्न *) ((वर्ण *) q->दत्त + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                स्मृतिं_प्रतिलिख(Q_f32 + tq * DK, pq, DK * परिमाण(भिन्न));
            }
            प्रति (पूर्णाङ्क tq = tile_पङ्क्तिकाः; tq < Q_TILE_SZ; tq++) {
                स्मृतिं_पूरय(Q_f32 + tq * DK, 0, DK * परिमाण(भिन्न));
            }
        }

        स्मृतिं_पूरय(K_f32, 0, DK * KV_TILE_SZ * परिमाण(भिन्न));
        स्मृतिं_पूरय(V32,   0, KV_TILE_SZ * DV * परिमाण(भिन्न));

        प्रति (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ध्रुव पूर्णाङ्क kv_tile = (पूर्णाङ्क)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            यदि (आवरणम्) {
                तार्किक can_skip = सत्य;
                प्रति (पूर्णाङ्क tq = 0; tq < tile_पङ्क्तिकाः; tq++) {
                    ध्रुव ggml_fp16_t * mp_row = (ध्रुव ggml_fp16_t *)((ध्रुव वर्ण *) आवरणम्->दत्त + (iq1 + tq)*आवरणम्->nb[1] + (iq2%आवरणम्->ne[2])*आवरणम्->nb[2] + (iq3%आवरणम्->ne[3])*आवरणम्->nb[3]);
                    प्रति (पूर्णाङ्क tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = नतिः * GGML_CPU_FP16_गन्तव्यम्_FP32(mp_row[ic + tk]);
                        यदि (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = असत्य;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    प्रति (पूर्णाङ्क tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                यदि (can_skip) {
                    अनुवर्त;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            प्रति (पूर्णाङ्क tk = 0; tk < kv_tile; tk++) {
                ध्रुव वर्ण * k_दत्त = (ध्रुव वर्ण *)k->दत्त + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                यदि (kv_type == GGML_TYPE_F16) {
                    ध्रुव ggml_fp16_t * k_f16 = (ध्रुव ggml_fp16_t *)k_दत्त;
                    प्रति (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_गन्तव्यम्_FP32(k_f16[dk]);
                    }
                } अन्यथा {
                    ध्रुव भिन्न * k_f32_src = (ध्रुव भिन्न *)k_दत्त;
                    प्रति (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            स्मृतिं_पूरय(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * परिमाण(भिन्न));
            simd_gemm(KQ, (ध्रुव भिन्न *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            यदि (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            यदि (kv_tile < KV_TILE_SZ) {
                प्रति (पूर्णाङ्क tq = 0; tq < Q_TILE_SZ; tq++) {
                    प्रति (पूर्णाङ्क tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            यदि (आवरणम्) {
                ggml_vec_सङ्कलय_f32(tile_पङ्क्तिकाः * KV_TILE_SZ, KQ, KQ, mask32);
            }

            तार्किक skip[Q_TILE_SZ] = {};

            प्रति (पूर्णाङ्क tq = 0; tq < Q_TILE_SZ; tq++) {
                भिन्न * kq_row = KQ + tq * KV_TILE_SZ;

                भिन्न tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                यदि (tile_max == -INFINITY) {
                    skip[tq] = सत्य;
                    अनुवर्त;
                }

                ध्रुव भिन्न Mold = M[tq];
                ध्रुव भिन्न Mnew = fmaxf(Mold, tile_max);

                यदि (Mnew > Mold) {
                    ध्रुव भिन्न ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            प्रति (पूर्णाङ्क tk = 0; tk < kv_tile; tk++) {
                ध्रुव वर्ण * v_दत्त = (ध्रुव वर्ण *)v->दत्त + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                यदि (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_गन्तव्यम्_fp32((ध्रुव ggml_fp16_t *)v_दत्त, V32 + tk * DV, DV);
                } अन्यथा {
                    स्मृतिं_प्रतिलिख(V32 + tk * DV, v_दत्त, DV * परिमाण(भिन्न));
                }
            }
            प्रति (पूर्णाङ्क tq = 0; tq < Q_TILE_SZ; tq++) {
                यदि (skip[tq]) {
                    स्मृतिं_पूरय(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * परिमाण(भिन्न));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        यदि (sinks) {
            ध्रुव भिन्न s = ((भिन्न *)((वर्ण *) sinks->दत्त))[h];

            प्रति (पूर्णाङ्क tq = 0; tq < tile_पङ्क्तिकाः; tq++) {
                भिन्न ms = 1.0f;
                भिन्न vs = 1.0f;

                यदि (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } अन्यथा {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        प्रति (पूर्णाङ्क tq = 0; tq < tile_पङ्क्तिकाः; tq++) {
            // V /= S
            ध्रुव भिन्न S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ध्रुव पूर्णाङ्क i1 = iq1 + tq;
            ध्रुव पूर्णाङ्क i2 = iq2;
            ध्रुव पूर्णाङ्क i3 = iq3;

            // permute(0, 2, 1, 3)
            स्मृतिं_प्रतिलिख((वर्ण *) dst->दत्त + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_पङ्क्तिकाः;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
स्थिर शून्य ggml_flash_attn_ext_reduce_partials(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        ध्रुव int64_t n_chunks,
        ध्रुव int64_t chunk_आकार) {

    ध्रुव ggml_tensor * q = dst->src[0];
    ध्रुव ggml_tensor * k = dst->src[1];
    ध्रुव ggml_tensor * v = dst->src[2];

    ध्रुव int64_t DK        = k->ne[0];
    ध्रुव int64_t DV        = v->ne[0];
    ध्रुव int64_t nek1      = k->ne[1];
    ध्रुव int64_t n_q_heads = q->ne[2];

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t wdata_प्रत्येकम्_thread = DK + 2*DV + CACHE_पङ्क्तिः_आकार_F32;
    भिन्न *       thread_wdata     = (भिन्न *) params->wdata + ith * wdata_प्रत्येकम्_thread;

    ध्रुव int64_t partials_विस्थापनम्  = nth * (DK + 2*DV + CACHE_पङ्क्तिः_आकार_F32);
    ध्रुव int64_t partial_आकार     = 2 + DV;
    ध्रुव भिन्न * partials_आधारः    = (ध्रुव भिन्न *) params->wdata + partials_विस्थापनम्;

    // Output layout
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    प्रति (int64_t q_शीर्ष = ith; q_शीर्ष < n_q_heads; q_शीर्ष += nth) {
        भिन्न   M_final   = -INFINITY;
        भिन्न   S_final   = 0.0f;
        भिन्न * VKQ_final = thread_wdata;
        स्मृतिं_पूरय(VKQ_final, 0, DV * परिमाण(भिन्न));

        // Combine partials from all chunks
        प्रति (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ध्रुव int64_t ic_start = chunk_idx * chunk_आकार;
            यदि (ic_start >= nek1) अनुवर्त;

            ध्रुव भिन्न * partial   = partials_आधारः + (q_शीर्ष * n_chunks + chunk_idx) * partial_आकार;
            ध्रुव भिन्न   M_chunk   = partial[0];
            ध्रुव भिन्न   S_chunk   = partial[1];
            ध्रुव भिन्न * VKQ_chunk = partial + 2;

            यदि (S_chunk == 0.0f) अनुवर्त;

            ध्रुव भिन्न M_new     = fmaxf(M_final, M_chunk);
            ध्रुव भिन्न scale_old = expf(M_final - M_new);
            ध्रुव भिन्न scale_new = expf(M_chunk - M_new);

            प्रति (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        यदि (S_final != 0.0f) {
            ध्रुव भिन्न S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        स्मृतिं_प्रतिलिख((वर्ण *) dst->दत्त + (0*ne2*ne1 + q_शीर्ष + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

स्थिर शून्य ggml_compute_forward_flash_attn_ext_f16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * q     = dst->src[0];
    ध्रुव ggml_tensor * k     = dst->src[1];
    ध्रुव ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    ध्रुव int64_t DK = nek0;
    ध्रुव int64_t DV = nev0;
    ध्रुव int64_t N  = neq1;


    GGML_ASSERT(ne0 == DV);
    GGML_ASSERT(ne2 == N);

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_आकार(q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(k->type));
    GGML_ASSERT(nbv0 == ggml_type_आकार(v->type));

    GGML_ASSERT(neq0 == DK);
    GGML_ASSERT(nek0 == DK);
    GGML_ASSERT(nev0 == DV);

    GGML_ASSERT(neq1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    ध्रुव तार्किक use_ref = params->use_ref;

    ध्रुव तार्किक kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    ध्रुव तार्किक use_विभज_kv_मार्गः = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    यदि (use_विभज_kv_मार्गः) {
        ध्रुव int64_t chunk_आकार = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ध्रुव int64_t partial_आकार  = 2 + DV;
        भिन्न *       partials_आधारः = (भिन्न *) params->wdata + nth * (DK + 2*DV + CACHE_पङ्क्तिः_आकार_F32);

        ध्रुव int64_t ic_start = ith * chunk_आकार;
        ध्रुव int64_t ic_अन्तः   = std::min(ic_start + chunk_आकार, nek1);

        ध्रुव int64_t partial_stride = nth * partial_आकार;
        भिन्न *       chunk_partials = partials_आधारः + ith * partial_आकार;

        यदि (ic_start < nek1) {
            प्रति (int64_t q_शीर्ष = 0; q_शीर्ष < neq2; q_शीर्ष++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_शीर्ष, q_शीर्ष + 1, ic_start, ic_अन्तः,
                    chunk_partials, partial_stride);
            }
        } अन्यथा {
            प्रति (int64_t q_शीर्ष = 0; q_शीर्ष < neq2; q_शीर्ष++) {
                भिन्न * q_partials = chunk_partials + q_शीर्ष * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_आकार);
    } अन्यथा {

        // total rows in q
        ध्रुव int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ध्रुव तार्किक disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        पूर्णाङ्क nth_scaled = nth * 4;
        int64_t chunk_आकार = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_आकार - 1) / chunk_आकार;

        यदि (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        यदि (ith == 0) {
            ggml_threadpool_chunk_समुच्चयः(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        ध्रुव int64_t dr = (nr + nchunk - 1) / nchunk;

        स्थिर constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        तार्किक use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#यदि defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#यदि defined(__ARM_FEATURE_SVE)
        ध्रुव int64_t f32_epr = svcntw();
#अन्यथा
        ध्रुव int64_t f32_epr = GGML_F32_EPR;
#यदि_अन्तः
        use_tiled &= (DV % f32_epr == 0);
#यदि_अन्तः
        पूर्णाङ्क current_chunk = ith;

        यावत् (current_chunk < nchunk) {
            ध्रुव int64_t ir0 = dr * current_chunk;
            ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

            यदि (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } अन्यथा {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, शून्यसूचकः, 0);
            }

            current_chunk = ggml_threadpool_chunk_सङ्कलय(params->threadpool, 1);
        }
    }
}

शून्य ggml_compute_forward_flash_attn_ext(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    विकल्प (dst->op_params[3]) {
        पक्ष GGML_PREC_DEFAULT:
        पक्ष GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

स्थिर शून्य ggml_compute_forward_flash_attn_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
        ध्रुव तार्किक masked,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * q = dst->src[0];
    ध्रुव ggml_tensor * k = dst->src[1];
    ध्रुव ggml_tensor * v = dst->src[2];
    ध्रुव ggml_tensor * d = dst->src[3];

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

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t D = neq0;
    ध्रुव int64_t N = neq1;
    ध्रुव int64_t P = nek1 - N;
    ध्रुव int64_t M = P + N;

    ध्रुव पूर्णाङ्क Mup  = ggml_ऊर्ध्वम्(M, GGML_SOFT_MAX_UNROLL);
    ध्रुव पूर्णाङ्क mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == परिमाण(भिन्न));
    GGML_ASSERT(nbk0 == परिमाण(भिन्न));
    GGML_ASSERT(nbv0 == परिमाण(भिन्न));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    यदि (ith == 0) {
        स्मृतिं_पूरय(dst->दत्त, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ध्रुव int64_t elem_q = ggml_nelements(q);
    ध्रुव int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_आकार(result_type) == 1);
    ध्रुव size_t tsize = ggml_type_आकार(result_type);

    ध्रुव size_t offs_q = 0;
    ध्रुव size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ध्रुव size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    शून्य * grad_q = (वर्ण *) dst->दत्त;
    शून्य * grad_k = (वर्ण *) dst->दत्त + offs_k;
    शून्य * grad_v = (वर्ण *) dst->दत्त + offs_v;

    ध्रुव size_t nbgq1 = nb0*neq0;
    ध्रुव size_t nbgq2 = nb0*neq0*neq1;
    ध्रुव size_t nbgq3 = nb0*neq0*neq1*neq2;

    ध्रुव size_t nbgk1 = nb0*nek0;
    ध्रुव size_t nbgk2 = nb0*nek0*nek1;
    ध्रुव size_t nbgk3 = nb0*nek0*nek1*neq2;

    ध्रुव size_t nbgv1 = nb0*nev0;
    ध्रुव size_t nbgv2 = nb0*nev0*nev1;
    ध्रुव size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    ध्रुव पूर्णाङ्क nr = nek2*nek3;

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    ध्रुव भिन्न scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    पूर्णाङ्क nrep = neq2/nek2;

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // q indices
        ध्रुव पूर्णाङ्क ik3 = ir/(nek2);
        ध्रुव पूर्णाङ्क ik2 = ir - ik3*nek2;

        ध्रुव पूर्णाङ्क iq3 = ik3;
        ध्रुव पूर्णाङ्क id3 = ik3;
        ध्रुव पूर्णाङ्क iv3 = ik3;
        ध्रुव पूर्णाङ्क iv2 = ik2;

        प्रति (पूर्णाङ्क irep = 0; irep < nrep; ++irep) {
            ध्रुव पूर्णाङ्क iq2 = ik2 + irep*nek2;
            ध्रुव पूर्णाङ्क id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            प्रति (पूर्णाङ्क iq1 = 0; iq1 < neq1; ++iq1) {
                ध्रुव पूर्णाङ्क id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                भिन्न * S  = (भिन्न *) params->wdata + ith*2*(mxDM + CACHE_पङ्क्तिः_आकार_F32) + 0*(mxDM+CACHE_पङ्क्तिः_आकार_F32);
                भिन्न * SM = (भिन्न *) params->wdata + ith*2*(mxDM + CACHE_पङ्क्तिः_आकार_F32) + 1*(mxDM+CACHE_पङ्क्तिः_आकार_F32);

                प्रति (पूर्णाङ्क i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                ध्रुव int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                प्रति (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    ध्रुव पूर्णाङ्क ik1 = ic;

                    // S indices
                    ध्रुव पूर्णाङ्क i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (भिन्न *) ((वर्ण *) k->दत्त + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (भिन्न *) ((वर्ण *) q->दत्त + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                प्रति (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    भिन्न max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float योग_2 = 0.0;
                    {
#यदि_परिभाषितम् GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_योग_f32(Mup, &योग_2, SM);
#अन्यथा
                        योग_2 = ggml_vec_soft_max_f32(Mup, SM, S, max);
#यदि_अन्तः
                    }

                    प्रतिज्ञा(योग_2 > 0.0);

                    योग_2 = 1.0/योग_2;
                    ggml_vec_scale_f32(masked_begin, SM, योग_2);

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
                ggml_vec_समुच्चयः_f32(masked_begin, S, 0);
                प्रति (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (भिन्न *) ((वर्ण *) v->दत्त + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(भिन्न *) ((वर्ण *) d->दत्त + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                भिन्न dot_SM_gradSM = 0;
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
                प्रति (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (भिन्न *) ((वर्ण *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (भिन्न *) ((वर्ण *) k->दत्त + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                प्रति (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (भिन्न *) ((वर्ण *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (भिन्न *) ((वर्ण *) q->दत्त + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                प्रति (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (भिन्न *) ((वर्ण *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(भिन्न *) ((वर्ण *) d->दत्त + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_flash_attn_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ध्रुव तार्किक masked,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * q = dst->src[0];

    विकल्प (q->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_पृष्ठभागः_f32(params, masked, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

स्थिर शून्य ggml_compute_forward_ssm_conv_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0]; // conv_x
    ध्रुव ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nc  = src1->ne[0]; // d_conv
    ध्रुव पूर्णाङ्क ncs = src0->ne[0]; // d_conv - 1 + n_t
    ध्रुव पूर्णाङ्क nr  = src0->ne[1]; // d_inner
    ध्रुव पूर्णाङ्क n_t =  dst->ne[1]; // tokens per sequence
    ध्रुव पूर्णाङ्क n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src1->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);
    ध्रुव पूर्णाङ्क ir  = ir1 - ir0;

    प्रति (पूर्णाङ्क i3 = 0; i3 < n_s; ++i3) {
        प्रति (पूर्णाङ्क i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            ध्रुव भिन्न * s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0->दत्त + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ध्रुव भिन्न * c = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1->दत्त + ir0*(src1->nb[1])); // {d_conv, d_inner}
            भिन्न * x = (भिन्न *) ((वर्ण *) dst->दत्त + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            प्रति (पूर्णाङ्क i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                भिन्न sumf = 0.0f;

                // d_conv
                प्रति (पूर्णाङ्क i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

शून्य ggml_compute_forward_ssm_conv(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    विकल्प (dst->src[0]->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

स्थिर शून्य ggml_compute_forward_ssm_अवलोकनम्_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    ध्रुव ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    ध्रुव ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    ध्रुव ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    ध्रुव ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    ध्रुव ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    ध्रुव ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t nc = src0->ne[0]; // d_state
    ध्रुव int64_t nr = src0->ne[1]; // dim
    ध्रुव int64_t nh = src1->ne[1]; // n_head
    ध्रुव int64_t ng = src4->ne[1];
    ध्रुव int64_t nt = src1->ne[2]; // number of tokens per sequence
    ध्रुव int64_t ns = src1->ne[3]; // number of sequences in the batch
    ध्रुव int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    ध्रुव int64_t s_off = ggml_nelements(src1) * ggml_element_आकार(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src1->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src2->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src3->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src4->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src5->nb[0] == परिमाण(भिन्न));
    GGML_ASSERT(src6->nb[0] == परिमाण(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    ध्रुव पूर्णाङ्क dh = (nh + nth - 1)/nth;

    // head range for this thread
    ध्रुव पूर्णाङ्क ih0 = dh*ith;
    ध्रुव पूर्णाङ्क ih1 = MIN(ih0 + dh, nh);

    ध्रुव int32_t * ids = (ध्रुव int32_t *) src6->दत्त;

    प्रति (पूर्णाङ्क i3 = 0; i3 < ns; ++i3) {
        ध्रुव भिन्न * s0 = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0->दत्त + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              भिन्न * s  = (      भिन्न *) ((      वर्ण *) dst->दत्त  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        प्रति (पूर्णाङ्क i2 = 0; i2 < nt; ++i2) {
            ध्रुव भिन्न * x  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1->दत्त + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ध्रुव भिन्न * dt = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src2->दत्त + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ध्रुव भिन्न * A  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src3->दत्त); // {d_state, nh} or {1, nh}
            ध्रुव भिन्न * B  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src4->दत्त + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ध्रुव भिन्न * C  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src5->दत्त + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  भिन्न * y  = (      भिन्न *) ((      वर्ण *) dst->दत्त + i2*(nh*nr*परिमाण(भिन्न)) + i3*(nt*nh*nr*परिमाण(भिन्न))); // {dim, nh, nt, ns}

            यदि (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                प्रति (पूर्णाङ्क h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ध्रुव भिन्न dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ध्रुव भिन्न dA = expf(dt_soft_plus * A[h]);
                    ध्रुव पूर्णाङ्क g = h / (nh / ng); // repeat_interleave

                    // dim
                    प्रति (पूर्णाङ्क i1 = 0; i1 < nr; ++i1) {
                        ध्रुव पूर्णाङ्क ii = i1 + h*nr;
                        ध्रुव भिन्न x_dt = x[ii] * dt_soft_plus;
                        भिन्न sumf = 0.0f;
#यदि defined(GGML_SIMD)
    #यदि defined(__ARM_FEATURE_SVE)
                        ध्रुव पूर्णाङ्क ggml_f32_epr = svcntw();
                        ध्रुव पूर्णाङ्क ggml_f32_चरणः = 1 * ggml_f32_epr;

                        ध्रुव पूर्णाङ्क np = (nc & ~(ggml_f32_चरणः - 1));

                        GGML_F32_VEC योग_2 = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        प्रति (पूर्णाङ्क i = 0; i < np; i += ggml_f32_चरणः) {
                            // TODO: maybe unroll more?
                            प्रति (पूर्णाङ्क j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_आरोपय(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_आरोपय(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_आरोपय(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_सङ्कलय(t0, t1);

                                योग_2 = GGML_F32_VEC_FMA(योग_2, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(योग_2);
    #नोचेद्यदि defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ध्रुव पूर्णाङ्क np = 0;
    #अन्यथा
                        ध्रुव पूर्णाङ्क np = (nc & ~(GGML_F32_चरणः - 1));

                        GGML_F32_VEC योग_2[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        प्रति (पूर्णाङ्क i = 0; i < np; i += GGML_F32_चरणः) {
                            प्रति (पूर्णाङ्क j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_आरोपय(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_आरोपय(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_आरोपय(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_सङ्कलय(ax[j], ay[j]);

                                योग_2[j] = GGML_F32_VEC_FMA(योग_2[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, योग_2);
    #यदि_अन्तः
#अन्यथा
                        ध्रुव पूर्णाङ्क np = 0;
#यदि_अन्तः
                        // d_state
                        प्रति (पूर्णाङ्क i0 = np; i0 < nc; ++i0) {
                            ध्रुव पूर्णाङ्क i = i0 + ii*nc;
                            ध्रुव पूर्णाङ्क ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ध्रुव भिन्न state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } अन्यथा {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                प्रति (पूर्णाङ्क h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    ध्रुव भिन्न dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    ध्रुव पूर्णाङ्क g = h / (nh / ng); // repeat_interleave

                    // dim
                    प्रति (पूर्णाङ्क i1 = 0; i1 < nr; ++i1) {
                        ध्रुव पूर्णाङ्क ii = i1 + h*nr;
                        ध्रुव भिन्न x_dt = x[ii] * dt_soft_plus;
#यदि defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_सदिशम् = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        प्रति (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_आरोपय(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_आरोपय(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_आरोपय(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_आरोपय(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_सदिशम् = GGML_F32_VEC_सङ्कलय(GGML_F32_VEC_MUL(vs0, vC), r1_सदिशम्);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_सदिशम्);
#अन्यथा
                        भिन्न sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        प्रति (पूर्णाङ्क i0 = 0; i0 < nc; ++i0) {
                            ध्रुव पूर्णाङ्क i = i0 + ii*nc;
                            ध्रुव पूर्णाङ्क ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            ध्रुव भिन्न state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#यदि_अन्तः
                    }
                }
            }
            ध्रुव int64_t slot = nt - 1 - i2;
            यदि (K > 1 && slot > 0 && slot < K) {
                भिन्न * s_snapshot = (भिन्न *) ((वर्ण *) dst->दत्त + s_off + (slot*ns + i3)*(src0->nb[3]));
                प्रति (पूर्णाङ्क h = ih0; h < ih1; ++h) {
                    स्मृतिं_प्रतिलिख((वर्ण *) s_snapshot + h*src0->nb[2], (वर्ण *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

शून्य ggml_compute_forward_ssm_अवलोकनम्(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    विकल्प (dst->src[0]->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_अवलोकनम्_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

स्थिर शून्य ggml_compute_forward_win_part_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_अप्रयुक्तम्(params);

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ध्रुव int32_t nep0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t nep1 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t w    = ((ध्रुव int32_t *)(dst->op_params))[2];

    प्रतिज्ञा(ne00 == ne0);
    प्रतिज्ञा(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    प्रति (पूर्णाङ्क py = 0; py < nep1; ++py) {
        प्रति (पूर्णाङ्क px = 0; px < nep0; ++px) {
            ध्रुव int64_t i3 = py*nep0 + px;
            प्रति (int64_t i2 = 0; i2 < ne2; ++i2) {
                प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        ध्रुव int64_t i02 = py*w + i2;
                        ध्रुव int64_t i01 = px*w + i1;
                        ध्रुव int64_t i00 = i0;

                        ध्रुव int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        ध्रुव int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        यदि (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((भिन्न *) dst->दत्त)[i] = 0.0f;
                        } अन्यथा {
                            ((भिन्न *) dst->दत्त)[i] = ((भिन्न *) src0->दत्त)[j];
                        }
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_win_part(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

स्थिर शून्य ggml_compute_forward_win_unpart_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_अप्रयुक्तम्(params);

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    ध्रुव int32_t w = ((ध्रुव int32_t *)(dst->op_params))[0];

    // padding
    ध्रुव पूर्णाङ्क px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    ध्रुव पूर्णाङ्क npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    प्रतिज्ञा(ne0 == ne00);

    // TODO: optimize / multi-thread
    प्रति (int64_t i2 = 0; i2 < ne2; ++i2) {
        प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
            प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                ध्रुव पूर्णाङ्क ip2 = i2/w;
                ध्रुव पूर्णाङ्क ip1 = i1/w;

                ध्रुव int64_t i02 = i2%w;
                ध्रुव int64_t i01 = i1%w;
                ध्रुव int64_t i00 = i0;

                ध्रुव int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                ध्रुव int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((भिन्न *) dst->दत्त)[j] = ((भिन्न *) src0->दत्त)[i];
            }
        }
    }
}

शून्य ggml_compute_forward_win_unpart(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

शून्य ggml_compute_forward_unary(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_unary_op op = ggml_get_unary_op(dst);

    विकल्प (op) {
        पक्ष GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_चरणः:
            {
                ggml_compute_forward_चरणः(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_GELU_शीघ्र:
            {
                ggml_compute_forward_gelu_शीघ्र(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_आवर्तनम्:
            {
                ggml_compute_forward_आवर्तनम्(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } भङ्ग;
        पक्ष GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

शून्य ggml_compute_forward_glu(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_glu_op op = ggml_get_glu_op(dst);

    विकल्प (op) {
        पक्ष GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } भङ्ग;
        पक्ष GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } भङ्ग;
        पक्ष GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } भङ्ग;
        पक्ष GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } भङ्ग;
        पक्ष GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } भङ्ग;
        पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
            {
                ggml_compute_forward_geglu_शीघ्र(params, dst);
            } भङ्ग;
        पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

स्थिर शून्य ggml_compute_forward_get_rel_pos_f16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_अप्रयुक्तम्(params);

    ध्रुव ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव int64_t w = ne1;

    ggml_fp16_t * src0_दत्त = (ggml_fp16_t *) src0->दत्त;
    ggml_fp16_t * dst_दत्त  = (ggml_fp16_t *) dst->दत्त;

    प्रति (int64_t i2 = 0; i2 < ne2; ++i2) {
        प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
            ध्रुव int64_t pos = (w - i1 - 1) + i2;
            प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_दत्त[i2*ne1*ne0 + i1*ne0 + i0] = src0_दत्त[pos*ne00 + i0];
            }
        }
    }
}

शून्य ggml_compute_forward_get_rel_pos(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

स्थिर शून्य ggml_compute_forward_सङ्कलय_rel_pos_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    ध्रुव तार्किक inplace = (तार्किक) ((int32_t *) dst->op_params)[0];
    यदि (!inplace) {
        यदि (params->ith == 0) {
            स्मृतिं_प्रतिलिख((वर्ण *) dst->दत्त, (वर्ण *) src0->दत्त, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    भिन्न * src1_दत्त = (भिन्न *) src1->दत्त;
    भिन्न * src2_दत्त = (भिन्न *) src2->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    ध्रुव int64_t ne10 = src1->ne[0];
    ध्रुव int64_t ne11 = src1->ne[1];
    ध्रुव int64_t ne12 = src1->ne[2];
    ध्रुव int64_t ne13 = src1->ne[3];

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // total patches in dst
    ध्रुव पूर्णाङ्क np = ne13;

    // patches per thread
    ध्रुव पूर्णाङ्क dp = (np + nth - 1)/nth;

    // patch range for this thread
    ध्रुव पूर्णाङ्क ip0 = dp*ith;
    ध्रुव पूर्णाङ्क ip1 = MIN(ip0 + dp, np);

    प्रति (int64_t i13 = ip0; i13 < ip1; ++i13) {
        प्रति (int64_t i12 = 0; i12 < ne12; ++i12) {
            प्रति (int64_t i11 = 0; i11 < ne11; ++i11) {
                ध्रुव int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                प्रति (int64_t i10 = 0; i10 < ne10; ++i10) {
                    ध्रुव int64_t jp0  = jp1 + i10;
                    ध्रुव भिन्न src1_e = src1_दत्त[jp0];
                    ध्रुव भिन्न src2_e = src2_दत्त[jp0];

                    ध्रुव int64_t jdh = jp0 * ne10;
                    ध्रुव int64_t jdw = jdh - (ne10 - 1) * i10;

                    प्रति (int64_t j = 0; j < ne10; ++j) {
                        dst_दत्त[jdh + j     ] += src2_e;
                        dst_दत्त[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_सङ्कलय_rel_pos(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_सङ्कलय_rel_pos_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

स्थिर शून्य ggml_compute_forward_rwkv_wkv6_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव int64_t T = dst->src[1]->ne[2];
    ध्रुव int64_t C = dst->ne[0];
    ध्रुव int64_t HEADS = dst->src[1]->ne[1];
    ध्रुव int64_t n_seqs = dst->src[5]->ne[1];
    ध्रुव int64_t शीर्ष_आकार = C / HEADS;

    भिन्न * dst_दत्त = (भिन्न *) dst->दत्त;
    भिन्न * state = ((भिन्न *) dst->दत्त) + C * T;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क h_start =  (HEADS * (ith    )) / nth;
    ध्रुव पूर्णाङ्क h_अन्तः   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भिन्न * k =          (भिन्न *) dst->src[0]->दत्त;
    भिन्न * v =          (भिन्न *) dst->src[1]->दत्त;
    भिन्न * r =          (भिन्न *) dst->src[2]->दत्त;
    भिन्न * time_faaaa = (भिन्न *) dst->src[3]->दत्त;
    भिन्न * time_decay = (भिन्न *) dst->src[4]->दत्त;

    size_t t_stride = HEADS * शीर्ष_आकार; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = शीर्ष_आकार * शीर्ष_आकार;

    यदि (ith == 0) {
        स्मृतिं_पूरय(dst_दत्त, 0, T * C * परिमाण(भिन्न));
    }
    ggml_barrier(params->threadpool);


    #यदि defined(__AVX__) && !defined(__AVX512F__)
        #परिभाषय GGML_F32X GGML_F32x8
        #परिभाषय GGML_F32X_SET1 GGML_F32x8_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32x8_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32x8_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32x8_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32x8_FMA
        #परिभाषय WKV_सदिशम्_आकार 8
    #नोचेद्यदि defined(__AVX512F__)
        #परिभाषय GGML_F32X GGML_F32x16
        #परिभाषय GGML_F32X_SET1 GGML_F32x16_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32x16_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32x16_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32x16_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32x16_FMA
        #परिभाषय WKV_सदिशम्_आकार 16
    #नोचेद्यदि defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #परिभाषय GGML_F32X GGML_F32xt
        #परिभाषय GGML_F32X_SET1 GGML_F32xt_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32xt_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32xt_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32xt_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32xt_FMA
        #परिभाषय WKV_सदिशम्_आकार 8
    #नोचेद्यदि defined(__ARM_NEON) && defined(__aarch64__)
        #परिभाषय GGML_F32X GGML_F32x4
        #परिभाषय GGML_F32X_SET1 GGML_F32x4_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32x4_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32x4_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32x4_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32x4_FMA
        #परिभाषय WKV_सदिशम्_आकार 4
    #यदि_अन्तः

    #यदि_परिभाषितम् WKV_सदिशम्_आकार
        पूर्णाङ्क wkv_सदिशम्_आकार;
        #यदि defined(__ARM_FEATURE_SVE)
            wkv_सदिशम्_आकार = svcntw();
        #अन्यथा
            wkv_सदिशम्_आकार = WKV_सदिशम्_आकार;
        #यदि_अन्तः
        ध्रुव int64_t vec_गणनम् = शीर्ष_आकार / wkv_सदिशम्_आकार;

        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_विस्थापनम् = t * t_stride;
            size_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_विस्थापनम्;
            भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[5]->दत्त + state_विस्थापनम्;

            प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                size_t h_विस्थापनम् = h * h_stride;
                size_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                size_t h_2d_विस्थापनम् = h * h_stride_2d;

                प्रति (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    size_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + i;
                    size_t h_i_विस्थापनम् = h_विस्थापनम् + i;
                    size_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + i * h_stride;

                    भिन्न k_val = k[t_h_i_विस्थापनम्];
                    भिन्न r_val = r[t_h_i_विस्थापनम्];
                    भिन्न time_faaaa_val = time_faaaa[h_i_विस्थापनम्];
                    भिन्न time_decay_val = time_decay[t_h_i_विस्थापनम्];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    प्रति (int64_t j = 0; j < vec_गणनम्; j++) {
                        size_t आधारः_j = j * wkv_सदिशम्_आकार;
                        size_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + आधारः_j;
                        size_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + आधारः_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_आरोपय(&v[t_h_j_विस्थापनम्]);
                        GGML_F32X पूर्व_state_vec = GGML_F32_X_आरोपय(&state_पूर्व[h_2d_i_j_विस्थापनम्]);
                        GGML_F32X dst_vec = GGML_F32_X_आरोपय(&dst_दत्त[t_h_j_विस्थापनम्]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X अस्थायी_vec = GGML_F32X_FMA(पूर्व_state_vec, kv_vec, time_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, अस्थायी_vec, r_vec);
                        GGML_F32X_STORE(&dst_दत्त[t_h_j_विस्थापनम्], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, पूर्व_state_vec, time_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_विस्थापनम्], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    प्रति (int64_t j = vec_गणनम् * wkv_सदिशम्_आकार; j < शीर्ष_आकार; j++) {
                        size_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                        size_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;
                        भिन्न v_val = v[t_h_j_विस्थापनम्];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                        भिन्न अस्थायी_val = kv_val * time_faaaa_val + पूर्व_state_val;
                        dst_दत्त[t_h_j_विस्थापनम्] += अस्थायी_val * r_val;
                        state_cur[h_2d_i_j_विस्थापनम्] = पूर्व_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #अन्यथा
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_विस्थापनम् = t * t_stride;
            size_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_विस्थापनम्;
            भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[5]->दत्त + state_विस्थापनम्;

            प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                size_t h_विस्थापनम् = h * h_stride;
                size_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                size_t h_2d_विस्थापनम् = h * h_stride_2d;

                प्रति (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    size_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + i;
                    size_t h_i_विस्थापनम् = h_विस्थापनम् + i;
                    size_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + i * h_stride;

                    भिन्न k_val = k[t_h_i_विस्थापनम्];
                    भिन्न r_val = r[t_h_i_विस्थापनम्];
                    भिन्न time_faaaa_val = time_faaaa[h_i_विस्थापनम्];
                    // RWKV v6: different time_decay for each token.
                    भिन्न time_decay_val = time_decay[t_h_i_विस्थापनम्];

                    प्रति (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        size_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                        size_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;

                        भिन्न v_val = v[t_h_j_विस्थापनम्];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                        भिन्न अस्थायी_val = kv_val * time_faaaa_val + पूर्व_state_val;
                        dst_दत्त[t_h_j_विस्थापनम्] += अस्थायी_val * r_val;
                        state_cur[h_2d_i_j_विस्थापनम्] = पूर्व_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #यदि_अन्तः
}


शून्य ggml_compute_forward_rwkv_wkv6(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

स्थिर शून्य ggml_compute_forward_gla_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव int64_t T = dst->src[1]->ne[2];
    ध्रुव int64_t C = dst->ne[0];
    ध्रुव int64_t HEADS = dst->src[1]->ne[1];
    ध्रुव int64_t n_seqs = dst->src[4]->ne[1];
    ध्रुव int64_t शीर्ष_आकार = C / HEADS;
    ध्रुव भिन्न scale = ggml_get_op_params_f32(dst, 0);

    भिन्न * dst_दत्त = (भिन्न *) dst->दत्त;
    भिन्न * state = ((भिन्न *) dst->दत्त) + C * T;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क h_start =  (HEADS * (ith    )) / nth;
    ध्रुव पूर्णाङ्क h_अन्तः   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भिन्न * k = (भिन्न *) dst->src[0]->दत्त;
    भिन्न * v = (भिन्न *) dst->src[1]->दत्त;
    भिन्न * q = (भिन्न *) dst->src[2]->दत्त;
    भिन्न * g = (भिन्न *) dst->src[3]->दत्त;

    size_t t_stride = HEADS * शीर्ष_आकार; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = शीर्ष_आकार * शीर्ष_आकार;

    यदि (ith == 0) {
        स्मृतिं_पूरय(dst_दत्त, 0, T * C * परिमाण(भिन्न));
    }
    ggml_barrier(params->threadpool);


    #यदि defined(__AVX__) && !defined(__AVX512F__)
        #परिभाषय GGML_F32X GGML_F32x8
        #परिभाषय GGML_F32X_SET1 GGML_F32x8_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32x8_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32x8_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32x8_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32x8_FMA
        #परिभाषय GLA_सदिशम्_आकार 8
    #नोचेद्यदि defined(__AVX512F__)
        #परिभाषय GGML_F32X GGML_F32x16
        #परिभाषय GGML_F32X_SET1 GGML_F32x16_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32x16_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32x16_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32x16_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32x16_FMA
        #परिभाषय GLA_सदिशम्_आकार 16
    #नोचेद्यदि defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #परिभाषय GGML_F32X GGML_F32xt
        #परिभाषय GGML_F32X_SET1 GGML_F32xt_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32xt_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32xt_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32xt_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32xt_FMA
        #परिभाषय GLA_सदिशम्_आकार 8
    #नोचेद्यदि defined(__ARM_NEON) && defined(__aarch64__)
        #परिभाषय GGML_F32X GGML_F32x4
        #परिभाषय GGML_F32X_SET1 GGML_F32x4_SET1
        #परिभाषय GGML_F32_X_आरोपय GGML_F32x4_आरोपय
        #परिभाषय GGML_F32X_STORE GGML_F32x4_STORE
        #परिभाषय GGML_F32X_MUL GGML_F32x4_MUL
        #परिभाषय GGML_F32X_FMA GGML_F32x4_FMA
        #परिभाषय GLA_सदिशम्_आकार 4
    #यदि_अन्तः

    #यदि_परिभाषितम् GLA_सदिशम्_आकार
        पूर्णाङ्क gla_सदिशम्_आकार;
        #यदि defined(__ARM_FEATURE_SVE)
            gla_सदिशम्_आकार = svcntw();
        #अन्यथा
            gla_सदिशम्_आकार = GLA_सदिशम्_आकार;
        #यदि_अन्तः
        ध्रुव int64_t vec_गणनम् = शीर्ष_आकार / gla_सदिशम्_आकार;

        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_विस्थापनम् = t * t_stride;
            size_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_विस्थापनम्;
            भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[4]->दत्त + state_विस्थापनम्;

            प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                size_t h_विस्थापनम् = h * h_stride;
                size_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                size_t h_2d_विस्थापनम् = h * h_stride_2d;

                प्रति (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    size_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + i;
                    size_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + i * h_stride;

                    भिन्न k_val = k[t_h_i_विस्थापनम्];
                    भिन्न q_val = q[t_h_i_विस्थापनम्] * scale;
                    भिन्न g_val = g[t_h_i_विस्थापनम्];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    प्रति (int64_t j = 0; j < vec_गणनम्; j++) {
                        size_t आधारः_j = j * gla_सदिशम्_आकार;
                        size_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + आधारः_j;
                        size_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + आधारः_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32_X_आरोपय(&v[t_h_j_विस्थापनम्]);
                        GGML_F32X पूर्व_state_vec = GGML_F32_X_आरोपय(&state_पूर्व[h_2d_i_j_विस्थापनम्]);
                        GGML_F32X dst_vec = GGML_F32_X_आरोपय(&dst_दत्त[t_h_j_विस्थापनम्]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X अस्थायी_vec = GGML_F32X_FMA(kv_vec, पूर्व_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, अस्थायी_vec, q_vec);
                        GGML_F32X_STORE(&dst_दत्त[t_h_j_विस्थापनम्], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_विस्थापनम्], अस्थायी_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    प्रति (int64_t j = vec_गणनम् * gla_सदिशम्_आकार; j < शीर्ष_आकार; j++) {
                        size_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                        size_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;
                        भिन्न v_val = v[t_h_j_विस्थापनम्];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                        भिन्न अस्थायी_val = kv_val + पूर्व_state_val * g_val;
                        dst_दत्त[t_h_j_विस्थापनम्] += अस्थायी_val * q_val;
                        state_cur[h_2d_i_j_विस्थापनम्] = अस्थायी_val;
                    }
                }
            }
        }

    #अन्यथा
        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_विस्थापनम् = t * t_stride;
            size_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_विस्थापनम्;
            भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[4]->दत्त + state_विस्थापनम्;

            प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                size_t h_विस्थापनम् = h * h_stride;
                size_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                size_t h_2d_विस्थापनम् = h * h_stride_2d;

                प्रति (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    size_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + i;
                    size_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + i * h_stride;

                    भिन्न k_val = k[t_h_i_विस्थापनम्];
                    भिन्न q_val = q[t_h_i_विस्थापनम्] * scale;
                    भिन्न g_val = g[t_h_i_विस्थापनम्];

                    प्रति (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        size_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                        size_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;

                        भिन्न v_val = v[t_h_j_विस्थापनम्];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                        भिन्न अस्थायी_val = पूर्व_state_val * g_val + kv_val;
                        dst_दत्त[t_h_j_विस्थापनम्] += अस्थायी_val * q_val;
                        state_cur[h_2d_i_j_विस्थापनम्] = अस्थायी_val;
                    }
                }
            }
        }
    #यदि_अन्तः
}


शून्य ggml_compute_forward_gla(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_solve_tri_f32(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) {
    ध्रुव संरचना ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    ध्रुव संरचना ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_द्विमान_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t k = ne10;   // number of RHS columns
    ध्रुव int64_t n = ne11;   // A is n×n
    ध्रुव int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    ध्रुव भिन्न * A = (ध्रुव भिन्न *) src0->दत्त;  // [n, n, B1, B2]
    ध्रुव भिन्न * B = (ध्रुव भिन्न *) src1->दत्त;  // [n, k, B1, B2]
          भिन्न * X = (      भिन्न *) dst->दत्त;   // [n, k, B1, B2]

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*k);
        ध्रुव int64_t i02 = (ir - i03*ne02*k)/k;
        ध्रुव int64_t i01 = (ir - i03*ne02*k - i02*k);

        ध्रुव भिन्न * A_batch = A + i02 * nb02 / परिमाण(भिन्न) + i03 * nb03 / परिमाण(भिन्न);
        ध्रुव भिन्न * B_batch = B + i02 * nb12 / परिमाण(भिन्न) + i03 * nb13 / परिमाण(भिन्न);

        भिन्न * X_batch = X + i02 * nb2 / परिमाण(भिन्न) + i03 * nb3 / परिमाण(भिन्न);

        प्रति (int64_t i00 = 0; i00 < n; ++i00) {
            भिन्न योग_2 = 0.0f;
            प्रति (int64_t t = 0; t < i00; ++t) {
                योग_2 += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ध्रुव भिन्न diag = A_batch[i00 * n + i00];
            प्रतिज्ञा(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - योग_2) / diag;
        }
    }
}

शून्य ggml_compute_forward_solve_tri(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } अन्यथा {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
स्थिर शून्य ggml_compute_forward_gated_delta_net_one_chunk(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    ध्रुव int64_t S_v      = src_v->ne[0];
    ध्रुव int64_t H        = src_v->ne[1];
    ध्रुव int64_t n_tokens = src_v->ne[2];
    ध्रुव int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src_q));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src_k));
    GGML_ASSERT(ggml_is_contiguous_पङ्क्तिकाः(src_v));
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

    ध्रुव तार्किक kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    ध्रुव int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    ध्रुव int64_t state_seq_stride = src_state->nb[3] / परिमाण(भिन्न);

    ध्रुव int64_t प्रत्येकम्_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ध्रुव पूर्णाङ्क ith = params->ith;

    भिन्न * delta       = (भिन्न *)params->wdata + ith * प्रत्येकम्_thread + CACHE_पङ्क्तिः_आकार_F32;
    भिन्न * state_work  = K > 1 ? (delta + S_v) : शून्यसूचकः;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ध्रुव int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    ध्रुव int64_t state_आकार_प्रत्येकम्_snap = S_v * S_v * H * n_seqs;
    भिन्न * attn_बहिर्गमनम्_आधारः  = (भिन्न *)dst->दत्त;
    भिन्न * state_बहिर्गमनम्_आधारः = (भिन्न *)dst->दत्त + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ध्रुव भिन्न * state_अन्तर्_आधारः = (ध्रुव भिन्न *)src_state->दत्त;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    ध्रुव int64_t rq3 = nev3 / neq3;
    ध्रुव int64_t rk3 = nev3 / nek3;

    ध्रुव भिन्न scale = 1.0f / sqrtf((भिन्न) S_v);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t iv1 = ir % H; // head_index
        ध्रुव int64_t iv3 = ir / H; // sequence

        ध्रुव int64_t iq1 = iv1 % neq1;
        ध्रुव int64_t ik1 = iv1 % nek1;

        ध्रुव int64_t iq3 = iv3 / rq3;
        ध्रुव int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        भिन्न * s_बहिर्गमनम् = (K > 1)
            ? state_work
            : state_बहिर्गमनम्_आधारः + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ध्रुव भिन्न * s_अन्तर् = state_अन्तर्_आधारः + iv3 * state_seq_stride + iv1 * S_v * S_v;
        स्मृतिं_प्रतिलिख(s_बहिर्गमनम्, s_अन्तर्, S_v * S_v * परिमाण(भिन्न));

        // attn output pointer for first token of this (head, seq)
        भिन्न * attn_दत्त = attn_बहिर्गमनम्_आधारः + (iv3 * n_tokens * H + iv1) * S_v;

        प्रति (int64_t t = 0; t < n_tokens; t++) {
            ध्रुव भिन्न * q_d = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_q->दत्त + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ध्रुव भिन्न * k_d = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_k->दत्त + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ध्रुव भिन्न * v_d = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_v->दत्त + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ध्रुव भिन्न beta_val = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src_beta->दत्त + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ध्रुव भिन्न * g_d    =  (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_g->दत्त    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            यदि (kda) {
                // precompute exp(g) into delta scratch (reused below)
                प्रति (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                प्रति (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_बहिर्गमनम्[j * S_v], &s_बहिर्गमनम्[j * S_v], delta);
                }
            } अन्यथा {
                ggml_vec_scale_f32(S_v * S_v, s_बहिर्गमनम्, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            प्रति (int64_t j = 0; j < S_v; ++j) {
                भिन्न योग_2 = 0.0f;
                ggml_vec_dot_f32(S_v, &योग_2, 0, &s_बहिर्गमनम्[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - योग_2) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            प्रति (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_बहिर्गमनम्[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            प्रति (int64_t j = 0; j < S_v; ++j) {
                भिन्न योग_2 = 0.0f;
                ggml_vec_dot_f32(S_v, &योग_2, 0, &s_बहिर्गमनम्[j * S_v], 0, q_d, 0, 1);
                attn_दत्त[j] = योग_2 * scale;
            }

            attn_दत्त += S_v * H; // advance to next token

            यदि (K > 1) {
                ध्रुव int64_t target_slot = n_tokens - 1 - t;
                यदि (target_slot >= 0 && target_slot < K) {
                    भिन्न * curr_state_o = state_बहिर्गमनम्_आधारः + target_slot * state_आकार_प्रत्येकम्_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    स्मृतिं_प्रतिलिख(curr_state_o, s_बहिर्गमनम्, S_v * S_v * परिमाण(भिन्न));
                }
            }
        }
    }
}


स्थिर शून्य ggml_compute_forward_gated_delta_net_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    ध्रुव तार्किक disable_chunking = ggml_is_numa();

    पूर्णाङ्क nth = params->nth;
    पूर्णाङ्क ith = params->ith;

    // 4x chunks per thread
    पूर्णाङ्क nth_scaled = nth * 4;
    int64_t chunk_आकार = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_आकार - 1) / chunk_आकार;

    यदि (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    यदि (ith == 0) {
      ggml_threadpool_chunk_समुच्चयः(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ध्रुव int64_t dr = (nr + nchunk - 1) / nchunk;

    पूर्णाङ्क current_chunk = ith;

    यावत् (current_chunk < nchunk) {
        ध्रुव int64_t ir0 = dr * current_chunk;
        ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_सङ्कलय(params->threadpool, 1);
    }
}

शून्य ggml_compute_forward_gated_delta_net(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

स्थिर शून्य ggml_dsv4_hc_comb_norm_स्तम्भसङ्ख्या(भिन्न * comb, भिन्न eps) {
    constexpr int64_t hc = 4;

    प्रति (int64_t idst = 0; idst < hc; ++idst) {
        भिन्न योग_2 = eps;
        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            योग_2 += comb[idst + hc*isrc];
        }

        ध्रुव भिन्न inv_योग = 1.0f / योग_2;
        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_योग;
        }
    }
}

स्थिर शून्य ggml_dsv4_hc_comb_norm_पङ्क्तिकाः(भिन्न * comb, भिन्न eps) {
    constexpr int64_t hc = 4;

    प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
        भिन्न योग_2 = eps;
        प्रति (int64_t idst = 0; idst < hc; ++idst) {
            योग_2 += comb[idst + hc*isrc];
        }

        ध्रुव भिन्न inv_योग = 1.0f / योग_2;
        प्रति (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_योग;
        }
    }
}

स्थिर शून्य ggml_compute_forward_dsv4_hc_comb_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * mixes = dst->src[0];
    ध्रुव ggml_tensor * scale = dst->src[1];
    ध्रुव ggml_tensor * आधारः  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(आधारः->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_विस्थापनम् = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ध्रुव int64_t n_tokens = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(आधारः->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, आधारः,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ध्रुव भिन्न eps = ggml_get_op_params_f32(dst, 0);
    ध्रुव int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t dr  = (n_tokens + nth - 1) / nth;
    ध्रुव int64_t it0 = dr * ith;
    ध्रुव int64_t it1 = MIN(it0 + dr, n_tokens);

    ध्रुव भिन्न scale_comb = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) scale->दत्त + 2*nbs0);

    प्रति (int64_t it = it0; it < it1; ++it) {
        भिन्न comb[hc*hc];

        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            भिन्न max = -INFINITY;
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                ध्रुव भिन्न xv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) mixes->दत्त + (comb_विस्थापनम् + idx)*nbm0 + it*nbm1);
                ध्रुव भिन्न bv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) आधारः->दत्त  + (comb_विस्थापनम् + idx)*nbb0);
                ध्रुव भिन्न v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            भिन्न योग_2 = 0.0f;
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                ध्रुव भिन्न v = expf(comb[idx] - max);
                comb[idx] = v;
                योग_2 += v;
            }

            ध्रुव भिन्न inv_योग = 1.0f / योग_2;
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_योग + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_स्तम्भसङ्ख्या(comb, eps);
        प्रति (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_पङ्क्तिकाः(comb, eps);
            ggml_dsv4_hc_comb_norm_स्तम्भसङ्ख्या(comb, eps);
        }

        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                *(भिन्न *) ((वर्ण *) dst->दत्त + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

शून्य ggml_compute_forward_dsv4_hc_comb(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

स्थिर शून्य ggml_compute_forward_dsv4_hc_pre_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * x       = dst->src[0];
    ध्रुव ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ध्रुव int64_t n_embd   = x->ne[0];
    ध्रुव int64_t hc       = x->ne[1];
    ध्रुव int64_t n_tokens = x->ne[2];

    ध्रुव भिन्न scale = ggml_get_op_params_f32(dst, 0);
    ध्रुव तार्किक  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    यदि (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } अन्यथा {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t nr  = n_embd * n_tokens;
    ध्रुव int64_t dr  = (nr + nth - 1) / nth;
    ध्रुव int64_t ir0 = dr * ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i0 = ir % n_embd;
        ध्रुव int64_t it = ir / n_embd;

        भिन्न योग_2 = 0.0f;
        प्रति (int64_t ih = 0; ih < hc; ++ih) {
            ध्रुव भिन्न xv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) x->दत्त + i0*nbx0 + ih*nbx1 + it*nbx2);
            भिन्न wv;
            यदि (gated) {
                ध्रुव भिन्न gv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) weights->दत्त + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } अन्यथा {
                wv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) weights->दत्त + ih*nbw0 + it*nbw1);
            }
            योग_2 += xv * wv;
        }

        *(भिन्न *) ((वर्ण *) dst->दत्त + i0*nbd0 + it*nbd1) = scale * योग_2;
    }
}

शून्य ggml_compute_forward_dsv4_hc_pre(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

स्थिर शून्य ggml_compute_forward_dsv4_hc_post_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * x        = dst->src[0];
    ध्रुव ggml_tensor * residual = dst->src[1];
    ध्रुव ggml_tensor * post     = dst->src[2];
    ध्रुव ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ध्रुव int64_t n_embd   = x->ne[0];
    ध्रुव int64_t n_tokens = x->ne[1];
    ध्रुव int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_tokens);

    // comb == NULL: identity mixing, each stream keeps its own residual
    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    यदि (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t nr  = n_embd * hc * n_tokens;
    ध्रुव int64_t dr  = (nr + nth - 1) / nth;
    ध्रुव int64_t ir0 = dr * ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i0     = ir % n_embd;
        ध्रुव int64_t idst   = (ir / n_embd) % hc;
        ध्रुव int64_t it     = ir / (n_embd * hc);

        ध्रुव भिन्न xv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) x->दत्त    + i0*nbx0 + it*nbx1);
        ध्रुव भिन्न pv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) post->दत्त + idst*nbp0 + it*nbp1);

        भिन्न योग_2 = xv * pv;
        यदि (comb) {
            प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
                ध्रुव भिन्न rv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) residual->दत्त + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ध्रुव भिन्न cv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) comb->दत्त     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                योग_2 += rv * cv;
            }
        } अन्यथा {
            योग_2 += *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) residual->दत्त + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(भिन्न *) ((वर्ण *) dst->दत्त + i0*nbd0 + idst*nbd1 + it*nbd2) = योग_2;
    }
}

शून्य ggml_compute_forward_dsv4_hc_post(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

शून्य ggml_compute_forward_शिखरम्_k_आवरणम्(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * आवरणम्   = dst->src[0];
    ध्रुव ggml_tensor * शिखरम्_k  = dst->src[1];
    ध्रुव ggml_tensor * score  = dst->src[2];
    ध्रुव ggml_tensor * खण्डानि = dst->src[3];
    ध्रुव ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == आवरणम्->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_आकार(dst->type));

    ध्रुव int64_t n_kv    = आवरणम्->ne[0];
    ध्रुव int64_t ne1     = dst->ne[1];
    ध्रुव int64_t ne2     = dst->ne[2];
    ध्रुव int64_t n_पङ्क्तिकाः  = ggml_nrows(dst);
    ध्रुव int64_t n_शिखरम्   = शिखरम्_k->ne[0];
    ध्रुव int64_t n_blk   = खण्डानि ? खण्डानि->ne[0] : 1;
    ध्रुव int64_t n_pick  = खण्डानि ? खण्डानि->ne[1] : score ? score->ne[0] : n_kv;
    ध्रुव int64_t n_extra = extra ? extra->ne[0] : 0;

    ध्रुव size_t ts = ggml_type_आकार(dst->type);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    प्रति (int64_t ir = ith; ir < n_पङ्क्तिकाः; ir += nth) {
        ध्रुव int64_t i1 = ir % ne1;
        ध्रुव int64_t i2 = (ir / ne1) % ne2;
        ध्रुव int64_t i3 = ir / (ne1*ne2);

        ध्रुव वर्ण * आवरणम्_row = (ध्रुव वर्ण *) आवरणम्->दत्त
            + (i1 % आवरणम्->ne[1])*आवरणम्->nb[1] + (i2 % आवरणम्->ne[2])*आवरणम्->nb[2] + (i3 % आवरणम्->ne[3])*आवरणम्->nb[3];
        वर्ण * dst_row = (वर्ण *) dst->दत्त + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        यदि (dst->type == GGML_TYPE_F16) {
            ध्रुव ggml_fp16_t ninf = GGML_CPU_FP32_गन्तव्यम्_FP16(-INFINITY);
            प्रति (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } अन्यथा {
            प्रति (int64_t c = 0; c < n_kv; ++c) {
                ((भिन्न *) dst_row)[c] = -INFINITY;
            }
        }

        स्वचालित keep = [&](int32_t c) {
            यदि (c >= 0 && c < n_kv) {
                स्मृतिं_प्रतिलिख(dst_row + c*ts, आवरणम्_row + c*ts, ts);
            }
        };

        ध्रुव वर्ण * शिखरम्_k_row = (ध्रुव वर्ण *) शिखरम्_k->दत्त + i1*शिखरम्_k->nb[1] + i2*शिखरम्_k->nb[2] + i3*शिखरम्_k->nb[3];

        प्रति (int64_t j = 0; j < n_शिखरम्; ++j) {
            ध्रुव int32_t p = *(ध्रुव int32_t *) (शिखरम्_k_row + j*शिखरम्_k->nb[0]);
            यदि (p < 0 || p >= n_pick) {
                अनुवर्त;
            }
            यदि (score) {
                ध्रुव भिन्न s = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) score->दत्त
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                यदि (!(s > -INFINITY)) {
                    अनुवर्त;
                }
            }
            यदि (!खण्डानि) {
                keep(p);
                अनुवर्त;
            }
            प्रति (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ध्रुव int32_t *) ((ध्रुव वर्ण *) खण्डानि->दत्त + k*खण्डानि->nb[0] + p*खण्डानि->nb[1]));
            }
        }

        यदि (extra) {
            ध्रुव वर्ण * extra_row = (ध्रुव वर्ण *) extra->दत्त + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            प्रति (int64_t j = 0; j < n_extra; ++j) {
                keep(*(ध्रुव int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

स्थिर शून्य ggml_compute_forward_rwkv_wkv7_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव int64_t T = dst->src[1]->ne[2];
    ध्रुव int64_t C = dst->ne[0];
    ध्रुव int64_t HEADS = dst->src[1]->ne[1];
    ध्रुव int64_t n_seqs = dst->src[6]->ne[1];
    ध्रुव int64_t शीर्ष_आकार = C / HEADS;

    भिन्न * dst_दत्त = (भिन्न *) dst->दत्त;
    भिन्न * state = ((भिन्न *) dst->दत्त) + C * T;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क h_start =  (HEADS * (ith    )) / nth;
    ध्रुव पूर्णाङ्क h_अन्तः   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भिन्न * r = (भिन्न *) dst->src[0]->दत्त;
    भिन्न * w = (भिन्न *) dst->src[1]->दत्त;
    भिन्न * k = (भिन्न *) dst->src[2]->दत्त;
    भिन्न * v = (भिन्न *) dst->src[3]->दत्त;
    भिन्न * a = (भिन्न *) dst->src[4]->दत्त;
    भिन्न * b = (भिन्न *) dst->src[5]->दत्त;

    int64_t t_stride = HEADS * शीर्ष_आकार; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = शीर्ष_आकार * शीर्ष_आकार;

    #यदि defined(GGML_SIMD)
        #यदि defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            प्रति (int64_t t = 0; t < T; t++) {
                int64_t t_विस्थापनम् = t * t_stride;
                int64_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
                भिन्न * state_cur = state + state_विस्थापनम्;
                भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[6]->दत्त + state_विस्थापनम्;

                प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                    int64_t h_विस्थापनम् = h * h_stride;
                    int64_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                    int64_t h_2d_विस्थापनम् = h * h_stride_2d;

                    प्रति (int64_t i = 0; i < शीर्ष_आकार; i++) {
                        int64_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + i;
                        int64_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + i * h_stride;

                        भिन्न v_val = v[t_h_i_विस्थापनम्];

                        भिन्न sa = 0, परिणाम = 0;
                        प्रति (int64_t j = 0; j < शीर्ष_आकार; j++) {
                            sa += a[t_h_विस्थापनम् + j] * state_पूर्व[h_2d_i_विस्थापनम् + j];
                        }

                        प्रति (int64_t j = 0; j < शीर्ष_आकार; j++) {
                            int64_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                            int64_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;

                            भिन्न r_val = r[t_h_j_विस्थापनम्];
                            भिन्न w_val = w[t_h_j_विस्थापनम्];
                            भिन्न k_val = k[t_h_j_विस्थापनम्];
                            भिन्न b_val = b[t_h_j_विस्थापनम्];
                            भिन्न kv_val = v_val * k_val;
                            भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                            state_cur[h_2d_i_j_विस्थापनम्] = पूर्व_state_val * w_val + kv_val + sa * b_val;
                            परिणाम += state_cur[h_2d_i_j_विस्थापनम्] * r_val;
                        }
                        dst_दत्त[t_h_i_विस्थापनम्] = परिणाम;
                    }
                }
            }
        #अन्यथा
            प्रति (int64_t t = 0; t < T; t++) {
                int64_t t_विस्थापनम् = t * t_stride;
                int64_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
                भिन्न * state_cur = state + state_विस्थापनम्;
                भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[6]->दत्त + state_विस्थापनम्;

                प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                    int64_t h_विस्थापनम् = h * h_stride;
                    int64_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                    int64_t h_2d_विस्थापनम् = h * h_stride_2d;

                    प्रति (int64_t ii = 0; ii < शीर्ष_आकार; ii++) {
                        int64_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + ii;
                        int64_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_विस्थापनम्]);

                        भिन्न sa = 0;
                        {
                            GGML_F32_VEC योग_2[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            प्रति (int64_t j = 0; j < शीर्ष_आकार; j += GGML_F32_चरणः) {
                                प्रति (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_आरोपय(&a[t_h_विस्थापनम् + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_आरोपय(&state_पूर्व[h_2d_i_विस्थापनम् + j + kk * GGML_F32_EPR]);
                                    योग_2[kk] = GGML_F32_VEC_FMA(योग_2[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, योग_2);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC परिणाम_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        प्रति (; j < शीर्ष_आकार; j += GGML_F32_चरणः) {
                            प्रति (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_आरोपय(&r[t_h_j_विस्थापनम्]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_आरोपय(&w[t_h_j_विस्थापनम्]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_आरोपय(&k[t_h_j_विस्थापनम्]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_आरोपय(&b[t_h_j_विस्थापनम्]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_आरोपय(&state_पूर्व[h_2d_i_j_विस्थापनम्]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_विस्थापनम्], state_vec);

                                परिणाम_vec[kk] = GGML_F32_VEC_FMA(परिणाम_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_दत्त[t_h_i_विस्थापनम्], परिणाम_vec);

                        // There shouldn't be left-overs though.
                        प्रति (; j < शीर्ष_आकार; j++) {
                            int64_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                            int64_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;

                            भिन्न r_val = r[t_h_j_विस्थापनम्];
                            भिन्न w_val = w[t_h_j_विस्थापनम्];
                            भिन्न k_val = k[t_h_j_विस्थापनम्];
                            भिन्न b_val = b[t_h_j_विस्थापनम्];
                            भिन्न kv_val = v[t_h_i_विस्थापनम्] * k_val;

                            भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                            state_cur[h_2d_i_j_विस्थापनम्] = पूर्व_state_val * w_val + kv_val + sa * b_val;
                            dst_दत्त[t_h_i_विस्थापनम्] += state_cur[h_2d_i_j_विस्थापनम्] * r_val;
                        }
                    }
                }
            }
        #यदि_अन्तः
    #अन्यथा
        प्रति (int64_t t = 0; t < T; t++) {
            int64_t t_विस्थापनम् = t * t_stride;
            int64_t state_विस्थापनम् = शीर्ष_आकार * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_विस्थापनम्;
            भिन्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[6]->दत्त + state_विस्थापनम्;

            प्रति (int64_t h = h_start; h < h_अन्तः; h++) {
                int64_t h_विस्थापनम् = h * h_stride;
                int64_t t_h_विस्थापनम् = t_विस्थापनम् + h_विस्थापनम्;
                int64_t h_2d_विस्थापनम् = h * h_stride_2d;

                प्रति (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    int64_t t_h_i_विस्थापनम् = t_h_विस्थापनम् + i;
                    int64_t h_2d_i_विस्थापनम् = h_2d_विस्थापनम् + i * h_stride;

                    भिन्न v_val = v[t_h_i_विस्थापनम्];

                    भिन्न sa = 0, परिणाम = 0;
                    प्रति (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        sa += a[t_h_विस्थापनम् + j] * state_पूर्व[h_2d_i_विस्थापनम् + j];
                    }

                    प्रति (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        int64_t t_h_j_विस्थापनम् = t_h_विस्थापनम् + j;
                        int64_t h_2d_i_j_विस्थापनम् = h_2d_i_विस्थापनम् + j;

                        भिन्न r_val = r[t_h_j_विस्थापनम्];
                        भिन्न w_val = w[t_h_j_विस्थापनम्];
                        भिन्न k_val = k[t_h_j_विस्थापनम्];
                        भिन्न b_val = b[t_h_j_विस्थापनम्];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न पूर्व_state_val = state_पूर्व[h_2d_i_j_विस्थापनम्];
                        state_cur[h_2d_i_j_विस्थापनम्] = पूर्व_state_val * w_val + kv_val + sa * b_val;
                        परिणाम += state_cur[h_2d_i_j_विस्थापनम्] * r_val;
                    }
                    dst_दत्त[t_h_i_विस्थापनम्] = परिणाम;
                }
            }
        }
    #यदि_अन्तः
}


शून्य ggml_compute_forward_rwkv_wkv7(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

शून्य ggml_compute_forward_map_custom1(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * a = dst->src[0];

    संरचना ggml_map_custom1_op_params p;
    स्मृतिं_प्रतिलिख(&p, dst->op_params, परिमाण(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

शून्य ggml_compute_forward_map_custom2(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * a = dst->src[0];
    ध्रुव ggml_tensor * b = dst->src[1];

    संरचना ggml_map_custom2_op_params p;
    स्मृतिं_प्रतिलिख(&p, dst->op_params, परिमाण(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

शून्य ggml_compute_forward_map_custom3(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * a = dst->src[0];
    ध्रुव ggml_tensor * b = dst->src[1];
    ध्रुव ggml_tensor * c = dst->src[2];

    संरचना ggml_map_custom3_op_params p;
    स्मृतिं_प्रतिलिख(&p, dst->op_params, परिमाण(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

शून्य ggml_compute_forward_custom(
    ध्रुव संरचना ggml_compute_params * params,
          संरचना ggml_tensor * dst) {

    संरचना ggml_custom_op_params p;
    स्मृतिं_प्रतिलिख(&p, dst->op_params, परिमाण(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

स्थिर शून्य ggml_compute_forward_cross_entropy_loss_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_आकार(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_आकार(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    ध्रुव int64_t nc = src0->ne[0];
    ध्रुव int64_t nr = ggml_nrows(src0);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    भिन्न * sums =  (भिन्न *) params->wdata;
    भिन्न * st   = ((भिन्न *) params->wdata) + nth + ith*nc;
    भिन्न योग_thread = 0.0f;

    GGML_ASSERT(params->wsize >= परिमाण(भिन्न) * (nth + nth * nc));

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ध्रुव भिन्न * s0 = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src0->दत्त + i1*src0->nb[1]);
        ध्रुव भिन्न * s1 = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src1->दत्त + i1*src1->nb[1]);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(s0[i]));
            प्रतिज्ञा(!isnan(s1[i]));
        }
#यदि_अन्तः // NDEBUG

        भिन्न max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ध्रुव ggml_float योग_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        प्रतिज्ञा(योग_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -योग_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        भिन्न योग_st = 0.0f;
        ggml_vec_योग_f32(nc, &योग_st, st);
        योग_thread += योग_st;

#यदि_अपरिभाषितम् NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(st[i]));
            प्रतिज्ञा(!isinf(st[i]));
        }
#यदि_अन्तः // NDEBUG
    }
    sums[ith] = योग_thread;
    ggml_barrier(params->threadpool);

    यदि (ith == 0) {
        भिन्न * dp = (भिन्न *) dst->दत्त;
        ggml_vec_योग_f32(nth, dp, sums);
        dp[0] *= -1.0f / (भिन्न) nr;
    }
}

शून्य ggml_compute_forward_cross_entropy_loss(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

स्थिर शून्य ggml_compute_forward_cross_entropy_loss_पृष्ठभागः_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    ध्रुव ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    ध्रुव ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    ध्रुव int64_t ith = params->ith;
    ध्रुव int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    ध्रुव int64_t nc = src0f->ne[0];
    ध्रुव int64_t nr = ggml_nrows(src0f);

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    ध्रुव भिन्न d_by_nr = ((ध्रुव भिन्न *) grad->दत्त)[0] / (भिन्न) nr;

    प्रति (int64_t i1 = ir0; i1 < ir1; i1++) {
        भिन्न       * ds0 = (भिन्न       *)((वर्ण       *) dst->दत्त   + i1*dst->nb[1]);
        ध्रुव भिन्न * s0  = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src0f->दत्त + i1*src0f->nb[1]);
        ध्रुव भिन्न * s1  = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src1f->दत्त + i1*src1f->nb[1]);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(s0[i]));
            प्रतिज्ञा(!isnan(s1[i]));
        }
#यदि_अन्तः // NDEBUG

        // soft_max
        भिन्न max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ध्रुव ggml_float योग_2 = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        प्रतिज्ञा(योग_2 > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/योग_2);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#यदि_अपरिभाषितम् NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(ds0[i]));
            प्रतिज्ञा(!isinf(ds0[i]));
        }
#यदि_अन्तः // NDEBUG
    }
}

शून्य ggml_compute_forward_cross_entropy_loss_पृष्ठभागः(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_पृष्ठभागः_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_opt_चरणः_adamw_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0         = dst->src[0];
    ध्रुव ggml_tensor * src0_grad    = dst->src[1];
    ध्रुव ggml_tensor * src0_grad_m  = dst->src[2];
    ध्रुव ggml_tensor * src0_grad_v  = dst->src[3];
    ध्रुव ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    ध्रुव भिन्न * adamw_params_ptr = ggml_get_दत्त_f32(adamw_params);

    ध्रुव भिन्न alpha  = adamw_params_ptr[0];
    ध्रुव भिन्न beta1  = adamw_params_ptr[1];
    ध्रुव भिन्न beta2  = adamw_params_ptr[2];
    ध्रुव भिन्न eps    = adamw_params_ptr[3];
    ध्रुव भिन्न wd     = adamw_params_ptr[4];
    ध्रुव भिन्न beta1h = adamw_params_ptr[5];
    ध्रुव भिन्न beta2h = adamw_params_ptr[6];
    ध्रुव भिन्न keep   = 1.f - alpha * wd;
    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ध्रुव size_t विस्थापनम् = i03*nb03 + i02*nb02 + i01*nb01;

        भिन्न       * w = (भिन्न       *) ((वर्ण       *) src0->दत्त        + विस्थापनम्); // weight
        ध्रुव भिन्न * g = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_grad->दत्त   + विस्थापनम्); // grad
        भिन्न       * m = (भिन्न       *) ((वर्ण       *) src0_grad_m->दत्त + विस्थापनम्);
        भिन्न       * v = (भिन्न       *) ((वर्ण       *) src0_grad_v->दत्त + विस्थापनम्);

        प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            ध्रुव भिन्न mh =       m[i00]*beta1h;
            ध्रुव भिन्न vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

शून्य ggml_compute_forward_opt_चरणः_adamw(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_चरणः_adamw_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_opt_चरणः_sgd_f32(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0       = dst->src[0];
    ध्रुव ggml_tensor * src0_grad  = dst->src[1];
    ध्रुव ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == परिमाण(भिन्न));

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1) / nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    ध्रुव भिन्न * sgd_params_ptr   = ggml_get_दत्त_f32(sgd_params);
    ध्रुव भिन्न   alpha            = sgd_params_ptr[0];
    ध्रुव भिन्न   keep             = 1.f - alpha * sgd_params_ptr[1];

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir / (ne02 * ne01);
        ध्रुव int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ध्रुव int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ध्रुव size_t विस्थापनम् = i03 * nb03 + i02 * nb02 + i01 * nb01;

        भिन्न *       w = (भिन्न *) ((वर्ण *) src0->दत्त + विस्थापनम्);                   // weight
        ध्रुव भिन्न * g = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_grad->दत्त + विस्थापनम्);  // grad

        प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

शून्य ggml_compute_forward_opt_चरणः_sgd(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_चरणः_sgd_f32(params, dst);
            }
            भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

स्थिर अन्तःस्थ भिन्न ggml_fwht_आरोपय(ध्रुव भिन्न value) {
    प्रत्यागम value;
}

स्थिर अन्तःस्थ भिन्न ggml_fwht_आरोपय(ध्रुव ggml_fp16_t value) {
    प्रत्यागम ggml_fp16_गन्तव्यम्_fp32(value);
}

प्रतिरूपम्<प्रकारनाम src_t>
स्थिर शून्य ggml_compute_forward_fwht_impl(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, भिन्न> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    ध्रुव int64_t nr = ne11 * ne12 * ne13;
    ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_thread = (nr + nth - 1) / nth;
    ध्रुव int64_t start_row = ith * पङ्क्तिकाः_प्रत्येकम्_thread;
    ध्रुव int64_t अन्तः_row = MIN(start_row + पङ्क्तिकाः_प्रत्येकम्_thread, nr);

    ध्रुव भिन्न scale = 1.0f / sqrtf((भिन्न)n);

#यदि defined(GGML_SIMD)
    ध्रुव GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#यदि_अन्तः

    प्रति (int64_t r = start_row; r < अन्तः_row; r++) {
        ध्रुव int64_t i13 = r / (ne11 * ne12);
        ध्रुव int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ध्रुव int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ध्रुव src_t * src_row = (ध्रुव src_t *) ((ध्रुव वर्ण *) src1->दत्त + i11 * nb11 + i12 * nb12 + i13 * nb13);
        भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst->दत्त + i11 * nb1 + i12 * nb2 + i13 * nb3);

        प्रति (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_आरोपय(src_row[j]) * scale;
        }

        // Scalar passes
#यदि defined(GGML_SIMD)
#यदि defined(__ARM_FEATURE_SVE)
        ध्रुव पूर्णाङ्क चरणः = svcntw();
#अन्यथा
        ध्रुव पूर्णाङ्क चरणः = GGML_F32_EPR;
#यदि_अन्तः
#अन्यथा
        ध्रुव पूर्णाङ्क चरणः = n;
#यदि_अन्तः
        प्रति (int64_t len = 1; len < चरणः && len < n; len <<= 1) {
            प्रति (int64_t i = 0; i < n; i += 2 * len) {
                प्रति (int64_t j = 0; j < len; j++) {
                    भिन्न u = dst_row[i + j];
                    भिन्न v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#यदि defined(GGML_SIMD)
        प्रति (int64_t len = चरणः; len < n; len <<= 1) {
            प्रति (int64_t i = 0; i < n; i += 2 * len) {
                प्रति (int64_t j = 0; j < len; j += चरणः) {
                    GGML_F32_VEC u = GGML_F32_VEC_आरोपय(dst_row + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_आरोपय(dst_row + i + len + j);

                    GGML_F32_VEC_STORE(dst_row + i + j,       GGML_F32_VEC_सङ्कलय(u, v));
                    GGML_F32_VEC_STORE(dst_row + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#यदि_अन्तः
    }
}

शून्य ggml_compute_forward_fwht(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src1 = dst->src[1];

    विकल्प (src1->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<भिन्न>(params, dst);
            }
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

शून्य ggml_compute_forward_lightning_indexer(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * q = dst->src[0];
    ध्रुव ggml_tensor * k = dst->src[1];
    ध्रुव ggml_tensor * w = dst->src[2]; // weights
    ध्रुव ggml_tensor * m = dst->src[3]; //आवरणम्

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, नूतन,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_आकार(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_आकार(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_आकार(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_आकार(  m->type));

    ध्रुव पूर्णाङ्क n_embd    = q->ne[0];
    ध्रुव पूर्णाङ्क n_शीर्ष    = q->ne[1];
    ध्रुव पूर्णाङ्क n_tokens  = q->ne[2];
    ध्रुव पूर्णाङ्क n_प्रवाहः  = q->ne[3];
    ध्रुव पूर्णाङ्क n_kv      = k->ne[2];

    ggml_गन्तव्यम्_float_t ध्रुव k_गन्तव्यम्_float = ggml_get_type_traits(k->type)->गन्तव्यम्_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_गन्तव्यम्_float) && "lightning indexer: unsupported K-type");

    ध्रुव पूर्णाङ्क nr  = n_kv;
    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // (temporary) buffer for K converted to float
    भिन्न * k_row_f32 = (भिन्न *) params->wdata + ith*(1*n_embd + CACHE_पङ्क्तिः_आकार_F32);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क s = 0; s < n_प्रवाहः; ++s) {
        प्रति (पूर्णाङ्क t = 0; t < n_tokens; ++t) {
            ध्रुव भिन्न       *   w_row =       (भिन्न *) ((वर्ण *)   w->दत्त + t*nbw1 +        s*nbw3);
            ध्रुव ggml_fp16_t *   m_row = (ggml_fp16_t *) ((वर्ण *)   m->दत्त + t*nbm1 + (s%nem3)*nbm3);
            भिन्न             * dst_row =       (भिन्न *) ((वर्ण *) dst->दत्त + t*nb1  +        s*nb3 );
            प्रति (पूर्णाङ्क ik = ir0; ik < ir1; ++ik) {
                वर्ण * k_row = (वर्ण *) k->दत्त + ik*nbk2 + s*nbk3;
                यदि (k_गन्तव्यम्_float) {
                    k_गन्तव्यम्_float(k_row, k_row_f32, n_embd);
                } अन्यथा {
                    k_row_f32 = (भिन्न *) k_row;
                }
                भिन्न score = 0.0f;
                प्रति (पूर्णाङ्क h = 0; h < n_शीर्ष; ++h) {
                    // dot product of q and k for head h
                    भिन्न qk = 0.0f;
                    ध्रुव भिन्न * q_row = (भिन्न *) ((वर्ण *) q->दत्त + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_row, 0, k_row_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_row[h];
                }
                // apply mask
                dst_row[ik] = score + GGML_CPU_FP16_गन्तव्यम्_FP32(m_row[ik]);
            }
        }
    }
}
