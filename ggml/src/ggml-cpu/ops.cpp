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

    ध्रुव size_t nb0 = ggml_type_size(src0->type);

    ध्रुव पूर्णाङ्क ith = params->ith; // thread index
    ध्रुव पूर्णाङ्क nth = params->nth; // number of threads

    // parallelize by blocks
    ध्रुव पूर्णाङ्क nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    ध्रुव पूर्णाङ्क dr = (nk + nth - 1) / nth;
    ध्रुव पूर्णाङ्क k0 = dr * ith;
    ध्रुव पूर्णाङ्क k1 = MIN(k0 + dr, nk);

    यदि (k0 < k1) {
        memcpy(
            ((वर्ण *)  dst->data + k0*nb0),
            ((वर्ण *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
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
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        ध्रुव size_t rs = ne00*nb00;
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((वर्ण *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
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
                वर्ण * dst_ptr = (वर्ण *) dst->data;

                प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
                    प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                            ध्रुव वर्ण * src0_ptr = (वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } अन्यथा {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
                    प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                            ध्रुव src_t * src0_ptr = (src_t *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                                भिन्न tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
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
            dst_t * dst_ptr = (dst_t *) dst->data;

            प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
                प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                        प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                            ध्रुव src_t * src0_ptr = (src_t *) ((वर्ण *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            भिन्न tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
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
                        ध्रुव वर्ण * src0_ptr = ((वर्ण *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              वर्ण * dst_ptr  = ((वर्ण *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, परिमाण(dst_t));

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
                        ध्रुव वर्ण * src0_ptr = ((वर्ण *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              वर्ण * dst_ptr  = ((वर्ण *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        भिन्न tmp = type_conversion_table<src_t>::to_f32(*(ध्रुव src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

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


template<typename src_t>
स्थिर शून्य ggml_compute_forward_dup_to_q(
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
        भिन्न * src0_f32 = (भिन्न *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        वर्ण * dst_ptr = (वर्ण *) dst->data;

        प्रति (पूर्णाङ्क i03 = 0; i03 < ne03; i03++) {
            प्रति (पूर्णाङ्क i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                प्रति (पूर्णाङ्क i01 = ir0; i01 < ir1; i01++) {
                    ध्रुव src_t * src0_ptr = (src_t *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
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
स्थिर शून्य ggml_compute_forward_dup_bytes(
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

    ध्रुव size_t type_size = ggml_type_size(src0->type);

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
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        ध्रुव size_t rs = ggml_row_size(src0->type, ne00);
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((वर्ण *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        प्रत्यागम;
    }

    यदि (ggml_is_contiguous(dst)) {
        size_t id = 0;
        वर्ण * dst_ptr = (वर्ण *) dst->data;
        ध्रुव size_t rs = ne00 * type_size;

        यदि (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
                प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    प्रति (int64_t i01 = ir0; i01 < ir1; i01++) {
                        ध्रुव वर्ण * src0_ptr = (वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
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
                            ध्रुव वर्ण * src0_ptr = (वर्ण *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
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
    ध्रुव int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    ध्रुव int64_t nk0  = ne0  / ggml_blck_size(dst->type);

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
                    ध्रुव वर्ण * src0_ptr = ((वर्ण *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          वर्ण * dst_ptr  = ((वर्ण *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

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
    ggml_to_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    ध्रुव int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
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
        ध्रुव int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        ध्रुव int64_t i13 = i/(ne10 * ne11 * ne12);
        ध्रुव int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        ध्रुव int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        ध्रुव int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        ध्रुव int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (ध्रुव शून्य *) ((वर्ण *) src0->data + x_offset),
                     (भिन्न *) ((वर्ण *)  dst->data + dst_offset), qk);
    }
}

शून्य ggml_compute_forward_dup(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        प्रत्यागम;
    }

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                /**/ यदि (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, भिन्न      >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                /**/ यदि (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, भिन्न      >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                /**/ यदि (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<भिन्न, ggml_fp16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<भिन्न, ggml_bf16_t>(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<भिन्न, भिन्न      >(params, dst);
                अन्यथा यदि (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<भिन्न, int32_t    >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<भिन्न>(params, dst);
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

स्थिर शून्य ggml_compute_forward_add_q_f32(
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
    ggml_to_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t ध्रुव quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
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

    भिन्न * wdata = (भिन्न *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

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

        शून्य  * src0_row = (शून्य *) ((वर्ण *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        भिन्न * src1_row = (भिन्न *)((वर्ण *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        शून्य  * dst_row  = (शून्य *) ((वर्ण *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        प्रतिज्ञा(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        यदि (quantize_row_q != रिक्त) {
            quantize_row_q(wdata, dst_row, ne00);
        } अन्यथा {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

शून्य ggml_compute_forward_add(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
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
                ggml_compute_forward_add_q_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

स्थिर शून्य ggml_compute_forward_add_id_f32(
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
        ध्रुव पूर्णाङ्क i11 = *(int32_t *) ((वर्ण *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (भिन्न *) ((वर्ण *) src1->data + i11*nb11));
    }
}

शून्य ggml_compute_forward_add_id(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
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

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (भिन्न *) ((वर्ण *) src1->data), 0,
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#अन्यथा
        ggml_vec_add1_f32(ne0,
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(भिन्न *) src1->data);
#endif
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
    ध्रुव भिन्न v = *(भिन्न *) src1->data;

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

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
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
    ध्रुव भिन्न v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

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

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
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
    ध्रुव भिन्न v = *(भिन्न *) src1->data;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव ggml_type type = src0->type;
    ggml_to_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t ध्रुव quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

    // we don't support permuted src0
    GGML_ASSERT(nb00 == ggml_type_size(type));

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

    भिन्न * wdata = (भिन्न *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32) * ith;

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        ध्रुव पूर्णाङ्क i3 = ir/(ne2*ne1);
        ध्रुव पूर्णाङ्क i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव पूर्णाङ्क i1 = (ir - i3*ne2*ne1 - i2*ne1);

        शून्य  * src0_row = (शून्य *) ((वर्ण *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        शून्य  * dst_row  = (शून्य *) ((वर्ण *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

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
    ध्रुव भिन्न v = *(भिन्न *) src1->data;

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

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
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
    ध्रुव भिन्न v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

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

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        प्रति (पूर्णाङ्क i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
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
    size_t offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    यदि (!inplace) {
        यदि (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((वर्ण *)  dst->data),
                ((वर्ण *) src0->data),
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
    ध्रुव size_t nb0 = ggml_element_size(src0);

    ध्रुव size_t nb00 = nb0;
    ध्रुव size_t nb01 = nb1;
    ध्रुव size_t nb02 = nb2;
    ध्रुव size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

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

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (भिन्न *) ((वर्ण *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#अन्यथा
        ggml_vec_add_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (भिन्न *) ((वर्ण *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
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

स्थिर शून्य ggml_compute_forward_sum_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum = 0;

    यदि (nb00 == परिमाण(भिन्न)) {
        ggml_float row_sum = 0;
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (भिन्न *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } अन्यथा {
        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव वर्ण * row = (ध्रुव वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(ध्रुव भिन्न *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((भिन्न *) dst->data)[0] = sum;
}

स्थिर शून्य ggml_compute_forward_sum_f16(
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

    भिन्न sum = 0;
    भिन्न row_sum = 0;

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((वर्ण *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

स्थिर शून्य ggml_compute_forward_sum_bf16(
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

    भिन्न sum = 0;
    भिन्न row_sum = 0;

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((वर्ण *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

शून्य ggml_compute_forward_sum(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
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

    ध्रुव auto [ir0, ir1] = get_thread_range(params, src0);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        भिन्न * src_row = (भिन्न *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

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

स्थिर शून्य ggml_compute_forward_sum_rows_f32(
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
                भिन्न * src_row = (भिन्न *) ((वर्ण *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                भिन्न row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

शून्य ggml_compute_forward_sum_rows(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

स्थिर शून्य ggml_compute_forward_mean_f32(
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

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (भिन्न *) ((वर्ण *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (भिन्न *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(भिन्न *) ((वर्ण *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (भिन्न) ne00;
            }
        }
    }
}

शून्य ggml_compute_forward_mean(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
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
        भिन्न * src = (भिन्न *) ((वर्ण *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((वर्ण *)  dst->data + i1*nb0);
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

स्थिर शून्य ggml_compute_forward_मोजणी_equal_i32(
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
    int64_t sum_thread = 0;

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 =  ir                        / (ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne03)            /       ne01;
        ध्रुव int64_t i01 =  ir - i03*ne03 - i02*ne02;

        ध्रुव वर्ण * data0 = (ध्रुव वर्ण *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        ध्रुव वर्ण * data1 = (ध्रुव वर्ण *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        प्रति (int64_t i00 = 0; i00 < ne00; ++i00) {
            ध्रुव int32_t val0 = *((ध्रुव int32_t *) (data0 + i00*nb00));
            ध्रुव int32_t val1 = *((ध्रुव int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    यदि (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    यदि (ith != 0) {
        प्रत्यागम;
    }

    प्रति (पूर्णाङ्क ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

शून्य ggml_compute_forward_मोजणी_equal(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_मोजणी_equal_i32(params, dst);
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
                                        (भिन्न *) ((वर्ण *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (भिन्न *) ((वर्ण *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
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
                                ggml_fp16_t * y = (ggml_fp16_t *) ((वर्ण *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((वर्ण *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
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

स्थिर शून्य ggml_compute_forward_repeat_back_f32(
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
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->data, 0);
    } अन्यथा {
        प्रति         (पूर्णाङ्क k3 = 0; k3 < ne3; k3++) {
            प्रति     (पूर्णाङ्क k2 = 0; k2 < ne2; k2++) {
                प्रति (पूर्णाङ्क k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (भिन्न *) ((वर्ण *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
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
                                        (भिन्न *) ((वर्ण *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (भिन्न *) ((वर्ण *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_repeat_back(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
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
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    यदि (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } अन्यथा {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne03; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne02; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne01; i1++) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      वर्ण * y = (      वर्ण *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    प्रति (पूर्णाङ्क i3 = 0; i3 < ne13; i3++) {
        प्रति (पूर्णाङ्क i2 = ith; i2 < ne12; i2 += nth) {
            प्रति (पूर्णाङ्क i1 = 0; i1 < ne11; i1++) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      वर्ण * y = (      वर्ण *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                memcpy(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_i8(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == परिमाण(int8_t));

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
                        x = (ध्रुव int8_t *) ((ध्रुव वर्ण *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (ध्रुव int8_t *) ((ध्रुव वर्ण *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((वर्ण *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

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

    GGML_ASSERT(ggml_type_size(src0->type) == परिमाण(ggml_fp16_t));

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
                        x = (ध्रुव ggml_fp16_t *) ((ध्रुव वर्ण *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (ध्रुव ggml_fp16_t *) ((ध्रुव वर्ण *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((वर्ण *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

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

    GGML_ASSERT(ggml_type_size(src0->type) == परिमाण(भिन्न));

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
                        x = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    भिन्न * y = (भिन्न *)((वर्ण *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

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
template <शून्य (*vec_op)(ध्रुव पूर्णाङ्क, ggml_bf16_t *, ध्रुव ggml_bf16_t *)>
स्थिर शून्य ggml_compute_forward_unary_bf16_rows(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (ggml_bf16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

स्थिर शून्य ggml_compute_forward_gelu_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (ggml_fp16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
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

    ध्रुव auto [ir0, ir1] = get_thread_range(params, dst);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne2*ne1);
        ध्रुव int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ध्रुव int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        भिन्न * dst_ptr  = (भिन्न *) ((वर्ण *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

स्थिर शून्य ggml_compute_forward_fill_f16(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    ध्रुव auto [ir0, ir1] = get_thread_range(params, dst);

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne2*ne1);
        ध्रुव int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        ध्रुव int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
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

    ध्रुव auto [ir0, ir1] = get_thread_range(params, src0);

    तार्किक (*bipred)(पूर्णाङ्क, पूर्णाङ्क);

    विकल्प (ttype) {
        पक्ष GGML_TRI_TYPE_LOWER:      bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i <  r; }; भङ्ग;
        पक्ष GGML_TRI_TYPE_LOWER_DIAG: bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i <= r; }; भङ्ग;
        पक्ष GGML_TRI_TYPE_UPPER:      bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i >  r; }; भङ्ग;
        पक्ष GGML_TRI_TYPE_UPPER_DIAG: bipred = [](पूर्णाङ्क i, पूर्णाङ्क r) { प्रत्यागम i >= r; }; भङ्ग;
        स्वभाव: GGML_ABORT("invalid tri type");
    }

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*ne01);
        ध्रुव int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        ध्रुव int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न  *) ((ध्रुव वर्ण *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              भिन्न * dst_ptr = (      भिन्न  *) ((      वर्ण *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

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

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_erf_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (ggml_fp16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_erf_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
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

स्थिर शून्य ggml_compute_forward_gelu_quick_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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

        ggml_vec_gelu_quick_f32(nc,
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_quick_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_quick_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu_quick(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
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

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (भिन्न *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
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
                (ggml_fp16_t *) ((वर्ण *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
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

    भिन्न negative_slope;
    memcpy(&negative_slope, dst->op_params, परिमाण(भिन्न));

    प्रतिज्ञा(dst->nb[0]  == परिमाण(भिन्न));
    प्रतिज्ञा(src0->nb[0] == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (भिन्न *) ((वर्ण *) dst->data  + i*( dst->nb[1])),
                (भिन्न *) ((वर्ण *) src0->data + i*(src0->nb[1])), negative_slope);
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

    भिन्न negative_slope;
    memcpy(&negative_slope, dst->op_params, परिमाण(भिन्न));

    प्रतिज्ञा(dst->nb[0]  == परिमाण(ggml_fp16_t));
    प्रतिज्ञा(src0->nb[0] == परिमाण(ggml_fp16_t));

    प्रति (पूर्णाङ्क i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((वर्ण *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((वर्ण *) src0->data + i*(src0->nb[1])), negative_slope);
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

स्थिर शून्य ggml_compute_forward_silu_back_f32(
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
                (भिन्न *) ((वर्ण *) dst->data  + i1*( dst->nb[1])),
                (भिन्न *) ((वर्ण *) src1->data + i1*(src1->nb[1])),
                (भिन्न *) ((वर्ण *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_back_f16(
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
                (ggml_fp16_t *) ((वर्ण *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((वर्ण *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((वर्ण *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

शून्य ggml_compute_forward_silu_back(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
template <शून्य (*vec_op)(ध्रुव पूर्णाङ्क, ggml_bf16_t *, ध्रुव ggml_bf16_t *, ध्रुव ggml_bf16_t *)>
स्थिर शून्य ggml_compute_forward_glu_bf16_rows(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1]));

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
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_reglu_f32(nc, (भिन्न *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_reglu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_reglu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
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
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_geglu_f32(nc, (भिन्न *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
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
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_swiglu_f32(nc, (भिन्न *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
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
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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
    ध्रुव भिन्न limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1*src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1*src1_o);
        भिन्न * dst_p  = (भिन्न *) ((वर्ण *) dst->data + i1*(dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = std::min(src0_p[k], limit);
            ध्रुव भिन्न y = std::clamp(src1_p[k], -limit, limit);
            ध्रुव भिन्न out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = dst_p[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_oai_f16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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
    ध्रुव भिन्न limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            ध्रुव भिन्न y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            ध्रुव भिन्न out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
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
    वर्ण *              src0_d = (वर्ण *) src0->data;
    वर्ण *              src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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
    ध्रुव भिन्न   limit   = ggml_get_op_params_f32(dst, 3);

    ध्रुव पूर्णाङ्क dr  = (nr + nth - 1) / nth;
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        भिन्न * src0_p = (भिन्न *) (src0_d + i1 * src0_o);
        भिन्न * src1_p = (भिन्न *) (src1_d + i1 * src1_o);
        भिन्न * dst_p  = (भिन्न *) ((वर्ण *) dst->data + i1 * (dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न gate = std::min(src0_p[k], limit);
            ध्रुव भिन्न up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = dst_p[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif  // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_clamp_f16(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0   = dst->src[0];
    ध्रुव ggml_tensor * src1   = dst->src[1];
    वर्ण *              src0_d = (वर्ण *) src0->data;
    वर्ण *              src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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
    ध्रुव भिन्न   limit   = ggml_get_op_params_f32(dst, 3);

    ध्रुव पूर्णाङ्क dr  = (nr + nth - 1) / nth;
    ध्रुव पूर्णाङ्क ir0 = dr * ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((वर्ण *) dst->data + i1 * (dst->nb[1]));

        यदि (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            ध्रुव भिन्न up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif  // NDEBUG
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
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_geglu_erf_f32(nc, (भिन्न *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_erf_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_erf_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
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

स्थिर शून्य ggml_compute_forward_geglu_quick_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_geglu_quick_f32(nc, (भिन्न *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव भिन्न x = ((भिन्न *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_quick_f16(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    वर्ण * src0_d = (वर्ण *) src0->data;
    वर्ण * src1_d = (वर्ण *) (src1 ? src1->data : src0->data);
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

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((वर्ण *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            ध्रुव ggml_fp16_t x = ((ggml_fp16_t *) ((वर्ण *) dst->data + i1*( dst->nb[1])))[k];
            ध्रुव भिन्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_quick_bf16(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu_quick(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
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
    memcpy(&eps, dst->op_params, परिमाण(भिन्न));

    GGML_ASSERT(eps >= 0.0f);

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                वर्ण * y = (वर्ण *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                यदि (nb00 == परिमाण(भिन्न) && nb0 == परिमाण(भिन्न)) {
                    ध्रुव भिन्न * xf = (ध्रुव भिन्न *) x;

                    भिन्न sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    भिन्न mean = sum/ne00;

                    भिन्न * yf = (भिन्न *) y;
                    भिन्न variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#अन्यथा
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    ध्रुव भिन्न scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } अन्यथा {
                    भिन्न sum = 0.0;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(ध्रुव भिन्न *) (x + i00*nb00);
                    }
                    ध्रुव भिन्न mean = sum/ne00;

                    भिन्न variance = 0.0f;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        ध्रुव भिन्न v = *(ध्रुव भिन्न *) (x + i00*nb00) - mean;
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

template <ggml_rms_norm_fuse_op FUSE_OP>
स्थिर शून्य ggml_compute_forward_rms_norm_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    ध्रुव ggml_tensor * src0 = dst_rms_norm->src[0];
    ध्रुव ggml_tensor * src1 = nullptr;
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
    memcpy(&eps, dst_rms_norm->op_params, परिमाण(भिन्न));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव भिन्न * x = (भिन्न *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                ध्रुव भिन्न mean  = sum/ne00;
                ध्रुव भिन्न scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                प्रतिज्ञा(scale > 0.0f);

                भिन्न * y = (भिन्न *) ((वर्ण *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                यदि constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    ध्रुव int64_t i11 = i01 % ne11;
                    ध्रुव int64_t i12 = i02 % ne12;
                    ध्रुव int64_t i13 = i03 % ne13;
                    ध्रुव भिन्न * w = (भिन्न *) ((वर्ण *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } अन्यथा {
                    memcpy(y, x, ne00 * परिमाण(भिन्न));
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

    GGML_ASSERT(dst_mul != nullptr);
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

स्थिर शून्य ggml_compute_forward_rms_norm_back_f32(
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
    memcpy(&eps, dst->op_params, परिमाण(भिन्न));

    // TODO: optimize
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                ध्रुव int64_t i11 = i01;
                ध्रुव int64_t i12 = i02;
                ध्रुव int64_t i13 = i03;

                ध्रुव भिन्न * dz = (भिन्न *) ((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                ध्रुव भिन्न * x  = (भिन्न *) ((वर्ण *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                ध्रुव भिन्न mean_eps = (भिन्न)(sum_xx)/ne00 + eps;
                ध्रुव भिन्न sum_eps  = (भिन्न)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                ध्रुव भिन्न rrms     = 1.0f / sqrtf(mean_eps);
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
                भिन्न * dx = (भिन्न *) ((वर्ण *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                ध्रुव भिन्न scale_x = (भिन्न) (-sum_xdz) / sum_eps;
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_rms_norm_back(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

स्थिर शून्य ggml_compute_forward_group_norm_f32(
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
    memcpy(&eps, dst->op_params + 1, परिमाण(भिन्न));

    पूर्णाङ्क n_channels = src0->ne[2];
    पूर्णाङ्क n_groups = dst->op_params[0];
    पूर्णाङ्क n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    प्रति (पूर्णाङ्क i = ith; i < n_groups; i += nth) {
        पूर्णाङ्क start = i * n_channels_per_group;
        पूर्णाङ्क end = start + n_channels_per_group;
        यदि (end > n_channels) {
            end = n_channels;
        }
        पूर्णाङ्क step = end - start;

        प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            प्रति (int64_t i02 = start; i02 < end; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव भिन्न * x = (भिन्न *)((वर्ण *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            ध्रुव भिन्न mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            प्रति (int64_t i02 = start; i02 < end; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव भिन्न * x = (भिन्न *)((वर्ण *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    भिन्न * y = (भिन्न *)((वर्ण *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        भिन्न v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            ध्रुव भिन्न variance = sum2 / (ne00 * ne01 * step);
            ध्रुव भिन्न scale = 1.0f / sqrtf(variance + eps);

            प्रति (int64_t i02 = start; i02 < end; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    भिन्न * y = (भिन्न *)((वर्ण *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

शून्य ggml_compute_forward_group_norm(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
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
    memcpy(&eps, dst->op_params, परिमाण(भिन्न));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव वर्ण * x = (ध्रुव वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                    ध्रुव भिन्न xi = *(ध्रुव भिन्न *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                ध्रुव भिन्न scale = 1.0f/fmaxf(sqrtf(sum), eps);

                वर्ण * y = (वर्ण *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                यदि (nb00 == परिमाण(भिन्न) && nb0 == परिमाण(भिन्न)) {
                    memcpy(y, x, ne00 * परिमाण(भिन्न));
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

स्थिर शून्य ggml_compute_forward_out_prod_f32(
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
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->data, 0);
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

                    भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                प्रति (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    ध्रुव int64_t i11 = i01;

                    भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#अन्यथा
                प्रति (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    ध्रुव int64_t i11 = i01;

                    भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_out_prod_q_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव ggml_type type = src0->type;
    ggml_to_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

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
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->data, 0);
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

    भिन्न * wdata = (भिन्न *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32) * ith;

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

            भिन्न * s0 = (भिन्न *) ((वर्ण *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            भिन्न * d  = (भिन्न *) ((वर्ण *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

स्थिर शून्य ggml_compute_forward_out_prod_f16_f32(
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
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भिन्न *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    ध्रुव int64_t nr = ne1*ne2*ne3;
    ध्रुव int64_t dr = (nr + nth - 1)/nth;
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    भिन्न * wdata = (भिन्न *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32) * ith;

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i3 = ir/(ne2*ne1);
        ध्रुव int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        ध्रुव int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ध्रुव int64_t i02 = i2;
        ध्रुव int64_t i03 = i3;

        ध्रुव int64_t i12 = i2;
        ध्रुव int64_t i13 = i3;

        भिन्न * d = (भिन्न *) ((वर्ण *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        प्रति (int64_t i01 = 0; i01 < ne01; ++i01) {
            ध्रुव int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((वर्ण *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            भिन्न * s1 = (भिन्न *) ((वर्ण *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

शून्य ggml_compute_forward_out_prod(
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
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
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

    memcpy(&s, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&b, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

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
            यदि (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((वर्ण *)dst->data + i1*nb1, (वर्ण *)src0->data + i1*nb01, nc * परिमाण(भिन्न));
            }
            ggml_vec_scale_f32(nc, (भिन्न *) ((वर्ण *) dst->data + i1*nb1), s);
        }
    } अन्यथा {
        प्रति (पूर्णाङ्क i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (भिन्न *) ((वर्ण *) dst->data  + i1*nb1),
                (भिन्न *) ((वर्ण *) src0->data + i1*nb1),
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

    memcpy(&s, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&b, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

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
        ध्रुव ggml_bf16_t * x = (ध्रुव ggml_bf16_t *) ((ध्रुव वर्ण *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((वर्ण *)       dst->data  + i1*dst->nb[1]);
        प्रति (पूर्णाङ्क k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
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

स्थिर शून्य ggml_compute_forward_set_f32(
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
    size_t offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    यदि (!inplace) {
        यदि (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((वर्ण *)  dst->data),
                ((वर्ण *) src0->data),
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
    ध्रुव size_t nb0 = ggml_element_size(src0);

    ध्रुव पूर्णाङ्क im0 = (ne10 == 0 ? 0 : ne10-1);
    ध्रुव पूर्णाङ्क im1 = (ne11 == 0 ? 0 : ne11-1);
    ध्रुव पूर्णाङ्क im2 = (ne12 == 0 ? 0 : ne12-1);
    ध्रुव पूर्णाङ्क im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

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
                (भिन्न *) ((वर्ण *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (भिन्न *) ((वर्ण *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

स्थिर शून्य ggml_compute_forward_set_i32(
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
    size_t offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    यदि (!inplace) {
        यदि (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((वर्ण *)  dst->data),
                ((वर्ण *) src0->data),
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
    ध्रुव size_t nb0 = ggml_element_size(src0);

    ध्रुव पूर्णाङ्क im0 = (ne10 == 0 ? 0 : ne10-1);
    ध्रुव पूर्णाङ्क im1 = (ne11 == 0 ? 0 : ne11-1);
    ध्रुव पूर्णाङ्क im2 = (ne12 == 0 ? 0 : ne12-1);
    ध्रुव पूर्णाङ्क im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

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
                (int32_t *) ((वर्ण *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((वर्ण *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

शून्य ggml_compute_forward_set(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
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

स्थिर शून्य ggml_compute_forward_get_rows_q(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव int64_t nc = ne00;
    ध्रुव int64_t nr = ggml_nelements(src1);

    ध्रुव ggml_type type = src0->type;
    ggml_to_float_t ध्रुव dequantize_row_q = ggml_get_type_traits(type)->to_float;

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == ggml_type_size(type));
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
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (ध्रुव शून्य *) ((वर्ण *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (भिन्न *) ((वर्ण *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_f16(
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
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (ध्रुव ggml_fp16_t*) ((वर्ण *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (भिन्न *) ((वर्ण *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_bf16(
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
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (ध्रुव ggml_bf16_t *) ((वर्ण *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (भिन्न *) ((वर्ण *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_f32(
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
        ध्रुव int64_t i01 = *(int32_t *) ((वर्ण *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (भिन्न *) ((वर्ण *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

शून्य ggml_compute_forward_get_rows(
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
                ggml_compute_forward_get_rows_q(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
        पक्ष GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
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

template<typename src_t, typename idx_t>
स्थिर शून्य ggml_compute_forward_set_rows_impl(
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

    ध्रुव size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t ध्रुव from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    प्रति (int64_t i03 = 0; i03 < ne03; ++i03) {
        प्रति (int64_t i02 = 0; i02 < ne02; ++i02) {
            प्रति (int64_t i = ir0; i < ir1; ++i) {
                ध्रुव int64_t i12 = i03%ne12;
                ध्रुव int64_t i11 = i02%ne11;
                ध्रुव int64_t i10 = i;

                ध्रुव int64_t i1 = *(idx_t *) ((वर्ण *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                यदि constexpr (std::is_same_v<src_t, भिन्न>) {
                    from_float(
                            (ध्रुव भिन्न *) ((वर्ण *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((वर्ण *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } अन्यथा यदि constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    यदि (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((वर्ण *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((वर्ण *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } अन्यथा {
                        भिन्न * wdata = (भिन्न *) params->wdata + (nc + CACHE_ओळ_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (ध्रुव ggml_fp16_t *) ((वर्ण *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((वर्ण *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } अन्यथा {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_set_rows(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                यदि (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<भिन्न, int64_t>(params, dst);
                } अन्यथा यदि (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<भिन्न, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                यदि (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } अन्यथा यदि (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
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

स्थिर शून्य ggml_compute_forward_get_rows_back_f32_f16(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == परिमाण(ggml_fp16_t));

    प्रति (पूर्णाङ्क i = 0; i < nr; ++i) {
        ध्रुव पूर्णाङ्क r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        प्रति (पूर्णाङ्क j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((वर्ण *) src0->data + i*src0->nb[1]))[j];
            ((भिन्न *) ((वर्ण *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_back_f32(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    यदि (params->ith != 0) {
        प्रत्यागम;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    ध्रुव पूर्णाङ्क nc = src0->ne[0];
    ध्रुव पूर्णाङ्क nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == परिमाण(भिन्न));

    प्रति (पूर्णाङ्क i = 0; i < nr; ++i) {
        ध्रुव पूर्णाङ्क r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (भिन्न *) ((वर्ण *)  dst->data + r*dst->nb[1]),
                (भिन्न *) ((वर्ण *)  dst->data + r*dst->nb[1]),
                (भिन्न *) ((वर्ण *) src0->data + i*src0->nb[1]));
    }
}

शून्य ggml_compute_forward_get_rows_back(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } भङ्ग;
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
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
                भिन्न * d = (भिन्न *)((वर्ण *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                भिन्न * s = (भिन्न *)((वर्ण *) src0->data + i3*nb03 + i2*nb02);
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

स्थिर शून्य ggml_compute_forward_diag_mask_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        ध्रुव भिन्न value) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क  n_past  = ((int32_t *) dst->op_params)[0];
    ध्रुव तार्किक inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    यदि (!inplace) {
        यदि (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((वर्ण *)  dst->data),
                ((वर्ण *) src0->data),
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
                    *(भिन्न *)((वर्ण *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_diag_mask_inf(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_compute_forward_diag_mask_zero(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
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

    memcpy(&scale,    (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&max_bias, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

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
    ध्रुव uint32_t n_head      = ne02;
    ध्रुव uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    भिन्न * wp = (भिन्न *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

    ध्रुव तार्किक use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    ध्रुव भिन्न * sk = src2 ? (भिन्न *)((वर्ण *) src2->data) : nullptr;

    प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
        प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
            प्रति (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                ध्रुव int64_t i11 = i01;
                ध्रुव int64_t i12 = i02%ne12;
                ध्रुव int64_t i13 = i03%ne13;

                // ALiBi
                ध्रुव uint32_t h = i02; // head
                ध्रुव भिन्न slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                भिन्न * sp = (भिन्न *)((वर्ण *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                भिन्न * dp = (भिन्न *)((वर्ण *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((वर्ण *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : रिक्त;
                भिन्न       * mp_f32 = src1 ? (भिन्न       *)((वर्ण *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : रिक्त;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                यदि (mp_f32) {
                    यदि (use_f16) {
                        प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } अन्यथा {
                        प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    प्रतिज्ञा(!isnan(wp[i]));
                }
#endif // NDEBUG

                भिन्न max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                यदि (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                प्रतिज्ञा(sum > 0.0);

                यदि (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                प्रति (पूर्णाङ्क i = 0; i < ne00; ++i) {
                    प्रतिज्ञा(!isnan(dp[i]));
                    प्रतिज्ञा(!isinf(dp[i]));
                }
#endif // NDEBUG
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

स्थिर शून्य ggml_compute_forward_soft_max_ext_back_f32(
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

    memcpy(&scale,    (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&max_bias, (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

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
        भिन्न *dy = (भिन्न *)((वर्ण *) src0->data + i1*src0->nb[1]);
        भिन्न *y  = (भिन्न *)((वर्ण *) src1->data + i1*src1->nb[1]);
        भिन्न *dx = (भिन्न *)((वर्ण *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        प्रति (पूर्णाङ्क i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(dy[i]));
            प्रतिज्ञा(!isnan(y[i]));
        }
#endif // NDEBUG
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

#ifndef NDEBUG
        प्रति (पूर्णाङ्क i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(dx[i]));
            प्रतिज्ञा(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

शून्य ggml_compute_forward_soft_max_ext_back(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
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
    memcpy(&min, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&max, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

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

        भिन्न * dst_ptr  = (भिन्न *) ((वर्ण *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        भिन्न * src0_ptr = (भिन्न *) ((वर्ण *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

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
    memcpy(&min, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&max, (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

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

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((वर्ण *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((वर्ण *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        प्रति (पूर्णाङ्क i = 0; i < nc; i++) {
            भिन्न v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
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
        पक्ष GGML_TYPE_मोजणी:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

स्थिर भिन्न rope_yarn_ramp(ध्रुव भिन्न खालचा, ध्रुव भिन्न वरचा, ध्रुव पूर्णाङ्क i0) {
    ध्रुव भिन्न y = (i0 / 2 - खालचा) / MAX(0.001f, वरचा - खालचा);
    प्रत्यागम 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
स्थिर शून्य rope_yarn(
    भिन्न theta_extrap, भिन्न freq_scale, भिन्न corr_dims[2], int64_t i0, भिन्न ext_factor, भिन्न mscale,
    भिन्न * cos_theta, भिन्न * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    भिन्न theta_interp = freq_scale * theta_extrap;
    भिन्न theta = theta_interp;
    यदि (ext_factor != 0.0f) {
        भिन्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

स्थिर शून्य ggml_rope_cache_init(
     भिन्न theta_base, भिन्न freq_scale, ध्रुव भिन्न * freq_factors, भिन्न corr_dims[2], int64_t ne0, भिन्न ext_factor, भिन्न mscale,
     भिन्न * cache, भिन्न sin_sign, भिन्न theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    भिन्न theta = theta_base;
    प्रति (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        ध्रुव भिन्न ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

स्थिर शून्य ggml_mrope_cache_init(
     भिन्न theta_base_t, भिन्न theta_base_h, भिन्न theta_base_w, भिन्न theta_base_e, पूर्णाङ्क sections[4], तार्किक is_imrope, तार्किक indep_sects,
     भिन्न freq_scale, ध्रुव भिन्न * freq_factors, भिन्न corr_dims[2], int64_t ne0, भिन्न ext_factor, भिन्न mscale,
     भिन्न * cache, भिन्न sin_sign, भिन्न theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    भिन्न theta_t = theta_base_t;
    भिन्न theta_h = theta_base_h;
    भिन्न theta_w = theta_base_w;
    भिन्न theta_e = theta_base_e;  // extra position id for vision encoder
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
                theta_t = theta_base_t;
            }
            अन्यथा यदि (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            अन्यथा यदि (sector == sec_w) {
                theta_w = theta_base_w;
            }
            अन्यथा यदि (sector == sec_e) {
                theta_e = theta_base_e;
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
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
    }
}


template<typename T>
स्थिर शून्य rotate_pairs(ध्रुव int64_t n, ध्रुव int64_t n_offset, ध्रुव भिन्न * cache, ध्रुव T * src_data, T * dst_data, ध्रुव पूर्णाङ्क scale = 2) {
  प्रति (int64_t i0 = 0; i0 < n; i0 += 2) {
    ध्रुव int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    ध्रुव भिन्न cos_theta = cache[i0 + 0];
    ध्रुव भिन्न sin_theta = cache[i0 + 1];

    ध्रुव T * ध्रुव src = src_data + ic;
    T * dst             = dst_data + ic;

    ध्रुव भिन्न x0 = type_conversion_table<T>::to_f32(src[0]);
    ध्रुव भिन्न x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
स्थिर शून्य ggml_compute_forward_rope_flt(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        ध्रुव तार्किक forward) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    भिन्न freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    पूर्णाङ्क sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    ध्रुव पूर्णाङ्क n_dims     = ((int32_t *) dst->op_params)[1];
    ध्रुव पूर्णाङ्क mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    ध्रुव पूर्णाङ्क n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, परिमाण(भिन्न));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, परिमाण(भिन्न));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, परिमाण(भिन्न));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, परिमाण(भिन्न));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, परिमाण(भिन्न));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, परिमाण(भिन्न));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, परिमाण(पूर्णाङ्क)*4);

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

    ध्रुव भिन्न theta_scale = powf(freq_base, -2.0f/n_dims);

    भिन्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    ध्रुव तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    ध्रुव तार्किक mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    ध्रुव तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

    यदि (mrope_used) {
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
        freq_factors = (ध्रुव भिन्न *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    ध्रुव भिन्न sin_sign = forward ? 1.0f : -1.0f;

    ध्रुव int32_t * pos = (ध्रुव int32_t *) src1->data;

    int64_t last_i2 = -1;

    प्रति (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        प्रति (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            प्रति (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                यदि (ir++ < ir0) अनुवर्त; // skip rows mapped to other threads
                यदि (ir   > ir1) भङ्ग;

                भिन्न * cache = (भिन्न *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32)*ith;
                यदि (last_i2 != i2) {
                    यदि (!mrope_used) {
                        ध्रुव int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    अन्यथा {
                        ध्रुव int64_t p_t = pos[i2];
                        ध्रुव int64_t p_h = pos[i2 + ne2];
                        ध्रुव int64_t p_w = pos[i2 + ne2 * 2];
                        ध्रुव int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((वर्ण *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                विकल्प (mode) {
                    पक्ष GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        भङ्ग;
                    पक्ष GGML_ROPE_TYPE_NEOX:
                    पक्ष GGML_ROPE_TYPE_MROPE:
                    पक्ष GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        भङ्ग;
                    पक्ष GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
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
                        ध्रुव T * ध्रुव src = (T *)((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((वर्ण *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
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

शून्य ggml_compute_forward_rope_back(
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
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) params->wdata + 0;

            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव ggml_fp16_t * ध्रुव src = (ggml_fp16_t *)((वर्ण *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            प्रति (int64_t i11 = 0; i11 < ne11; i11++) {
                ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src1->data + i11*nb11);
                प्रति (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
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
        भिन्न * dst_data = (भिन्न *)((वर्ण *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
            ध्रुव पूर्णाङ्क i1n = i10*ne11;
            प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                भिन्न v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
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
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            भिन्न * ध्रुव wdata = (भिन्न *) params->wdata + 0;

            प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                    ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src0->data + i02*nb02 + i01*nb01);
                    भिन्न * dst_data = wdata + i01*ne00*ne02;
                    प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            भिन्न * ध्रुव wdata = (भिन्न *) params->wdata + nk;
            भिन्न * dst_data = wdata;

            प्रति (int64_t i11 = 0; i11 < ne11; i11++) {
                ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src1->data + i11*nb11);
                प्रति (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
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
        भिन्न * dst_data = (भिन्न *)((वर्ण *) dst->data + i1*nb1);
        भिन्न * wdata_kernel = wdata + i1*ne02*ne00;
        प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
            ध्रुव पूर्णाङ्क i1n = i10*ne11;
            प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; i00++) {
                भिन्न v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
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
        भिन्न * ध्रुव wdata = (भिन्न *) dst->data;

        प्रति (int64_t in = 0; in < N; in++) {
            प्रति (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                प्रति (int64_t iow = 0; iow < OW; iow++) {
                    प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        भिन्न * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ध्रुव भिन्न * ध्रुव src_data = (भिन्न *)((वर्ण *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        प्रति (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;

                                यदि (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
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

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) dst->data;

        प्रति (int64_t in = 0; in < N; in++) {
            प्रति (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                प्रति (int64_t iow = 0; iow < OW; iow++) {
                    प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        ध्रुव भिन्न * ध्रुव src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (ध्रुव भिन्न *)((ध्रुव वर्ण *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        ध्रुव ggml_fp16_t * ध्रुव src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (ध्रुव ggml_fp16_t *)((ध्रुव वर्ण *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        प्रति (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;

                                यदि (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    यदि (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } अन्यथा {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = src_data_f16[iih*IW + iiw];
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

शून्य ggml_compute_forward_im2col_back_f32(
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
        भिन्न * ध्रुव wdata = (भिन्न *) dst->data;

        प्रति (int64_t in = 0; in < N; in++) {
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

                                ध्रुव भिन्न * ध्रुव grad_in = (ध्रुव भिन्न *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        भिन्न * dst_data = (भिन्न *)((वर्ण *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_data[iih*IW + iiw] = grad;
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
    GGML_UNUSED(OC);
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
        ggml_fp16_t * ध्रुव wdata = (ggml_fp16_t *) dst->data;

        प्रति (int64_t in = 0; in < N; in++) {
            प्रति (int64_t iod = 0; iod < OD; iod++) {
                प्रति (int64_t ioh = 0; ioh < OH; ioh++) {
                    प्रति (int64_t iow = 0; iow < OW; iow++) {
                        प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ध्रुव भिन्न * ध्रुव src_data = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            प्रति (int64_t ikd = 0; ikd < KD; ikd++) {
                                प्रति (int64_t ikh = 0; ikh < KH; ikh++) {
                                    प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ध्रुव int64_t iid = iod*s2 + ikd*d2 - p2;

                                        यदि (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            ध्रुव भिन्न * ध्रुव s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(*s);
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
    GGML_UNUSED(OC);
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
        भिन्न * ध्रुव wdata = (भिन्न *) dst->data;

        प्रति (int64_t in = 0; in < N; in++) {
            प्रति (int64_t iod = 0; iod < OD; iod++) {
                प्रति (int64_t ioh = 0; ioh < OH; ioh++) {
                    प्रति (int64_t iow = 0; iow < OW; iow++) {
                        प्रति (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            भिन्न * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            ध्रुव भिन्न * ध्रुव src_data = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            प्रति (int64_t ikd = 0; ikd < KD; ikd++) {
                                प्रति (int64_t ikh = 0; ikh < KH; ikh++) {
                                    प्रति (int64_t ikw = 0; ikw < KW; ikw++) {
                                        ध्रुव int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        ध्रुव int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        ध्रुव int64_t iid = iod*s2 + ikd*d2 - p2;

                                        यदि (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            ध्रुव भिन्न * ध्रुव s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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
    src1.nb[0] = traits->type_size;
    src1.nb[1] = k * traits->type_size;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.data  = a;

    संरचना ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_size;
    src0.nb[1] = k * traits->type_size;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.data  = b;

    संरचना ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = परिमाण(भिन्न);
    dst.nb[1] = n * परिमाण(भिन्न);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
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

template <typename elem_t>
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
    ध्रुव int64_t T_in = src->ne[1];
    ध्रुव int64_t K    = K_OC / OC;
    ध्रुव int64_t T_out = dst->ne[0];

    ध्रुव elem_t * col_data = (ध्रुव elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    ध्रुव int64_t dr = (T_out + nth - 1) / nth;
    ध्रुव int64_t it0 = dr * ith;
    ध्रुव int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    प्रति (int64_t oc = 0; oc < OC; oc++) {
        प्रति (int64_t t_out = it0; t_out < it1; t_out++) {
            ध्रुव int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            यदि (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            यदि (t_in_max >= T_in) t_in_max = T_in - 1;

            भिन्न sum = 0.0f;
            प्रति (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                यदि (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
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

    ध्रुव int64_t c_in  = src->ne[2];
    ध्रुव int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    ध्रुव int64_t src_w = src->ne[0];
    ध्रुव int64_t src_h = src->ne[1];
    ध्रुव int64_t knl_w = kernel->ne[0];
    ध्रुव int64_t knl_h = kernel->ne[1];
    ध्रुव int64_t dst_w = dst->ne[0];
    ध्रुव int64_t dst_h = dst->ne[1];

    ध्रुव भिन्न * src_data = (भिन्न *) src->data;
    शून्य  * knl_data       = kernel->data;
    भिन्न * dst_data       = (भिन्न *) dst->data;

    ध्रुव int64_t knl_n           = knl_w * knl_h * c_in;
    ध्रुव int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    ध्रुव int64_t space_per_patch   = knl_n * traits->type_size + c_out * परिमाण(भिन्न);
    ध्रुव int64_t batch_size        = params->wsize / space_per_patch;
    ध्रुव int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    ध्रुव int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    शून्य * tmp = params->wdata;

    प्रति (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        ध्रुव int64_t patch_start_batch = batch_i * patches_per_batch;
        ध्रुव int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        ध्रुव int64_t patch_n           = patch_end_batch - patch_start_batch;

        ध्रुव int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        ध्रुव int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        ध्रुव int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        प्रति (int64_t p = patch_start; p < patch_end; ++p) {
            ध्रुव int64_t  batch_n     =  p / (dst_w * dst_h);
            ध्रुव int64_t  src_x       = (p / dst_w) % dst_h;
            ध्रुव int64_t  src_y       =  p % dst_w;

            ध्रुव भिन्न * src_base = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_data + batch_n * src->nb[3]);
            वर्ण *        dst_row  = (वर्ण *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            प्रति (int64_t ic = 0; ic < c_in; ++ic) {
                प्रति (int64_t ky = 0; ky < knl_h; ++ky) {
                    प्रति (int64_t kx = 0; kx < knl_w; ++kx) {
                        ध्रुव int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        ध्रुव int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        भिन्न src_val;
                        यदि (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } अन्यथा {
                            ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        वर्ण * element_ptr = dst_row + dst_idx * traits->type_size;
                        यदि (kernel_type == GGML_TYPE_F32) {
                            *(भिन्न *) element_ptr = src_val;
                        } अन्यथा यदि (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        भिन्न * gemm_output = (भिन्न *) ((वर्ण *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (भिन्न*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        ध्रुव int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        ध्रुव int64_t permute_start = params->ith * permute_per_thread;
        ध्रुव int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        प्रति (int64_t i = permute_start; i < permute_end; ++i) {
            ध्रुव int64_t p       = patch_start_batch + i;
            ध्रुव int64_t batch_n = p / (dst_w * dst_h);
            ध्रुव int64_t dst_y   = (p / dst_w) % dst_h;
            ध्रुव int64_t dst_x   = p % dst_w;

            प्रति (int64_t oc = 0; oc < c_out; ++oc) {
                ध्रुव भिन्न value = gemm_output[i * c_out + oc];
                भिन्न * dst_ptr = (भिन्न *)((वर्ण *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
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

    ध्रुव भिन्न * src_data = (भिन्न *) src->data;
    शून्य  * knl_data       = kernel->data;
    भिन्न * dst_data       = (भिन्न *) dst->data;

    ध्रुव int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    ध्रुव int64_t knl_n_total       = knl_n_per_channel * c;
    ध्रुव int64_t patch_total       = n * dst_w * dst_h * dst_d;

    ध्रुव int64_t space_per_patch   = knl_n_total * traits->type_size + oc * परिमाण(भिन्न);
    ध्रुव int64_t batch_size        = params->wsize / space_per_patch;
    ध्रुव int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    ध्रुव int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    शून्य * tmp = params->wdata;

    प्रति (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        ध्रुव int64_t patch_start_batch = batch_i * patches_per_batch;
        ध्रुव int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        ध्रुव int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        ध्रुव int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        ध्रुव int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        ध्रुव int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        प्रति (int64_t p = patch_start; p < patch_end; ++p) {
            ध्रुव int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            ध्रुव int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            ध्रुव int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ध्रुव int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            ध्रुव int64_t dst_y      = p_in_depth / dst_w;
            ध्रुव int64_t dst_x      = p_in_depth % dst_w;

            वर्ण * dst_row = (वर्ण *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            प्रति (int64_t ic = 0; ic < c; ++ic) {
                प्रति (int64_t kz = 0; kz < knl_d; ++kz) {
                    प्रति (int64_t ky = 0; ky < knl_h; ++ky) {
                        प्रति (int64_t kx = 0; kx < knl_w; ++kx) {
                            ध्रुव int64_t sz = dst_z * s2 + kz * d2 - p2;
                            ध्रुव int64_t sy = dst_y * s1 + ky * d1 - p1;
                            ध्रुव int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            भिन्न src_val;
                            यदि (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } अन्यथा {
                                ध्रुव int64_t cn_idx = batch_idx * c + ic;
                                ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            वर्ण * element_ptr = dst_row + dst_idx * traits->type_size;
                            यदि (kernel_type == GGML_TYPE_F32) {
                                *(भिन्न *)element_ptr = src_val;
                            } अन्यथा यदि (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        भिन्न * gemm_output = (भिन्न *) ((वर्ण *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        ध्रुव int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        ध्रुव int64_t permute_start = params->ith * permute_per_thread;
        ध्रुव int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        प्रति (int64_t i = permute_start; i < permute_end; ++i) {
            ध्रुव int64_t p = patch_start_batch + i;
            ध्रुव int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            ध्रुव int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            ध्रुव int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            ध्रुव int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            ध्रुव int64_t dst_y      = p_in_depth / dst_w;
            ध्रुव int64_t dst_x      = p_in_depth % dst_w;

            प्रति (int64_t ioc = 0; ioc < oc; ++ioc) {
                ध्रुव भिन्न value = gemm_output[i * oc + ioc];
                ध्रुव int64_t ocn_idx = batch_idx * oc + ioc;
                भिन्न * dst_ptr = (भिन्न *)((वर्ण *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
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

template <typename kernel_t>
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

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == परिमाण(भिन्न));

    यदि (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * ध्रुव wdata = (kernel_t *) params->wdata + 0;

            प्रति (int64_t i03 = 0; i03 < ne03; i03++) {
                प्रति (int64_t i02 = 0; i02 < ne02; i02++) {
                    ध्रुव kernel_t * ध्रुव src = (kernel_t *)((वर्ण *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    प्रति (int64_t i01 = 0; i01 < ne01; i01++) {
                        प्रति (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
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
                        ध्रुव भिन्न * ध्रुव src = (भिन्न *)((वर्ण *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        प्रति (पूर्णाङ्क i10 = 0; i10 < ne10; i10++) {
                            यदि constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } अन्यथा {
                                dst_data[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        memset(dst->data, 0, ggml_nbytes(dst));
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
            भिन्न * dst_data = (भिन्न *)((वर्ण *) dst->data + i3*nb3 + i2*nb2);
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
                            dst_data[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
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

स्थिर अन्तःस्थ भिन्न ggml_conv_2d_dw_knl_f32(ध्रुव वर्ण * data, int64_t i, ggml_type type) {
    यदि (type == GGML_TYPE_F16) {
        प्रत्यागम GGML_FP16_TO_FP32(((ध्रुव ggml_fp16_t *)data)[i]);
    }
    प्रत्यागम ((ध्रुव भिन्न *)data)[i];
}

स्थिर शून्य ggml_compute_forward_conv_2d_dw_cwhn(
        ध्रुव ggml_compute_params * params,
        ध्रुव ggml_tensor * src,
        ध्रुव ggml_tensor * kernel,
        ggml_tensor * dst,
        ध्रुव ggml_conv_2d_dw_params & p) {

    ध्रुव int64_t c = p.channels;
    ध्रुव वर्ण * knl_data = (ध्रुव वर्ण *)kernel->data;
    ध्रुव ggml_type knl_type = kernel->type;

    ध्रुव int64_t rows_total = p.dst_h * p.batch;
    ध्रुव int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    ध्रुव int64_t row_start = params->ith * rows_per_thread;
    ध्रुव int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    यदि (knl_type == GGML_TYPE_F32) {
    #यदि defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #अन्यथा
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#अन्यथा
    ध्रुव int64_t c_pkg_end = 0;
#endif

    प्रति (int64_t row = row_start; row < row_end; ++row) {
        ध्रुव int64_t dst_y = row % p.dst_h;
        ध्रुव भिन्न * src_data = (ध्रुव भिन्न *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        प्रति (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            भिन्न * dst_data = (भिन्न *)dst->data + (row * p.dst_w + dst_x) * c;
            ध्रुव int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            ध्रुव int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            प्रति (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                प्रति (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ध्रुव int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    यदि (src_y < 0 || src_y >= p.src_h) {
                        अनुवर्त;
                    }
                    प्रति (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ध्रुव int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        यदि (src_x < 0 || src_x >= p.src_w) {
                            अनुवर्त;
                        }
                        ध्रुव भिन्न * kp = (ध्रुव भिन्न *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            प्रति (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                भिन्न sum = 0.0f;
                प्रति (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    ध्रुव int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    यदि (src_y < 0 || src_y >= p.src_h) {
                        अनुवर्त;
                    }
                    प्रति (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        ध्रुव int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        यदि (src_x < 0 || src_x >= p.src_w) {
                            अनुवर्त;
                        }
                        sum += ggml_conv_2d_dw_knl_f32(knl_data, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_data[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_data[c_i] = sum;
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
    ध्रुव int64_t per_thread = (n + params->nth - 1) / params->nth;
    ध्रुव int64_t start = params->ith * per_thread;
    ध्रुव int64_t end = MIN(start + per_thread, n);
    ध्रुव वर्ण * knl_base = (ध्रुव वर्ण *)kernel->data;
    ध्रुव ggml_type knl_type = kernel->type;

    प्रति (int64_t i = start; i < end; ++i) {
        ध्रुव int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        ध्रुव भिन्न * src_data = (ध्रुव भिन्न *)src->data + i * p.src_w * p.src_h;
        भिन्न * dst_data = (भिन्न *)dst->data + i * p.dst_w * p.dst_h;

        प्रति (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            प्रति (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                भिन्न sum = 0.0f;
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
                        sum += ggml_conv_2d_dw_knl_f32(knl_base, knl_offset + knl_y * p.knl_w + knl_x, knl_type)
                             * src_data[src_y * p.src_w + src_x];
                    }
                }
                dst_data[dst_y * p.dst_w + dst_x] = sum;
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
        ध्रुव वर्ण * srow_bytes =            (ध्रुव वर्ण *) src->data + ir * src->nb[1];
        भिन्न      * drow       = (भिन्न *) ((      वर्ण *) dst->data + ir * dst->nb[1]);

        प्रति (int64_t ow = 0; ow < OW; ++ow) {
            भिन्न res = 0;
            विकल्प (op) {
                पक्ष GGML_OP_POOL_AVG: res = 0.0f;     भङ्ग;
                पक्ष GGML_OP_POOL_MAX: res = -FLT_MAX; भङ्ग;
                पक्ष GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
            }

            पूर्णाङ्क count = 0;
            ध्रुव पूर्णाङ्क base = (पूर्णाङ्क) ow * s - p;

            प्रति (पूर्णाङ्क ki = 0; ki < k; ++ki) {
                ध्रुव पूर्णाङ्क j = base + ki;
                यदि (j < 0 || j >= (पूर्णाङ्क) IW) {
                    अनुवर्त;
                }

                भिन्न v;
                यदि (src->type == GGML_TYPE_F32) {
                    v = ((ध्रुव भिन्न *) srow_bytes)[j];
                } अन्यथा {
                    v = GGML_CPU_FP16_TO_FP32(((ध्रुव ggml_fp16_t *) srow_bytes)[j]);
                }

                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG: res += v;                भङ्ग;
                    पक्ष GGML_OP_POOL_MAX: res =  std::max(v, res); भङ्ग;
                    पक्ष GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
                }

                ++count;
            }

            विकल्प (op) {
                पक्ष GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; भङ्ग;
                पक्ष GGML_OP_POOL_MAX:                                           भङ्ग;
                पक्ष GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
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
    ध्रुव वर्ण * cdata = (ध्रुव वर्ण*)src->data;
    ध्रुव वर्ण * ध्रुव data_end = cdata + ggml_nbytes(src);

    ध्रुव int64_t px = dst->ne[0];
    ध्रुव int64_t py = dst->ne[1];
    ध्रुव int64_t pa = px * py;

    भिन्न * dplane = (भिन्न *)dst->data;

    ध्रुव पूर्णाङ्क ka = k0 * k1;
    ध्रुव पूर्णाङ्क offset0 = -p0;
    ध्रुव पूर्णाङ्क offset1 = -p1;

    यावत् (cdata < data_end) {
        प्रति (पूर्णाङ्क oy = 0; oy < py; ++oy) {
            भिन्न * ध्रुव drow = dplane + oy * px;
            भिन्न * ध्रुव out  = drow;

            प्रति (पूर्णाङ्क ox = 0; ox < px; ++ox) {
                भिन्न res = 0;
                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG: res = 0;        भङ्ग;
                    पक्ष GGML_OP_POOL_MAX: res = -FLT_MAX; भङ्ग;
                    पक्ष GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
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

                        ध्रुव भिन्न srow_j = (src->type == GGML_TYPE_F32) ? ((ध्रुव भिन्न*)srow)[j] : GGML_CPU_FP16_TO_FP32(((ध्रुव ggml_fp16_t*)srow)[j]);
                        विकल्प (op) {
                            पक्ष GGML_OP_POOL_AVG: res += srow_j;                भङ्ग;
                            पक्ष GGML_OP_POOL_MAX: res =  std::max(srow_j, res); भङ्ग;
                            पक्ष GGML_OP_POOL_मोजणी:               GGML_ABORT("fatal error");
                        }
                    }
                }
                विकल्प (op) {
                    पक्ष GGML_OP_POOL_AVG:           res /= ka; भङ्ग;
                    पक्ष GGML_OP_POOL_MAX:                      भङ्ग;
                    पक्ष GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

शून्य ggml_compute_forward_pool_2d_back(
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

    वर्ण       * cdata  = (वर्ण       *) dst->data;
    ध्रुव वर्ण * cdataf = (ध्रुव वर्ण *) dstf->data;
    ध्रुव वर्ण * ध्रुव data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    ध्रुव int64_t px = src->ne[0];
    ध्रुव int64_t py = src->ne[1];
    ध्रुव int64_t pa = px * py;

    ध्रुव भिन्न * splane = (ध्रुव भिन्न *) src->data;

    ध्रुव पूर्णाङ्क ka = k0 * k1;
    ध्रुव पूर्णाङ्क offset0 = -p0;
    ध्रुव पूर्णाङ्क offset1 = -p1;

    यावत् (cdata < data_end) {
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
                                ((ध्रुव भिन्न *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((ध्रुव ggml_fp16_t *) drowf)[j]);
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
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((ध्रुव ggml_fp16_t *) drow)[j]));
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
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
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
    भिन्न pixel_offset = 0.5f;

    ध्रुव int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    ध्रुव ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    यदि (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (भिन्न)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (भिन्न)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    यदि (mode == GGML_SCALE_MODE_NEAREST) {
        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            ध्रुव int64_t i03 = i3 / sf3;
            प्रति (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ध्रुव int64_t i02 = i2 / sf2;
                प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
                    ध्रुव int64_t i01 = i1 / sf1;
                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव int64_t i00 = i0 / sf0;

                        ध्रुव भिन्न * x = (भिन्न *)((वर्ण *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              भिन्न * y = (भिन्न *)((वर्ण *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } अन्यथा यदि (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](भिन्न x) -> भिन्न {
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
                    ध्रुव भिन्न y = ((भिन्न) i1 + pixel_offset) / sf1;
                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव भिन्न x = ((भिन्न) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        ध्रुव int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        ध्रुव int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        ध्रुव int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        ध्रुव int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        भिन्न val = 0.0f;
                        भिन्न total_weight = 0.0f;

                        प्रति (int64_t sy = y_min; sy < y_max; sy++) {
                            ध्रुव भिन्न weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            प्रति (int64_t sx = x_min; sx < x_max; sx++) {
                                ध्रुव भिन्न weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                ध्रुव भिन्न weight = weight_x * weight_y;

                                यदि (weight <= 0.0f) {
                                    अनुवर्त;
                                }

                                ध्रुव भिन्न pixel = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        यदि (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        भिन्न * dst_ptr = (भिन्न *)((वर्ण *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } अन्यथा यदि (mode == GGML_SCALE_MODE_BILINEAR) {
        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            ध्रुव int64_t i03 = i3 / sf3;
            प्रति (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                ध्रुव int64_t i02 = i2 / sf2;
                प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
                    ध्रुव भिन्न y = ((भिन्न)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    भिन्न dy = y - (भिन्न)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव भिन्न x = ((भिन्न)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        भिन्न dx = x - (भिन्न)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        ध्रुव भिन्न a = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ध्रुव भिन्न b = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        ध्रुव भिन्न c = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        ध्रुव भिन्न d = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        ध्रुव भिन्न val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        भिन्न * y_dst = (भिन्न *)((वर्ण *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } अन्यथा यदि (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        ध्रुव भिन्न a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](भिन्न x) { प्रत्यागम ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](भिन्न x) { प्रत्यागम ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](भिन्न p0, भिन्न p1, भिन्न p2, भिन्न p3, भिन्न x) {
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
                    ध्रुव भिन्न y = ((भिन्न)i1 + pixel_offset) / sf1 - pixel_offset;
                    ध्रुव int64_t y0 = (int64_t)floorf(y);
                    ध्रुव भिन्न dy = y - (भिन्न)y0;

                    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                        ध्रुव भिन्न x = ((भिन्न)i0 + pixel_offset) / sf0 - pixel_offset;
                        ध्रुव int64_t x0 = (int64_t)floorf(x);
                        ध्रुव भिन्न dx = x - (भिन्न)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> भिन्न {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            प्रत्यागम *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        ध्रुव भिन्न val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        भिन्न * y_dst = (भिन्न *)((वर्ण *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
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

template<तार्किक circular_t>
स्थिर शून्य ggml_compute_forward_pad_f32(
    ध्रुव ggml_compute_params * params,
          ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(dst->nb[0] == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भिन्न * dst_ptr = (भिन्न *) dst->data;
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
                dst_row[i0] = *(ध्रुव भिन्न *)((ध्रुव वर्ण *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } अन्यथा {
            ध्रुव तार्किक interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            यदि (!interior) {
                memset(dst_row, 0, ne0 * परिमाण(भिन्न));
                अनुवर्त;
            }

            ध्रुव वर्ण * src_row = (ध्रुव वर्ण *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            यदि (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_row, 0, lp0 * परिमाण(भिन्न));
                memcpy(dst_row + lp0, src_row, ne00 * परिमाण(भिन्न));
                memset(dst_row + lp0 + ne00, 0, rp0 * परिमाण(भिन्न));
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
                भिन्न * left  = (भिन्न *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                भिन्न * right = (भिन्न *) ((वर्ण *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (भिन्न *) ((वर्ण *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                प्रति (पूर्णाङ्क i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                प्रति (पूर्णाङ्क i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

स्थिर int64_t ggml_wrap_index(int64_t i, int64_t ne) {
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
    ध्रुव भिन्न * src_data = (ध्रुव भिन्न *) src0->data;
    भिन्न * dst_data = (भिन्न *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव पूर्णाङ्क s0 = ggml_get_op_params_i32(dst, 0);
    ध्रुव पूर्णाङ्क s1 = ggml_get_op_params_i32(dst, 1);
    ध्रुव पूर्णाङ्क s2 = ggml_get_op_params_i32(dst, 2);
    ध्रुव पूर्णाङ्क s3 = ggml_get_op_params_i32(dst, 3);

    ध्रुव int64_t total = ne1 * ne2 * ne3;
    ध्रुव int64_t per_thread = (total + params->nth) / params->nth;
    ध्रुव int64_t start = params->ith * per_thread;
    ध्रुव int64_t end   = std::min(start + per_thread, total);

    प्रति (int64_t i = start; i < end; ++i) {
        ध्रुव int64_t i1 = i % ne1;
        ध्रुव int64_t i2 = (i / ne1) % ne2;
        ध्रुव int64_t i3 = i / (ne2 * ne1);
        भिन्न * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / परिमाण(भिन्न);

        ध्रुव int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        ध्रुव int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        ध्रुव int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        ध्रुव भिन्न * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / परिमाण(भिन्न);

        ध्रुव int64_t s = ggml_wrap_index(-s0, ne00);
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
    ध्रुव भिन्न step  = ggml_get_op_params_f32(dst, 2);

    ध्रुव int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    प्रति (int64_t i = ith; i < steps; i+= nth) {
        भिन्न value = start + step * i;
        ((भिन्न *)dst->data)[i] = value;
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
        भिन्न * embed_data = (भिन्न *)((वर्ण *)  dst->data +  i*nb1);
        प्रति (int64_t j = ith; j < half; j += nth) {
            भिन्न timestep = ((भिन्न *)src0->data)[i];
            भिन्न freq = (भिन्न)expf(-logf(max_period) * j / half);
            भिन्न arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        यदि (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
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

template<गणना ggml_sort_order order>
संरचना cmp_argsort {
    ध्रुव भिन्न * data;
    तार्किक operator()(int32_t a, int32_t b) ध्रुव {
        यदि constexpr (order == GGML_SORT_ORDER_ASC) {
            प्रत्यागम data[a] < data[b];
        } अन्यथा {
            प्रत्यागम data[a] > data[b];
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

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    प्रति (int64_t i = ith; i < nr; i += nth) {
        ध्रुव भिन्न * src_data = (भिन्न *)((वर्ण *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((वर्ण *) dst->data + i*nb1);

        प्रति (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        विकल्प (order) {
            पक्ष GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                भङ्ग;

            पक्ष GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
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

संरचना cmp_शीर्ष_k {
    ध्रुव भिन्न * data;
    तार्किक operator()(int32_t a, int32_t b) ध्रुव {
        प्रत्यागम data[a] > data[b];
    }
};

स्थिर शून्य ggml_compute_forward_शीर्ष_k_f32(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == परिमाण(भिन्न));

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t nr = ggml_nrows(src0);

    ध्रुव पूर्णाङ्क शीर्ष_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

    प्रति (int64_t i = ith; i < nr; i += nth) {
        ध्रुव भिन्न * src_data = (भिन्न *)((वर्ण *) src0->data + i*nb01);

        प्रति (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + शीर्ष_k, tmp + ne00, cmp_शीर्ष_k{src_data});

        int32_t * dst_data = (int32_t *)((वर्ण *) dst->data + i*nb1);

        std::copy(tmp, tmp + शीर्ष_k, dst_data);

        // emphasize that the order is not important
        यदि (शीर्ष_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

शून्य ggml_compute_forward_शीर्ष_k(
    ध्रुव ggml_compute_params * params,
    ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_शीर्ष_k_f32(params, dst);
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
        int64_t ic_start, int64_t ic_end,
        भिन्न * partials, int64_t partial_stride) {

    ध्रुव तार्किक write_partials = (partials != nullptr);
    ध्रुव ggml_tensor * q     = dst->src[0];
    ध्रुव ggml_tensor * k     = dst->src[1];
    ध्रुव ggml_tensor * v     = dst->src[2];
    ध्रुव ggml_tensor * mask  = dst->src[3];
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
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

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

    memcpy(&scale,         (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&max_bias,      (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    memcpy(&logit_softcap, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ध्रुव uint32_t n_head      = neq2;
    ध्रुव uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         ध्रुव k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t ध्रुव q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    ध्रुव kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   ध्रुव v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    पूर्णाङ्क ith = params->ith;

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        // q indices
        ध्रुव पूर्णाङ्क iq3 = ir/(neq2*neq1);
        ध्रुव पूर्णाङ्क iq2 = (ir - iq3*neq2*neq1)/neq1;
        ध्रुव पूर्णाङ्क iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        ध्रुव uint32_t h = iq2; // head index
        ध्रुव भिन्न slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        भिन्न S = 0.0f;      // sum
        भिन्न M = -INFINITY; // maximum KQ value

        भिन्न       * VKQ32 = (भिन्न       *) params->wdata + ith*(1*DK + 2*DV + CACHE_ओळ_SIZE_F32); // FP32 VKQ accumulator
        भिन्न       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        यदि (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*परिमाण(ggml_fp16_t));
        } अन्यथा {
            memset(VKQ32, 0, DV*परिमाण(भिन्न));
        }

        ध्रुव ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((वर्ण *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : रिक्त;

        // k indices
        ध्रुव पूर्णाङ्क ik3 = iq3 / rk3;
        ध्रुव पूर्णाङ्क ik2 = iq2 / rk2;

        // v indices
        ध्रुव पूर्णाङ्क iv3 = iq3 / rv3;
        ध्रुव पूर्णाङ्क iv2 = iq2 / rv2;

        ध्रुव भिन्न * pq = (ध्रुव भिन्न *) ((वर्ण *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        प्रति (int64_t ic = ic_start; ic < ic_end; ++ic) {
            ध्रुव भिन्न mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            यदि (mv == -INFINITY) {
                अनुवर्त;
            }

            भिन्न s; // KQ value

            ध्रुव वर्ण * k_data = (ध्रुव वर्ण *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            यदि (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            ध्रुव भिन्न Mold = M;

            भिन्न ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            भिन्न vs = 1.0f; // post-softmax KQ value, expf(s - M)

            ध्रुव वर्ण * v_data = ((ध्रुव वर्ण *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

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
                ggml_vec_mad_f16(DV, VKQ16, (ध्रुव ggml_fp16_t *) v_data, vs);
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
                यदि (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } अन्यथा {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (ध्रुव भिन्न *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        यदि (v->type == GGML_TYPE_F16) {
            प्रति (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        यदि (sinks && ic_start == 0) {
            ध्रुव भिन्न s = ((भिन्न *)((वर्ण *) sinks->data))[h];

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
            memcpy(partial + 2, VKQ32, DV * परिमाण(भिन्न));
        } अन्यथा {
            // V /= S
            ध्रुव भिन्न S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            ध्रुव पूर्णाङ्क i1 = iq1;
            ध्रुव पूर्णाङ्क i2 = iq2;
            ध्रुव पूर्णाङ्क i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((वर्ण *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
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
    ध्रुव ggml_tensor * mask  = dst->src[3];
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
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

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

    memcpy(&scale,         (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    memcpy(&max_bias,      (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));
    memcpy(&logit_softcap, (भिन्न *) dst->op_params + 2, परिमाण(भिन्न));

    यदि (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    ध्रुव uint32_t n_head      = neq2;
    ध्रुव uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

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
        ध्रुव पूर्णाङ्क tile_rows = MIN(Q_TILE_SZ, MIN((पूर्णाङ्क)(ir1 - ir), (पूर्णाङ्क)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        ध्रुव uint32_t h = iq2; // head index
        ध्रुव भिन्न slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

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
        भिन्न * base  = (भिन्न *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_ओळ_SIZE_F32);

        शून्य  * Q_q    = base;
        भिन्न * KQ     = (भिन्न *)((वर्ण *)base + Q_TILE_SZ * DK * परिमाण(भिन्न));
        भिन्न * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        भिन्न * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        भिन्न * V32    = VKQ32 + Q_TILE_SZ * DV;
        भिन्न * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * परिमाण(भिन्न));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * परिमाण(भिन्न));

        // k indices
        ध्रुव पूर्णाङ्क ik3 = iq3 / rk3;
        ध्रुव पूर्णाङ्क ik2 = iq2 / rk2;

        // v indices
        ध्रुव पूर्णाङ्क iv3 = iq3 / rv3;
        ध्रुव पूर्णाङ्क iv2 = iq2 / rv2;

        {
            भिन्न * Q_f32 = (भिन्न *)Q_q;
            प्रति (पूर्णाङ्क tq = 0; tq < tile_rows; tq++) {
                ध्रुव भिन्न * pq = (ध्रुव भिन्न *) ((वर्ण *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * परिमाण(भिन्न));
            }
            प्रति (पूर्णाङ्क tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * परिमाण(भिन्न));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * परिमाण(भिन्न));
        memset(V32,   0, KV_TILE_SZ * DV * परिमाण(भिन्न));

        प्रति (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            ध्रुव पूर्णाङ्क kv_tile = (पूर्णाङ्क)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            यदि (mask) {
                तार्किक can_skip = सत्य;
                प्रति (पूर्णाङ्क tq = 0; tq < tile_rows; tq++) {
                    ध्रुव ggml_fp16_t * mp_row = (ध्रुव ggml_fp16_t *)((ध्रुव वर्ण *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    प्रति (पूर्णाङ्क tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
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
                ध्रुव वर्ण * k_data = (ध्रुव वर्ण *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                यदि (kv_type == GGML_TYPE_F16) {
                    ध्रुव ggml_fp16_t * k_f16 = (ध्रुव ggml_fp16_t *)k_data;
                    प्रति (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } अन्यथा {
                    ध्रुव भिन्न * k_f32_src = (ध्रुव भिन्न *)k_data;
                    प्रति (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * परिमाण(भिन्न));
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

            यदि (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
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
                ध्रुव वर्ण * v_data = (ध्रुव वर्ण *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                यदि (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((ध्रुव ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } अन्यथा {
                    memcpy(V32 + tk * DV, v_data, DV * परिमाण(भिन्न));
                }
            }
            प्रति (पूर्णाङ्क tq = 0; tq < Q_TILE_SZ; tq++) {
                यदि (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * परिमाण(भिन्न));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        यदि (sinks) {
            ध्रुव भिन्न s = ((भिन्न *)((वर्ण *) sinks->data))[h];

            प्रति (पूर्णाङ्क tq = 0; tq < tile_rows; tq++) {
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

        प्रति (पूर्णाङ्क tq = 0; tq < tile_rows; tq++) {
            // V /= S
            ध्रुव भिन्न S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            ध्रुव पूर्णाङ्क i1 = iq1 + tq;
            ध्रुव पूर्णाङ्क i2 = iq2;
            ध्रुव पूर्णाङ्क i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((वर्ण *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
स्थिर शून्य ggml_flash_attn_ext_reduce_partials(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst,
        ध्रुव int64_t n_chunks,
        ध्रुव int64_t chunk_size) {

    ध्रुव ggml_tensor * q = dst->src[0];
    ध्रुव ggml_tensor * k = dst->src[1];
    ध्रुव ggml_tensor * v = dst->src[2];

    ध्रुव int64_t DK        = k->ne[0];
    ध्रुव int64_t DV        = v->ne[0];
    ध्रुव int64_t nek1      = k->ne[1];
    ध्रुव int64_t n_q_heads = q->ne[2];

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t wdata_per_thread = DK + 2*DV + CACHE_ओळ_SIZE_F32;
    भिन्न *       thread_wdata     = (भिन्न *) params->wdata + ith * wdata_per_thread;

    ध्रुव int64_t partials_offset  = nth * (DK + 2*DV + CACHE_ओळ_SIZE_F32);
    ध्रुव int64_t partial_size     = 2 + DV;
    ध्रुव भिन्न * partials_base    = (ध्रुव भिन्न *) params->wdata + partials_offset;

    // Output layout
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    प्रति (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        भिन्न   M_final   = -INFINITY;
        भिन्न   S_final   = 0.0f;
        भिन्न * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * परिमाण(भिन्न));

        // Combine partials from all chunks
        प्रति (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            ध्रुव int64_t ic_start = chunk_idx * chunk_size;
            यदि (ic_start >= nek1) अनुवर्त;

            ध्रुव भिन्न * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
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
        memcpy((वर्ण *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
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
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbv0 == ggml_type_size(v->type));

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
    ध्रुव तार्किक use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    यदि (use_split_kv_path) {
        ध्रुव int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        ध्रुव int64_t partial_size  = 2 + DV;
        भिन्न *       partials_base = (भिन्न *) params->wdata + nth * (DK + 2*DV + CACHE_ओळ_SIZE_F32);

        ध्रुव int64_t ic_start = ith * chunk_size;
        ध्रुव int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        ध्रुव int64_t partial_stride = nth * partial_size;
        भिन्न *       chunk_partials = partials_base + ith * partial_size;

        यदि (ic_start < nek1) {
            प्रति (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } अन्यथा {
            प्रति (int64_t q_head = 0; q_head < neq2; q_head++) {
                भिन्न * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } अन्यथा {

        // total rows in q
        ध्रुव int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        ध्रुव तार्किक disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        पूर्णाङ्क nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        यदि (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        यदि (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
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
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        पूर्णाङ्क current_chunk = ith;

        यावत् (current_chunk < nchunk) {
            ध्रुव int64_t ir0 = dr * current_chunk;
            ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

            यदि (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } अन्यथा {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
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

स्थिर शून्य ggml_compute_forward_flash_attn_back_f32(
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

    ध्रुव पूर्णाङ्क Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
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
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    ध्रुव int64_t elem_q = ggml_nelements(q);
    ध्रुव int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    ध्रुव size_t tsize = ggml_type_size(result_type);

    ध्रुव size_t offs_q = 0;
    ध्रुव size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    ध्रुव size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    शून्य * grad_q = (वर्ण *) dst->data;
    शून्य * grad_k = (वर्ण *) dst->data + offs_k;
    शून्य * grad_v = (वर्ण *) dst->data + offs_v;

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
                भिन्न * S  = (भिन्न *) params->wdata + ith*2*(mxDM + CACHE_ओळ_SIZE_F32) + 0*(mxDM+CACHE_ओळ_SIZE_F32);
                भिन्न * SM = (भिन्न *) params->wdata + ith*2*(mxDM + CACHE_ओळ_SIZE_F32) + 1*(mxDM+CACHE_ओळ_SIZE_F32);

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
                            (भिन्न *) ((वर्ण *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (भिन्न *) ((वर्ण *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
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

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#अन्यथा
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    प्रतिज्ञा(sum > 0.0);

                    sum = 1.0/sum;
                    ggml_vec_scale_f32(masked_begin, SM, sum);

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
                ggml_vec_set_f32(masked_begin, S, 0);
                प्रति (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (भिन्न *) ((वर्ण *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(भिन्न *) ((वर्ण *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
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
                            (भिन्न *) ((वर्ण *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
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
                            (भिन्न *) ((वर्ण *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
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
                            *(भिन्न *) ((वर्ण *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_flash_attn_back(
        ध्रुव ggml_compute_params * params,
        ध्रुव तार्किक masked,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * q = dst->src[0];

    विकल्प (q->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
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
            ध्रुव भिन्न * s = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            ध्रुव भिन्न * c = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            भिन्न * x = (भिन्न *) ((वर्ण *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

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

स्थिर शून्य ggml_compute_forward_ssm_scan_f32(
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
    ध्रुव int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

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

    ध्रुव int32_t * ids = (ध्रुव int32_t *) src6->data;

    प्रति (पूर्णाङ्क i3 = 0; i3 < ns; ++i3) {
        ध्रुव भिन्न * s0 = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              भिन्न * s  = (      भिन्न *) ((      वर्ण *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        प्रति (पूर्णाङ्क i2 = 0; i2 < nt; ++i2) {
            ध्रुव भिन्न * x  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            ध्रुव भिन्न * dt = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            ध्रुव भिन्न * A  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src3->data); // {d_state, nh} or {1, nh}
            ध्रुव भिन्न * B  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            ध्रुव भिन्न * C  = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  भिन्न * y  = (      भिन्न *) ((      वर्ण *) dst->data + i2*(nh*nr*परिमाण(भिन्न)) + i3*(nt*nh*nr*परिमाण(भिन्न))); // {dim, nh, nt, ns}

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
                        ध्रुव पूर्णाङ्क ggml_f32_step = 1 * ggml_f32_epr;

                        ध्रुव पूर्णाङ्क np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        प्रति (पूर्णाङ्क i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            प्रति (पूर्णाङ्क j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_LOAD(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_LOAD(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_LOAD(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_ADD(t0, t1);

                                sum = GGML_F32_VEC_FMA(sum, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(sum);
    #नोचेद्यदि defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        ध्रुव पूर्णाङ्क np = 0;
    #अन्यथा
                        ध्रुव पूर्णाङ्क np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        प्रति (पूर्णाङ्क i = 0; i < np; i += GGML_F32_STEP) {
                            प्रति (पूर्णाङ्क j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_LOAD(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_LOAD(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_LOAD(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);

                                sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, sum);
    #endif
#अन्यथा
                        ध्रुव पूर्णाङ्क np = 0;
#endif
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
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        प्रति (int64_t k = 0; k < nc; k += svcntw()) {
                            svfloat32_t vA = GGML_F32_VEC_LOAD(&A[h*nc + k]);
                            svfloat32_t vB = GGML_F32_VEC_LOAD(&B[k + g*nc]);
                            svfloat32_t vC = GGML_F32_VEC_LOAD(&C[k + g*nc]);
                            svfloat32_t vs0 = GGML_F32_VEC_LOAD(&s0[ii*nc + k]);

                            svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
                            t1 = exp_ps_sve(svptrue_b32(), t1);
                            svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

                            vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
                            r1_vector = GGML_F32_VEC_ADD(GGML_F32_VEC_MUL(vs0, vC), r1_vector);

                            GGML_F32_VEC_STORE(&s[ii*nc + k], vs0);
                        }
                        y[ii] = GGML_F32xt_REDUCE_ONE(r1_vector);
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
#endif
                    }
                }
            }
            ध्रुव int64_t slot = nt - 1 - i2;
            यदि (K > 1 && slot > 0 && slot < K) {
                भिन्न * s_snapshot = (भिन्न *) ((वर्ण *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                प्रति (पूर्णाङ्क h = ih0; h < ih1; ++h) {
                    memcpy((वर्ण *) s_snapshot + h*src0->nb[2], (वर्ण *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

शून्य ggml_compute_forward_ssm_scan(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    विकल्प (dst->src[0]->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
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
    GGML_UNUSED(params);

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
                            ((भिन्न *) dst->data)[i] = 0.0f;
                        } अन्यथा {
                            ((भिन्न *) dst->data)[i] = ((भिन्न *) src0->data)[j];
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
    GGML_UNUSED(params);

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

                ((भिन्न *) dst->data)[j] = ((भिन्न *) src0->data)[i];
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
        पक्ष GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
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
        पक्ष GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
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
        पक्ष GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
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
        पक्ष GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
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
    GGML_UNUSED(params);

    ध्रुव ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    ध्रुव int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    प्रति (int64_t i2 = 0; i2 < ne2; ++i2) {
        प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
            ध्रुव int64_t pos = (w - i1 - 1) + i2;
            प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
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

स्थिर शून्य ggml_compute_forward_add_rel_pos_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    ध्रुव तार्किक inplace = (तार्किक) ((int32_t *) dst->op_params)[0];
    यदि (!inplace) {
        यदि (params->ith == 0) {
            memcpy((वर्ण *) dst->data, (वर्ण *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    भिन्न * src1_data = (भिन्न *) src1->data;
    भिन्न * src2_data = (भिन्न *) src2->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

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
                    ध्रुव भिन्न src1_e = src1_data[jp0];
                    ध्रुव भिन्न src2_e = src2_data[jp0];

                    ध्रुव int64_t jdh = jp0 * ne10;
                    ध्रुव int64_t jdw = jdh - (ne10 - 1) * i10;

                    प्रति (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_add_rel_pos(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
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
    ध्रुव int64_t head_size = C / HEADS;

    भिन्न * dst_data = (भिन्न *) dst->data;
    भिन्न * state = ((भिन्न *) dst->data) + C * T;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क h_start =  (HEADS * (ith    )) / nth;
    ध्रुव पूर्णाङ्क h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भिन्न * k =          (भिन्न *) dst->src[0]->data;
    भिन्न * v =          (भिन्न *) dst->src[1]->data;
    भिन्न * r =          (भिन्न *) dst->src[2]->data;
    भिन्न * time_faaaa = (भिन्न *) dst->src[3]->data;
    भिन्न * time_decay = (भिन्न *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    यदि (ith == 0) {
        memset(dst_data, 0, T * C * परिमाण(भिन्न));
    }
    ggml_barrier(params->threadpool);


    #यदि defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #नोचेद्यदि defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #नोचेद्यदि defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #नोचेद्यदि defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        पूर्णाङ्क wkv_vector_size;
        #यदि defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #अन्यथा
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        ध्रुव int64_t vec_मोजणी = head_size / wkv_vector_size;

        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_offset;
            भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[5]->data + state_offset;

            प्रति (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                प्रति (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    भिन्न k_val = k[t_h_i_offset];
                    भिन्न r_val = r[t_h_i_offset];
                    भिन्न time_faaaa_val = time_faaaa[h_i_offset];
                    भिन्न time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    प्रति (int64_t j = 0; j < vec_मोजणी; j++) {
                        size_t base_j = j * wkv_vector_size;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X temp_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, time_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, r_vec);
                        GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, time_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    प्रति (int64_t j = vec_मोजणी * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        भिन्न v_val = v[t_h_j_offset];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                        भिन्न temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
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
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_offset;
            भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[5]->data + state_offset;

            प्रति (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                प्रति (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    भिन्न k_val = k[t_h_i_offset];
                    भिन्न r_val = r[t_h_i_offset];
                    भिन्न time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    भिन्न time_decay_val = time_decay[t_h_i_offset];

                    प्रति (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        भिन्न v_val = v[t_h_j_offset];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                        भिन्न temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
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
    ध्रुव int64_t head_size = C / HEADS;
    ध्रुव भिन्न scale = ggml_get_op_params_f32(dst, 0);

    भिन्न * dst_data = (भिन्न *) dst->data;
    भिन्न * state = ((भिन्न *) dst->data) + C * T;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क h_start =  (HEADS * (ith    )) / nth;
    ध्रुव पूर्णाङ्क h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भिन्न * k = (भिन्न *) dst->src[0]->data;
    भिन्न * v = (भिन्न *) dst->src[1]->data;
    भिन्न * q = (भिन्न *) dst->src[2]->data;
    भिन्न * g = (भिन्न *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    यदि (ith == 0) {
        memset(dst_data, 0, T * C * परिमाण(भिन्न));
    }
    ggml_barrier(params->threadpool);


    #यदि defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #नोचेद्यदि defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #नोचेद्यदि defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #नोचेद्यदि defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        पूर्णाङ्क gla_vector_size;
        #यदि defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #अन्यथा
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        ध्रुव int64_t vec_मोजणी = head_size / gla_vector_size;

        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_offset;
            भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[4]->data + state_offset;

            प्रति (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                प्रति (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    भिन्न k_val = k[t_h_i_offset];
                    भिन्न q_val = q[t_h_i_offset] * scale;
                    भिन्न g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    प्रति (int64_t j = 0; j < vec_मोजणी; j++) {
                        size_t base_j = j * gla_vector_size;
                        size_t t_h_j_offset = t_h_offset + base_j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X temp_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, q_vec);
                        GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], temp_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    प्रति (int64_t j = vec_मोजणी * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        भिन्न v_val = v[t_h_j_offset];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                        भिन्न temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #अन्यथा
        प्रति (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_offset;
            भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[4]->data + state_offset;

            प्रति (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                प्रति (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    भिन्न k_val = k[t_h_i_offset];
                    भिन्न q_val = q[t_h_i_offset] * scale;
                    भिन्न g_val = g[t_h_i_offset];

                    प्रति (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        भिन्न v_val = v[t_h_j_offset];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                        भिन्न temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
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

    ध्रुव भिन्न * A = (ध्रुव भिन्न *) src0->data;  // [n, n, B1, B2]
    ध्रुव भिन्न * B = (ध्रुव भिन्न *) src1->data;  // [n, k, B1, B2]
          भिन्न * X = (      भिन्न *) dst->data;   // [n, k, B1, B2]

    प्रति (int64_t ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir/(ne02*k);
        ध्रुव int64_t i02 = (ir - i03*ne02*k)/k;
        ध्रुव int64_t i01 = (ir - i03*ne02*k - i02*k);

        ध्रुव भिन्न * A_batch = A + i02 * nb02 / परिमाण(भिन्न) + i03 * nb03 / परिमाण(भिन्न);
        ध्रुव भिन्न * B_batch = B + i02 * nb12 / परिमाण(भिन्न) + i03 * nb13 / परिमाण(भिन्न);

        भिन्न * X_batch = X + i02 * nb2 / परिमाण(भिन्न) + i03 * nb3 / परिमाण(भिन्न);

        प्रति (int64_t i00 = 0; i00 < n; ++i00) {
            भिन्न sum = 0.0f;
            प्रति (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            ध्रुव भिन्न diag = A_batch[i00 * n + i00];
            प्रतिज्ञा(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
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

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
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

    ध्रुव int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    ध्रुव पूर्णाङ्क ith = params->ith;

    भिन्न * delta       = (भिन्न *)params->wdata + ith * per_thread + CACHE_ओळ_SIZE_F32;
    भिन्न * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    ध्रुव int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    ध्रुव int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    भिन्न * attn_out_base  = (भिन्न *)dst->data;
    भिन्न * state_out_base = (भिन्न *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    ध्रुव भिन्न * state_in_base = (ध्रुव भिन्न *)src_state->data;

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
        भिन्न * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        ध्रुव भिन्न * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * परिमाण(भिन्न));

        // attn output pointer for first token of this (head, seq)
        भिन्न * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        प्रति (int64_t t = 0; t < n_tokens; t++) {
            ध्रुव भिन्न * q_d = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            ध्रुव भिन्न * k_d = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            ध्रुव भिन्न * v_d = (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            ध्रुव भिन्न beta_val = *(ध्रुव भिन्न *)((ध्रुव वर्ण *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            ध्रुव भिन्न * g_d    =  (ध्रुव भिन्न *)((ध्रुव वर्ण *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            यदि (kda) {
                // precompute exp(g) into delta scratch (reused below)
                प्रति (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                प्रति (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } अन्यथा {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            प्रति (int64_t j = 0; j < S_v; ++j) {
                भिन्न sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            प्रति (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            प्रति (int64_t j = 0; j < S_v; ++j) {
                भिन्न sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            यदि (K > 1) {
                ध्रुव int64_t target_slot = n_tokens - 1 - t;
                यदि (target_slot >= 0 && target_slot < K) {
                    भिन्न * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * परिमाण(भिन्न));
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
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    यदि (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    यदि (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    ध्रुव int64_t dr = (nr + nchunk - 1) / nchunk;

    पूर्णाङ्क current_chunk = ith;

    यावत् (current_chunk < nchunk) {
        ध्रुव int64_t ir0 = dr * current_chunk;
        ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
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

स्थिर शून्य ggml_dsv4_hc_comb_norm_cols(भिन्न * comb, भिन्न eps) {
    constexpr int64_t hc = 4;

    प्रति (int64_t idst = 0; idst < hc; ++idst) {
        भिन्न sum = eps;
        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        ध्रुव भिन्न inv_sum = 1.0f / sum;
        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

स्थिर शून्य ggml_dsv4_hc_comb_norm_rows(भिन्न * comb, भिन्न eps) {
    constexpr int64_t hc = 4;

    प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
        भिन्न sum = eps;
        प्रति (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        ध्रुव भिन्न inv_sum = 1.0f / sum;
        प्रति (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

स्थिर शून्य ggml_compute_forward_dsv4_hc_comb_f32(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * mixes = dst->src[0];
    ध्रुव ggml_tensor * scale = dst->src[1];
    ध्रुव ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    ध्रुव int64_t n_tokens = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ध्रुव भिन्न eps = ggml_get_op_params_f32(dst, 0);
    ध्रुव int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव int64_t dr  = (n_tokens + nth - 1) / nth;
    ध्रुव int64_t it0 = dr * ith;
    ध्रुव int64_t it1 = MIN(it0 + dr, n_tokens);

    ध्रुव भिन्न scale_comb = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) scale->data + 2*nbs0);

    प्रति (int64_t it = it0; it < it1; ++it) {
        भिन्न comb[hc*hc];

        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            भिन्न max = -INFINITY;
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                ध्रुव भिन्न xv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                ध्रुव भिन्न bv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) base->data  + (comb_offset + idx)*nbb0);
                ध्रुव भिन्न v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            भिन्न sum = 0.0f;
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                ध्रुव भिन्न v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            ध्रुव भिन्न inv_sum = 1.0f / sum;
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        प्रति (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
            प्रति (int64_t idst = 0; idst < hc; ++idst) {
                ध्रुव int64_t idx = idst + hc*isrc;
                *(भिन्न *) ((वर्ण *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
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

        भिन्न sum = 0.0f;
        प्रति (int64_t ih = 0; ih < hc; ++ih) {
            ध्रुव भिन्न xv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            भिन्न wv;
            यदि (gated) {
                ध्रुव भिन्न gv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } अन्यथा {
                wv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(भिन्न *) ((वर्ण *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
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

        ध्रुव भिन्न xv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) x->data    + i0*nbx0 + it*nbx1);
        ध्रुव भिन्न pv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) post->data + idst*nbp0 + it*nbp1);

        भिन्न sum = xv * pv;
        यदि (comb) {
            प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
                ध्रुव भिन्न rv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                ध्रुव भिन्न cv = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } अन्यथा {
            sum += *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(भिन्न *) ((वर्ण *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
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

शून्य ggml_compute_forward_शीर्ष_k_mask(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {
    ध्रुव ggml_tensor * mask   = dst->src[0];
    ध्रुव ggml_tensor * शीर्ष_k  = dst->src[1];
    ध्रुव ggml_tensor * score  = dst->src[2];
    ध्रुव ggml_tensor * blocks = dst->src[3];
    ध्रुव ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    ध्रुव int64_t n_kv    = mask->ne[0];
    ध्रुव int64_t ne1     = dst->ne[1];
    ध्रुव int64_t ne2     = dst->ne[2];
    ध्रुव int64_t n_rows  = ggml_nrows(dst);
    ध्रुव int64_t n_शीर्ष   = शीर्ष_k->ne[0];
    ध्रुव int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    ध्रुव int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    ध्रुव int64_t n_extra = extra ? extra->ne[0] : 0;

    ध्रुव size_t ts = ggml_type_size(dst->type);

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    प्रति (int64_t ir = ith; ir < n_rows; ir += nth) {
        ध्रुव int64_t i1 = ir % ne1;
        ध्रुव int64_t i2 = (ir / ne1) % ne2;
        ध्रुव int64_t i3 = ir / (ne1*ne2);

        ध्रुव वर्ण * mask_row = (ध्रुव वर्ण *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        वर्ण * dst_row = (वर्ण *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        यदि (dst->type == GGML_TYPE_F16) {
            ध्रुव ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-INFINITY);
            प्रति (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } अन्यथा {
            प्रति (int64_t c = 0; c < n_kv; ++c) {
                ((भिन्न *) dst_row)[c] = -INFINITY;
            }
        }

        auto keep = [&](int32_t c) {
            यदि (c >= 0 && c < n_kv) {
                memcpy(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        ध्रुव वर्ण * शीर्ष_k_row = (ध्रुव वर्ण *) शीर्ष_k->data + i1*शीर्ष_k->nb[1] + i2*शीर्ष_k->nb[2] + i3*शीर्ष_k->nb[3];

        प्रति (int64_t j = 0; j < n_शीर्ष; ++j) {
            ध्रुव int32_t p = *(ध्रुव int32_t *) (शीर्ष_k_row + j*शीर्ष_k->nb[0]);
            यदि (p < 0 || p >= n_pick) {
                अनुवर्त;
            }
            यदि (score) {
                ध्रुव भिन्न s = *(ध्रुव भिन्न *) ((ध्रुव वर्ण *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                यदि (!(s > -INFINITY)) {
                    अनुवर्त;
                }
            }
            यदि (!blocks) {
                keep(p);
                अनुवर्त;
            }
            प्रति (int64_t k = 0; k < n_blk; ++k) {
                keep(*(ध्रुव int32_t *) ((ध्रुव वर्ण *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        यदि (extra) {
            ध्रुव वर्ण * extra_row = (ध्रुव वर्ण *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
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
    ध्रुव int64_t head_size = C / HEADS;

    भिन्न * dst_data = (भिन्न *) dst->data;
    भिन्न * state = ((भिन्न *) dst->data) + C * T;

    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    ध्रुव पूर्णाङ्क h_start =  (HEADS * (ith    )) / nth;
    ध्रुव पूर्णाङ्क h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भिन्न * r = (भिन्न *) dst->src[0]->data;
    भिन्न * w = (भिन्न *) dst->src[1]->data;
    भिन्न * k = (भिन्न *) dst->src[2]->data;
    भिन्न * v = (भिन्न *) dst->src[3]->data;
    भिन्न * a = (भिन्न *) dst->src[4]->data;
    भिन्न * b = (भिन्न *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #यदि defined(GGML_SIMD)
        #यदि defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            प्रति (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                भिन्न * state_cur = state + state_offset;
                भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[6]->data + state_offset;

                प्रति (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    प्रति (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        भिन्न v_val = v[t_h_i_offset];

                        भिन्न sa = 0, result = 0;
                        प्रति (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        प्रति (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            भिन्न r_val = r[t_h_j_offset];
                            भिन्न w_val = w[t_h_j_offset];
                            भिन्न k_val = k[t_h_j_offset];
                            भिन्न b_val = b[t_h_j_offset];
                            भिन्न kv_val = v_val * k_val;
                            भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #अन्यथा
            प्रति (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                भिन्न * state_cur = state + state_offset;
                भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[6]->data + state_offset;

                प्रति (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    प्रति (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        भिन्न sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            प्रति (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                प्रति (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_LOAD(&a[t_h_offset + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_offset + j + kk * GGML_F32_EPR]);
                                    sum[kk] = GGML_F32_VEC_FMA(sum[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, sum);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC result_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        प्रति (; j < head_size; j += GGML_F32_STEP) {
                            प्रति (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_offset = t_h_offset + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_offset = h_2d_i_offset + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_LOAD(&r[t_h_j_offset]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_LOAD(&w[t_h_j_offset]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_LOAD(&k[t_h_j_offset]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_LOAD(&b[t_h_j_offset]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_j_offset]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_offset], state_vec);

                                result_vec[kk] = GGML_F32_VEC_FMA(result_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_data[t_h_i_offset], result_vec);

                        // There shouldn't be left-overs though.
                        प्रति (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            भिन्न r_val = r[t_h_j_offset];
                            भिन्न w_val = w[t_h_j_offset];
                            भिन्न k_val = k[t_h_j_offset];
                            भिन्न b_val = b[t_h_j_offset];
                            भिन्न kv_val = v[t_h_i_offset] * k_val;

                            भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #अन्यथा
        प्रति (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            भिन्न * state_cur = state + state_offset;
            भिन्न * state_prev = t % (T / n_seqs) ? state_cur : (भिन्न*)dst->src[6]->data + state_offset;

            प्रति (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                प्रति (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    भिन्न v_val = v[t_h_i_offset];

                    भिन्न sa = 0, result = 0;
                    प्रति (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    प्रति (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        भिन्न r_val = r[t_h_j_offset];
                        भिन्न w_val = w[t_h_j_offset];
                        भिन्न k_val = k[t_h_j_offset];
                        भिन्न b_val = b[t_h_j_offset];
                        भिन्न kv_val = v_val * k_val;
                        भिन्न prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
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
    memcpy(&p, dst->op_params, परिमाण(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

शून्य ggml_compute_forward_map_custom2(
        ध्रुव ggml_compute_params * params,
              ggml_tensor * dst) {

    ध्रुव ggml_tensor * a = dst->src[0];
    ध्रुव ggml_tensor * b = dst->src[1];

    संरचना ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, परिमाण(p));

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
    memcpy(&p, dst->op_params, परिमाण(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

शून्य ggml_compute_forward_custom(
    ध्रुव संरचना ggml_compute_params * params,
          संरचना ggml_tensor * dst) {

    संरचना ggml_custom_op_params p;
    memcpy(&p, dst->op_params, परिमाण(p));

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
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
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
    भिन्न sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= परिमाण(भिन्न) * (nth + nth * nc));

    // rows per thread
    ध्रुव int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव int64_t ir0 = dr*ith;
    ध्रुव int64_t ir1 = MIN(ir0 + dr, nr);

    प्रति (int64_t i1 = ir0; i1 < ir1; ++i1) {
        ध्रुव भिन्न * s0 = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src0->data + i1*src0->nb[1]);
        ध्रुव भिन्न * s1 = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(s0[i]));
            प्रतिज्ञा(!isnan(s1[i]));
        }
#endif // NDEBUG

        भिन्न max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ध्रुव ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        प्रतिज्ञा(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        भिन्न sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(st[i]));
            प्रतिज्ञा(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    यदि (ith == 0) {
        भिन्न * dp = (भिन्न *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
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

स्थिर शून्य ggml_compute_forward_cross_entropy_loss_back_f32(
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

    ध्रुव भिन्न d_by_nr = ((ध्रुव भिन्न *) grad->data)[0] / (भिन्न) nr;

    प्रति (int64_t i1 = ir0; i1 < ir1; i1++) {
        भिन्न       * ds0 = (भिन्न       *)((वर्ण       *) dst->data   + i1*dst->nb[1]);
        ध्रुव भिन्न * s0  = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src0f->data + i1*src0f->nb[1]);
        ध्रुव भिन्न * s1  = (ध्रुव भिन्न *)((ध्रुव वर्ण *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(s0[i]));
            प्रतिज्ञा(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        भिन्न max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        ध्रुव ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        प्रतिज्ञा(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        प्रति (int64_t i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(ds0[i]));
            प्रतिज्ञा(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

शून्य ggml_compute_forward_cross_entropy_loss_back(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_opt_step_adamw_f32(
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

    ध्रुव भिन्न * adamw_params_ptr = ggml_get_data_f32(adamw_params);

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

        ध्रुव size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        भिन्न       * w = (भिन्न       *) ((वर्ण       *) src0->data        + offset); // weight
        ध्रुव भिन्न * g = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_grad->data   + offset); // grad
        भिन्न       * m = (भिन्न       *) ((वर्ण       *) src0_grad_m->data + offset);
        भिन्न       * v = (भिन्न       *) ((वर्ण       *) src0_grad_v->data + offset);

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

शून्य ggml_compute_forward_opt_step_adamw(
        ध्रुव ggml_compute_params * params,
        ggml_tensor * dst) {

    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_opt_step_sgd_f32(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
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
    ध्रुव भिन्न * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    ध्रुव भिन्न   alpha            = sgd_params_ptr[0];
    ध्रुव भिन्न   keep             = 1.f - alpha * sgd_params_ptr[1];

    प्रति (पूर्णाङ्क ir = ir0; ir < ir1; ++ir) {
        ध्रुव int64_t i03 = ir / (ne02 * ne01);
        ध्रुव int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ध्रुव int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        ध्रुव size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        भिन्न *       w = (भिन्न *) ((वर्ण *) src0->data + offset);                   // weight
        ध्रुव भिन्न * g = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_grad->data + offset);  // grad

        प्रति (पूर्णाङ्क i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

शून्य ggml_compute_forward_opt_step_sgd(ध्रुव ggml_compute_params * params, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            भङ्ग;
        स्वभाव:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

स्थिर अन्तःस्थ भिन्न ggml_fwht_load(ध्रुव भिन्न value) {
    प्रत्यागम value;
}

स्थिर अन्तःस्थ भिन्न ggml_fwht_load(ध्रुव ggml_fp16_t value) {
    प्रत्यागम ggml_fp16_to_fp32(value);
}

template<typename src_t>
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
    ध्रुव int64_t rows_per_thread = (nr + nth - 1) / nth;
    ध्रुव int64_t start_row = ith * rows_per_thread;
    ध्रुव int64_t end_row = MIN(start_row + rows_per_thread, nr);

    ध्रुव भिन्न scale = 1.0f / sqrtf((भिन्न)n);

#यदि defined(GGML_SIMD)
    ध्रुव GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    प्रति (int64_t r = start_row; r < end_row; r++) {
        ध्रुव int64_t i13 = r / (ne11 * ne12);
        ध्रुव int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        ध्रुव int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        ध्रुव src_t * src_row = (ध्रुव src_t *) ((ध्रुव वर्ण *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        प्रति (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#यदि defined(GGML_SIMD)
#यदि defined(__ARM_FEATURE_SVE)
        ध्रुव पूर्णाङ्क step = svcntw();
#अन्यथा
        ध्रुव पूर्णाङ्क step = GGML_F32_EPR;
#endif
#अन्यथा
        ध्रुव पूर्णाङ्क step = n;
#endif
        प्रति (int64_t len = 1; len < step && len < n; len <<= 1) {
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
        प्रति (int64_t len = step; len < n; len <<= 1) {
            प्रति (int64_t i = 0; i < n; i += 2 * len) {
                प्रति (int64_t j = 0; j < len; j += step) {
                    GGML_F32_VEC u = GGML_F32_VEC_LOAD(dst_row + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_LOAD(dst_row + i + len + j);

                    GGML_F32_VEC_STORE(dst_row + i + j,       GGML_F32_VEC_ADD(u, v));
                    GGML_F32_VEC_STORE(dst_row + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#endif
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
    ध्रुव ggml_tensor * m = dst->src[3]; // mask

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

    GGML_ASSERT( nb0 == ggml_type_size(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_size(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(  m->type));

    ध्रुव पूर्णाङ्क n_embd    = q->ne[0];
    ध्रुव पूर्णाङ्क n_head    = q->ne[1];
    ध्रुव पूर्णाङ्क n_tokens  = q->ne[2];
    ध्रुव पूर्णाङ्क n_stream  = q->ne[3];
    ध्रुव पूर्णाङ्क n_kv      = k->ne[2];

    ggml_to_float_t ध्रुव k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    ध्रुव पूर्णाङ्क nr  = n_kv;
    ध्रुव पूर्णाङ्क ith = params->ith;
    ध्रुव पूर्णाङ्क nth = params->nth;

    // (temporary) buffer for K converted to float
    भिन्न * k_row_f32 = (भिन्न *) params->wdata + ith*(1*n_embd + CACHE_ओळ_SIZE_F32);

    // rows per thread
    ध्रुव पूर्णाङ्क dr = (nr + nth - 1)/nth;

    // row range for this thread
    ध्रुव पूर्णाङ्क ir0 = dr*ith;
    ध्रुव पूर्णाङ्क ir1 = MIN(ir0 + dr, nr);

    प्रति (पूर्णाङ्क s = 0; s < n_stream; ++s) {
        प्रति (पूर्णाङ्क t = 0; t < n_tokens; ++t) {
            ध्रुव भिन्न       *   w_row =       (भिन्न *) ((वर्ण *)   w->data + t*nbw1 +        s*nbw3);
            ध्रुव ggml_fp16_t *   m_row = (ggml_fp16_t *) ((वर्ण *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            भिन्न             * dst_row =       (भिन्न *) ((वर्ण *) dst->data + t*nb1  +        s*nb3 );
            प्रति (पूर्णाङ्क ik = ir0; ik < ir1; ++ik) {
                वर्ण * k_row = (वर्ण *) k->data + ik*nbk2 + s*nbk3;
                यदि (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } अन्यथा {
                    k_row_f32 = (भिन्न *) k_row;
                }
                भिन्न score = 0.0f;
                प्रति (पूर्णाङ्क h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    भिन्न qk = 0.0f;
                    ध्रुव भिन्न * q_row = (भिन्न *) ((वर्ण *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
                    ggml_vec_dot_f32(n_embd, &qk, 0, q_row, 0, k_row_f32, 0, 1);
                    // ReLU and weights (prescaled)
                    score += MAX(qk, 0.0f) * w_row[h];
                }
                // apply mask
                dst_row[ik] = score + GGML_CPU_FP16_TO_FP32(m_row[ik]);
            }
        }
    }
}
