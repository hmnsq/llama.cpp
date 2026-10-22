#समावेश "ops.h"

#समावेश "ggml-cpu.h"
#समावेश "ggml-impl.h"
#समावेश "binary-ops.h"
#समावेश "simd-gemm.h"
#समावेश "ggml.h"
#समावेश "unary-ops.h"
#समावेश "vec.h"

#समावेश <algorithm>
#समावेश <cfloat>
#समावेश <cmath>

// ggml_compute_forward_dup

अटल व्योम ggml_compute_forward_dup_same_cont(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    स्थिर माप_प्रकार nb0 = ggml_type_आकार(src0->type);

    स्थिर पूर्णांक ith = params->ith; // thread index
    स्थिर पूर्णांक nth = params->nth; // number of threads

    // parallelize by blocks
    स्थिर पूर्णांक nk = ggml_nelements(src0)/ggml_blck_आकार(src0->type);
    स्थिर पूर्णांक dr = (nk + nth - 1) / nth;
    स्थिर पूर्णांक k0 = dr * ith;
    स्थिर पूर्णांक k1 = MIN(k0 + dr, nk);

    अगर (k0 < k1) {
        स_नकल(
            ((अक्षर *)  dst->तथ्य + k0*nb0),
            ((अक्षर *) src0->तथ्य + k0*nb0),
            (k1 - k0) * nb0);
    }
}

ढाँचा<typename src_t, typename dst_t>
अटल व्योम ggml_compute_forward_dup_flt(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith; // thread index
    स्थिर पूर्णांक nth = params->nth; // number of threads

    // parallelize by rows
    स्थिर पूर्णांक nr = ne01;
    // number of rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1) / nth;
    // row range for this thread
    स्थिर पूर्णांक ir0 = dr * ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    अगर (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_आकार(src0->type) && nb0 == ggml_type_आकार(dst->type)) {
        // copy by rows
        स्थिर माप_प्रकार rs = ne00*nb00;
        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                    स_नकल(
                        ((अक्षर *)  dst->तथ्य + i01*nb1  + i02*nb2  + i03*nb3),
                        ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        वापस;
    }

    // case: dst tensor is contiguous
    अगर (ggml_is_contiguous(dst)) {
        अगर (nb00 == माप(src_t)) {
            अगर constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                माप_प्रकार id = 0;
                स्थिर माप_प्रकार rs = ne00 * nb00;
                अक्षर * dst_ptr = (अक्षर *) dst->तथ्य;

                क्रम (पूर्णांक i03 = 0; i03 < ne03; i03++) {
                    क्रम (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        क्रम (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                            स्थिर अक्षर * src0_ptr = (अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
                            स_नकल(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } अन्यथा {
                // casting between non-quantized types
                माप_प्रकार id = 0;
                dst_t * dst_ptr = (dst_t *) dst->तथ्य;

                क्रम (पूर्णांक i03 = 0; i03 < ne03; i03++) {
                    क्रम (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        क्रम (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                            स्थिर src_t * src0_ptr = (src_t *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03);
                            क्रम (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                                भग्न tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
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

            माप_प्रकार id = 0;
            dst_t * dst_ptr = (dst_t *) dst->तथ्य;

            क्रम (पूर्णांक i03 = 0; i03 < ne03; i03++) {
                क्रम (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    क्रम (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                        क्रम (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                            स्थिर src_t * src0_ptr = (src_t *) ((अक्षर *) src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            भग्न tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        वापस;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    अगर constexpr (std::is_same_v<dst_t, src_t>) {
        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                जबतक (i10 >= ne0) {
                    i10 -= ne0;
                    अगर (++i11 == ne1) {
                        i11 = 0;
                        अगर (++i12 == ne2) {
                            i12 = 0;
                            अगर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिर अक्षर * src0_ptr = ((अक्षर *) src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              अक्षर * dst_ptr  = ((अक्षर *)  dst->तथ्य + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        स_नकल(dst_ptr, src0_ptr, माप(dst_t));

                        अगर (++i10 == ne00) {
                            i10 = 0;
                            अगर (++i11 == ne01) {
                                i11 = 0;
                                अगर (++i12 == ne02) {
                                    i12 = 0;
                                    अगर (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                जबतक (i10 >= ne0) {
                    i10 -= ne0;
                    अगर (++i11 == ne1) {
                        i11 = 0;
                        अगर (++i12 == ne2) {
                            i12 = 0;
                            अगर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } अन्यथा {
        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                जबतक (i10 >= ne0) {
                    i10 -= ne0;
                    अगर (++i11 == ne1) {
                        i11 = 0;
                        अगर (++i12 == ne2) {
                            i12 = 0;
                            अगर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिर अक्षर * src0_ptr = ((अक्षर *) src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              अक्षर * dst_ptr  = ((अक्षर *)  dst->तथ्य + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        भग्न tmp = type_conversion_table<src_t>::to_f32(*(स्थिर src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        अगर (++i10 == ne0) {
                            i10 = 0;
                            अगर (++i11 == ne1) {
                                i11 = 0;
                                अगर (++i12 == ne2) {
                                    i12 = 0;
                                    अगर (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                जबतक (i10 >= ne0) {
                    i10 -= ne0;
                    अगर (++i11 == ne1) {
                        i11 = 0;
                        अगर (++i12 == ne2) {
                            i12 = 0;
                            अगर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}


ढाँचा<typename src_t>
अटल व्योम ggml_compute_forward_dup_to_q(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith; // thread index
    स्थिर पूर्णांक nth = params->nth; // number of threads

    // parallelize by rows
    स्थिर पूर्णांक nr = ne01;
    // number of rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1) / nth;
    // row range for this thread
    स्थिर पूर्णांक ir0 = dr * ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    अगर (ggml_is_contiguous(dst) &&
            nb00 == माप(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t स्थिर quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        भग्न * src0_f32 = (भग्न *) params->wdata + (ne00 + CACHE_पंक्ति_आकार_F32) * ith;

        माप_प्रकार id = 0;
        माप_प्रकार rs = nb0 * (ne00 / ggml_blck_आकार(dst->type));
        अक्षर * dst_ptr = (अक्षर *) dst->तथ्य;

        क्रम (पूर्णांक i03 = 0; i03 < ne03; i03++) {
            क्रम (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                क्रम (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                    स्थिर src_t * src0_ptr = (src_t *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03);

                    क्रम (पूर्णांक i00 = 0; i00 < ne00; i00++) {
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
अटल व्योम ggml_compute_forward_dup_bytes(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    अगर (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        वापस;
    }

    स्थिर माप_प्रकार type_आकार = ggml_type_आकार(src0->type);

    स्थिर पूर्णांक ith = params->ith; // thread index
    स्थिर पूर्णांक nth = params->nth; // number of threads

    // parallelize by rows
    स्थिर पूर्णांक nr = ne01;
    // number of rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1) / nth;
    // row range for this thread
    स्थिर पूर्णांक ir0 = dr * ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    अगर (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_आकार && nb0 == type_आकार) {
        // copy by rows
        स्थिर माप_प्रकार rs = ggml_row_आकार(src0->type, ne00);
        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                    स_नकल(
                        ((अक्षर *)  dst->तथ्य + i01*nb1  + i02*nb2  + i03*nb3),
                        ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        वापस;
    }

    अगर (ggml_is_contiguous(dst)) {
        माप_प्रकार id = 0;
        अक्षर * dst_ptr = (अक्षर *) dst->तथ्य;
        स्थिर माप_प्रकार rs = ne00 * type_आकार;

        अगर (nb00 == type_आकार) {
            // src0 is contiguous on first dimension, copy by rows
            क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
                क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                        स्थिर अक्षर * src0_ptr = (अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
                        स_नकल(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } अन्यथा {
            //printf("%s: this is not optimal - fix me\n", __func__);

            क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
                क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                        क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                            स्थिर अक्षर * src0_ptr = (अक्षर *) src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            स_नकल(dst_ptr + id, src0_ptr, type_आकार);

                            id += type_आकार;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        वापस;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    स्थिर int64_t nk00 = ne00 / ggml_blck_आकार(src0->type);
    स्थिर int64_t nk0  = ne0  / ggml_blck_आकार(dst->type);

    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            जबतक (k10 >= nk0) {
                k10 -= nk0;
                अगर (++i11 == ne1) {
                    i11 = 0;
                    अगर (++i12 == ne2) {
                        i12 = 0;
                        अगर (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            क्रम (int64_t i01 = ir0; i01 < ir1; i01++) {
                क्रम (int64_t k00 = 0; k00 < nk00; k00++) {
                    स्थिर अक्षर * src0_ptr = ((अक्षर *) src0->तथ्य + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          अक्षर * dst_ptr  = ((अक्षर *)  dst->तथ्य + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    स_नकल(dst_ptr, src0_ptr, type_आकार);

                    अगर (++k10 == nk0) {
                        k10 = 0;
                        अगर (++i11 == ne1) {
                            i11 = 0;
                            अगर (++i12 == ne2) {
                                i12 = 0;
                                अगर (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            जबतक (k10 >= nk0) {
                k10 -= nk0;
                अगर (++i11 == ne1) {
                    i11 = 0;
                    अगर (++i12 == ne2) {
                        i12 = 0;
                        अगर (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

अटल व्योम ggml_compute_forward_dup_from_q(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर ggml_type type = src0->type;
    ggml_to_float_t स्थिर dequantize_row_q = ggml_get_type_traits(type)->to_float;

    माप_प्रकार qk = ggml_blck_आकार(type);
    स्थिर int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_आकार(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        स्थिर int64_t i03 = i/(ne00 * ne01 * ne02);
        स्थिर int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        स्थिर int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        स्थिर int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        स्थिर int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        स्थिर int64_t i13 = i/(ne10 * ne11 * ne12);
        स्थिर int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        स्थिर int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        स्थिर int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        स्थिर int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (स्थिर व्योम *) ((अक्षर *) src0->तथ्य + x_offset),
                     (भग्न *) ((अक्षर *)  dst->तथ्य + dst_offset), qk);
    }
}

व्योम ggml_compute_forward_dup(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        वापस;
    }

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
            {
                /**/ अगर (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, भग्न      >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                /**/ अगर (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, भग्न      >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                /**/ अगर (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<भग्न, ggml_fp16_t>(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<भग्न, ggml_bf16_t>(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<भग्न, भग्न      >(params, dst);
                अन्यथा अगर (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<भग्न, int32_t    >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<भग्न>(params, dst);
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                अगर (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, भग्न>(params, dst);
                अन्यथा GGML_ABORT("not implemented");
            } अवरोध;
        शेष:
            {
                अगर (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    अवरोध;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

अटल व्योम ggml_compute_forward_add_q_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर ggml_type type = src0->type;
    स्थिर ggml_type dtype = dst->type;
    ggml_to_float_t स्थिर dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t स्थिर quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == माप(भग्न));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    भग्न * wdata = (भग्न *) params->wdata + (ne00 + CACHE_पंक्ति_आकार_F32) * ith;

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        स्थिर पूर्णांक i03 = ir/(ne02*ne01);
        स्थिर पूर्णांक i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर पूर्णांक i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        स्थिर पूर्णांक i13 = i03;
        स्थिर पूर्णांक i12 = i02;
        स्थिर पूर्णांक i11 = i01;

        स्थिर पूर्णांक i3 = i03;
        स्थिर पूर्णांक i2 = i02;
        स्थिर पूर्णांक i1 = i01;

        व्योम  * src0_row = (व्योम *) ((अक्षर *) src0->तथ्य + (i01*nb01 + i02*nb02 + i03*nb03));
        भग्न * src1_row = (भग्न *)((अक्षर *) src1->तथ्य + (i11*nb11 + i12*nb12 + i13*nb13));
        व्योम  * dst_row  = (व्योम *) ((अक्षर *)  dst->तथ्य + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        निश्चित(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        अगर (quantize_row_q != शून्य) {
            quantize_row_q(wdata, dst_row, ne00);
        } अन्यथा {
            स_नकल(dst_row, wdata, ne0*nb0);
        }
    }
}

व्योम ggml_compute_forward_add(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } अवरोध;
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

अटल व्योम ggml_compute_forward_add_id_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == माप(भग्न));
    GGML_ASSERT(src1->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == माप(भग्न));
    GGML_ASSERT(nb10 == माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        स्थिर पूर्णांक i11 = *(int32_t *) ((अक्षर *) src2->तथ्य + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01),
                (भग्न *) ((अक्षर *) src1->तथ्य + i11*nb11));
    }
}

व्योम ggml_compute_forward_add_id(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

अटल व्योम ggml_compute_forward_add1_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == माप(भग्न));
    GGML_ASSERT(nb00 == माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

#अगर_घोषित GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (भग्न *) ((अक्षर *) src1->तथ्य), 0,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#अन्यथा
        ggml_vec_add1_f32(ne0,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01),
               *(भग्न *) src1->तथ्य);
#पूर्ण_अगर
    }
}

अटल व्योम ggml_compute_forward_add1_f16_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिर भग्न v = *(भग्न *) src1->तथ्य;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == माप(ggml_fp16_t));
    GGML_ASSERT(nb00 == माप(ggml_fp16_t));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01);
        क्रम (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

अटल व्योम ggml_compute_forward_add1_f16_f16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिर भग्न v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->तथ्य);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == माप(ggml_fp16_t));
    GGML_ASSERT(nb00 == माप(ggml_fp16_t));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01);
        क्रम (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

अटल व्योम ggml_compute_forward_add1_q_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिर भग्न v = *(भग्न *) src1->तथ्य;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर ggml_type type = src0->type;
    ggml_to_float_t स्थिर dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t स्थिर quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    भग्न * wdata = (भग्न *) params->wdata + (ne0 + CACHE_पंक्ति_आकार_F32) * ith;

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        व्योम  * src0_row = (व्योम *) ((अक्षर *) src0->तथ्य + (i1*nb01 + i2*nb02 + i3*nb03));
        व्योम  * dst_row  = (व्योम *) ((अक्षर *)  dst->तथ्य + (i1*nb1  + i2*nb2  + i3*nb0 ));

        निश्चित(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

अटल व्योम ggml_compute_forward_add1_bf16_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिर भग्न v = *(भग्न *) src1->तथ्य;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == माप(ggml_bf16_t));
    GGML_ASSERT(nb00 == माप(ggml_bf16_t));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01);
        क्रम (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

अटल व्योम ggml_compute_forward_add1_bf16_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिर भग्न v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->तथ्य);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == माप(ggml_bf16_t));
    GGML_ASSERT(nb00 == माप(ggml_bf16_t));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिर पूर्णांक i3 = ir/(ne2*ne1);
        स्थिर पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01);
        क्रम (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

व्योम ggml_compute_forward_add1(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                अगर (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                अन्यथा अगर (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                अन्यथा {
                    GGML_ABORT("fatal error");
                }
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                अगर (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                अन्यथा अगर (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                अन्यथा {
                    GGML_ABORT("fatal error");
                }
            } अवरोध;
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q8_1:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

अटल व्योम ggml_compute_forward_acc_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    अगर (!inplace) {
        अगर (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स_नकल(
                ((अक्षर *)  dst->तथ्य),
                ((अक्षर *) src0->तथ्य),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr = ggml_nrows(src1);
    स्थिर पूर्णांक nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    स्थिर माप_प्रकार nb0 = ggml_element_आकार(src0);

    स्थिर माप_प्रकार nb00 = nb0;
    स्थिर माप_प्रकार nb01 = nb1;
    स्थिर माप_प्रकार nb02 = nb2;
    स्थिर माप_प्रकार nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        स्थिर पूर्णांक i3 = ir/(ne12*ne11);
        स्थिर पूर्णांक i2 = (ir - i3*ne12*ne11)/ne11;
        स्थिर पूर्णांक i1 = (ir - i3*ne12*ne11 - i2*ne11);

#अगर_घोषित GGML_USE_ACCELERATE
        vDSP_vadd(
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (भग्न *) ((अक्षर *) src1->तथ्य + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#अन्यथा
        ggml_vec_add_f32(nc,
                (भग्न *) ((अक्षर *)  dst->तथ्य + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (भग्न *) ((अक्षर *) src1->तथ्य + i3*nb13 + i2*nb12 + i1*nb11));
#पूर्ण_अगर
    }
}

व्योम ggml_compute_forward_acc(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q8_1:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

अटल व्योम ggml_compute_forward_योग_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)

    ggml_float योग_2 = 0;

    अगर (nb00 == माप(भग्न)) {
        ggml_float row_योग = 0;
        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_योग_f32_ggf(ne00,
                            &row_योग,
                            (भग्न *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03));
                    योग_2 += row_योग;
                }
            }
        }
    } अन्यथा {
        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिर अक्षर * row = (स्थिर अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        योग_2 += (ggml_float) *(स्थिर भग्न *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((भग्न *) dst->तथ्य)[0] = योग_2;
}

अटल व्योम ggml_compute_forward_योग_f16(
    स्थिर ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(ggml_is_scalar(dst));

    निश्चित(src0->nb[0] == माप(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)

    भग्न योग_2 = 0;
    भग्न row_योग = 0;

    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_योग_f16_ggf(ne00,
                    &row_योग,
                    (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03));
                योग_2 += row_योग;
            }
        }
    }
    ((ggml_fp16_t *) dst->तथ्य)[0] = GGML_CPU_FP32_TO_FP16(योग_2);
}

अटल व्योम ggml_compute_forward_योग_bf16(
    स्थिर ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(ggml_is_scalar(dst));

    निश्चित(src0->nb[0] == माप(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)

    भग्न योग_2 = 0;
    भग्न row_योग = 0;

    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_योग_bf16_ggf(ne00,
                    &row_योग,
                    (ggml_bf16_t *) ((अक्षर *) src0->तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03));
                योग_2 += row_योग;
            }
        }
    }
    ((ggml_bf16_t *) dst->तथ्य)[0] = GGML_FP32_TO_BF16(योग_2);
}

व्योम ggml_compute_forward_योग(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_योग_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_योग_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_योग_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

अटल व्योम ggml_compute_forward_cumsum_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == माप(भग्न));
    GGML_ASSERT(dst->nb[0] == माप(भग्न));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, src0);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        भग्न * src_row = (भग्न *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03);
        भग्न * dst_row = (भग्न *) ((अक्षर *) dst->तथ्य  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

व्योम ggml_compute_forward_cumsum(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

अटल व्योम ggml_compute_forward_योग_rows_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    GGML_ASSERT(src0->nb[0] == माप(भग्न));
    GGML_ASSERT(dst->nb[0] == माप(भग्न));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    क्रम (int64_t i3 = 0; i3 < ne03; i3++) {
        क्रम (int64_t i2 = 0; i2 < ne02; i2++) {
            क्रम (int64_t i1 = 0; i1 < ne01; i1++) {
                भग्न * src_row = (भग्न *) ((अक्षर *) src0->तथ्य + i1*nb01 + i2*nb02 + i3*nb03);
                भग्न * dst_row = (भग्न *) ((अक्षर *) dst->तथ्य  + i1*nb1  + i2*nb2  + i3*nb3);
                भग्न row_योग = 0;
                ggml_vec_योग_f32(ne00, &row_योग, src_row);
                dst_row[0] = row_योग;
            }
        }
    }
}

व्योम ggml_compute_forward_योग_rows(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_योग_rows_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

अटल व्योम ggml_compute_forward_mean_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(src0->nb[0] == माप(भग्न));

    GGML_TENSOR_UNARY_OP_LOCALS

    निश्चित(ne0 == 1);
    निश्चित(ne1 == ne01);
    निश्चित(ne2 == ne02);
    निश्चित(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_योग_f32(ne00,
                        (भग्न *) ((अक्षर *)  dst->तथ्य + i01*nb1  + i02*nb2  + i03*nb3),
                        (भग्न *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03));

                *(भग्न *) ((अक्षर *) dst->तथ्य + i01*nb1 + i02*nb2 + i03*nb3) /= (भग्न) ne00;
            }
        }
    }
}

व्योम ggml_compute_forward_mean(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

अटल व्योम ggml_compute_forward_argmax_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(src0->nb[0] == माप(भग्न));
    निश्चित(dst->nb[0] == माप(भग्न));

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];

    स्थिर माप_प्रकार nb01 = src0->nb[1];
    स्थिर माप_प्रकार nb0 = dst->nb[0];

    क्रम (int64_t i1 = 0; i1 < ne01; i1++) {
        भग्न * src = (भग्न *) ((अक्षर *) src0->तथ्य + i1*nb01);
        int32_t * dst_ = (int32_t *) ((अक्षर *)  dst->तथ्य + i1*nb0);
        पूर्णांक v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

व्योम ggml_compute_forward_argmax(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

अटल व्योम ggml_compute_forward_गिनती_equal_i32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    स्थिर int64_t nr = ggml_nrows(src0);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t योग_thread = 0;

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 =  ir                        / (ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne03)            /       ne01;
        स्थिर int64_t i01 =  ir - i03*ne03 - i02*ne02;

        स्थिर अक्षर * data0 = (स्थिर अक्षर *) src0->तथ्य + i03*nb03 + i02*nb02 + i01*nb01;
        स्थिर अक्षर * data1 = (स्थिर अक्षर *) src1->तथ्य + i03*nb13 + i02*nb12 + i01*nb11;

        क्रम (int64_t i00 = 0; i00 < ne00; ++i00) {
            स्थिर int32_t val0 = *((स्थिर int32_t *) (data0 + i00*nb00));
            स्थिर int32_t val1 = *((स्थिर int32_t *) (data1 + i00*nb10));

            योग_thread += val0 == val1;
        }
    }
    अगर (ith != 0) {
        sums[ith] = योग_thread;
    }
    ggml_barrier(params->threadpool);

    अगर (ith != 0) {
        वापस;
    }

    क्रम (पूर्णांक ith_other = 1; ith_other < nth; ++ith_other) {
        योग_thread += sums[ith_other];
    }
    *((int64_t *) dst->तथ्य) = योग_thread;
}

व्योम ggml_compute_forward_गिनती_equal(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_I32:
            {
                ggml_compute_forward_गिनती_equal_i32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

अटल व्योम ggml_compute_forward_repeat_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    स्थिर पूर्णांक nr0 = (पूर्णांक)(ne0/ne00);
    स्थिर पूर्णांक nr1 = (पूर्णांक)(ne1/ne01);
    स्थिर पूर्णांक nr2 = (पूर्णांक)(ne2/ne02);
    स्थिर पूर्णांक nr3 = (पूर्णांक)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == माप(भग्न));
    GGML_ASSERT(nb00 == माप(भग्न));

    // TODO: maybe this is not optimal?
    क्रम                         (पूर्णांक i3 = 0; i3 < nr3;  i3++) {
        क्रम                     (पूर्णांक k3 = 0; k3 < ne03; k3++) {
            क्रम                 (पूर्णांक i2 = 0; i2 < nr2;  i2++) {
                क्रम             (पूर्णांक k2 = 0; k2 < ne02; k2++) {
                    क्रम         (पूर्णांक i1 = 0; i1 < nr1;  i1++) {
                        क्रम     (पूर्णांक k1 = 0; k1 < ne01; k1++) {
                            क्रम (पूर्णांक i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (भग्न *) ((अक्षर *)  dst->तथ्य + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (भग्न *) ((अक्षर *) src0->तथ्य + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

अटल व्योम ggml_compute_forward_repeat_f16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    स्थिर पूर्णांक nr0 = (पूर्णांक)(ne0/ne00);
    स्थिर पूर्णांक nr1 = (पूर्णांक)(ne1/ne01);
    स्थिर पूर्णांक nr2 = (पूर्णांक)(ne2/ne02);
    स्थिर पूर्णांक nr3 = (पूर्णांक)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == माप(ggml_fp16_t));
    GGML_ASSERT(nb00 == माप(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    क्रम                         (पूर्णांक i3 = 0; i3 < nr3;  i3++) {
        क्रम                     (पूर्णांक k3 = 0; k3 < ne03; k3++) {
            क्रम                 (पूर्णांक i2 = 0; i2 < nr2;  i2++) {
                क्रम             (पूर्णांक k2 = 0; k2 < ne02; k2++) {
                    क्रम         (पूर्णांक i1 = 0; i1 < nr1;  i1++) {
                        क्रम     (पूर्णांक k1 = 0; k1 < ne01; k1++) {
                            क्रम (पूर्णांक i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((अक्षर *)  dst->तथ्य + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                क्रम (पूर्णांक i = 0; i < ne00; ++i) {
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

व्योम ggml_compute_forward_repeat(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } अवरोध;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

अटल व्योम ggml_compute_forward_repeat_back_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    स्थिर पूर्णांक nr0 = (पूर्णांक)(ne00/ne0);
    स्थिर पूर्णांक nr1 = (पूर्णांक)(ne01/ne1);
    स्थिर पूर्णांक nr2 = (पूर्णांक)(ne02/ne2);
    स्थिर पूर्णांक nr3 = (पूर्णांक)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == माप(भग्न));
    GGML_ASSERT(nb00 == माप(भग्न));

    अगर (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भग्न *)dst->तथ्य, 0);
    } अन्यथा {
        क्रम         (पूर्णांक k3 = 0; k3 < ne3; k3++) {
            क्रम     (पूर्णांक k2 = 0; k2 < ne2; k2++) {
                क्रम (पूर्णांक k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (भग्न *) ((अक्षर *) dst->तथ्य + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    क्रम                         (पूर्णांक i3 = 0; i3 < nr3; i3++) {
        क्रम                     (पूर्णांक k3 = 0; k3 < ne3; k3++) {
            क्रम                 (पूर्णांक i2 = 0; i2 < nr2; i2++) {
                क्रम             (पूर्णांक k2 = 0; k2 < ne2; k2++) {
                    क्रम         (पूर्णांक i1 = 0; i1 < nr1; i1++) {
                        क्रम     (पूर्णांक k1 = 0; k1 < ne1; k1++) {
                            क्रम (पूर्णांक i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (भग्न *) ((अक्षर *)  dst->तथ्य + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (भग्न *) ((अक्षर *) src0->तथ्य + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

व्योम ggml_compute_forward_repeat_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
अटल व्योम ggml_compute_forward_concat_any(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    अगर (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_आकार(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_आकार(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_आकार(src0->type);
    } अन्यथा {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    क्रम (पूर्णांक i3 = 0; i3 < ne03; i3++) {
        क्रम (पूर्णांक i2 = ith; i2 < ne02; i2 += nth) {
            क्रम (पूर्णांक i1 = 0; i1 < ne01; i1++) {
                स्थिर अक्षर * x = (स्थिर अक्षर *) src0->तथ्य + i1*nb01 + i2*nb02 + i3*nb03;
                      अक्षर * y = (      अक्षर *) dst->तथ्य  + i1*nb1  + i2*nb2  + i3*nb3;
                स_नकल(y, x, ggml_row_आकार(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    क्रम (पूर्णांक i3 = 0; i3 < ne13; i3++) {
        क्रम (पूर्णांक i2 = ith; i2 < ne12; i2 += nth) {
            क्रम (पूर्णांक i1 = 0; i1 < ne11; i1++) {
                स्थिर अक्षर * x = (स्थिर अक्षर *) src1->तथ्य + i1*nb11         + i2*nb12         + i3*nb13;
                      अक्षर * y = (      अक्षर *)  dst->तथ्य + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                स_नकल(y, x, ggml_row_आकार(src1->type, ne10));
            }
        }
    }
}

अटल व्योम ggml_compute_forward_concat_i8(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_आकार(src0->type) == माप(int8_t));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    स्थिर int8_t * x;

    // TODO: smarter multi-theading
    क्रम (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        क्रम (पूर्णांक i2 = ith; i2 < ne2; i2 += nth) {
            क्रम (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                क्रम (पूर्णांक i0 = 0; i0 < ne0; i0++) {
                    अगर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (स्थिर int8_t *) ((स्थिर अक्षर *)src0->तथ्य + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (स्थिर int8_t *) ((स्थिर अक्षर *)src1->तथ्य + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((अक्षर *)dst->तथ्य + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

अटल व्योम ggml_compute_forward_concat_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_आकार(src0->type) == माप(ggml_fp16_t));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    स्थिर ggml_fp16_t * x;

    // TODO: smarter multi-theading
    क्रम (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        क्रम (पूर्णांक i2 = ith; i2 < ne2; i2 += nth) {
            क्रम (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                क्रम (पूर्णांक i0 = 0; i0 < ne0; i0++) {
                    अगर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (स्थिर ggml_fp16_t *) ((स्थिर अक्षर *)src0->तथ्य + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (स्थिर ggml_fp16_t *) ((स्थिर अक्षर *)src1->तथ्य + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((अक्षर *)dst->तथ्य + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

अटल व्योम ggml_compute_forward_concat_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_आकार(src0->type) == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    स्थिर भग्न * x;

    // TODO: smarter multi-theading
    क्रम (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        क्रम (पूर्णांक i2 = ith; i2 < ne2; i2 += nth) {
            क्रम (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                क्रम (पूर्णांक i0 = 0; i0 < ne0; i0++) {
                    अगर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (स्थिर भग्न *) ((स्थिर अक्षर *)src0->तथ्य + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (स्थिर भग्न *) ((स्थिर अक्षर *)src1->तथ्य + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    भग्न * y = (भग्न *)((अक्षर *)dst->तथ्य + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

व्योम ggml_compute_forward_concat(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } अवरोध;
        शेष:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
ढाँचा <व्योम (*vec_op)(स्थिर पूर्णांक, ggml_bf16_t *, स्थिर ggml_bf16_t *)>
अटल व्योम ggml_compute_forward_unary_bf16_rows(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

अटल व्योम ggml_compute_forward_gelu_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_gelu_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_gelu_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_gelu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

अटल व्योम ggml_compute_forward_fill_f32(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर भग्न c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb);

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, dst);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne2*ne1);
        स्थिर int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        स्थिर int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        भग्न * dst_ptr  = (भग्न *) ((अक्षर *) dst->तथ्य + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

अटल व्योम ggml_compute_forward_fill_f16(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb);

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, dst);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne2*ne1);
        स्थिर int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        स्थिर int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

व्योम ggml_compute_forward_fill(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

अटल व्योम ggml_compute_forward_tri_f32(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    स्थिर ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर स्वतः [ir0, ir1] = get_thread_range(params, src0);

    bool (*bipred)(पूर्णांक, पूर्णांक);

    चयन (ttype) {
        हाल GGML_TRI_TYPE_LOWER:      bipred = [](पूर्णांक i, पूर्णांक r) { वापस i <  r; }; अवरोध;
        हाल GGML_TRI_TYPE_LOWER_DIAG: bipred = [](पूर्णांक i, पूर्णांक r) { वापस i <= r; }; अवरोध;
        हाल GGML_TRI_TYPE_UPPER:      bipred = [](पूर्णांक i, पूर्णांक r) { वापस i >  r; }; अवरोध;
        हाल GGML_TRI_TYPE_UPPER_DIAG: bipred = [](पूर्णांक i, पूर्णांक r) { वापस i >= r; }; अवरोध;
        शेष: GGML_ABORT("invalid tri type");
    }

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        स्थिर भग्न * src_ptr = (स्थिर भग्न  *) ((स्थिर अक्षर *) src0->तथ्य + i03*nb03 + i02*nb02 + i01*nb01);
              भग्न * dst_ptr = (      भग्न  *) ((      अक्षर *) dst->तथ्य  + i03*nb3  + i02*nb2  + i01*nb1);

        क्रम (पूर्णांक i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

व्योम ggml_compute_forward_tri(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

अटल व्योम ggml_compute_forward_gelu_erf_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_gelu_erf_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_gelu_erf_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_gelu_erf(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

अटल व्योम ggml_compute_forward_gelu_quick_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_gelu_quick_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_gelu_quick_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_gelu_quick(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

अटल व्योम ggml_compute_forward_silu_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_silu_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(ggml_is_contiguous_rows(src0));
    निश्चित(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर पूर्णांक i3 = ir/(ne02*ne01);
        स्थिर पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिर पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_silu_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_silu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

अटल व्योम ggml_compute_forward_leaky_relu_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(ggml_is_contiguous_1(src0));
    निश्चित(ggml_is_contiguous_1(dst));
    निश्चित(ggml_are_same_shape(src0, dst));

    स्थिर पूर्णांक n  = ggml_nrows(src0);
    स्थिर पूर्णांक nc = src0->ne[0];

    भग्न negative_slope;
    स_नकल(&negative_slope, dst->op_params, माप(भग्न));

    निश्चित(dst->nb[0]  == माप(भग्न));
    निश्चित(src0->nb[0] == माप(भग्न));

    क्रम (पूर्णांक i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i*( dst->nb[1])),
                (भग्न *) ((अक्षर *) src0->तथ्य + i*(src0->nb[1])), negative_slope);
    }
}

अटल व्योम ggml_compute_forward_leaky_relu_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    निश्चित(ggml_is_contiguous_1(src0));
    निश्चित(ggml_is_contiguous_1(dst));
    निश्चित(ggml_are_same_shape(src0, dst));

    स्थिर पूर्णांक n  = ggml_nrows(src0);
    स्थिर पूर्णांक nc = src0->ne[0];

    भग्न negative_slope;
    स_नकल(&negative_slope, dst->op_params, माप(भग्न));

    निश्चित(dst->nb[0]  == माप(ggml_fp16_t));
    निश्चित(src0->nb[0] == माप(ggml_fp16_t));

    क्रम (पूर्णांक i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i*(src0->nb[1])), negative_slope);
    }
}

व्योम ggml_compute_forward_leaky_relu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

अटल व्योम ggml_compute_forward_silu_back_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * grad = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    निश्चित(ggml_is_contiguous_1(grad));
    निश्चित(ggml_is_contiguous_1(src1));
    निश्चित(ggml_is_contiguous_1(dst));
    निश्चित(ggml_are_same_shape(src1, dst));
    निश्चित(ggml_are_same_shape(src1, grad));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i1*( dst->nb[1])),
                (भग्न *) ((अक्षर *) src1->तथ्य + i1*(src1->nb[1])),
                (भग्न *) ((अक्षर *) grad->तथ्य + i1*(grad->nb[1])));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_silu_back_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * grad = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    निश्चित(ggml_is_contiguous_1(grad));
    निश्चित(ggml_is_contiguous_1(src1));
    निश्चित(ggml_is_contiguous_1(dst));
    निश्चित(ggml_are_same_shape(src1, dst));
    निश्चित(ggml_are_same_shape(src1, grad));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->तथ्य  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((अक्षर *) src1->तथ्य + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((अक्षर *) grad->तथ्य + i1*(grad->nb[1])));

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

व्योम ggml_compute_forward_silu_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
ढाँचा <व्योम (*vec_op)(स्थिर पूर्णांक, ggml_bf16_t *, स्थिर ggml_bf16_t *, स्थिर ggml_bf16_t *)>
अटल व्योम ggml_compute_forward_glu_bf16_rows(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        स्थिर ggml_bf16_t * src0_p = (स्थिर ggml_bf16_t *) (src0_d + i1*src0_o);
        स्थिर ggml_bf16_t * src1_p = (स्थिर ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1]));

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

अटल व्योम ggml_compute_forward_reglu_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1*src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (भग्न *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_reglu_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_reglu_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_reglu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

अटल व्योम ggml_compute_forward_geglu_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1*src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (भग्न *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_geglu_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_geglu_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_geglu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

अटल व्योम ggml_compute_forward_swiglu_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1*src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (भग्न *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_swiglu_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_swiglu_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_swiglu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

अटल व्योम ggml_compute_forward_swiglu_oai_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिर भग्न alpha = ggml_get_op_params_f32(dst, 2);
    स्थिर भग्न limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1*src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1*src1_o);
        भग्न * dst_p  = (भग्न *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1]));

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = std::min(src0_p[k], limit);
            स्थिर भग्न y = std::clamp(src1_p[k], -limit, limit);
            स्थिर भग्न out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = dst_p[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_swiglu_oai_f16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिर भग्न alpha = ggml_get_op_params_f32(dst, 2);
    स्थिर भग्न limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1]));

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            स्थिर भग्न y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            स्थिर भग्न out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_swiglu_oai(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

अटल व्योम ggml_compute_forward_swiglu_clamp_f32(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0   = dst->src[0];
    स्थिर ggml_tensor * src1   = dst->src[1];
    अक्षर *              src0_d = (अक्षर *) src0->तथ्य;
    अक्षर *              src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार        src0_o = src0->nb[1];
    स्थिर माप_प्रकार        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिर भग्न   limit   = ggml_get_op_params_f32(dst, 3);

    स्थिर पूर्णांक dr  = (nr + nth - 1) / nth;
    स्थिर पूर्णांक ir0 = dr * ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1 * src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1 * src1_o);
        भग्न * dst_p  = (भग्न *) ((अक्षर *) dst->तथ्य + i1 * (dst->nb[1]));

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न gate = std::min(src0_p[k], limit);
            स्थिर भग्न up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = dst_p[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर  // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_swiglu_clamp_f16(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0   = dst->src[0];
    स्थिर ggml_tensor * src1   = dst->src[1];
    अक्षर *              src0_d = (अक्षर *) src0->तथ्य;
    अक्षर *              src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार        src0_o = src0->nb[1];
    स्थिर माप_प्रकार        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिर भग्न   limit   = ggml_get_op_params_f32(dst, 3);

    स्थिर पूर्णांक dr  = (nr + nth - 1) / nth;
    स्थिर पूर्णांक ir0 = dr * ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1 * (dst->nb[1]));

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            स्थिर भग्न up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर  // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_swiglu_clamp(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    चयन (dst->src[0]->type) {
        हाल GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            अवरोध;
        हाल GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            अवरोध;
        शेष:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

अटल व्योम ggml_compute_forward_geglu_erf_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1*src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (भग्न *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_geglu_erf_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_geglu_erf_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_geglu_erf(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

अटल व्योम ggml_compute_forward_geglu_quick_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * src0_p = (भग्न *) (src0_d + i1*src0_o);
        भग्न * src1_p = (भग्न *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (भग्न *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर भग्न x = ((भग्न *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            निश्चित(!isnan(x));
            निश्चित(!isinf(x));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_geglu_quick_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->तथ्य;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->तथ्य : src0->तथ्य);
    स्थिर माप_प्रकार src0_o = src0->nb[1];
    स्थिर माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    अगर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिर int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        अगर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*(dst->nb[1])), src0_p, src1_p);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            स्थिर ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->तथ्य + i1*( dst->nb[1])))[k];
            स्थिर भग्न v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            निश्चित(!isnan(v));
            निश्चित(!isinf(v));
        }
#पूर्ण_अगर // NDEBUG
    }
}

अटल व्योम ggml_compute_forward_geglu_quick_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

अटल व्योम ggml_compute_forward_geglu_quick(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

अटल व्योम ggml_compute_forward_norm_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भग्न eps;
    स_नकल(&eps, dst->op_params, माप(भग्न));

    GGML_ASSERT(eps >= 0.0f);

    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिर अक्षर * x = (स्थिर अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;
                अक्षर * y = (अक्षर *) dst->तथ्य + i01*nb1 + i02*nb2 + i03*nb3;

                अगर (nb00 == माप(भग्न) && nb0 == माप(भग्न)) {
                    स्थिर भग्न * xf = (स्थिर भग्न *) x;

                    भग्न योग_2 = 0.0;
                    ggml_vec_योग_f32(ne00, &योग_2, xf);
                    भग्न mean = योग_2/ne00;

                    भग्न * yf = (भग्न *) y;
                    भग्न variance = 0;

#अगर_घोषित GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#अन्यथा
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#पूर्ण_अगर //GGML_USE_ACCELERATE

                    स्थिर भग्न scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } अन्यथा {
                    भग्न योग_2 = 0.0;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        योग_2 += *(स्थिर भग्न *) (x + i00*nb00);
                    }
                    स्थिर भग्न mean = योग_2/ne00;

                    भग्न variance = 0.0f;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिर भग्न v = *(स्थिर भग्न *) (x + i00*nb00) - mean;
                        *(भग्न *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    स्थिर भग्न scale = 1.0f/sqrtf(variance + eps);
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(भग्न *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

व्योम ggml_compute_forward_norm(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
क्रमागत ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

ढाँचा <ggml_rms_norm_fuse_op FUSE_OP>
अटल व्योम ggml_compute_forward_rms_norm_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    स्थिर ggml_tensor * src0 = dst_rms_norm->src[0];
    स्थिर ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    अगर constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    भग्न eps;
    स_नकल(&eps, dst_rms_norm->op_params, माप(भग्न));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिर भग्न * x = (भग्न *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float योग_2 = 0.0;
                // worth switching to explicit SIMD?
                क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                    योग_2 += (ggml_float)(x[i00] * x[i00]);
                }

                स्थिर भग्न mean  = योग_2/ne00;
                स्थिर भग्न scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                निश्चित(scale > 0.0f);

                भग्न * y = (भग्न *) ((अक्षर *) dst->तथ्य + i01*nb1 + i02*nb2 + i03*nb3);

                अगर constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    स्थिर int64_t i11 = i01 % ne11;
                    स्थिर int64_t i12 = i02 % ne12;
                    स्थिर int64_t i13 = i03 % ne13;
                    स्थिर भग्न * w = (भग्न *) ((अक्षर *) src1->तथ्य + i11*nb11 + i12*nb12 + i13*nb13);

                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } अन्यथा {
                    स_नकल(y, x, ne00 * माप(भग्न));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

व्योम ggml_compute_forward_rms_norm(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
व्योम ggml_compute_forward_rms_norm_mul_fused(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    स्थिर ggml_tensor * src0 = dst_rms_norm->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_compute_forward_rms_norm_back_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    स्थिर ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == माप(भग्न));
    GGML_ASSERT(src1->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    भग्न eps;
    स_नकल(&eps, dst->op_params, माप(भग्न));

    // TODO: optimize
    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                स्थिर int64_t i11 = i01;
                स्थिर int64_t i12 = i02;
                स्थिर int64_t i13 = i03;

                स्थिर भग्न * dz = (भग्न *) ((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03);
                स्थिर भग्न * x  = (भग्न *) ((अक्षर *) src1->तथ्य + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float योग_xx  = 0.0;
                ggml_float योग_xdz = 0.0;

                क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                    योग_xx  += (ggml_float)(x[i00] * x[i00]);
                    योग_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                स्थिर भग्न mean_eps = (भग्न)(योग_xx)/ne00 + eps;
                स्थिर भग्न योग_eps  = (भग्न)(योग_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                स्थिर भग्न rrms     = 1.0f / sqrtf(mean_eps);
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
                भग्न * dx = (भग्न *) ((अक्षर *) dst->तथ्य + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                स्थिर भग्न scale_x = (भग्न) (-योग_xdz) / योग_eps;
                क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

व्योम ggml_compute_forward_rms_norm_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

अटल व्योम ggml_compute_forward_group_norm_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    भग्न eps;
    स_नकल(&eps, dst->op_params + 1, माप(भग्न));

    पूर्णांक n_channels = src0->ne[2];
    पूर्णांक n_groups = dst->op_params[0];
    पूर्णांक n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    क्रम (पूर्णांक i = ith; i < n_groups; i += nth) {
        पूर्णांक start = i * n_channels_per_group;
        पूर्णांक end = start + n_channels_per_group;
        अगर (end > n_channels) {
            end = n_channels;
        }
        पूर्णांक step = end - start;

        क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float योग_2 = 0.0;
            क्रम (int64_t i02 = start; i02 < end; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिर भग्न * x = (भग्न *)((अक्षर *) src0->तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    योग_2 += sumr;
                }
            }
            स्थिर भग्न mean = योग_2 / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            क्रम (int64_t i02 = start; i02 < end; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिर भग्न * x = (भग्न *)((अक्षर *) src0->तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    भग्न * y = (भग्न *)((अक्षर *) dst->तथ्य + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        भग्न v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            स्थिर भग्न variance = sum2 / (ne00 * ne01 * step);
            स्थिर भग्न scale = 1.0f / sqrtf(variance + eps);

            क्रम (int64_t i02 = start; i02 < end; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    भग्न * y = (भग्न *)((अक्षर *) dst->तथ्य + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

व्योम ggml_compute_forward_group_norm(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

अटल व्योम ggml_compute_forward_l2_norm_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भग्न eps;
    स_नकल(&eps, dst->op_params, माप(भग्न));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिर अक्षर * x = (स्थिर अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float योग_2 = 0.0;
                क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                    स्थिर भग्न xi = *(स्थिर भग्न *) (x + i00*nb00);
                    योग_2 += (ggml_float)(xi * xi);
                }

                स्थिर भग्न scale = 1.0f/fmaxf(sqrtf(योग_2), eps);

                अक्षर * y = (अक्षर *) dst->तथ्य + i01*nb1 + i02*nb2 + i03*nb3;

                अगर (nb00 == माप(भग्न) && nb0 == माप(भग्न)) {
                    स_नकल(y, x, ne00 * माप(भग्न));
                    ggml_vec_scale_f32(ne00, (भग्न *) y, scale);
                } अन्यथा {
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिर भग्न xi = *(स्थिर भग्न *) (x + i00*nb00);
                        *(भग्न *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

व्योम ggml_compute_forward_l2_norm(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

अटल व्योम ggml_compute_forward_out_prod_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == माप(भग्न));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    अगर (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भग्न *)dst->तथ्य, 0);
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
    स्थिर int64_t nr = ne1*ne2*ne3;

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    स्थिर int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    स्थिर int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    स्थिर int64_t dps2 = ne2 / ne02;
    स्थिर int64_t dps3 = ne3 / ne03;

    क्रम (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        स्थिर int64_t bir1 = MIN(bir + blck_1, ir1);
        क्रम (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            स्थिर int64_t bne01 = MIN(bi01 + blck_0, ne01);
            क्रम (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                स्थिर int64_t i3 = ir/(ne2*ne1);
                स्थिर int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                स्थिर int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                स्थिर int64_t i02 = i2 / dps2;
                स्थिर int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                स्थिर int64_t i12 = i2;
                स्थिर int64_t i13 = i3;

#अगर GGML_VEC_MAD_UNROLL > 2
                स्थिर int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                क्रम (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    स्थिर int64_t i11 = i01;

                    भग्न * s0 = (भग्न *) ((अक्षर *) src0->तथ्य + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भग्न * s1 = (भग्न *) ((अक्षर *) src1->तथ्य + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भग्न * d  = (भग्न *) ((अक्षर *)  dst->तथ्य + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                क्रम (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    स्थिर int64_t i11 = i01;

                    भग्न * s0 = (भग्न *) ((अक्षर *) src0->तथ्य + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भग्न * s1 = (भग्न *) ((अक्षर *) src1->तथ्य + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भग्न * d  = (भग्न *) ((अक्षर *)  dst->तथ्य + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#अन्यथा
                क्रम (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    स्थिर int64_t i11 = i01;

                    भग्न * s0 = (भग्न *) ((अक्षर *) src0->तथ्य + (          i01*nb01 + i02*nb02 + i03*nb03));
                    भग्न * s1 = (भग्न *) ((अक्षर *) src1->तथ्य + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    भग्न * d  = (भग्न *) ((अक्षर *)  dst->तथ्य + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#पूर्ण_अगर
            }
        }
    }
}

अटल व्योम ggml_compute_forward_out_prod_q_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर ggml_type type = src0->type;
    ggml_to_float_t स्थिर dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_आकार(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    अगर (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भग्न *)dst->तथ्य, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    स्थिर int64_t nr = ne1*ne2*ne3;

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    भग्न * wdata = (भग्न *) params->wdata + (ne0 + CACHE_पंक्ति_आकार_F32) * ith;

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        स्थिर int64_t i3 = ir/(ne2*ne1);
        स्थिर int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        स्थिर int64_t i02 = i2;
        स्थिर int64_t i03 = i3;

        //const int64_t i10 = i1;
        स्थिर int64_t i12 = i2;
        स्थिर int64_t i13 = i3;

        क्रम (int64_t i01 = 0; i01 < ne01; ++i01) {
            स्थिर int64_t i11 = i01;

            भग्न * s0 = (भग्न *) ((अक्षर *) src0->तथ्य + (          i01*nb01 + i02*nb02 + i03*nb03));
            भग्न * s1 = (भग्न *) ((अक्षर *) src1->तथ्य + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            भग्न * d  = (भग्न *) ((अक्षर *)  dst->तथ्य + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

अटल व्योम ggml_compute_forward_out_prod_f16_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == माप(ggml_fp16_t));
    GGML_ASSERT(nb0  == माप(भग्न));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    अगर (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (भग्न *)dst->तथ्य, 0);
    }
    ggml_barrier(params->threadpool);

    स्थिर int64_t nr = ne1*ne2*ne3;
    स्थिर int64_t dr = (nr + nth - 1)/nth;
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    भग्न * wdata = (भग्न *) params->wdata + (ne0 + CACHE_पंक्ति_आकार_F32) * ith;

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i3 = ir/(ne2*ne1);
        स्थिर int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिर int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        स्थिर int64_t i02 = i2;
        स्थिर int64_t i03 = i3;

        स्थिर int64_t i12 = i2;
        स्थिर int64_t i13 = i3;

        भग्न * d = (भग्न *) ((अक्षर *) dst->तथ्य + (i1*nb1 + i2*nb2 + i3*nb3));

        क्रम (int64_t i01 = 0; i01 < ne01; ++i01) {
            स्थिर int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + (i01*nb01 + i02*nb02 + i03*nb03));
            भग्न * s1 = (भग्न *) ((अक्षर *) src1->तथ्य + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

व्योम ggml_compute_forward_out_prod(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

अटल व्योम ggml_compute_forward_scale_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    भग्न s; // scale factor
    भग्न b; // bias

    स_नकल(&s, (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&b, (भग्न *) dst->op_params + 1, माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    स्थिर माप_प्रकार nb01 = src0->nb[1];

    स्थिर माप_प्रकार nb1 = dst->nb[1];

    अगर (b == 0.0f) {
        क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
            अगर (dst->तथ्य != src0->तथ्य) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                स_नकल((अक्षर *)dst->तथ्य + i1*nb1, (अक्षर *)src0->तथ्य + i1*nb01, nc * माप(भग्न));
            }
            ggml_vec_scale_f32(nc, (भग्न *) ((अक्षर *) dst->तथ्य + i1*nb1), s);
        }
    } अन्यथा {
        क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (भग्न *) ((अक्षर *) dst->तथ्य  + i1*nb1),
                (भग्न *) ((अक्षर *) src0->तथ्य + i1*nb1),
                s, b);
        }
    }
}

अटल व्योम ggml_compute_forward_scale_bf16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    भग्न s; // scale factor
    भग्न b; // bias

    स_नकल(&s, (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&b, (भग्न *) dst->op_params + 1, माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        स्थिर ggml_bf16_t * x = (स्थिर ggml_bf16_t *) ((स्थिर अक्षर *) src0->तथ्य + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((अक्षर *)       dst->तथ्य  + i1*dst->nb[1]);
        क्रम (पूर्णांक k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

व्योम ggml_compute_forward_scale(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

अटल व्योम ggml_compute_forward_set_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    अगर (!inplace) {
        अगर (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स_नकल(
                ((अक्षर *)  dst->तथ्य),
                ((अक्षर *) src0->तथ्य),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr = ggml_nrows(src1);
    स्थिर पूर्णांक nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb1, src1, nb)

    // src0 and dst as viewed during set
    स्थिर माप_प्रकार nb0 = ggml_element_आकार(src0);

    स्थिर पूर्णांक im0 = (ne10 == 0 ? 0 : ne10-1);
    स्थिर पूर्णांक im1 = (ne11 == 0 ? 0 : ne11-1);
    स्थिर पूर्णांक im2 = (ne12 == 0 ? 0 : ne12-1);
    स्थिर पूर्णांक im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        स्थिर पूर्णांक i3 = ir/(ne12*ne11);
        स्थिर पूर्णांक i2 = (ir - i3*ne12*ne11)/ne11;
        स्थिर पूर्णांक i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (भग्न *) ((अक्षर *)  dst->तथ्य + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (भग्न *) ((अक्षर *) src1->तथ्य + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

अटल व्योम ggml_compute_forward_set_i32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    bool   inplace = (bool) ((int32_t *) dst->op_params)[4];

    अगर (!inplace) {
        अगर (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स_नकल(
                ((अक्षर *)  dst->तथ्य),
                ((अक्षर *) src0->तथ्य),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr = ggml_nrows(src1);
    स्थिर पूर्णांक nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb1, src1, nb)

    // src0 and dst as viewed during set
    स्थिर माप_प्रकार nb0 = ggml_element_आकार(src0);

    स्थिर पूर्णांक im0 = (ne10 == 0 ? 0 : ne10-1);
    स्थिर पूर्णांक im1 = (ne11 == 0 ? 0 : ne11-1);
    स्थिर पूर्णांक im2 = (ne12 == 0 ? 0 : ne12-1);
    स्थिर पूर्णांक im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == माप(int32_t));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        स्थिर पूर्णांक i3 = ir/(ne12*ne11);
        स्थिर पूर्णांक i2 = (ir - i3*ne12*ne11)/ne11;
        स्थिर पूर्णांक i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((अक्षर *)  dst->तथ्य + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((अक्षर *) src1->तथ्य + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

व्योम ggml_compute_forward_set(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q8_1:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

व्योम ggml_compute_forward_cpy(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

व्योम ggml_compute_forward_cont(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

अटल व्योम ggml_compute_forward_get_rows_q(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int64_t nc = ne00;
    स्थिर int64_t nr = ggml_nelements(src1);

    स्थिर ggml_type type = src0->type;
    ggml_to_float_t स्थिर dequantize_row_q = ggml_get_type_traits(type)->to_float;

    निश्चित(ne0  == nc);
    निश्चित(ne02 == ne11);
    निश्चित(nb00 == ggml_type_आकार(type));
    निश्चित(ggml_nrows(dst) == nr);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t i = ir0; i < ir1; ++i) {
        स्थिर int64_t i12 = i/(ne11*ne10);
        स्थिर int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिर int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिर int64_t i01 = *(int32_t *) ((अक्षर *) src1->तथ्य + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (स्थिर व्योम *) ((अक्षर *) src0->तथ्य + i01*nb01 + i11*nb02 + i12*nb03),
                     (भग्न *) ((अक्षर *)  dst->तथ्य + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

अटल व्योम ggml_compute_forward_get_rows_f16(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int64_t nc = ne00;
    स्थिर int64_t nr = ggml_nelements(src1);

    निश्चित(ne0  == nc);
    निश्चित(ne02 == ne11);
    निश्चित(nb00 == माप(ggml_fp16_t));
    निश्चित(ggml_nrows(dst) == nr);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t i = ir0; i < ir1; ++i) {
        स्थिर int64_t i12 = i/(ne11*ne10);
        स्थिर int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिर int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिर int64_t i01 = *(int32_t *) ((अक्षर *) src1->तथ्य + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (स्थिर ggml_fp16_t*) ((अक्षर *) src0->तथ्य + i01*nb01 + i11*nb02 + i12*nb03),
                       (भग्न *) ((अक्षर *)  dst->तथ्य + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

अटल व्योम ggml_compute_forward_get_rows_bf16(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int64_t nc = ne00;
    स्थिर int64_t nr = ggml_nelements(src1);

    निश्चित(ne0  == nc);
    निश्चित(ne02 == ne11);
    निश्चित(nb00 == माप(ggml_bf16_t));
    निश्चित(ggml_nrows(dst) == nr);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t i = ir0; i < ir1; ++i) {
        स्थिर int64_t i12 = i/(ne11*ne10);
        स्थिर int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिर int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिर int64_t i01 = *(int32_t *) ((अक्षर *) src1->तथ्य + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (स्थिर ggml_bf16_t *) ((अक्षर *) src0->तथ्य + i01*nb01 + i11*nb02 + i12*nb03),
                        (भग्न *) ((अक्षर *)  dst->तथ्य + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

अटल व्योम ggml_compute_forward_get_rows_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int64_t nc = ne00;
    स्थिर int64_t nr = ggml_nelements(src1);

    निश्चित(ne0  == nc);
    निश्चित(ne02 == ne11);
    निश्चित(nb00 == माप(भग्न));
    निश्चित(ggml_nrows(dst) == nr);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t i = ir0; i < ir1; ++i) {
        स्थिर int64_t i12 = i/(ne11*ne10);
        स्थिर int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिर int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिर int64_t i01 = *(int32_t *) ((अक्षर *) src1->तथ्य + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (भग्न *) ((अक्षर *)  dst->तथ्य + i10*nb1  + i11*nb2  + i12*nb3),
                (भग्न *) ((अक्षर *) src0->तथ्य + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

व्योम ggml_compute_forward_get_rows(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q8_1:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
        हाल GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } अवरोध;
        शेष:
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

ढाँचा<typename src_t, typename idx_t>
अटल व्योम ggml_compute_forward_set_rows_impl(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर int64_t nc = ne00;
    स्थिर int64_t nr = ne01;

    निश्चित(ne0  == nc);
    निश्चित(ne2  == ne02);
    निश्चित(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    निश्चित(ne02 % ne11 == 0);
    निश्चित(ne03 % ne12 == 0);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = std::min(ir0 + dr, nr);

    स्थिर माप_प्रकार rs = ggml_row_आकार(src0->type, nc);

    ggml_from_float_t स्थिर from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    क्रम (int64_t i03 = 0; i03 < ne03; ++i03) {
        क्रम (int64_t i02 = 0; i02 < ne02; ++i02) {
            क्रम (int64_t i = ir0; i < ir1; ++i) {
                स्थिर int64_t i12 = i03%ne12;
                स्थिर int64_t i11 = i02%ne11;
                स्थिर int64_t i10 = i;

                स्थिर int64_t i1 = *(idx_t *) ((अक्षर *) src1->तथ्य + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                अगर constexpr (std::is_same_v<src_t, भग्न>) {
                    from_float(
                            (स्थिर भग्न *) ((अक्षर *) src0->तथ्य +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((अक्षर *)  dst->तथ्य + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } अन्यथा अगर constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    अगर (dst->type == GGML_TYPE_F16) {
                        स_नकल(
                                            ((अक्षर *)  dst->तथ्य + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((अक्षर *) src0->तथ्य +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } अन्यथा {
                        भग्न * wdata = (भग्न *) params->wdata + (nc + CACHE_पंक्ति_आकार_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (स्थिर ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((अक्षर *) dst->तथ्य + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } अन्यथा {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

व्योम ggml_compute_forward_set_rows(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                अगर (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<भग्न, int64_t>(params, dst);
                } अन्यथा अगर (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<भग्न, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                अगर (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } अन्यथा अगर (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } अवरोध;
        शेष:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

अटल व्योम ggml_compute_forward_get_rows_back_f32_f16(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    अगर (params->ith != 0) {
        वापस;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    स_रखो(dst->तथ्य, 0, ggml_nbytes(dst));

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == माप(ggml_fp16_t));

    क्रम (पूर्णांक i = 0; i < nr; ++i) {
        स्थिर पूर्णांक r = ((int32_t *) src1->तथ्य)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        क्रम (पूर्णांक j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i*src0->nb[1]))[j];
            ((भग्न *) ((अक्षर *) dst->तथ्य + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

अटल व्योम ggml_compute_forward_get_rows_back_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    अगर (params->ith != 0) {
        वापस;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    स_रखो(dst->तथ्य, 0, ggml_nbytes(dst));

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == माप(भग्न));

    क्रम (पूर्णांक i = 0; i < nr; ++i) {
        स्थिर पूर्णांक r = ((int32_t *) src1->तथ्य)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (भग्न *) ((अक्षर *)  dst->तथ्य + r*dst->nb[1]),
                (भग्न *) ((अक्षर *)  dst->तथ्य + r*dst->nb[1]),
                (भग्न *) ((अक्षर *) src0->तथ्य + i*src0->nb[1]));
    }
}

व्योम ggml_compute_forward_get_rows_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } अवरोध;
        शेष:
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

अटल व्योम ggml_compute_forward_diag_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    अगर (params->ith != 0) {
        वापस;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == माप(भग्न));
    GGML_ASSERT(nb0  == माप(भग्न));

    क्रम (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        क्रम (पूर्णांक i2 = 0; i2 < ne2; i2++) {
            क्रम (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                भग्न * d = (भग्न *)((अक्षर *)  dst->तथ्य + i3*nb3  + i2*nb2 + i1*nb1);
                भग्न * s = (भग्न *)((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02);
                क्रम (पूर्णांक i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                क्रम (पूर्णांक i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

व्योम ggml_compute_forward_diag(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

अटल व्योम ggml_compute_forward_diag_mask_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst,
        स्थिर भग्न value) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक  n_past  = ((int32_t *) dst->op_params)[0];
    स्थिर bool inplace = src0->तथ्य == dst->तथ्य;

    GGML_ASSERT(n_past >= 0);

    अगर (!inplace) {
        अगर (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            स_नकल(
                ((अक्षर *)  dst->तथ्य),
                ((अक्षर *) src0->तथ्य),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    स्थिर पूर्णांक n  = ggml_nrows(src0);
    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = src0->ne[1];
    स्थिर पूर्णांक nz = n/nr;

    GGML_ASSERT( dst->nb[0] == माप(भग्न));
    GGML_ASSERT(src0->nb[0] == माप(भग्न));

    क्रम (पूर्णांक k = 0; k < nz; k++) {
        क्रम (पूर्णांक j = ith; j < nr; j += nth) {
            क्रम (पूर्णांक i = n_past; i < nc; i++) {
                अगर (i > n_past + j) {
                    *(भग्न *)((अक्षर *) dst->तथ्य + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

व्योम ggml_compute_forward_diag_mask_inf(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -अनन्त);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

व्योम ggml_compute_forward_diag_mask_zero(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

अटल व्योम ggml_compute_forward_soft_max_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];

    निश्चित(ggml_is_contiguous(dst));
    निश्चित(ggml_are_same_shape(src0, dst));

    भग्न scale    = 1.0f;
    भग्न max_bias = 0.0f;

    स_नकल(&scale,    (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias, (भग्न *) dst->op_params + 1, माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर int64_t nb11 = src1 ? src1->nb[1] : 1;
    स्थिर int64_t nb12 = src1 ? src1->nb[2] : 1;
    स्थिर int64_t nb13 = src1 ? src1->nb[3] : 1;

    स्थिर int64_t ne12 = src1 ? src1->ne[2] : 1;
    स्थिर int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    स्थिर uint32_t n_शीर्ष      = ne02;
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) न्यूनमान(log2(n_शीर्ष));

    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    भग्न * wp = (भग्न *) params->wdata + (ne00 + CACHE_पंक्ति_आकार_F32) * ith;

    स्थिर bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    स्थिर भग्न * sk = src2 ? (भग्न *)((अक्षर *) src2->तथ्य) : nullptr;

    क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
        क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
            क्रम (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिर int64_t i11 = i01;
                स्थिर int64_t i12 = i02%ne12;
                स्थिर int64_t i13 = i03%ne13;

                // ALiBi
                स्थिर uint32_t h = i02; //शीर्ष
                स्थिर भग्न slope = (max_bias > 0.0f) ? h < n_शीर्ष_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_शीर्ष_log2) + 1) : 1.0f;

                भग्न * sp = (भग्न *)((अक्षर *) src0->तथ्य + i01*nb01 + i02*nb02 + i03*nb03);
                भग्न * dp = (भग्न *)((अक्षर *)  dst->तथ्य + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((अक्षर *) src1->तथ्य + i11*nb11 + i12*nb12 + i13*nb13) : शून्य;
                भग्न       * mp_f32 = src1 ? (भग्न       *)((अक्षर *) src1->तथ्य + i11*nb11 + i12*nb12 + i13*nb13) : शून्य;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                अगर (mp_f32) {
                    अगर (use_f16) {
                        क्रम (पूर्णांक i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } अन्यथा {
                        क्रम (पूर्णांक i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#अगर_अघोषित न_संशोधन
                क्रम (पूर्णांक i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    निश्चित(!isnan(wp[i]));
                }
#पूर्ण_अगर // NDEBUG

                भग्न max = -अनन्त;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                अगर (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float योग_2 = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                निश्चित(योग_2 > 0.0);

                अगर (sk) {
                    योग_2 += (ggml_float) expf(sk[i02] - max);
                }

                योग_2 = 1.0/योग_2;
                ggml_vec_scale_f32(ne00, dp, योग_2);

#अगर_अघोषित न_संशोधन
                क्रम (पूर्णांक i = 0; i < ne00; ++i) {
                    निश्चित(!isnan(dp[i]));
                    निश्चित(!isinf(dp[i]));
                }
#पूर्ण_अगर // NDEBUG
            }
        }
    }
}

व्योम ggml_compute_forward_soft_max(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

अटल व्योम ggml_compute_forward_soft_max_ext_back_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    भग्न scale    = 1.0f;
    भग्न max_bias = 0.0f;

    स_नकल(&scale,    (स्थिर भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias, (स्थिर भग्न *) dst->op_params + 1, माप(भग्न));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc = src0->ne[0];
    स्थिर पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न *dy = (भग्न *)((अक्षर *) src0->तथ्य + i1*src0->nb[1]);
        भग्न *y  = (भग्न *)((अक्षर *) src1->तथ्य + i1*src1->nb[1]);
        भग्न *dx = (भग्न *)((अक्षर *) dst->तथ्य  + i1*dst->nb[1]);

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            निश्चित(!isnan(dy[i]));
            निश्चित(!isnan(y[i]));
        }
#पूर्ण_अगर // NDEBUG
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
        भग्न dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        क्रम (पूर्णांक i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#अगर_अघोषित न_संशोधन
        क्रम (पूर्णांक i = 0; i < nc; ++i) {
            निश्चित(!isnan(dx[i]));
            निश्चित(!isinf(dx[i]));
        }
#पूर्ण_अगर // NDEBUG
    }
}

व्योम ggml_compute_forward_soft_max_ext_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

अटल व्योम ggml_compute_forward_clamp_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    भग्न min;
    भग्न max;
    स_नकल(&min, (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max, (भग्न *) dst->op_params + 1, माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक n  = ggml_nrows(src0);
    स्थिर पूर्णांक nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == माप(भग्न));
    GGML_ASSERT(nb00 == माप(भग्न));

    क्रम (पूर्णांक j = ith; j < n; j += nth) {
        स्थिर int64_t i1 = j % ne01;
        स्थिर int64_t i2 = (j / ne01) % ne02;
        स्थिर int64_t i3 = j / (ne01*ne02);

        भग्न * dst_ptr  = (भग्न *) ((अक्षर *)  dst->तथ्य + i1*nb1  + i2*nb2  + i3*nb3);
        भग्न * src0_ptr = (भग्न *) ((अक्षर *) src0->तथ्य + i1*nb01 + i2*nb02 + i3*nb03);

        क्रम (पूर्णांक i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

अटल व्योम ggml_compute_forward_clamp_f16(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    भग्न min;
    भग्न max;
    स_नकल(&min, (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max, (भग्न *) dst->op_params + 1, माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक n  = ggml_nrows(src0);
    स्थिर पूर्णांक nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == माप(ggml_fp16_t));
    GGML_ASSERT(nb00 == माप(ggml_fp16_t));

    क्रम (पूर्णांक j = ith; j < n; j += nth) {
        स्थिर int64_t i1 = j % ne01;
        स्थिर int64_t i2 = (j / ne01) % ne02;
        स्थिर int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *)  dst->तथ्य + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((अक्षर *) src0->तथ्य + i1*nb01 + i2*nb02 + i3*nb03);

        क्रम (पूर्णांक i = 0; i < nc; i++) {
            भग्न v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

व्योम ggml_compute_forward_clamp(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_BF16:
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q2_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q8_1:
        हाल GGML_TYPE_MXFP4:
        हाल GGML_TYPE_NVFP4:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_TQ1_0:
        हाल GGML_TYPE_TQ2_0:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ4_NL:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ3_S:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_Q8_K:
        हाल GGML_TYPE_I8:
        हाल GGML_TYPE_I16:
        हाल GGML_TYPE_I32:
        हाल GGML_TYPE_I64:
        हाल GGML_TYPE_F64:
        हाल GGML_TYPE_गिनती:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

अटल भग्न rope_yarn_ramp(स्थिर भग्न निचला, स्थिर भग्न ऊपरी, स्थिर पूर्णांक i0) {
    स्थिर भग्न y = (i0 / 2 - निचला) / MAX(0.001f, ऊपरी - निचला);
    वापस 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
अटल व्योम rope_yarn(
    भग्न theta_extrap, भग्न freq_scale, भग्न corr_dims[2], int64_t i0, भग्न ext_factor, भग्न mscale,
    भग्न * cos_theta, भग्न * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    भग्न theta_interp = freq_scale * theta_extrap;
    भग्न theta = theta_interp;
    अगर (ext_factor != 0.0f) {
        भग्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

अटल व्योम ggml_rope_cache_init(
     भग्न theta_base, भग्न freq_scale, स्थिर भग्न * freq_factors, भग्न corr_dims[2], int64_t ne0, भग्न ext_factor, भग्न mscale,
     भग्न * cache, भग्न sin_sign, भग्न theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    भग्न theta = theta_base;
    क्रम (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        स्थिर भग्न ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

अटल व्योम ggml_mrope_cache_init(
     भग्न theta_base_t, भग्न theta_base_h, भग्न theta_base_w, भग्न theta_base_e, पूर्णांक sections[4], bool is_imrope, bool indep_sects,
     भग्न freq_scale, स्थिर भग्न * freq_factors, भग्न corr_dims[2], int64_t ne0, भग्न ext_factor, भग्न mscale,
     भग्न * cache, भग्न sin_sign, भग्न theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    भग्न theta_t = theta_base_t;
    भग्न theta_h = theta_base_h;
    भग्न theta_w = theta_base_w;
    भग्न theta_e = theta_base_e;  // extra position id for vision encoder
    पूर्णांक sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    पूर्णांक sec_w = sections[1] + sections[0];
    पूर्णांक sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    क्रम (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        स्थिर भग्न ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        पूर्णांक sector = (i0 / 2) % sect_dims;
        अगर (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            अगर (sector == 0) {
                theta_t = theta_base_t;
            }
            अन्यथा अगर (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            अन्यथा अगर (sector == sec_w) {
                theta_w = theta_base_w;
            }
            अन्यथा अगर (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        भग्न theta = theta_t;
        अगर (is_imrope) { // qwen3vl apply interleaved mrope
            अगर (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } अन्यथा अगर (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } अन्यथा अगर (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } अन्यथा {
                theta = theta_e;
            }
        } अन्यथा {
            अगर (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            अन्यथा अगर (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            अन्यथा अगर (sector >= sec_w + sections[2]) {
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


ढाँचा<typename T>
अटल व्योम rotate_pairs(स्थिर int64_t n, स्थिर int64_t n_offset, स्थिर भग्न * cache, स्थिर T * src_तथ्य, T * dst_तथ्य, स्थिर पूर्णांक scale = 2) {
  क्रम (int64_t i0 = 0; i0 < n; i0 += 2) {
    स्थिर int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    स्थिर भग्न cos_theta = cache[i0 + 0];
    स्थिर भग्न sin_theta = cache[i0 + 1];

    स्थिर T * स्थिर src = src_तथ्य + ic;
    T * dst             = dst_तथ्य + ic;

    स्थिर भग्न x0 = type_conversion_table<T>::to_f32(src[0]);
    स्थिर भग्न x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

ढाँचा<typename T> //float or ggml_fp16_t
अटल व्योम ggml_compute_forward_rope_flt(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst,
        स्थिर bool forward) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    भग्न freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    पूर्णांक sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    स्थिर पूर्णांक n_dims     = ((int32_t *) dst->op_params)[1];
    स्थिर पूर्णांक mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    स्थिर पूर्णांक n_ctx_orig = ((int32_t *) dst->op_params)[4];

    स_नकल(&freq_base,   (int32_t *) dst->op_params +  5, माप(भग्न));
    स_नकल(&freq_scale,  (int32_t *) dst->op_params +  6, माप(भग्न));
    स_नकल(&ext_factor,  (int32_t *) dst->op_params +  7, माप(भग्न));
    स_नकल(&attn_factor, (int32_t *) dst->op_params +  8, माप(भग्न));
    स_नकल(&beta_fast,   (int32_t *) dst->op_params +  9, माप(भग्न));
    स_नकल(&beta_slow,   (int32_t *) dst->op_params + 10, माप(भग्न));
    स_नकल(&sections,    (int32_t *) dst->op_params + 11, माप(पूर्णांक)*4);

    स्थिर पूर्णांक n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == माप(T));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    पूर्णांक ir = 0;

    स्थिर भग्न theta_scale = powf(freq_base, -2.0f/n_dims);

    भग्न corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    स्थिर bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    स्थिर bool mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    स्थिर bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    अगर (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    अगर (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    स्थिर भग्न * freq_factors = शून्य;
    अगर (src2 != शून्य) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (स्थिर भग्न *) src2->तथ्य;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    स्थिर भग्न sin_sign = forward ? 1.0f : -1.0f;

    स्थिर int32_t * pos = (स्थिर int32_t *) src1->तथ्य;

    int64_t last_i2 = -1;

    क्रम (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        क्रम (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            क्रम (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                अगर (ir++ < ir0) जारी; // skip rows mapped to other threads
                अगर (ir   > ir1) अवरोध;

                भग्न * cache = (भग्न *) params->wdata + (ne0 + CACHE_पंक्ति_आकार_F32)*ith;
                अगर (last_i2 != i2) {
                    अगर (!mrope_used) {
                        स्थिर int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    अन्यथा {
                        स्थिर int64_t p_t = pos[i2];
                        स्थिर int64_t p_h = pos[i2 + ne2];
                        स्थिर int64_t p_w = pos[i2 + ne2 * 2];
                        स्थिर int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_तथ्य  = (T *)((अक्षर *)  dst->तथ्य + i3*nb3  + i2*nb2  + i1*nb1);

                चयन (mode) {
                    हाल GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_तथ्य + n_offs, 1);
                        अवरोध;
                    हाल GGML_ROPE_TYPE_NEOX:
                    हाल GGML_ROPE_TYPE_MROPE:
                    हाल GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_तथ्य + n_offs);
                        अवरोध;
                    हाल GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_तथ्य);
                        अवरोध;
                    शेष:
                        GGML_ABORT("rope type not supported");
                }

                अगर (!is_vision) {
                    // fill the remain channels with data from src tensor
                    क्रम (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        अगर (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            जारी;
                        }
                        स्थिर T * स्थिर src = (T *)((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_तथ्य  = (T *)((अक्षर *)  dst->तथ्य + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_तथ्य[0] = src[0];
                        dst_तथ्य[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

व्योम ggml_compute_forward_rope(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, true);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<भग्न>(params, dst, true);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

व्योम ggml_compute_forward_rope_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, false);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<भग्न>(params, dst, false);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

अटल व्योम ggml_compute_forward_conv_transpose_1d_f16_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == माप(ggml_fp16_t));
    GGML_ASSERT(nb10 == माप(भग्न));

    अगर (ith == 0) {
        स_रखो(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * स्थिर wdata = (ggml_fp16_t *) params->wdata + 0;

            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिर ggml_fp16_t * स्थिर src = (ggml_fp16_t *)((अक्षर *) src0->तथ्य + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_तथ्य = wdata + i01*ne00*ne02;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_तथ्य[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * स्थिर wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_तथ्य = wdata;

            क्रम (int64_t i11 = 0; i11 < ne11; i11++) {
                स्थिर भग्न * स्थिर src = (भग्न *)((अक्षर *) src1->तथ्य + i11*nb11);
                क्रम (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_तथ्य[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        स_रखो(dst->तथ्य, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    स्थिर int32_t s0 = ((स्थिर int32_t*)(dst->op_params))[0];

    // total rows in dst
    स्थिर पूर्णांक nr = ne1;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * स्थिर wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * स्थिर wdata_src = wdata + nk;

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * dst_तथ्य = (भग्न *)((अक्षर *) dst->तथ्य + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        क्रम (पूर्णांक i10 = 0; i10 < ne10; i10++) {
            स्थिर पूर्णांक i1n = i10*ne11;
            क्रम (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                भग्न v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_तथ्य[i10*s0 + i00] += v;
            }
        }
    }
}

अटल व्योम ggml_compute_forward_conv_transpose_1d_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == माप(भग्न));
    GGML_ASSERT(nb10 == माप(भग्न));

    अगर (ith == 0) {
        स_रखो(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            भग्न * स्थिर wdata = (भग्न *) params->wdata + 0;

            क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिर भग्न * स्थिर src = (भग्न *)((अक्षर *) src0->तथ्य + i02*nb02 + i01*nb01);
                    भग्न * dst_तथ्य = wdata + i01*ne00*ne02;
                    क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_तथ्य[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            भग्न * स्थिर wdata = (भग्न *) params->wdata + nk;
            भग्न * dst_तथ्य = wdata;

            क्रम (int64_t i11 = 0; i11 < ne11; i11++) {
                स्थिर भग्न * स्थिर src = (भग्न *)((अक्षर *) src1->तथ्य + i11*nb11);
                क्रम (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_तथ्य[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        स_रखो(dst->तथ्य, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    स्थिर int32_t s0 = ((स्थिर int32_t*)(dst->op_params))[0];

    // total rows in dst
    स्थिर पूर्णांक nr = ne1;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    भग्न * स्थिर wdata     = (भग्न *) params->wdata + 0;
    भग्न * स्थिर wdata_src = wdata + nk;

    क्रम (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        भग्न * dst_तथ्य = (भग्न *)((अक्षर *) dst->तथ्य + i1*nb1);
        भग्न * wdata_kernel = wdata + i1*ne02*ne00;
        क्रम (पूर्णांक i10 = 0; i10 < ne10; i10++) {
            स्थिर पूर्णांक i1n = i10*ne11;
            क्रम (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                भग्न v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_तथ्य[i10*s0 + i00] += v;
            }
        }
    }
}

व्योम ggml_compute_forward_conv_transpose_1d(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
अटल व्योम ggml_compute_forward_im2col_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t s1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[2];
    स्थिर int32_t p1 = ((स्थिर int32_t *)(dst->op_params))[3];
    स्थिर int32_t d0 = ((स्थिर int32_t *)(dst->op_params))[4];
    स्थिर int32_t d1 = ((स्थिर int32_t *)(dst->op_params))[5];
    स्थिर bool is_2D = ((स्थिर int32_t *)(dst->op_params))[6] == 1;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t N  = is_2D ? ne13 : ne12;
    स्थिर int64_t IC = is_2D ? ne12 : ne11;
    स्थिर int64_t IH = is_2D ? ne11 : 1;
    स्थिर int64_t IW = ne10;

    स्थिर int64_t KH = is_2D ? ne01 : 1;
    स्थिर int64_t KW = ne00;

    स्थिर int64_t OH = is_2D ? ne2 : 1;
    स्थिर int64_t OW = ne1;

    पूर्णांक ofs0 = is_2D ? nb13 : nb12;
    पूर्णांक ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == माप(भग्न));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        भग्न * स्थिर wdata = (भग्न *) dst->तथ्य;

        क्रम (int64_t in = 0; in < N; in++) {
            क्रम (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                क्रम (int64_t iow = 0; iow < OW; iow++) {
                    क्रम (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        भग्न * dst_तथ्य = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        स्थिर भग्न * स्थिर src_तथ्य = (भग्न *)((अक्षर *) src1->तथ्य + in*ofs0 + iic*ofs1); // [IH, IW]

                        क्रम (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            क्रम (int64_t ikw = 0; ikw < KW; ikw++) {
                                स्थिर int64_t iiw = iow*s0 + ikw*d0 - p0;
                                स्थिर int64_t iih = ioh*s1 + ikh*d1 - p1;

                                अगर (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_तथ्य[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    dst_तथ्य[iic*(KH*KW) + ikh*KW + ikw] = (src_तथ्य[iih*IW + iiw]);
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
अटल व्योम ggml_compute_forward_im2col_f16(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t s1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[2];
    स्थिर int32_t p1 = ((स्थिर int32_t *)(dst->op_params))[3];
    स्थिर int32_t d0 = ((स्थिर int32_t *)(dst->op_params))[4];
    स्थिर int32_t d1 = ((स्थिर int32_t *)(dst->op_params))[5];
    स्थिर bool is_2D = ((स्थिर int32_t *)(dst->op_params))[6] == 1;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t N  = is_2D ? ne13 : ne12;
    स्थिर int64_t IC = is_2D ? ne12 : ne11;
    स्थिर int64_t IH = is_2D ? ne11 : 1;
    स्थिर int64_t IW = ne10;

    स्थिर int64_t KH = is_2D ? ne01 : 1;
    स्थिर int64_t KW = ne00;

    स्थिर int64_t OH = is_2D ? ne2 : 1;
    स्थिर int64_t OW = ne1;

    पूर्णांक ofs0 = is_2D ? nb13 : nb12;
    पूर्णांक ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * स्थिर wdata = (ggml_fp16_t *) dst->तथ्य;

        क्रम (int64_t in = 0; in < N; in++) {
            क्रम (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                क्रम (int64_t iow = 0; iow < OW; iow++) {
                    क्रम (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_तथ्य = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        स्थिर भग्न * स्थिर src_तथ्य_f32 = src1->type == GGML_TYPE_F32
                            ? (स्थिर भग्न *)((स्थिर अक्षर *) src1->तथ्य + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        स्थिर ggml_fp16_t * स्थिर src_तथ्य_f16 = src1->type == GGML_TYPE_F16
                            ? (स्थिर ggml_fp16_t *)((स्थिर अक्षर *) src1->तथ्य + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        क्रम (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            क्रम (int64_t ikw = 0; ikw < KW; ikw++) {
                                स्थिर int64_t iiw = iow*s0 + ikw*d0 - p0;
                                स्थिर int64_t iih = ioh*s1 + ikh*d1 - p1;

                                अगर (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_तथ्य[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    अगर (src_तथ्य_f32 != nullptr) {
                                        dst_तथ्य[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_तथ्य_f32[iih*IW + iiw]);
                                    } अन्यथा {
                                        dst_तथ्य[iic*(KH*KW) + ikh*KW + ikw] = src_तथ्य_f16[iih*IW + iiw];
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

व्योम ggml_compute_forward_im2col(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {
    चयन (dst->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

व्योम ggml_compute_forward_im2col_back_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    स्थिर ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t s1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[2];
    स्थिर int32_t p1 = ((स्थिर int32_t *)(dst->op_params))[3];
    स्थिर int32_t d0 = ((स्थिर int32_t *)(dst->op_params))[4];
    स्थिर int32_t d1 = ((स्थिर int32_t *)(dst->op_params))[5];
    स्थिर bool is_2D = ((स्थिर int32_t *)(dst->op_params))[6] == 1;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t N  = is_2D ? ne3 : ne2;
    स्थिर int64_t IC = is_2D ? ne2 : ne1;
    स्थिर int64_t IH = is_2D ? ne1 : 1;
    स्थिर int64_t IW = ne0;

    स्थिर int64_t KH = is_2D ? ne11 : 1;
    स्थिर int64_t KW = ne10;

    स्थिर int64_t OH = is_2D ? ne02 : 1;
    स्थिर int64_t OW = ne01;

    पूर्णांक ofs0 = is_2D ? nb3 : nb2;
    पूर्णांक ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == माप(भग्न));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        भग्न * स्थिर wdata = (भग्न *) dst->तथ्य;

        क्रम (int64_t in = 0; in < N; in++) {
            क्रम (int64_t iic = ith; iic < IC; iic += nth) {
                क्रम (int64_t iih = 0; iih < IH; iih++) {
                    क्रम (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        भग्न grad = 0.0f;
                        क्रम (int64_t ikh = 0; ikh < KH; ikh++) {
                            क्रम (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                स्थिर int64_t tmpw = (iiw + p0 - ikw*d0);
                                अगर (tmpw % s0 != 0) {
                                    जारी;
                                }
                                स्थिर int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                अगर (is_2D) {
                                    स्थिर int64_t tmph = iih + p1 - ikh*d1;

                                    अगर (tmph % s1 != 0) {
                                        जारी;
                                    }

                                    ioh = tmph / s1;
                                } अन्यथा {
                                    ioh = 0;
                                }

                                अगर (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    जारी;
                                }

                                स्थिर भग्न * स्थिर grad_in = (स्थिर भग्न *) src0->तथ्य
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        भग्न * dst_तथ्य = (भग्न *)((अक्षर *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
                        dst_तथ्य[iih*IW + iiw] = grad;
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
अटल व्योम ggml_compute_forward_im2col_3d_f16(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t s1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t s2 = ((स्थिर int32_t *)(dst->op_params))[2];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[3];
    स्थिर int32_t p1 = ((स्थिर int32_t *)(dst->op_params))[4];
    स्थिर int32_t p2 = ((स्थिर int32_t *)(dst->op_params))[5];
    स्थिर int32_t d0 = ((स्थिर int32_t *)(dst->op_params))[6];
    स्थिर int32_t d1 = ((स्थिर int32_t *)(dst->op_params))[7];
    स्थिर int32_t d2 = ((स्थिर int32_t *)(dst->op_params))[8];
    स्थिर int32_t IC = ((स्थिर int32_t *)(dst->op_params))[9];


    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t N  = ne13 / IC;
    स्थिर int64_t ID = ne12;
    स्थिर int64_t IH = ne11;
    स्थिर int64_t IW = ne10;

    स्थिर int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    स्थिर int64_t KD = ne02;
    स्थिर int64_t KH = ne01;
    स्थिर int64_t KW = ne00;

    स्थिर int64_t OD = ne3 / N;
    स्थिर int64_t OH = ne2;
    स्थिर int64_t OW = ne1;
    स्थिर int64_t OH_OW = OH*OW;
    स्थिर int64_t KD_KH_KW = KD*KH*KW;
    स्थिर int64_t KH_KW = KH*KW;
    स्थिर int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == माप(भग्न));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * स्थिर wdata = (ggml_fp16_t *) dst->तथ्य;

        क्रम (int64_t in = 0; in < N; in++) {
            क्रम (int64_t iod = 0; iod < OD; iod++) {
                क्रम (int64_t ioh = 0; ioh < OH; ioh++) {
                    क्रम (int64_t iow = 0; iow < OW; iow++) {
                        क्रम (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_तथ्य = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            स्थिर भग्न * स्थिर src_तथ्य = (स्थिर भग्न *) ((स्थिर अक्षर *)src1->तथ्य + (in*IC + iic)*nb13); // [ID, IH, IW]

                            क्रम (int64_t ikd = 0; ikd < KD; ikd++) {
                                क्रम (int64_t ikh = 0; ikh < KH; ikh++) {
                                    क्रम (int64_t ikw = 0; ikw < KW; ikw++) {
                                        स्थिर int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        स्थिर int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        स्थिर int64_t iid = iod*s2 + ikd*d2 - p2;

                                        अगर (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_तथ्य[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            स्थिर भग्न * स्थिर s = (स्थिर भग्न *) ((स्थिर अक्षर *)src_तथ्य + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_तथ्य[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(*s);
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
अटल व्योम ggml_compute_forward_im2col_3d_f32(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t s1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t s2 = ((स्थिर int32_t *)(dst->op_params))[2];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[3];
    स्थिर int32_t p1 = ((स्थिर int32_t *)(dst->op_params))[4];
    स्थिर int32_t p2 = ((स्थिर int32_t *)(dst->op_params))[5];
    स्थिर int32_t d0 = ((स्थिर int32_t *)(dst->op_params))[6];
    स्थिर int32_t d1 = ((स्थिर int32_t *)(dst->op_params))[7];
    स्थिर int32_t d2 = ((स्थिर int32_t *)(dst->op_params))[8];
    स्थिर int32_t IC = ((स्थिर int32_t *)(dst->op_params))[9];


    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t N  = ne13 / IC;
    स्थिर int64_t ID = ne12;
    स्थिर int64_t IH = ne11;
    स्थिर int64_t IW = ne10;

    स्थिर int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    स्थिर int64_t KD = ne02;
    स्थिर int64_t KH = ne01;
    स्थिर int64_t KW = ne00;

    स्थिर int64_t OD = ne3 / N;
    स्थिर int64_t OH = ne2;
    स्थिर int64_t OW = ne1;

    स्थिर int64_t OH_OW = OH*OW;
    स्थिर int64_t KD_KH_KW = KD*KH*KW;
    स्थिर int64_t KH_KW = KH*KW;
    स्थिर int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == माप(भग्न));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        भग्न * स्थिर wdata = (भग्न *) dst->तथ्य;

        क्रम (int64_t in = 0; in < N; in++) {
            क्रम (int64_t iod = 0; iod < OD; iod++) {
                क्रम (int64_t ioh = 0; ioh < OH; ioh++) {
                    क्रम (int64_t iow = 0; iow < OW; iow++) {
                        क्रम (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            भग्न * dst_तथ्य = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            स्थिर भग्न * स्थिर src_तथ्य = (स्थिर भग्न *) ((स्थिर अक्षर *)src1->तथ्य + (in*IC + iic)*nb13); // [ID, IH, IW]

                            क्रम (int64_t ikd = 0; ikd < KD; ikd++) {
                                क्रम (int64_t ikh = 0; ikh < KH; ikh++) {
                                    क्रम (int64_t ikw = 0; ikw < KW; ikw++) {
                                        स्थिर int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        स्थिर int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        स्थिर int64_t iid = iod*s2 + ikd*d2 - p2;

                                        अगर (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_तथ्य[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            स्थिर भग्न * स्थिर s = (स्थिर भग्न *) ((स्थिर अक्षर *)src_तथ्य + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
                                            dst_तथ्य[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = *s;
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


व्योम ggml_compute_forward_im2col_3d(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {
    चयन (dst->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_call_mul_mat(ggml_type type, स्थिर ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              व्योम * a, व्योम * b, भग्न * c) {
    स्थिर ggml_type_traits * traits = ggml_get_type_traits(type);
    काष्ठा ggml_tensor src1 = {};
    src1.type  = type;
    src1.ne[0] = k;
    src1.ne[1] = m;
    src1.ne[2] = 1;
    src1.ne[3] = 1;
    src1.nb[0] = traits->type_आकार;
    src1.nb[1] = k * traits->type_आकार;
    src1.nb[2] = src1.nb[1];
    src1.nb[3] = src1.nb[2];
    src1.तथ्य  = a;

    काष्ठा ggml_tensor src0 = {};
    src0.type  = type;
    src0.ne[0] = k;
    src0.ne[1] = n;
    src0.ne[2] = 1;
    src0.ne[3] = 1;
    src0.nb[0] = traits->type_आकार;
    src0.nb[1] = k * traits->type_आकार;
    src0.nb[2] = src0.nb[1];
    src0.nb[3] = src0.nb[2];
    src0.तथ्य  = b;

    काष्ठा ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = माप(भग्न);
    dst.nb[1] = n * माप(भग्न);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.तथ्य  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

अटल अंतरभूत int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    वापस (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

ढाँचा <typename elem_t>
अटल व्योम ggml_compute_forward_col2im_1d_impl(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    स्थिर int32_t s0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t OC = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t p0 = ((स्थिर int32_t *)(dst->op_params))[2];

    स्थिर int64_t K_OC = src->ne[0];
    स्थिर int64_t T_in = src->ne[1];
    स्थिर int64_t K    = K_OC / OC;
    स्थिर int64_t T_out = dst->ne[0];

    स्थिर elem_t * col_तथ्य = (स्थिर elem_t *) src->तथ्य;
    elem_t       * dst_तथ्य = (elem_t *) dst->तथ्य;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    स्थिर int64_t dr = (T_out + nth - 1) / nth;
    स्थिर int64_t it0 = dr * ith;
    स्थिर int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    क्रम (int64_t oc = 0; oc < OC; oc++) {
        क्रम (int64_t t_out = it0; t_out < it1; t_out++) {
            स्थिर int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            अगर (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            अगर (t_in_max >= T_in) t_in_max = T_in - 1;

            भग्न योग_2 = 0.0f;
            क्रम (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                अगर (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    योग_2 += type_conversion_table<elem_t>::to_f32(col_तथ्य[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_तथ्य[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(योग_2);
        }
    }
}

व्योम ggml_compute_forward_col2im_1d(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    चयन (dst->src[0]->type) {
        हाल GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<भग्न>      (params, dst); अवरोध;
        हाल GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); अवरोध;
        हाल GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); अवरोध;
        शेष: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


अटल व्योम ggml_compute_forward_conv_2d_impl(स्थिर ggml_compute_params * params,
                                              स्थिर ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              स्थिर ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    स्थिर ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    स्थिर int32_t stride_x   = dst->op_params[0];
    स्थिर int32_t stride_y   = dst->op_params[1];
    स्थिर int32_t pad_x      = dst->op_params[2];
    स्थिर int32_t pad_y      = dst->op_params[3];
    स्थिर int32_t dilation_x = dst->op_params[4];
    स्थिर int32_t dilation_y = dst->op_params[5];

    स्थिर int64_t c_in  = src->ne[2];
    स्थिर int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    स्थिर int64_t src_w = src->ne[0];
    स्थिर int64_t src_h = src->ne[1];
    स्थिर int64_t knl_w = kernel->ne[0];
    स्थिर int64_t knl_h = kernel->ne[1];
    स्थिर int64_t dst_w = dst->ne[0];
    स्थिर int64_t dst_h = dst->ne[1];

    स्थिर भग्न * src_तथ्य = (भग्न *) src->तथ्य;
    व्योम  * knl_तथ्य       = kernel->तथ्य;
    भग्न * dst_तथ्य       = (भग्न *) dst->तथ्य;

    स्थिर int64_t knl_n           = knl_w * knl_h * c_in;
    स्थिर int64_t patch_योग     = dst->ne[3] * dst_w * dst_h;

    स्थिर int64_t space_per_patch   = knl_n * traits->type_आकार + c_out * माप(भग्न);
    स्थिर int64_t batch_आकार        = params->wsize / space_per_patch;
    स्थिर int64_t patches_per_batch = batch_आकार > 8 ? (batch_आकार / 8) * 8 : batch_आकार;
    स्थिर int64_t batch_n           = (patch_योग + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_आकार >= 1);

    व्योम * tmp = params->wdata;

    क्रम (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        स्थिर int64_t patch_start_batch = batch_i * patches_per_batch;
        स्थिर int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_योग);
        स्थिर int64_t patch_n           = patch_end_batch - patch_start_batch;

        स्थिर int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        स्थिर int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        स्थिर int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        क्रम (int64_t p = patch_start; p < patch_end; ++p) {
            स्थिर int64_t  batch_n     =  p / (dst_w * dst_h);
            स्थिर int64_t  src_x       = (p / dst_w) % dst_h;
            स्थिर int64_t  src_y       =  p % dst_w;

            स्थिर भग्न * src_base = (स्थिर भग्न *)((स्थिर अक्षर *)src_तथ्य + batch_n * src->nb[3]);
            अक्षर *        dst_row  = (अक्षर *) tmp + (p % patches_per_batch) * knl_n * traits->type_आकार;

            क्रम (int64_t ic = 0; ic < c_in; ++ic) {
                क्रम (int64_t ky = 0; ky < knl_h; ++ky) {
                    क्रम (int64_t kx = 0; kx < knl_w; ++kx) {
                        स्थिर int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        स्थिर int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        भग्न src_val;
                        अगर (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } अन्यथा {
                            स्थिर भग्न * src_ptr = (स्थिर भग्न *)((स्थिर अक्षर *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        अक्षर * element_ptr = dst_row + dst_idx * traits->type_आकार;
                        अगर (kernel_type == GGML_TYPE_F32) {
                            *(भग्न *) element_ptr = src_val;
                        } अन्यथा अगर (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        भग्न * gemm_output = (भग्न *) ((अक्षर *) tmp + patches_per_batch * knl_n * traits->type_आकार);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (भग्न*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_तथ्य, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        स्थिर int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        स्थिर int64_t permute_start = params->ith * permute_per_thread;
        स्थिर int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        क्रम (int64_t i = permute_start; i < permute_end; ++i) {
            स्थिर int64_t p       = patch_start_batch + i;
            स्थिर int64_t batch_n = p / (dst_w * dst_h);
            स्थिर int64_t dst_y   = (p / dst_w) % dst_h;
            स्थिर int64_t dst_x   = p % dst_w;

            क्रम (int64_t oc = 0; oc < c_out; ++oc) {
                स्थिर भग्न value = gemm_output[i * c_out + oc];
                भग्न * dst_ptr = (भग्न *)((अक्षर *)dst_तथ्य + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

व्योम ggml_compute_forward_conv_2d(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

अटल व्योम ggml_compute_forward_conv_3d_impl(स्थिर ggml_compute_params * params,
                                              स्थिर ggml_tensor *         kernel,
                                              स्थिर ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    स्थिर ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    स्थिर int32_t s0 = dst->op_params[0];
    स्थिर int32_t s1 = dst->op_params[1];
    स्थिर int32_t s2 = dst->op_params[2];
    स्थिर int32_t p0 = dst->op_params[3];
    स्थिर int32_t p1 = dst->op_params[4];
    स्थिर int32_t p2 = dst->op_params[5];
    स्थिर int32_t d0 = dst->op_params[6];
    स्थिर int32_t d1 = dst->op_params[7];
    स्थिर int32_t d2 = dst->op_params[8];
    स्थिर int32_t c  = dst->op_params[9];
    स्थिर int32_t n  = dst->op_params[10];
    स्थिर int32_t oc = dst->op_params[11];

    स्थिर int64_t src_w = src->ne[0];
    स्थिर int64_t src_h = src->ne[1];
    स्थिर int64_t src_d = src->ne[2];
    स्थिर int64_t knl_w = kernel->ne[0];
    स्थिर int64_t knl_h = kernel->ne[1];
    स्थिर int64_t knl_d = kernel->ne[2];
    स्थिर int64_t dst_w = dst->ne[0];
    स्थिर int64_t dst_h = dst->ne[1];
    स्थिर int64_t dst_d = dst->ne[2];

    स्थिर भग्न * src_तथ्य = (भग्न *) src->तथ्य;
    व्योम  * knl_तथ्य       = kernel->तथ्य;
    भग्न * dst_तथ्य       = (भग्न *) dst->तथ्य;

    स्थिर int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    स्थिर int64_t knl_n_योग       = knl_n_per_channel * c;
    स्थिर int64_t patch_योग       = n * dst_w * dst_h * dst_d;

    स्थिर int64_t space_per_patch   = knl_n_योग * traits->type_आकार + oc * माप(भग्न);
    स्थिर int64_t batch_आकार        = params->wsize / space_per_patch;
    स्थिर int64_t patches_per_batch = batch_आकार > 8 ? (batch_आकार / 8) * 8 : batch_आकार;
    स्थिर int64_t batch_n           = (patch_योग + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_आकार >= 1);

    व्योम * tmp = params->wdata;

    क्रम (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        स्थिर int64_t patch_start_batch = batch_i * patches_per_batch;
        स्थिर int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_योग);
        स्थिर int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        स्थिर int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        स्थिर int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        स्थिर int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        क्रम (int64_t p = patch_start; p < patch_end; ++p) {
            स्थिर int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            स्थिर int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            स्थिर int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            स्थिर int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            स्थिर int64_t dst_y      = p_in_depth / dst_w;
            स्थिर int64_t dst_x      = p_in_depth % dst_w;

            अक्षर * dst_row = (अक्षर *) tmp + (p % patches_per_batch) * knl_n_योग * traits->type_आकार;

            क्रम (int64_t ic = 0; ic < c; ++ic) {
                क्रम (int64_t kz = 0; kz < knl_d; ++kz) {
                    क्रम (int64_t ky = 0; ky < knl_h; ++ky) {
                        क्रम (int64_t kx = 0; kx < knl_w; ++kx) {
                            स्थिर int64_t sz = dst_z * s2 + kz * d2 - p2;
                            स्थिर int64_t sy = dst_y * s1 + ky * d1 - p1;
                            स्थिर int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            भग्न src_val;
                            अगर (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } अन्यथा {
                                स्थिर int64_t cn_idx = batch_idx * c + ic;
                                स्थिर भग्न * src_ptr = (स्थिर भग्न *)((स्थिर अक्षर *)src_तथ्य + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            अक्षर * element_ptr = dst_row + dst_idx * traits->type_आकार;
                            अगर (kernel_type == GGML_TYPE_F32) {
                                *(भग्न *)element_ptr = src_val;
                            } अन्यथा अगर (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        भग्न * gemm_output = (भग्न *) ((अक्षर *) tmp + patches_per_batch * knl_n_योग * traits->type_आकार);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_योग, tmp, knl_तथ्य, gemm_output);

        ggml_barrier(params->threadpool);

        स्थिर int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        स्थिर int64_t permute_start = params->ith * permute_per_thread;
        स्थिर int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        क्रम (int64_t i = permute_start; i < permute_end; ++i) {
            स्थिर int64_t p = patch_start_batch + i;
            स्थिर int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            स्थिर int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            स्थिर int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            स्थिर int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            स्थिर int64_t dst_y      = p_in_depth / dst_w;
            स्थिर int64_t dst_x      = p_in_depth % dst_w;

            क्रम (int64_t ioc = 0; ioc < oc; ++ioc) {
                स्थिर भग्न value = gemm_output[i * oc + ioc];
                स्थिर int64_t ocn_idx = batch_idx * oc + ioc;
                भग्न * dst_ptr = (भग्न *)((अक्षर *)dst_तथ्य + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

व्योम ggml_compute_forward_conv_3d(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

ढाँचा <typename kernel_t>
अटल व्योम ggml_compute_forward_conv_transpose_2d_impl(
    स्थिर ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_आकार(src0->type));
    GGML_ASSERT(nb10 == माप(भग्न));

    अगर (ith == 0) {
        स_रखो(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * स्थिर wdata = (kernel_t *) params->wdata + 0;

            क्रम (int64_t i03 = 0; i03 < ne03; i03++) {
                क्रम (int64_t i02 = 0; i02 < ne02; i02++) {
                    स्थिर kernel_t * स्थिर src = (kernel_t *)((अक्षर *) src0->तथ्य + i03*nb03 + i02*nb02);
                    kernel_t * dst_तथ्य = wdata + i02*ne01*ne00*ne03;
                    क्रम (int64_t i01 = 0; i01 < ne01; i01++) {
                        क्रम (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_तथ्य[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * स्थिर wdata = (kernel_t *) params->wdata + nk;
            क्रम (पूर्णांक i13 = 0; i13 < ne13; i13++) {
                kernel_t * स्थिर wdata_b = wdata + i13*ne10*ne11*ne12;
                क्रम (पूर्णांक i12 = 0; i12 < ne12; i12++) {
                    क्रम (पूर्णांक i11 = 0; i11 < ne11; i11++) {
                        स्थिर भग्न * स्थिर src = (भग्न *)((अक्षर *) src1->तथ्य + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_तथ्य = wdata_b + i11*ne10*ne12;
                        क्रम (पूर्णांक i10 = 0; i10 < ne10; i10++) {
                            अगर constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_तथ्य[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } अन्यथा {
                                dst_तथ्य[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        स_रखो(dst->तथ्य, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    स्थिर int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    स्थिर पूर्णांक np = ne2;

    // patches per thread
    स्थिर पूर्णांक dp = (np + nth - 1)/nth;

    // patch range for this thread
    स्थिर पूर्णांक ip0 = dp*ith;
    स्थिर पूर्णांक ip1 = MIN(ip0 + dp, np);

    kernel_t * स्थिर wdata = (kernel_t *) params->wdata + 0;
    kernel_t * स्थिर wdata_src = wdata + nk;

    क्रम (पूर्णांक i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        क्रम (पूर्णांक i3 = 0; i3 < ne3; i3++) { // batch
            भग्न * dst_तथ्य = (भग्न *)((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            क्रम (पूर्णांक i11 = 0; i11 < ne11; i11++) {
                क्रम (पूर्णांक i10 = 0; i10 < ne10; i10++) {
                    स्थिर पूर्णांक i1n = i11*ne10*ne12 + i10*ne12;
                    क्रम (पूर्णांक i01 = 0; i01 < ne01; i01++) {
                        क्रम (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                            भग्न v = 0;
                            अगर constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } अन्यथा {
                                ggml_vec_dot_f32(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            }
                            dst_तथ्य[(i11*stride + i01)*ne0 + i10*stride + i00] += v;
                        }
                    }
                }
            }
        }
    }
}

व्योम ggml_compute_forward_conv_transpose_2d(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } अवरोध;
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<भग्न>(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

काष्ठा ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    पूर्णांक stride_x;
    पूर्णांक stride_y;
    पूर्णांक pad_x;
    पूर्णांक pad_y;
    पूर्णांक dilation_x;
    पूर्णांक dilation_y;
};

अटल अंतरभूत भग्न ggml_conv_2d_dw_knl_f32(स्थिर अक्षर * तथ्य, int64_t i, ggml_type type) {
    अगर (type == GGML_TYPE_F16) {
        वापस GGML_FP16_TO_FP32(((स्थिर ggml_fp16_t *)तथ्य)[i]);
    }
    वापस ((स्थिर भग्न *)तथ्य)[i];
}

अटल व्योम ggml_compute_forward_conv_2d_dw_cwhn(
        स्थिर ggml_compute_params * params,
        स्थिर ggml_tensor * src,
        स्थिर ggml_tensor * kernel,
        ggml_tensor * dst,
        स्थिर ggml_conv_2d_dw_params & p) {

    स्थिर int64_t c = p.channels;
    स्थिर अक्षर * knl_तथ्य = (स्थिर अक्षर *)kernel->तथ्य;
    स्थिर ggml_type knl_type = kernel->type;

    स्थिर int64_t rows_योग = p.dst_h * p.batch;
    स्थिर int64_t rows_per_thread = (rows_योग + params->nth - 1) / params->nth;
    स्थिर int64_t row_start = params->ith * rows_per_thread;
    स्थिर int64_t row_end = MIN(row_start + rows_per_thread, rows_योग);

#अगर_घोषित GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_आकार = GGML_F32_EPR;
    अगर (knl_type == GGML_TYPE_F32) {
    #अगर defined(__ARM_FEATURE_SVE)
        pkg_आकार = svcntw();
    #अन्यथा
        pkg_आकार = GGML_F32_EPR;
    #पूर्ण_अगर
        c_pkg_end = (c / pkg_आकार) * pkg_आकार;
    }
#अन्यथा
    स्थिर int64_t c_pkg_end = 0;
#पूर्ण_अगर

    क्रम (int64_t row = row_start; row < row_end; ++row) {
        स्थिर int64_t dst_y = row % p.dst_h;
        स्थिर भग्न * src_तथ्य = (स्थिर भग्न *)src->तथ्य + (row / p.dst_h) * p.src_w * p.src_h * c;
        क्रम (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            भग्न * dst_तथ्य = (भग्न *)dst->तथ्य + (row * p.dst_w + dst_x) * c;
            स्थिर int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            स्थिर int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#अगर_घोषित GGML_SIMD
            क्रम (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_आकार) {
                GGML_F32_VEC योग_2 = GGML_F32_VEC_ZERO;
                क्रम (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    स्थिर int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    अगर (src_y < 0 || src_y >= p.src_h) {
                        जारी;
                    }
                    क्रम (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        स्थिर int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        अगर (src_x < 0 || src_x >= p.src_w) {
                            जारी;
                        }
                        स्थिर भग्न * kp = (स्थिर भग्न *)knl_तथ्य + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_तथ्य + (src_y * p.src_w + src_x) * c + c_i);
                        योग_2 = GGML_F32_VEC_FMA(योग_2, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_तथ्य + c_i, योग_2);
            }
#पूर्ण_अगर
            क्रम (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                भग्न योग_2 = 0.0f;
                क्रम (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    स्थिर int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    अगर (src_y < 0 || src_y >= p.src_h) {
                        जारी;
                    }
                    क्रम (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        स्थिर int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        अगर (src_x < 0 || src_x >= p.src_w) {
                            जारी;
                        }
                        योग_2 += ggml_conv_2d_dw_knl_f32(knl_तथ्य, (knl_y * p.knl_w + knl_x) * c + c_i, knl_type)
                             * src_तथ्य[(src_y * p.src_w + src_x) * c + c_i];
                    }
                }
                dst_तथ्य[c_i] = योग_2;
            }
        }
    }
}

अटल व्योम ggml_compute_forward_conv_2d_dw_whcn(
        स्थिर ggml_compute_params * params,
        स्थिर ggml_tensor * src,
        स्थिर ggml_tensor * kernel,
        ggml_tensor * dst,
        स्थिर ggml_conv_2d_dw_params & p) {

    स्थिर int64_t n = p.channels * p.batch;
    स्थिर int64_t per_thread = (n + params->nth - 1) / params->nth;
    स्थिर int64_t start = params->ith * per_thread;
    स्थिर int64_t end = MIN(start + per_thread, n);
    स्थिर अक्षर * knl_base = (स्थिर अक्षर *)kernel->तथ्य;
    स्थिर ggml_type knl_type = kernel->type;

    क्रम (int64_t i = start; i < end; ++i) {
        स्थिर int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        स्थिर भग्न * src_तथ्य = (स्थिर भग्न *)src->तथ्य + i * p.src_w * p.src_h;
        भग्न * dst_तथ्य = (भग्न *)dst->तथ्य + i * p.dst_w * p.dst_h;

        क्रम (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            क्रम (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                भग्न योग_2 = 0.0f;
                क्रम (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    स्थिर int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    अगर (src_y < 0 || src_y >= p.src_h) {
                        जारी;
                    }
                    क्रम (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        स्थिर int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        अगर (src_x < 0 || src_x >= p.src_w) {
                            जारी;
                        }
                        योग_2 += ggml_conv_2d_dw_knl_f32(knl_base, knl_offset + knl_y * p.knl_w + knl_x, knl_type)
                             * src_तथ्य[src_y * p.src_w + src_x];
                    }
                }
                dst_तथ्य[dst_y * p.dst_w + dst_x] = योग_2;
            }
        }
    }
}

व्योम ggml_compute_forward_conv_2d_dw(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * kernel = dst->src[0];
    स्थिर ggml_tensor * src = dst->src[1];
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

    अगर (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } अन्यथा अगर (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } अन्यथा {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
अटल व्योम ggml_compute_forward_pool_1d_ksp(
        स्थिर ggml_compute_params * params,
        स्थिर ggml_op_pool op,
        स्थिर पूर्णांक k,
        स्थिर पूर्णांक s,
        स्थिर पूर्णांक p,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src = dst->src[0];

    निश्चित(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    अगर (params->ith != 0) {
        वापस;
    }

    स्थिर int64_t IW = src->ne[0];
    स्थिर int64_t OW = dst->ne[0];

    स्थिर int64_t nr = ggml_nrows(src);

    क्रम (int64_t ir = 0; ir < nr; ++ir) {
        स्थिर अक्षर * srow_bytes =            (स्थिर अक्षर *) src->तथ्य + ir * src->nb[1];
        भग्न      * drow       = (भग्न *) ((      अक्षर *) dst->तथ्य + ir * dst->nb[1]);

        क्रम (int64_t ow = 0; ow < OW; ++ow) {
            भग्न res = 0;
            चयन (op) {
                हाल GGML_OP_POOL_AVG: res = 0.0f;     अवरोध;
                हाल GGML_OP_POOL_MAX: res = -भग्न_उच्च; अवरोध;
                हाल GGML_OP_POOL_गिनती: GGML_ABORT("fatal error");
            }

            पूर्णांक count = 0;
            स्थिर पूर्णांक base = (पूर्णांक) ow * s - p;

            क्रम (पूर्णांक ki = 0; ki < k; ++ki) {
                स्थिर पूर्णांक j = base + ki;
                अगर (j < 0 || j >= (पूर्णांक) IW) {
                    जारी;
                }

                भग्न v;
                अगर (src->type == GGML_TYPE_F32) {
                    v = ((स्थिर भग्न *) srow_bytes)[j];
                } अन्यथा {
                    v = GGML_CPU_FP16_TO_FP32(((स्थिर ggml_fp16_t *) srow_bytes)[j]);
                }

                चयन (op) {
                    हाल GGML_OP_POOL_AVG: res += v;                अवरोध;
                    हाल GGML_OP_POOL_MAX: res =  std::max(v, res); अवरोध;
                    हाल GGML_OP_POOL_गिनती: GGML_ABORT("fatal error");
                }

                ++count;
            }

            चयन (op) {
                हाल GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; अवरोध;
                हाल GGML_OP_POOL_MAX:                                           अवरोध;
                हाल GGML_OP_POOL_गिनती: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

व्योम ggml_compute_forward_pool_1d(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर int32_t * opts = (स्थिर int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिर पूर्णांक k0 = opts[1];
    स्थिर पूर्णांक s0 = opts[2];
    स्थिर पूर्णांक p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

व्योम ggml_compute_forward_pool_2d(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src = dst->src[0];

    निश्चित(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    अगर (params->ith != 0) {
        वापस;
    }

    स्थिर int32_t * opts = (स्थिर int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिर पूर्णांक k0 = opts[1];
    स्थिर पूर्णांक k1 = opts[2];
    स्थिर पूर्णांक s0 = opts[3];
    स्थिर पूर्णांक s1 = opts[4];
    स्थिर पूर्णांक p0 = opts[5];
    स्थिर पूर्णांक p1 = opts[6];
    स्थिर अक्षर * cdata = (स्थिर अक्षर*)src->तथ्य;
    स्थिर अक्षर * स्थिर तथ्य_end = cdata + ggml_nbytes(src);

    स्थिर int64_t px = dst->ne[0];
    स्थिर int64_t py = dst->ne[1];
    स्थिर int64_t pa = px * py;

    भग्न * dplane = (भग्न *)dst->तथ्य;

    स्थिर पूर्णांक ka = k0 * k1;
    स्थिर पूर्णांक offset0 = -p0;
    स्थिर पूर्णांक offset1 = -p1;

    जबतक (cdata < तथ्य_end) {
        क्रम (पूर्णांक oy = 0; oy < py; ++oy) {
            भग्न * स्थिर drow = dplane + oy * px;
            भग्न * स्थिर out  = drow;

            क्रम (पूर्णांक ox = 0; ox < px; ++ox) {
                भग्न res = 0;
                चयन (op) {
                    हाल GGML_OP_POOL_AVG: res = 0;        अवरोध;
                    हाल GGML_OP_POOL_MAX: res = -भग्न_उच्च; अवरोध;
                    हाल GGML_OP_POOL_गिनती: GGML_ABORT("fatal error");
                }

                स्थिर पूर्णांक ix = offset0 + ox * s0;
                स्थिर पूर्णांक iy = offset1 + oy * s1;

                क्रम (पूर्णांक ky = 0; ky < k1; ++ky) {
                    अगर (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        जारी;
                    }

                    स्थिर व्योम * srow = (स्थिर व्योम *)(cdata + src->nb[1] * (iy + ky));
                    क्रम (पूर्णांक kx = 0; kx < k0; ++kx) {
                        पूर्णांक j = ix + kx;
                        अगर (j < 0 || j >= src->ne[0]) {
                            जारी;
                        }

                        स्थिर भग्न srow_j = (src->type == GGML_TYPE_F32) ? ((स्थिर भग्न*)srow)[j] : GGML_CPU_FP16_TO_FP32(((स्थिर ggml_fp16_t*)srow)[j]);
                        चयन (op) {
                            हाल GGML_OP_POOL_AVG: res += srow_j;                अवरोध;
                            हाल GGML_OP_POOL_MAX: res =  std::max(srow_j, res); अवरोध;
                            हाल GGML_OP_POOL_गिनती:               GGML_ABORT("fatal error");
                        }
                    }
                }
                चयन (op) {
                    हाल GGML_OP_POOL_AVG:           res /= ka; अवरोध;
                    हाल GGML_OP_POOL_MAX:                      अवरोध;
                    हाल GGML_OP_POOL_गिनती: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

व्योम ggml_compute_forward_pool_2d_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src  = dst->src[0];
    स्थिर ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    निश्चित(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    अगर (params->ith != 0) {
        वापस;
    }

    स्थिर int32_t * opts = (स्थिर int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिर पूर्णांक k0 = opts[1];
    स्थिर पूर्णांक k1 = opts[2];
    स्थिर पूर्णांक s0 = opts[3];
    स्थिर पूर्णांक s1 = opts[4];
    स्थिर पूर्णांक p0 = opts[5];
    स्थिर पूर्णांक p1 = opts[6];

    अक्षर       * cdata  = (अक्षर       *) dst->तथ्य;
    स्थिर अक्षर * cdataf = (स्थिर अक्षर *) dstf->तथ्य;
    स्थिर अक्षर * स्थिर तथ्य_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    स_रखो(cdata, 0, ggml_nbytes(dst));

    स्थिर int64_t px = src->ne[0];
    स्थिर int64_t py = src->ne[1];
    स्थिर int64_t pa = px * py;

    स्थिर भग्न * splane = (स्थिर भग्न *) src->तथ्य;

    स्थिर पूर्णांक ka = k0 * k1;
    स्थिर पूर्णांक offset0 = -p0;
    स्थिर पूर्णांक offset1 = -p1;

    जबतक (cdata < तथ्य_end) {
        क्रम (पूर्णांक oy = 0; oy < py; ++oy) {
            स्थिर भग्न * स्थिर srow = splane + oy * px;
            क्रम (पूर्णांक ox = 0; ox < px; ++ox) {
                स्थिर भग्न grad0 = srow[ox];

                स्थिर पूर्णांक ix = offset0 + ox * s0;
                स्थिर पूर्णांक iy = offset1 + oy * s1;

                अगर (op == GGML_OP_POOL_MAX) {
                    भग्न maxval = -भग्न_उच्च;
                    पूर्णांक kxmax = -1;
                    पूर्णांक kymax = -1;

                    क्रम (पूर्णांक ky = 0; ky < k1; ++ky) {
                        अगर (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            जारी;
                        }
                        स्थिर व्योम * drowf = (स्थिर व्योम *)(cdataf + dst->nb[1] * (iy + ky));
                        क्रम (पूर्णांक kx = 0; kx < k0; ++kx) {
                            पूर्णांक j = ix + kx;
                            अगर (j < 0 || j >= dst->ne[0]) {
                                जारी;
                            }

                            स्थिर भग्न val = dst->type == GGML_TYPE_F32 ?
                                ((स्थिर भग्न *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((स्थिर ggml_fp16_t *) drowf)[j]);
                            अगर (val <= maxval) {
                                जारी;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    अगर (kxmax == -1 || kymax == -1) {
                        जारी;
                    }

                    व्योम * drow = (व्योम *)(cdata + dst->nb[1] * (iy + kymax));
                    स्थिर पूर्णांक j = ix + kxmax;
                    अगर (dst->type == GGML_TYPE_F32) {
                        ((भग्न *) drow)[j] += grad0;
                    } अन्यथा {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((स्थिर ggml_fp16_t *) drow)[j]));
                    }
                } अन्यथा अगर (op == GGML_OP_POOL_AVG) {
                    स्थिर भग्न grad = grad0 / ka;

                    क्रम (पूर्णांक ky = 0; ky < k1; ++ky) {
                        अगर (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            जारी;
                        }
                        व्योम * drow = (व्योम *)(cdata + dst->nb[1] * (iy + ky));
                        क्रम (पूर्णांक kx = 0; kx < k0; ++kx) {
                            पूर्णांक j = ix + kx;
                            अगर (j < 0 || j >= dst->ne[0]) {
                                जारी;
                            }

                            अगर (dst->type == GGML_TYPE_F32) {
                                ((भग्न *) drow)[j] += grad;
                            } अन्यथा {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } अन्यथा {
                    GGML_ASSERT(false);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

अटल व्योम ggml_compute_forward_upscale_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भग्न sf0 = (भग्न)ne0/src0->ne[0];
    भग्न sf1 = (भग्न)ne1/src0->ne[1];
    भग्न sf2 = (भग्न)ne2/src0->ne[2];
    भग्न sf3 = (भग्न)ne3/src0->ne[3];
    भग्न pixel_offset = 0.5f;

    स्थिर int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    स्थिर ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    अगर (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (भग्न)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (भग्न)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    अगर (mode == GGML_SCALE_MODE_NEAREST) {
        क्रम (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिर int64_t i03 = i3 / sf3;
            क्रम (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिर int64_t i02 = i2 / sf2;
                क्रम (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिर int64_t i01 = i1 / sf1;
                    क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिर int64_t i00 = i0 / sf0;

                        स्थिर भग्न * x = (भग्न *)((अक्षर *) src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              भग्न * y = (भग्न *)((अक्षर *)  dst->तथ्य +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } अन्यथा अगर (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        स्वतः triangle_filter = [](भग्न x) -> भग्न {
            वापस std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        स्थिर भग्न support1  = std::max(1.0f, 1.0f / sf1);
        स्थिर भग्न invscale1 = 1.0f / support1;
        स्थिर भग्न support0  = std::max(1.0f, 1.0f / sf0);
        स्थिर भग्न invscale0 = 1.0f / support0;

        क्रम (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिर int64_t i03 = i3 / sf3;
            क्रम (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिर int64_t i02 = i2 / sf2;
                क्रम (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिर भग्न y = ((भग्न) i1 + pixel_offset) / sf1;
                    क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिर भग्न x = ((भग्न) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        स्थिर int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        स्थिर int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        स्थिर int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        स्थिर int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        भग्न val = 0.0f;
                        भग्न योग_weight = 0.0f;

                        क्रम (int64_t sy = y_min; sy < y_max; sy++) {
                            स्थिर भग्न weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            क्रम (int64_t sx = x_min; sx < x_max; sx++) {
                                स्थिर भग्न weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                स्थिर भग्न weight = weight_x * weight_y;

                                अगर (weight <= 0.0f) {
                                    जारी;
                                }

                                स्थिर भग्न pixel = *(स्थिर भग्न *)((स्थिर अक्षर *)src0->तथ्य + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                योग_weight += weight;
                            }
                        }

                        अगर (योग_weight > 0.0f) {
                            val /= योग_weight;
                        }

                        भग्न * dst_ptr = (भग्न *)((अक्षर *)dst->तथ्य + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } अन्यथा अगर (mode == GGML_SCALE_MODE_BILINEAR) {
        क्रम (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिर int64_t i03 = i3 / sf3;
            क्रम (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिर int64_t i02 = i2 / sf2;
                क्रम (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिर भग्न y = ((भग्न)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    भग्न dy = y - (भग्न)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिर भग्न x = ((भग्न)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        भग्न dx = x - (भग्न)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        स्थिर भग्न a = *(स्थिर भग्न *)((स्थिर अक्षर *)src0->तथ्य + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        स्थिर भग्न b = *(स्थिर भग्न *)((स्थिर अक्षर *)src0->तथ्य + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        स्थिर भग्न c = *(स्थिर भग्न *)((स्थिर अक्षर *)src0->तथ्य + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        स्थिर भग्न d = *(स्थिर भग्न *)((स्थिर अक्षर *)src0->तथ्य + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        स्थिर भग्न val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        भग्न * y_dst = (भग्न *)((अक्षर *)dst->तथ्य + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } अन्यथा अगर (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        स्थिर भग्न a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        स्वतः weight1 = [a](भग्न x) { वापस ((a + 2) * x - (a + 3)) * x * x + 1; };
        स्वतः weight2 = [a](भग्न x) { वापस ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        स्वतः bicubic = [=](भग्न p0, भग्न p1, भग्न p2, भग्न p3, भग्न x) {
            स्थिर भग्न w0 = weight2(x + 1);
            स्थिर भग्न w1 = weight1(x + 0);
            स्थिर भग्न w2 = weight1(1 - x);
            स्थिर भग्न w3 = weight2(2 - x);
            वापस p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        क्रम (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिर int64_t i03 = i3 / sf3;
            क्रम (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिर int64_t i02 = i2 / sf2;
                क्रम (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिर भग्न y = ((भग्न)i1 + pixel_offset) / sf1 - pixel_offset;
                    स्थिर int64_t y0 = (int64_t)floorf(y);
                    स्थिर भग्न dy = y - (भग्न)y0;

                    क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिर भग्न x = ((भग्न)i0 + pixel_offset) / sf0 - pixel_offset;
                        स्थिर int64_t x0 = (int64_t)floorf(x);
                        स्थिर भग्न dx = x - (भग्न)x0;

                        स्वतः p = [=](int64_t x_off, int64_t y_off) -> भग्न {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            वापस *(स्थिर भग्न *)((स्थिर अक्षर *)src0->तथ्य + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        स्थिर भग्न val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        भग्न * y_dst = (भग्न *)((अक्षर *)dst->तथ्य + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } अन्यथा {
        GGML_ABORT("unsupported upscale mode");
    }
}

व्योम ggml_compute_forward_upscale(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

ढाँचा<bool circular_t>
अटल व्योम ggml_compute_forward_pad_f32(
    स्थिर ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    निश्चित(dst->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    भग्न * dst_ptr = (भग्न *) dst->तथ्य;
    स्थिर int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    स्थिर int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    स्थिर int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    स्थिर int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    स्थिर int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    स्थिर int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    स्थिर int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    स्थिर int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    स्थिर int64_t nrows = ne1 * ne2 * ne3;
    स्थिर bool src_row_contig = src0->nb[0] == माप(भग्न);

    क्रम (int64_t row = ith; row < nrows; row += nth) {
        स्थिर int64_t i1 = row % ne1;
        स्थिर int64_t i2 = (row / ne1) % ne2;
        स्थिर int64_t i3 = row / (ne1 * ne2);

        भग्न * dst_row = dst_ptr + row * ne0;

        अगर constexpr (circular_t) {
            // circular padding wraps around on a torus
            स्थिर int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            स्थिर int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            स्थिर int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                स्थिर int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(स्थिर भग्न *)((स्थिर अक्षर *) src0->तथ्य + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } अन्यथा {
            स्थिर bool interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            अगर (!interior) {
                स_रखो(dst_row, 0, ne0 * माप(भग्न));
                जारी;
            }

            स्थिर अक्षर * src_row = (स्थिर अक्षर *) src0->तथ्य + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            अगर (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                स_रखो(dst_row, 0, lp0 * माप(भग्न));
                स_नकल(dst_row + lp0, src_row, ne00 * माप(भग्न));
                स_रखो(dst_row + lp0 + ne00, 0, rp0 * माप(भग्न));
            } अन्यथा {
                क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                    अगर (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(स्थिर भग्न *)(src_row + (i0 - lp0)*nb00);
                    } अन्यथा {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


व्योम ggml_compute_forward_pad(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर bool circular = (bool) ggml_get_op_params_i32(dst, 8);
    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                अगर (circular) {
                    ggml_compute_forward_pad_f32<true>(params, dst);
                } अन्यथा {
                    ggml_compute_forward_pad_f32<false>(params, dst);
                }
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

व्योम ggml_compute_forward_pad_reflect_1d(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int32_t * opts = (स्थिर int32_t *) dst->op_params;
    स्थिर पूर्णांक p0 = opts[0];
    स्थिर पूर्णांक p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    क्रम (int64_t i3 = 0; i3 < ne3; i3++) {
        क्रम (int64_t i2 = 0; i2 < ne2; i2++) {
            क्रम (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                भग्न * बाँए  = (भग्न *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                भग्न * दाँए = (भग्न *) ((अक्षर *) dst->तथ्य + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, बाँए, (भग्न *) ((अक्षर *) src0->तथ्य + i3*nb03 + i2*nb02 + i1*nb01));

                क्रम (पूर्णांक i0 = 1; i0 <= p0; i0++) { बाँए[-i0] = बाँए[i0];   }
                क्रम (पूर्णांक i0 = 1; i0 <= p1; i0++) { दाँए[i0] = दाँए[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

अटल int64_t ggml_wrap_सूचकांक(int64_t i, int64_t ne) {
    अगर (i < 0) {
        वापस i + ne;
    } अन्यथा अगर (i >= ne) {
        वापस i - ne;
    }
    वापस i;
}

अटल व्योम ggml_compute_forward_roll_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर भग्न * src_तथ्य = (स्थिर भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य = (भग्न *) dst->तथ्य;

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर पूर्णांक s0 = ggml_get_op_params_i32(dst, 0);
    स्थिर पूर्णांक s1 = ggml_get_op_params_i32(dst, 1);
    स्थिर पूर्णांक s2 = ggml_get_op_params_i32(dst, 2);
    स्थिर पूर्णांक s3 = ggml_get_op_params_i32(dst, 3);

    स्थिर int64_t योग = ne1 * ne2 * ne3;
    स्थिर int64_t per_thread = (योग + params->nth) / params->nth;
    स्थिर int64_t start = params->ith * per_thread;
    स्थिर int64_t end   = std::min(start + per_thread, योग);

    क्रम (int64_t i = start; i < end; ++i) {
        स्थिर int64_t i1 = i % ne1;
        स्थिर int64_t i2 = (i / ne1) % ne2;
        स्थिर int64_t i3 = i / (ne2 * ne1);
        भग्न * dst_row = dst_तथ्य + (i3*nb3 + i2*nb2 + i1*nb1) / माप(भग्न);

        स्थिर int64_t i01 = ggml_wrap_सूचकांक(i1 - s1, ne01);
        स्थिर int64_t i02 = ggml_wrap_सूचकांक(i2 - s2, ne02);
        स्थिर int64_t i03 = ggml_wrap_सूचकांक(i3 - s3, ne03);
        स्थिर भग्न * src_row = src_तथ्य + (i03*nb03 + i02*nb02 + i01*nb01) / माप(भग्न);

        स्थिर int64_t s = ggml_wrap_सूचकांक(-s0, ne00);
        स्थिर int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

व्योम ggml_compute_forward_roll(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

अटल व्योम ggml_compute_forward_arange_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर भग्न start = ggml_get_op_params_f32(dst, 0);
    स्थिर भग्न stop  = ggml_get_op_params_f32(dst, 1);
    स्थिर भग्न step  = ggml_get_op_params_f32(dst, 2);

    स्थिर int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    क्रम (int64_t i = ith; i < steps; i+= nth) {
        भग्न value = start + step * i;
        ((भग्न *)dst->तथ्य)[i] = value;
    }
}

व्योम ggml_compute_forward_arange(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {
    चयन (dst->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_compute_forward_timestep_embedding_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर पूर्णांक dim = ggml_get_op_params_i32(dst, 0);
    स्थिर पूर्णांक max_period = ggml_get_op_params_i32(dst, 1);

    पूर्णांक half = dim / 2;

    क्रम (int64_t i = 0; i < ne00; i++) {
        भग्न * embed_तथ्य = (भग्न *)((अक्षर *)  dst->तथ्य +  i*nb1);
        क्रम (int64_t j = ith; j < half; j += nth) {
            भग्न timestep = ((भग्न *)src0->तथ्य)[i];
            भग्न freq = (भग्न)expf(-logf(max_period) * j / half);
            भग्न arg = timestep * freq;
            embed_तथ्य[j] = cosf(arg);
            embed_तथ्य[j + half] = sinf(arg);
        }
        अगर (dim % 2 != 0 && ith == 0) {
            embed_तथ्य[2 * half] = 0.f;
        }
    }
}

व्योम ggml_compute_forward_timestep_embedding(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

ढाँचा<क्रमागत ggml_sort_order order>
काष्ठा cmp_argsort {
    स्थिर भग्न * तथ्य;
    bool चालक()(int32_t a, int32_t b) स्थिर {
        अगर constexpr (order == GGML_SORT_ORDER_ASC) {
            वापस तथ्य[a] < तथ्य[b];
        } अन्यथा {
            वापस तथ्य[a] > तथ्य[b];
        }
    }
};

अटल व्योम ggml_compute_forward_argsort_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    क्रम (int64_t i = ith; i < nr; i += nth) {
        स्थिर भग्न * src_तथ्य = (भग्न *)((अक्षर *) src0->तथ्य + i*nb01);

        int32_t * dst_तथ्य = (int32_t *)((अक्षर *) dst->तथ्य + i*nb1);

        क्रम (int64_t j = 0; j < ne0; j++) {
            dst_तथ्य[j] = j;
        }

        चयन (order) {
            हाल GGML_SORT_ORDER_ASC:
                std::sort(dst_तथ्य, dst_तथ्य + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_तथ्य});
                अवरोध;

            हाल GGML_SORT_ORDER_DESC:
                std::sort(dst_तथ्य, dst_तथ्य + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_तथ्य});
                अवरोध;

            शेष:
                GGML_ABORT("invalid sort order");
        }
    }
}

व्योम ggml_compute_forward_argsort(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

काष्ठा cmp_शीर्ष_k {
    स्थिर भग्न * तथ्य;
    bool चालक()(int32_t a, int32_t b) स्थिर {
        वापस तथ्य[a] > तथ्य[b];
    }
};

अटल व्योम ggml_compute_forward_शीर्ष_k_f32(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == माप(भग्न));

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t nr = ggml_nrows(src0);

    स्थिर पूर्णांक शीर्ष_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_पंक्ति_आकार_F32) * ith;

    क्रम (int64_t i = ith; i < nr; i += nth) {
        स्थिर भग्न * src_तथ्य = (भग्न *)((अक्षर *) src0->तथ्य + i*nb01);

        क्रम (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + शीर्ष_k, tmp + ne00, cmp_शीर्ष_k{src_तथ्य});

        int32_t * dst_तथ्य = (int32_t *)((अक्षर *) dst->तथ्य + i*nb1);

        std::copy(tmp, tmp + शीर्ष_k, dst_तथ्य);

        // emphasize that the order is not important
        अगर (शीर्ष_k > 1) {
            std::swap(dst_तथ्य[0], dst_तथ्य[1]);
        }
    }
}

व्योम ggml_compute_forward_शीर्ष_k(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_शीर्ष_k_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst,
        पूर्णांक ir0, पूर्णांक ir1,
        int64_t ic_start, int64_t ic_end,
        भग्न * partials, int64_t partial_stride) {

    स्थिर bool write_partials = (partials != nullptr);
    स्थिर ggml_tensor * q     = dst->src[0];
    स्थिर ggml_tensor * k     = dst->src[1];
    स्थिर ggml_tensor * v     = dst->src[2];
    स्थिर ggml_tensor * mask  = dst->src[3];
    स्थिर ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिर int64_t DK = nek0;
    स्थिर int64_t DV = nev0;
    स्थिर int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    स्थिर int64_t rk2 = neq2/nek2;
    स्थिर int64_t rk3 = neq3/nek3;

    स्थिर int64_t rv2 = neq2/nev2;
    स्थिर int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    भग्न scale         = 1.0f;
    भग्न max_bias      = 0.0f;
    भग्न logit_softcap = 0.0f;

    स_नकल(&scale,         (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias,      (भग्न *) dst->op_params + 1, माप(भग्न));
    स_नकल(&logit_softcap, (भग्न *) dst->op_params + 2, माप(भग्न));

    अगर (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    स्थिर uint32_t n_शीर्ष      = neq2;
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) न्यूनमान(log2(n_शीर्ष));

    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    ggml_type         स्थिर k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t स्थिर q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    स्थिर kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   स्थिर v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    पूर्णांक ith = params->ith;

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // q indices
        स्थिर पूर्णांक iq3 = ir/(neq2*neq1);
        स्थिर पूर्णांक iq2 = (ir - iq3*neq2*neq1)/neq1;
        स्थिर पूर्णांक iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        स्थिर uint32_t h = iq2; // head index
        स्थिर भग्न slope = (max_bias > 0.0f) ? h < n_शीर्ष_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_शीर्ष_log2) + 1) : 1.0f;

        भग्न S = 0.0f;      //योग
        भग्न M = -अनन्त; // maximum KQ value

        भग्न       * VKQ32 = (भग्न       *) params->wdata + ith*(1*DK + 2*DV + CACHE_पंक्ति_आकार_F32); // FP32 VKQ accumulator
        भग्न       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        अगर (v->type == GGML_TYPE_F16) {
            स_रखो(VKQ16, 0, DV*माप(ggml_fp16_t));
        } अन्यथा {
            स_रखो(VKQ32, 0, DV*माप(भग्न));
        }

        स्थिर ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((अक्षर *) mask->तथ्य + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : शून्य;

        // k indices
        स्थिर पूर्णांक ik3 = iq3 / rk3;
        स्थिर पूर्णांक ik2 = iq2 / rk2;

        // v indices
        स्थिर पूर्णांक iv3 = iq3 / rv3;
        स्थिर पूर्णांक iv2 = iq2 / rv2;

        स्थिर भग्न * pq = (स्थिर भग्न *) ((अक्षर *) q->तथ्य + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        क्रम (int64_t ic = ic_start; ic < ic_end; ++ic) {
            स्थिर भग्न mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            अगर (mv == -अनन्त) {
                जारी;
            }

            भग्न s; // KQ value

            स्थिर अक्षर * k_तथ्य = (स्थिर अक्षर *) k->तथ्य + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_तथ्य, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            अगर (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            स्थिर भग्न Mold = M;

            भग्न ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            भग्न vs = 1.0f; // post-softmax KQ value, expf(s - M)

            स्थिर अक्षर * v_तथ्य = ((स्थिर अक्षर *) v->तथ्य + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            अगर (v->type == GGML_TYPE_F16) {
                अगर (s > M) {
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
                ggml_vec_mad_f16(DV, VKQ16, (स्थिर ggml_fp16_t *) v_तथ्य, vs);
            } अन्यथा {
                अगर (s > M) {
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
                अगर (v_to_float) {
                    v_to_float(v_तथ्य, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } अन्यथा {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (स्थिर भग्न *) v_तथ्य, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        अगर (v->type == GGML_TYPE_F16) {
            क्रम (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        अगर (sinks && ic_start == 0) {
            स्थिर भग्न s = ((भग्न *)((अक्षर *) sinks->तथ्य))[h];

            भग्न ms = 1.0f;
            भग्न vs = 1.0f;

            अगर (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } अन्यथा {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        अगर (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            भग्न * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            स_नकल(partial + 2, VKQ32, DV * माप(भग्न));
        } अन्यथा {
            // V /= S
            स्थिर भग्न S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            स्थिर पूर्णांक i1 = iq1;
            स्थिर पूर्णांक i2 = iq2;
            स्थिर पूर्णांक i3 = iq3;

            // permute(0, 2, 1, 3)
            स_नकल((अक्षर *) dst->तथ्य + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

अटल व्योम ggml_compute_forward_flash_attn_ext_tiled(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst,
        पूर्णांक ir0, पूर्णांक ir1) {
    स्थिर ggml_tensor * q     = dst->src[0];
    स्थिर ggml_tensor * k     = dst->src[1];
    स्थिर ggml_tensor * v     = dst->src[2];
    स्थिर ggml_tensor * mask  = dst->src[3];
    स्थिर ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिर int64_t DK = nek0;
    स्थिर int64_t DV = nev0;
    स्थिर int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    स्थिर ggml_type kv_type = k->type;


    // broadcast factors
    स्थिर int64_t rk2 = neq2/nek2;
    स्थिर int64_t rk3 = neq3/nek3;

    स्थिर int64_t rv2 = neq2/nev2;
    स्थिर int64_t rv3 = neq3/nev3;

    भग्न scale         = 1.0f;
    भग्न max_bias      = 0.0f;
    भग्न logit_softcap = 0.0f;

    स_नकल(&scale,         (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&max_bias,      (भग्न *) dst->op_params + 1, माप(भग्न));
    स_नकल(&logit_softcap, (भग्न *) dst->op_params + 2, माप(भग्न));

    अगर (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    स्थिर uint32_t n_शीर्ष      = neq2;
    स्थिर uint32_t n_शीर्ष_log2 = 1u << (uint32_t) न्यूनमान(log2(n_शीर्ष));

    स्थिर भग्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    स्थिर भग्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);

    पूर्णांक ith = params->ith;

    अटल constexpr पूर्णांक Q_TILE_SZ  = ggml_fa_tile_config::Q;
    अटल constexpr पूर्णांक KV_TILE_SZ = ggml_fa_tile_config::KV;

    पूर्णांक ir = ir0;
    जबतक (ir < ir1) {
        // q indices for the start of this tile
        स्थिर पूर्णांक iq3 = ir/(neq2*neq1);
        स्थिर पूर्णांक iq2 = (ir - iq3*neq2*neq1)/neq1;
        स्थिर पूर्णांक iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        स्थिर पूर्णांक tile_rows = MIN(Q_TILE_SZ, MIN((पूर्णांक)(ir1 - ir), (पूर्णांक)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        स्थिर uint32_t h = iq2; // head index
        स्थिर भग्न slope = (max_bias > 0.0f) ? h < n_शीर्ष_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_शीर्ष_log2) + 1) : 1.0f;

        भग्न S[Q_TILE_SZ];
        भग्न M[Q_TILE_SZ];

        क्रम (पूर्णांक i = 0 ; i < Q_TILE_SZ; ++i) {
            S[i] = 0.;
            M[i] = -अनन्त;
        }

        // Per-thread scratch layout:
        // Q_q:    Q_TILE_SZ * DK (converted Q tile — F32 for GEMM, KV type for scalar)
        // KQ:     Q_TILE_SZ * KV_TILE_SZ (attention scores in float)
        // mask:   Q_TILE_SZ * KV_TILE_SZ (mask in float)
        // VKQ32:  Q_TILE_SZ * DV (FP32 output accumulator)
        // V32:    KV_TILE_SZ * DV (F32 buffer for V tile)
        // K_f32:  KV_TILE_SZ * DK (F32 buffer for K tile — GEMM path)
        भग्न * base  = (भग्न *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_पंक्ति_आकार_F32);

        व्योम  * Q_q    = base;
        भग्न * KQ     = (भग्न *)((अक्षर *)base + Q_TILE_SZ * DK * माप(भग्न));
        भग्न * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        भग्न * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        भग्न * V32    = VKQ32 + Q_TILE_SZ * DV;
        भग्न * K_f32  = V32 + KV_TILE_SZ * DV;

        स_रखो(VKQ32, 0, Q_TILE_SZ * DV * माप(भग्न));
        स_रखो(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * माप(भग्न));

        // k indices
        स्थिर पूर्णांक ik3 = iq3 / rk3;
        स्थिर पूर्णांक ik2 = iq2 / rk2;

        // v indices
        स्थिर पूर्णांक iv3 = iq3 / rv3;
        स्थिर पूर्णांक iv2 = iq2 / rv2;

        {
            भग्न * Q_f32 = (भग्न *)Q_q;
            क्रम (पूर्णांक tq = 0; tq < tile_rows; tq++) {
                स्थिर भग्न * pq = (स्थिर भग्न *) ((अक्षर *) q->तथ्य + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                स_नकल(Q_f32 + tq * DK, pq, DK * माप(भग्न));
            }
            क्रम (पूर्णांक tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                स_रखो(Q_f32 + tq * DK, 0, DK * माप(भग्न));
            }
        }

        स_रखो(K_f32, 0, DK * KV_TILE_SZ * माप(भग्न));
        स_रखो(V32,   0, KV_TILE_SZ * DV * माप(भग्न));

        क्रम (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            स्थिर पूर्णांक kv_tile = (पूर्णांक)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            अगर (mask) {
                bool can_skip = true;
                क्रम (पूर्णांक tq = 0; tq < tile_rows; tq++) {
                    स्थिर ggml_fp16_t * mp_row = (स्थिर ggml_fp16_t *)((स्थिर अक्षर *) mask->तथ्य + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    क्रम (पूर्णांक tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        अगर (mask32[tq * KV_TILE_SZ + tk] != -अनन्त) {
                            can_skip = false;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    क्रम (पूर्णांक tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -अनन्त;
                    }
                }

                अगर (can_skip) {
                    जारी;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            क्रम (पूर्णांक tk = 0; tk < kv_tile; tk++) {
                स्थिर अक्षर * k_तथ्य = (स्थिर अक्षर *)k->तथ्य + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                अगर (kv_type == GGML_TYPE_F16) {
                    स्थिर ggml_fp16_t * k_f16 = (स्थिर ggml_fp16_t *)k_तथ्य;
                    क्रम (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } अन्यथा {
                    स्थिर भग्न * k_f32_src = (स्थिर भग्न *)k_तथ्य;
                    क्रम (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            स_रखो(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * माप(भग्न));
            simd_gemm(KQ, (स्थिर भग्न *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            अगर (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            अगर (kv_tile < KV_TILE_SZ) {
                क्रम (पूर्णांक tq = 0; tq < Q_TILE_SZ; tq++) {
                    क्रम (पूर्णांक tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -अनन्त;
                    }
                }
            }

            अगर (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            bool skip[Q_TILE_SZ] = {};

            क्रम (पूर्णांक tq = 0; tq < Q_TILE_SZ; tq++) {
                भग्न * kq_row = KQ + tq * KV_TILE_SZ;

                भग्न tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                अगर (tile_max == -अनन्त) {
                    skip[tq] = true;
                    जारी;
                }

                स्थिर भग्न Mold = M[tq];
                स्थिर भग्न Mnew = fmaxf(Mold, tile_max);

                अगर (Mnew > Mold) {
                    स्थिर भग्न ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            क्रम (पूर्णांक tk = 0; tk < kv_tile; tk++) {
                स्थिर अक्षर * v_तथ्य = (स्थिर अक्षर *)v->तथ्य + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                अगर (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((स्थिर ggml_fp16_t *)v_तथ्य, V32 + tk * DV, DV);
                } अन्यथा {
                    स_नकल(V32 + tk * DV, v_तथ्य, DV * माप(भग्न));
                }
            }
            क्रम (पूर्णांक tq = 0; tq < Q_TILE_SZ; tq++) {
                अगर (skip[tq]) {
                    स_रखो(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * माप(भग्न));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        अगर (sinks) {
            स्थिर भग्न s = ((भग्न *)((अक्षर *) sinks->तथ्य))[h];

            क्रम (पूर्णांक tq = 0; tq < tile_rows; tq++) {
                भग्न ms = 1.0f;
                भग्न vs = 1.0f;

                अगर (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } अन्यथा {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        क्रम (पूर्णांक tq = 0; tq < tile_rows; tq++) {
            // V /= S
            स्थिर भग्न S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            स्थिर पूर्णांक i1 = iq1 + tq;
            स्थिर पूर्णांक i2 = iq2;
            स्थिर पूर्णांक i3 = iq3;

            // permute(0, 2, 1, 3)
            स_नकल((अक्षर *) dst->तथ्य + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
अटल व्योम ggml_flash_attn_ext_reduce_partials(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst,
        स्थिर int64_t n_chunks,
        स्थिर int64_t chunk_आकार) {

    स्थिर ggml_tensor * q = dst->src[0];
    स्थिर ggml_tensor * k = dst->src[1];
    स्थिर ggml_tensor * v = dst->src[2];

    स्थिर int64_t DK        = k->ne[0];
    स्थिर int64_t DV        = v->ne[0];
    स्थिर int64_t nek1      = k->ne[1];
    स्थिर int64_t n_q_heads = q->ne[2];

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t wdata_per_thread = DK + 2*DV + CACHE_पंक्ति_आकार_F32;
    भग्न *       thread_wdata     = (भग्न *) params->wdata + ith * wdata_per_thread;

    स्थिर int64_t partials_offset  = nth * (DK + 2*DV + CACHE_पंक्ति_आकार_F32);
    स्थिर int64_t partial_आकार     = 2 + DV;
    स्थिर भग्न * partials_base    = (स्थिर भग्न *) params->wdata + partials_offset;

    // Output layout
    स्थिर int64_t ne1 = dst->ne[1];
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर माप_प्रकार  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    क्रम (int64_t q_शीर्ष = ith; q_शीर्ष < n_q_heads; q_शीर्ष += nth) {
        भग्न   M_final   = -अनन्त;
        भग्न   S_final   = 0.0f;
        भग्न * VKQ_final = thread_wdata;
        स_रखो(VKQ_final, 0, DV * माप(भग्न));

        // Combine partials from all chunks
        क्रम (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            स्थिर int64_t ic_start = chunk_idx * chunk_आकार;
            अगर (ic_start >= nek1) जारी;

            स्थिर भग्न * partial   = partials_base + (q_शीर्ष * n_chunks + chunk_idx) * partial_आकार;
            स्थिर भग्न   M_chunk   = partial[0];
            स्थिर भग्न   S_chunk   = partial[1];
            स्थिर भग्न * VKQ_chunk = partial + 2;

            अगर (S_chunk == 0.0f) जारी;

            स्थिर भग्न M_new     = fmaxf(M_final, M_chunk);
            स्थिर भग्न scale_old = expf(M_final - M_new);
            स्थिर भग्न scale_new = expf(M_chunk - M_new);

            क्रम (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        अगर (S_final != 0.0f) {
            स्थिर भग्न S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        स_नकल((अक्षर *) dst->तथ्य + (0*ne2*ne1 + q_शीर्ष + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

अटल व्योम ggml_compute_forward_flash_attn_ext_f16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * q     = dst->src[0];
    स्थिर ggml_tensor * k     = dst->src[1];
    स्थिर ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिर int64_t DK = nek0;
    स्थिर int64_t DV = nev0;
    स्थिर int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    स्थिर bool use_ref = params->use_ref;

    स्थिर bool kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    स्थिर bool use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    अगर (use_split_kv_path) {
        स्थिर int64_t chunk_आकार = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        स्थिर int64_t partial_आकार  = 2 + DV;
        भग्न *       partials_base = (भग्न *) params->wdata + nth * (DK + 2*DV + CACHE_पंक्ति_आकार_F32);

        स्थिर int64_t ic_start = ith * chunk_आकार;
        स्थिर int64_t ic_end   = std::min(ic_start + chunk_आकार, nek1);

        स्थिर int64_t partial_stride = nth * partial_आकार;
        भग्न *       chunk_partials = partials_base + ith * partial_आकार;

        अगर (ic_start < nek1) {
            क्रम (int64_t q_शीर्ष = 0; q_शीर्ष < neq2; q_शीर्ष++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_शीर्ष, q_शीर्ष + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } अन्यथा {
            क्रम (int64_t q_शीर्ष = 0; q_शीर्ष < neq2; q_शीर्ष++) {
                भग्न * q_partials = chunk_partials + q_शीर्ष * partial_stride;
                q_partials[0] = -अनन्त;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_आकार);
    } अन्यथा {

        // total rows in q
        स्थिर int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        स्थिर bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        पूर्णांक nth_scaled = nth * 4;
        int64_t chunk_आकार = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_आकार - 1) / chunk_आकार;

        अगर (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        अगर (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        स्थिर int64_t dr = (nr + nchunk - 1) / nchunk;

        अटल constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        bool use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#अगर defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#अगर defined(__ARM_FEATURE_SVE)
        स्थिर int64_t f32_epr = svcntw();
#अन्यथा
        स्थिर int64_t f32_epr = GGML_F32_EPR;
#पूर्ण_अगर
        use_tiled &= (DV % f32_epr == 0);
#पूर्ण_अगर
        पूर्णांक current_chunk = ith;

        जबतक (current_chunk < nchunk) {
            स्थिर int64_t ir0 = dr * current_chunk;
            स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

            अगर (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } अन्यथा {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

व्योम ggml_compute_forward_flash_attn_ext(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    चयन (dst->op_params[3]) {
        हाल GGML_PREC_DEFAULT:
        हाल GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

अटल व्योम ggml_compute_forward_flash_attn_back_f32(
        स्थिर ggml_compute_params * params,
        स्थिर bool masked,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * q = dst->src[0];
    स्थिर ggml_tensor * k = dst->src[1];
    स्थिर ggml_tensor * v = dst->src[2];
    स्थिर ggml_tensor * d = dst->src[3];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ned, d,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbd, d,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t D = neq0;
    स्थिर int64_t N = neq1;
    स्थिर int64_t P = nek1 - N;
    स्थिर int64_t M = P + N;

    स्थिर पूर्णांक Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    स्थिर पूर्णांक mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == माप(भग्न));
    GGML_ASSERT(nbk0 == माप(भग्न));
    GGML_ASSERT(nbv0 == माप(भग्न));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    अगर (ith == 0) {
        स_रखो(dst->तथ्य, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    स्थिर int64_t elem_q = ggml_nelements(q);
    स्थिर int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_आकार(result_type) == 1);
    स्थिर माप_प्रकार tsize = ggml_type_आकार(result_type);

    स्थिर माप_प्रकार offs_q = 0;
    स्थिर माप_प्रकार offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    स्थिर माप_प्रकार offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    व्योम * grad_q = (अक्षर *) dst->तथ्य;
    व्योम * grad_k = (अक्षर *) dst->तथ्य + offs_k;
    व्योम * grad_v = (अक्षर *) dst->तथ्य + offs_v;

    स्थिर माप_प्रकार nbgq1 = nb0*neq0;
    स्थिर माप_प्रकार nbgq2 = nb0*neq0*neq1;
    स्थिर माप_प्रकार nbgq3 = nb0*neq0*neq1*neq2;

    स्थिर माप_प्रकार nbgk1 = nb0*nek0;
    स्थिर माप_प्रकार nbgk2 = nb0*nek0*nek1;
    स्थिर माप_प्रकार nbgk3 = nb0*nek0*nek1*neq2;

    स्थिर माप_प्रकार nbgv1 = nb0*nev0;
    स्थिर माप_प्रकार nbgv2 = nb0*nev0*nev1;
    स्थिर माप_प्रकार nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    स्थिर पूर्णांक nr = nek2*nek3;

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    स्थिर भग्न scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    पूर्णांक nrep = neq2/nek2;

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // q indices
        स्थिर पूर्णांक ik3 = ir/(nek2);
        स्थिर पूर्णांक ik2 = ir - ik3*nek2;

        स्थिर पूर्णांक iq3 = ik3;
        स्थिर पूर्णांक id3 = ik3;
        स्थिर पूर्णांक iv3 = ik3;
        स्थिर पूर्णांक iv2 = ik2;

        क्रम (पूर्णांक irep = 0; irep < nrep; ++irep) {
            स्थिर पूर्णांक iq2 = ik2 + irep*nek2;
            स्थिर पूर्णांक id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            क्रम (पूर्णांक iq1 = 0; iq1 < neq1; ++iq1) {
                स्थिर पूर्णांक id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                भग्न * S  = (भग्न *) params->wdata + ith*2*(mxDM + CACHE_पंक्ति_आकार_F32) + 0*(mxDM+CACHE_पंक्ति_आकार_F32);
                भग्न * SM = (भग्न *) params->wdata + ith*2*(mxDM + CACHE_पंक्ति_आकार_F32) + 1*(mxDM+CACHE_पंक्ति_आकार_F32);

                क्रम (पूर्णांक i = M; i < Mup; ++i) {
                    S[i] = -अनन्त;
                }

                स्थिर int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                क्रम (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    स्थिर पूर्णांक ik1 = ic;

                    // S indices
                    स्थिर पूर्णांक i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (भग्न *) ((अक्षर *) k->तथ्य + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (भग्न *) ((अक्षर *) q->तथ्य + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                क्रम (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -अनन्त;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    भग्न max = -अनन्त;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float योग_2 = 0.0;
                    {
#अगर_घोषित GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_योग_f32(Mup, &योग_2, SM);
#अन्यथा
                        योग_2 = ggml_vec_soft_max_f32(Mup, SM, S, max);
#पूर्ण_अगर
                    }

                    निश्चित(योग_2 > 0.0);

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
                ggml_vec_set_f32(masked_begin, S, 0);
                क्रम (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (भग्न *) ((अक्षर *) v->तथ्य + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(भग्न *) ((अक्षर *) d->तथ्य + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                भग्न dot_SM_gradSM = 0;
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
                क्रम (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (भग्न *) ((अक्षर *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (भग्न *) ((अक्षर *) k->तथ्य + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                क्रम (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (भग्न *) ((अक्षर *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (भग्न *) ((अक्षर *) q->तथ्य + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                क्रम (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (भग्न *) ((अक्षर *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(भग्न *) ((अक्षर *) d->तथ्य + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

व्योम ggml_compute_forward_flash_attn_back(
        स्थिर ggml_compute_params * params,
        स्थिर bool masked,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * q = dst->src[0];

    चयन (q->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

अटल व्योम ggml_compute_forward_ssm_conv_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0]; // conv_x
    स्थिर ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nc  = src1->ne[0]; // d_conv
    स्थिर पूर्णांक ncs = src0->ne[0]; // d_conv - 1 + n_t
    स्थिर पूर्णांक nr  = src0->ne[1]; // d_inner
    स्थिर पूर्णांक n_t =  dst->ne[1]; // tokens per sequence
    स्थिर पूर्णांक n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == माप(भग्न));
    GGML_ASSERT(src1->nb[0] == माप(भग्न));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);
    स्थिर पूर्णांक ir  = ir1 - ir0;

    क्रम (पूर्णांक i3 = 0; i3 < n_s; ++i3) {
        क्रम (पूर्णांक i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            स्थिर भग्न * s = (स्थिर भग्न *) ((स्थिर अक्षर *) src0->तथ्य + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            स्थिर भग्न * c = (स्थिर भग्न *) ((स्थिर अक्षर *) src1->तथ्य + ir0*(src1->nb[1])); // {d_conv, d_inner}
            भग्न * x = (भग्न *) ((अक्षर *) dst->तथ्य + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            क्रम (पूर्णांक i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                भग्न sumf = 0.0f;

                // d_conv
                क्रम (पूर्णांक i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

व्योम ggml_compute_forward_ssm_conv(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    चयन (dst->src[0]->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

अटल व्योम ggml_compute_forward_ssm_scan_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    स्थिर ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    स्थिर ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    स्थिर ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    स्थिर ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    स्थिर ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    स्थिर ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t nc = src0->ne[0]; // d_state
    स्थिर int64_t nr = src0->ne[1]; // dim
    स्थिर int64_t nh = src1->ne[1]; // n_head
    स्थिर int64_t ng = src4->ne[1];
    स्थिर int64_t nt = src1->ne[2]; // number of tokens per sequence
    स्थिर int64_t ns = src1->ne[3]; // number of sequences in the batch
    स्थिर int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    स्थिर int64_t s_off = ggml_nelements(src1) * ggml_element_आकार(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == माप(भग्न));
    GGML_ASSERT(src1->nb[0] == माप(भग्न));
    GGML_ASSERT(src2->nb[0] == माप(भग्न));
    GGML_ASSERT(src3->nb[0] == माप(भग्न));
    GGML_ASSERT(src4->nb[0] == माप(भग्न));
    GGML_ASSERT(src5->nb[0] == माप(भग्न));
    GGML_ASSERT(src6->nb[0] == माप(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    स्थिर पूर्णांक dh = (nh + nth - 1)/nth;

    // head range for this thread
    स्थिर पूर्णांक ih0 = dh*ith;
    स्थिर पूर्णांक ih1 = MIN(ih0 + dh, nh);

    स्थिर int32_t * ids = (स्थिर int32_t *) src6->तथ्य;

    क्रम (पूर्णांक i3 = 0; i3 < ns; ++i3) {
        स्थिर भग्न * s0 = (स्थिर भग्न *) ((स्थिर अक्षर *) src0->तथ्य + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              भग्न * s  = (      भग्न *) ((      अक्षर *) dst->तथ्य  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        क्रम (पूर्णांक i2 = 0; i2 < nt; ++i2) {
            स्थिर भग्न * x  = (स्थिर भग्न *) ((स्थिर अक्षर *) src1->तथ्य + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            स्थिर भग्न * dt = (स्थिर भग्न *) ((स्थिर अक्षर *) src2->तथ्य + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            स्थिर भग्न * A  = (स्थिर भग्न *) ((स्थिर अक्षर *) src3->तथ्य); // {d_state, nh} or {1, nh}
            स्थिर भग्न * B  = (स्थिर भग्न *) ((स्थिर अक्षर *) src4->तथ्य + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            स्थिर भग्न * C  = (स्थिर भग्न *) ((स्थिर अक्षर *) src5->तथ्य + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  भग्न * y  = (      भग्न *) ((      अक्षर *) dst->तथ्य + i2*(nh*nr*माप(भग्न)) + i3*(nt*nh*nr*माप(भग्न))); // {dim, nh, nt, ns}

            अगर (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                क्रम (पूर्णांक h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    स्थिर भग्न dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    स्थिर भग्न dA = expf(dt_soft_plus * A[h]);
                    स्थिर पूर्णांक g = h / (nh / ng); // repeat_interleave

                    // dim
                    क्रम (पूर्णांक i1 = 0; i1 < nr; ++i1) {
                        स्थिर पूर्णांक ii = i1 + h*nr;
                        स्थिर भग्न x_dt = x[ii] * dt_soft_plus;
                        भग्न sumf = 0.0f;
#अगर defined(GGML_SIMD)
    #अगर defined(__ARM_FEATURE_SVE)
                        स्थिर पूर्णांक ggml_f32_epr = svcntw();
                        स्थिर पूर्णांक ggml_f32_step = 1 * ggml_f32_epr;

                        स्थिर पूर्णांक np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC योग_2 = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        क्रम (पूर्णांक i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            क्रम (पूर्णांक j = 0; j < 1; j++) {
                                GGML_F32_VEC t0 = GGML_F32_VEC_LOAD(s0 + i + j*ggml_f32_epr + ii*nc);
                                GGML_F32_VEC t1 = GGML_F32_VEC_LOAD(B + i + j*ggml_f32_epr + g*nc);
                                GGML_F32_VEC t2 = GGML_F32_VEC_LOAD(C + i + j*ggml_f32_epr + g*nc);

                                t0 = GGML_F32_VEC_MUL(t0, adA);
                                t1 = GGML_F32_VEC_MUL(t1, axdt);

                                t0 = GGML_F32_VEC_ADD(t0, t1);

                                योग_2 = GGML_F32_VEC_FMA(योग_2, t0, t2);

                                GGML_F32_VEC_STORE(s + i + j*ggml_f32_epr + ii*nc, t0);
                            }
                        }

                        sumf = GGML_F32xt_REDUCE_ONE(योग_2);
    #या_अगर defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        स्थिर पूर्णांक np = 0;
    #अन्यथा
                        स्थिर पूर्णांक np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC योग_2[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        क्रम (पूर्णांक i = 0; i < np; i += GGML_F32_STEP) {
                            क्रम (पूर्णांक j = 0; j < GGML_F32_ARR; j++) {
                                ax[j] = GGML_F32_VEC_LOAD(s0 + i + j*GGML_F32_EPR + ii*nc);
                                ay[j] = GGML_F32_VEC_LOAD(B + i + j*GGML_F32_EPR + g*nc);
                                az[j] = GGML_F32_VEC_LOAD(C + i + j*GGML_F32_EPR + g*nc);

                                ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
                                ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

                                ax[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);

                                योग_2[j] = GGML_F32_VEC_FMA(योग_2[j], ax[j], az[j]);

                                GGML_F32_VEC_STORE(s + i + j*GGML_F32_EPR + ii*nc, ax[j]);
                            }
                        }

                        // reduce sum0..sum3 to sum0
                        GGML_F32_VEC_REDUCE(sumf, योग_2);
    #पूर्ण_अगर
#अन्यथा
                        स्थिर पूर्णांक np = 0;
#पूर्ण_अगर
                        // d_state
                        क्रम (पूर्णांक i0 = np; i0 < nc; ++i0) {
                            स्थिर पूर्णांक i = i0 + ii*nc;
                            स्थिर पूर्णांक ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            स्थिर भग्न state = (s0[i] * dA) + (B[ig] * x_dt);
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
                क्रम (पूर्णांक h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    स्थिर भग्न dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    स्थिर पूर्णांक g = h / (nh / ng); // repeat_interleave

                    // dim
                    क्रम (पूर्णांक i1 = 0; i1 < nr; ++i1) {
                        स्थिर पूर्णांक ii = i1 + h*nr;
                        स्थिर भग्न x_dt = x[ii] * dt_soft_plus;
#अगर defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        क्रम (int64_t k = 0; k < nc; k += svcntw()) {
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
                        भग्न sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        क्रम (पूर्णांक i0 = 0; i0 < nc; ++i0) {
                            स्थिर पूर्णांक i = i0 + ii*nc;
                            स्थिर पूर्णांक ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            स्थिर भग्न state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#पूर्ण_अगर
                    }
                }
            }
            स्थिर int64_t slot = nt - 1 - i2;
            अगर (K > 1 && slot > 0 && slot < K) {
                भग्न * s_snapshot = (भग्न *) ((अक्षर *) dst->तथ्य + s_off + (slot*ns + i3)*(src0->nb[3]));
                क्रम (पूर्णांक h = ih0; h < ih1; ++h) {
                    स_नकल((अक्षर *) s_snapshot + h*src0->nb[2], (अक्षर *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

व्योम ggml_compute_forward_ssm_scan(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    चयन (dst->src[0]->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

अटल व्योम ggml_compute_forward_win_part_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    स्थिर int32_t nep0 = ((स्थिर int32_t *)(dst->op_params))[0];
    स्थिर int32_t nep1 = ((स्थिर int32_t *)(dst->op_params))[1];
    स्थिर int32_t w    = ((स्थिर int32_t *)(dst->op_params))[2];

    निश्चित(ne00 == ne0);
    निश्चित(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    क्रम (पूर्णांक py = 0; py < nep1; ++py) {
        क्रम (पूर्णांक px = 0; px < nep0; ++px) {
            स्थिर int64_t i3 = py*nep0 + px;
            क्रम (int64_t i2 = 0; i2 < ne2; ++i2) {
                क्रम (int64_t i1 = 0; i1 < ne1; ++i1) {
                    क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                        स्थिर int64_t i02 = py*w + i2;
                        स्थिर int64_t i01 = px*w + i1;
                        स्थिर int64_t i00 = i0;

                        स्थिर int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        स्थिर int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        अगर (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((भग्न *) dst->तथ्य)[i] = 0.0f;
                        } अन्यथा {
                            ((भग्न *) dst->तथ्य)[i] = ((भग्न *) src0->तथ्य)[j];
                        }
                    }
                }
            }
        }
    }
}

व्योम ggml_compute_forward_win_part(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

अटल व्योम ggml_compute_forward_win_unpart_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    स्थिर int32_t w = ((स्थिर int32_t *)(dst->op_params))[0];

    // padding
    स्थिर पूर्णांक px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    स्थिर पूर्णांक npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    निश्चित(ne0 == ne00);

    // TODO: optimize / multi-thread
    क्रम (int64_t i2 = 0; i2 < ne2; ++i2) {
        क्रम (int64_t i1 = 0; i1 < ne1; ++i1) {
            क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                स्थिर पूर्णांक ip2 = i2/w;
                स्थिर पूर्णांक ip1 = i1/w;

                स्थिर int64_t i02 = i2%w;
                स्थिर int64_t i01 = i1%w;
                स्थिर int64_t i00 = i0;

                स्थिर int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                स्थिर int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((भग्न *) dst->तथ्य)[j] = ((भग्न *) src0->तथ्य)[i];
            }
        }
    }
}

व्योम ggml_compute_forward_win_unpart(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

व्योम ggml_compute_forward_unary(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_unary_op op = ggml_get_unary_op(dst);

    चयन (op) {
        हाल GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } अवरोध;
        हाल GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

व्योम ggml_compute_forward_glu(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_glu_op op = ggml_get_glu_op(dst);

    चयन (op) {
        हाल GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } अवरोध;
        हाल GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } अवरोध;
        हाल GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } अवरोध;
        हाल GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } अवरोध;
        हाल GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } अवरोध;
        हाल GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } अवरोध;
        हाल GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

अटल व्योम ggml_compute_forward_get_rel_pos_f16(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    स्थिर ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिर int64_t w = ne1;

    ggml_fp16_t * src0_तथ्य = (ggml_fp16_t *) src0->तथ्य;
    ggml_fp16_t * dst_तथ्य  = (ggml_fp16_t *) dst->तथ्य;

    क्रम (int64_t i2 = 0; i2 < ne2; ++i2) {
        क्रम (int64_t i1 = 0; i1 < ne1; ++i1) {
            स्थिर int64_t pos = (w - i1 - 1) + i2;
            क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_तथ्य[i2*ne1*ne0 + i1*ne0 + i0] = src0_तथ्य[pos*ne00 + i0];
            }
        }
    }
}

व्योम ggml_compute_forward_get_rel_pos(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

अटल व्योम ggml_compute_forward_add_rel_pos_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];
    स्थिर ggml_tensor * src2 = dst->src[2];

    स्थिर bool inplace = (bool) ((int32_t *) dst->op_params)[0];
    अगर (!inplace) {
        अगर (params->ith == 0) {
            स_नकल((अक्षर *) dst->तथ्य, (अक्षर *) src0->तथ्य, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    भग्न * src1_तथ्य = (भग्न *) src1->तथ्य;
    भग्न * src2_तथ्य = (भग्न *) src2->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    स्थिर int64_t ne10 = src1->ne[0];
    स्थिर int64_t ne11 = src1->ne[1];
    स्थिर int64_t ne12 = src1->ne[2];
    स्थिर int64_t ne13 = src1->ne[3];

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // total patches in dst
    स्थिर पूर्णांक np = ne13;

    // patches per thread
    स्थिर पूर्णांक dp = (np + nth - 1)/nth;

    // patch range for this thread
    स्थिर पूर्णांक ip0 = dp*ith;
    स्थिर पूर्णांक ip1 = MIN(ip0 + dp, np);

    क्रम (int64_t i13 = ip0; i13 < ip1; ++i13) {
        क्रम (int64_t i12 = 0; i12 < ne12; ++i12) {
            क्रम (int64_t i11 = 0; i11 < ne11; ++i11) {
                स्थिर int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                क्रम (int64_t i10 = 0; i10 < ne10; ++i10) {
                    स्थिर int64_t jp0  = jp1 + i10;
                    स्थिर भग्न src1_e = src1_तथ्य[jp0];
                    स्थिर भग्न src2_e = src2_तथ्य[jp0];

                    स्थिर int64_t jdh = jp0 * ne10;
                    स्थिर int64_t jdw = jdh - (ne10 - 1) * i10;

                    क्रम (int64_t j = 0; j < ne10; ++j) {
                        dst_तथ्य[jdh + j     ] += src2_e;
                        dst_तथ्य[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

व्योम ggml_compute_forward_add_rel_pos(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

अटल व्योम ggml_compute_forward_rwkv_wkv6_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर int64_t T = dst->src[1]->ne[2];
    स्थिर int64_t C = dst->ne[0];
    स्थिर int64_t HEADS = dst->src[1]->ne[1];
    स्थिर int64_t n_seqs = dst->src[5]->ne[1];
    स्थिर int64_t शीर्ष_आकार = C / HEADS;

    भग्न * dst_तथ्य = (भग्न *) dst->तथ्य;
    भग्न * state = ((भग्न *) dst->तथ्य) + C * T;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक h_start =  (HEADS * (ith    )) / nth;
    स्थिर पूर्णांक h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भग्न * k =          (भग्न *) dst->src[0]->तथ्य;
    भग्न * v =          (भग्न *) dst->src[1]->तथ्य;
    भग्न * r =          (भग्न *) dst->src[2]->तथ्य;
    भग्न * time_faaaa = (भग्न *) dst->src[3]->तथ्य;
    भग्न * time_decay = (भग्न *) dst->src[4]->तथ्य;

    माप_प्रकार t_stride = HEADS * शीर्ष_आकार; // Same to C

    माप_प्रकार h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    माप_प्रकार h_stride_2d = शीर्ष_आकार * शीर्ष_आकार;

    अगर (ith == 0) {
        स_रखो(dst_तथ्य, 0, T * C * माप(भग्न));
    }
    ggml_barrier(params->threadpool);


    #अगर defined(__AVX__) && !defined(__AVX512F__)
        #घोषणा GGML_F32X GGML_F32x8
        #घोषणा GGML_F32X_SET1 GGML_F32x8_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32x8_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32x8_STORE
        #घोषणा GGML_F32X_MUL GGML_F32x8_MUL
        #घोषणा GGML_F32X_FMA GGML_F32x8_FMA
        #घोषणा WKV_VECTOR_आकार 8
    #या_अगर defined(__AVX512F__)
        #घोषणा GGML_F32X GGML_F32x16
        #घोषणा GGML_F32X_SET1 GGML_F32x16_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32x16_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32x16_STORE
        #घोषणा GGML_F32X_MUL GGML_F32x16_MUL
        #घोषणा GGML_F32X_FMA GGML_F32x16_FMA
        #घोषणा WKV_VECTOR_आकार 16
    #या_अगर defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #घोषणा GGML_F32X GGML_F32xt
        #घोषणा GGML_F32X_SET1 GGML_F32xt_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32xt_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32xt_STORE
        #घोषणा GGML_F32X_MUL GGML_F32xt_MUL
        #घोषणा GGML_F32X_FMA GGML_F32xt_FMA
        #घोषणा WKV_VECTOR_आकार 8
    #या_अगर defined(__ARM_NEON) && defined(__aarch64__)
        #घोषणा GGML_F32X GGML_F32x4
        #घोषणा GGML_F32X_SET1 GGML_F32x4_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32x4_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32x4_STORE
        #घोषणा GGML_F32X_MUL GGML_F32x4_MUL
        #घोषणा GGML_F32X_FMA GGML_F32x4_FMA
        #घोषणा WKV_VECTOR_आकार 4
    #पूर्ण_अगर

    #अगर_घोषित WKV_VECTOR_आकार
        पूर्णांक wkv_vector_आकार;
        #अगर defined(__ARM_FEATURE_SVE)
            wkv_vector_आकार = svcntw();
        #अन्यथा
            wkv_vector_आकार = WKV_VECTOR_आकार;
        #पूर्ण_अगर
        स्थिर int64_t vec_गिनती = शीर्ष_आकार / wkv_vector_आकार;

        क्रम (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
            भग्न * state_cur = state + state_offset;
            भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[5]->तथ्य + state_offset;

            क्रम (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                क्रम (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_i_offset = h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    भग्न k_val = k[t_h_i_offset];
                    भग्न r_val = r[t_h_i_offset];
                    भग्न time_faaaa_val = time_faaaa[h_i_offset];
                    भग्न time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    क्रम (int64_t j = 0; j < vec_गिनती; j++) {
                        माप_प्रकार base_j = j * wkv_vector_आकार;
                        माप_प्रकार t_h_j_offset = t_h_offset + base_j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X पूर्व_state_vec = GGML_F32X_LOAD(&state_पूर्व[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_तथ्य[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = kv * time_faaaa + prev_state
                        GGML_F32X अस्थायी_vec = GGML_F32X_FMA(पूर्व_state_vec, kv_vec, time_faaaa_vec);

                        // Update dst: dst += temp * r
                        dst_vec = GGML_F32X_FMA(dst_vec, अस्थायी_vec, r_vec);
                        GGML_F32X_STORE(&dst_तथ्य[t_h_j_offset], dst_vec);

                        // Update state: state = prev_state * time_decay + kv
                        GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, पूर्व_state_vec, time_decay_vec);
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], new_state_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    क्रम (int64_t j = vec_गिनती * wkv_vector_आकार; j < शीर्ष_आकार; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;
                        भग्न v_val = v[t_h_j_offset];
                        भग्न kv_val = v_val * k_val;
                        भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                        भग्न अस्थायी_val = kv_val * time_faaaa_val + पूर्व_state_val;
                        dst_तथ्य[t_h_j_offset] += अस्थायी_val * r_val;
                        state_cur[h_2d_i_j_offset] = पूर्व_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #अन्यथा
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        क्रम (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
            भग्न * state_cur = state + state_offset;
            भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[5]->तथ्य + state_offset;

            क्रम (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                क्रम (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_i_offset = h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    भग्न k_val = k[t_h_i_offset];
                    भग्न r_val = r[t_h_i_offset];
                    भग्न time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    भग्न time_decay_val = time_decay[t_h_i_offset];

                    क्रम (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;

                        भग्न v_val = v[t_h_j_offset];
                        भग्न kv_val = v_val * k_val;
                        भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                        भग्न अस्थायी_val = kv_val * time_faaaa_val + पूर्व_state_val;
                        dst_तथ्य[t_h_j_offset] += अस्थायी_val * r_val;
                        state_cur[h_2d_i_j_offset] = पूर्व_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #पूर्ण_अगर
}


व्योम ggml_compute_forward_rwkv_wkv6(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

अटल व्योम ggml_compute_forward_gla_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर int64_t T = dst->src[1]->ne[2];
    स्थिर int64_t C = dst->ne[0];
    स्थिर int64_t HEADS = dst->src[1]->ne[1];
    स्थिर int64_t n_seqs = dst->src[4]->ne[1];
    स्थिर int64_t शीर्ष_आकार = C / HEADS;
    स्थिर भग्न scale = ggml_get_op_params_f32(dst, 0);

    भग्न * dst_तथ्य = (भग्न *) dst->तथ्य;
    भग्न * state = ((भग्न *) dst->तथ्य) + C * T;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक h_start =  (HEADS * (ith    )) / nth;
    स्थिर पूर्णांक h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भग्न * k = (भग्न *) dst->src[0]->तथ्य;
    भग्न * v = (भग्न *) dst->src[1]->तथ्य;
    भग्न * q = (भग्न *) dst->src[2]->तथ्य;
    भग्न * g = (भग्न *) dst->src[3]->तथ्य;

    माप_प्रकार t_stride = HEADS * शीर्ष_आकार; // Same to C

    माप_प्रकार h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    माप_प्रकार h_stride_2d = शीर्ष_आकार * शीर्ष_आकार;

    अगर (ith == 0) {
        स_रखो(dst_तथ्य, 0, T * C * माप(भग्न));
    }
    ggml_barrier(params->threadpool);


    #अगर defined(__AVX__) && !defined(__AVX512F__)
        #घोषणा GGML_F32X GGML_F32x8
        #घोषणा GGML_F32X_SET1 GGML_F32x8_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32x8_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32x8_STORE
        #घोषणा GGML_F32X_MUL GGML_F32x8_MUL
        #घोषणा GGML_F32X_FMA GGML_F32x8_FMA
        #घोषणा GLA_VECTOR_आकार 8
    #या_अगर defined(__AVX512F__)
        #घोषणा GGML_F32X GGML_F32x16
        #घोषणा GGML_F32X_SET1 GGML_F32x16_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32x16_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32x16_STORE
        #घोषणा GGML_F32X_MUL GGML_F32x16_MUL
        #घोषणा GGML_F32X_FMA GGML_F32x16_FMA
        #घोषणा GLA_VECTOR_आकार 16
    #या_अगर defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #घोषणा GGML_F32X GGML_F32xt
        #घोषणा GGML_F32X_SET1 GGML_F32xt_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32xt_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32xt_STORE
        #घोषणा GGML_F32X_MUL GGML_F32xt_MUL
        #घोषणा GGML_F32X_FMA GGML_F32xt_FMA
        #घोषणा GLA_VECTOR_आकार 8
    #या_अगर defined(__ARM_NEON) && defined(__aarch64__)
        #घोषणा GGML_F32X GGML_F32x4
        #घोषणा GGML_F32X_SET1 GGML_F32x4_SET1
        #घोषणा GGML_F32X_LOAD GGML_F32x4_LOAD
        #घोषणा GGML_F32X_STORE GGML_F32x4_STORE
        #घोषणा GGML_F32X_MUL GGML_F32x4_MUL
        #घोषणा GGML_F32X_FMA GGML_F32x4_FMA
        #घोषणा GLA_VECTOR_आकार 4
    #पूर्ण_अगर

    #अगर_घोषित GLA_VECTOR_आकार
        पूर्णांक gla_vector_आकार;
        #अगर defined(__ARM_FEATURE_SVE)
            gla_vector_आकार = svcntw();
        #अन्यथा
            gla_vector_आकार = GLA_VECTOR_आकार;
        #पूर्ण_अगर
        स्थिर int64_t vec_गिनती = शीर्ष_आकार / gla_vector_आकार;

        क्रम (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
            भग्न * state_cur = state + state_offset;
            भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[4]->तथ्य + state_offset;

            क्रम (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                क्रम (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    भग्न k_val = k[t_h_i_offset];
                    भग्न q_val = q[t_h_i_offset] * scale;
                    भग्न g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    क्रम (int64_t j = 0; j < vec_गिनती; j++) {
                        माप_प्रकार base_j = j * gla_vector_आकार;
                        माप_प्रकार t_h_j_offset = t_h_offset + base_j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + base_j;

                        // Load x elements at once
                        GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
                        GGML_F32X पूर्व_state_vec = GGML_F32X_LOAD(&state_पूर्व[h_2d_i_j_offset]);
                        GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_तथ्य[t_h_j_offset]);

                        // Compute kv = v * k
                        GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

                        // Compute temp = prev_state * g + kv
                        GGML_F32X अस्थायी_vec = GGML_F32X_FMA(kv_vec, पूर्व_state_vec, g_vec);

                        // Update dst: dst += temp * q
                        dst_vec = GGML_F32X_FMA(dst_vec, अस्थायी_vec, q_vec);
                        GGML_F32X_STORE(&dst_तथ्य[t_h_j_offset], dst_vec);

                        // Update state
                        GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], अस्थायी_vec);
                    }

                    // Handle remaining elements, this will not be used.
                    क्रम (int64_t j = vec_गिनती * gla_vector_आकार; j < शीर्ष_आकार; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;
                        भग्न v_val = v[t_h_j_offset];
                        भग्न kv_val = v_val * k_val;
                        भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                        भग्न अस्थायी_val = kv_val + पूर्व_state_val * g_val;
                        dst_तथ्य[t_h_j_offset] += अस्थायी_val * q_val;
                        state_cur[h_2d_i_j_offset] = अस्थायी_val;
                    }
                }
            }
        }

    #अन्यथा
        क्रम (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
            भग्न * state_cur = state + state_offset;
            भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[4]->तथ्य + state_offset;

            क्रम (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                क्रम (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    भग्न k_val = k[t_h_i_offset];
                    भग्न q_val = q[t_h_i_offset] * scale;
                    भग्न g_val = g[t_h_i_offset];

                    क्रम (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;

                        भग्न v_val = v[t_h_j_offset];
                        भग्न kv_val = v_val * k_val;
                        भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                        भग्न अस्थायी_val = पूर्व_state_val * g_val + kv_val;
                        dst_तथ्य[t_h_j_offset] += अस्थायी_val * q_val;
                        state_cur[h_2d_i_j_offset] = अस्थायी_val;
                    }
                }
            }
        }
    #पूर्ण_अगर
}


व्योम ggml_compute_forward_gla(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_compute_forward_solve_tri_f32(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst) {
    स्थिर काष्ठा ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    स्थिर काष्ठा ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t k = ne10;   // number of RHS columns
    स्थिर int64_t n = ne11;   // A is n×n
    स्थिर int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    स्थिर भग्न * A = (स्थिर भग्न *) src0->तथ्य;  // [n, n, B1, B2]
    स्थिर भग्न * B = (स्थिर भग्न *) src1->तथ्य;  // [n, k, B1, B2]
          भग्न * X = (      भग्न *) dst->तथ्य;   // [n, k, B1, B2]

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*k);
        स्थिर int64_t i02 = (ir - i03*ne02*k)/k;
        स्थिर int64_t i01 = (ir - i03*ne02*k - i02*k);

        स्थिर भग्न * A_batch = A + i02 * nb02 / माप(भग्न) + i03 * nb03 / माप(भग्न);
        स्थिर भग्न * B_batch = B + i02 * nb12 / माप(भग्न) + i03 * nb13 / माप(भग्न);

        भग्न * X_batch = X + i02 * nb2 / माप(भग्न) + i03 * nb3 / माप(भग्न);

        क्रम (int64_t i00 = 0; i00 < n; ++i00) {
            भग्न योग_2 = 0.0f;
            क्रम (int64_t t = 0; t < i00; ++t) {
                योग_2 += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            स्थिर भग्न diag = A_batch[i00 * n + i00];
            निश्चित(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - योग_2) / diag;
        }
    }
}

व्योम ggml_compute_forward_solve_tri(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } अन्यथा {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
अटल व्योम ggml_compute_forward_gated_delta_net_one_chunk(
    स्थिर ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    स्थिर int64_t S_v      = src_v->ne[0];
    स्थिर int64_t H        = src_v->ne[1];
    स्थिर int64_t n_tokens = src_v->ne[2];
    स्थिर int64_t n_seqs   = src_v->ne[3];

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    GGML_ASSERT(src_g->ne[0] == 1 || src_g->ne[0] == S_v);
    GGML_ASSERT(src_beta->ne[0] == 1);

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(int64_t, neg, src_g, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbg, src_g, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbb, src_beta, nb);

    स्थिर bool kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    स्थिर int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    स्थिर int64_t state_seq_stride = src_state->nb[3] / माप(भग्न);

    स्थिर int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    स्थिर पूर्णांक ith = params->ith;

    भग्न * delta       = (भग्न *)params->wdata + ith * per_thread + CACHE_पंक्ति_आकार_F32;
    भग्न * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    स्थिर int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    स्थिर int64_t state_आकार_per_snap = S_v * S_v * H * n_seqs;
    भग्न * attn_out_base  = (भग्न *)dst->तथ्य;
    भग्न * state_out_base = (भग्न *)dst->तथ्य + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    स्थिर भग्न * state_in_base = (स्थिर भग्न *)src_state->तथ्य;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    स्थिर int64_t rq3 = nev3 / neq3;
    स्थिर int64_t rk3 = nev3 / nek3;

    स्थिर भग्न scale = 1.0f / sqrtf((भग्न) S_v);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t iv1 = ir % H; // head_index
        स्थिर int64_t iv3 = ir / H; // sequence

        स्थिर int64_t iq1 = iv1 % neq1;
        स्थिर int64_t ik1 = iv1 % nek1;

        स्थिर int64_t iq3 = iv3 / rq3;
        स्थिर int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        भग्न * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        स्थिर भग्न * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        स_नकल(s_out, s_in, S_v * S_v * माप(भग्न));

        // attn output pointer for first token of this (head, seq)
        भग्न * attn_तथ्य = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        क्रम (int64_t t = 0; t < n_tokens; t++) {
            स्थिर भग्न * q_d = (स्थिर भग्न *)((स्थिर अक्षर *)src_q->तथ्य + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            स्थिर भग्न * k_d = (स्थिर भग्न *)((स्थिर अक्षर *)src_k->तथ्य + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            स्थिर भग्न * v_d = (स्थिर भग्न *)((स्थिर अक्षर *)src_v->तथ्य + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            स्थिर भग्न beta_val = *(स्थिर भग्न *)((स्थिर अक्षर *)src_beta->तथ्य + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            स्थिर भग्न * g_d    =  (स्थिर भग्न *)((स्थिर अक्षर *)src_g->तथ्य    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            अगर (kda) {
                // precompute exp(g) into delta scratch (reused below)
                क्रम (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                क्रम (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } अन्यथा {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            क्रम (int64_t j = 0; j < S_v; ++j) {
                भग्न योग_2 = 0.0f;
                ggml_vec_dot_f32(S_v, &योग_2, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - योग_2) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            क्रम (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            क्रम (int64_t j = 0; j < S_v; ++j) {
                भग्न योग_2 = 0.0f;
                ggml_vec_dot_f32(S_v, &योग_2, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_तथ्य[j] = योग_2 * scale;
            }

            attn_तथ्य += S_v * H; // advance to next token

            अगर (K > 1) {
                स्थिर int64_t target_slot = n_tokens - 1 - t;
                अगर (target_slot >= 0 && target_slot < K) {
                    भग्न * curr_state_o = state_out_base + target_slot * state_आकार_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    स_नकल(curr_state_o, s_out, S_v * S_v * माप(भग्न));
                }
            }
        }
    }
}


अटल व्योम ggml_compute_forward_gated_delta_net_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    स्थिर bool disable_chunking = ggml_is_numa();

    पूर्णांक nth = params->nth;
    पूर्णांक ith = params->ith;

    // 4x chunks per thread
    पूर्णांक nth_scaled = nth * 4;
    int64_t chunk_आकार = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_आकार - 1) / chunk_आकार;

    अगर (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    अगर (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    स्थिर int64_t dr = (nr + nchunk - 1) / nchunk;

    पूर्णांक current_chunk = ith;

    जबतक (current_chunk < nchunk) {
        स्थिर int64_t ir0 = dr * current_chunk;
        स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

व्योम ggml_compute_forward_gated_delta_net(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

अटल व्योम ggml_dsv4_hc_comb_norm_cols(भग्न * comb, भग्न eps) {
    constexpr int64_t hc = 4;

    क्रम (int64_t idst = 0; idst < hc; ++idst) {
        भग्न योग_2 = eps;
        क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
            योग_2 += comb[idst + hc*isrc];
        }

        स्थिर भग्न inv_योग = 1.0f / योग_2;
        क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_योग;
        }
    }
}

अटल व्योम ggml_dsv4_hc_comb_norm_rows(भग्न * comb, भग्न eps) {
    constexpr int64_t hc = 4;

    क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
        भग्न योग_2 = eps;
        क्रम (int64_t idst = 0; idst < hc; ++idst) {
            योग_2 += comb[idst + hc*isrc];
        }

        स्थिर भग्न inv_योग = 1.0f / योग_2;
        क्रम (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_योग;
        }
    }
}

अटल व्योम ggml_compute_forward_dsv4_hc_comb_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * mixes = dst->src[0];
    स्थिर ggml_tensor * scale = dst->src[1];
    स्थिर ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    स्थिर int64_t n_tokens = mixes->ne[1];

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(माप_प्रकार, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbs, scale, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbb, base,  nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbd, dst,   nb);

    स्थिर भग्न eps = ggml_get_op_params_f32(dst, 0);
    स्थिर int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t dr  = (n_tokens + nth - 1) / nth;
    स्थिर int64_t it0 = dr * ith;
    स्थिर int64_t it1 = MIN(it0 + dr, n_tokens);

    स्थिर भग्न scale_comb = *(स्थिर भग्न *) ((स्थिर अक्षर *) scale->तथ्य + 2*nbs0);

    क्रम (int64_t it = it0; it < it1; ++it) {
        भग्न comb[hc*hc];

        क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
            भग्न max = -अनन्त;
            क्रम (int64_t idst = 0; idst < hc; ++idst) {
                स्थिर int64_t idx = idst + hc*isrc;
                स्थिर भग्न xv = *(स्थिर भग्न *) ((स्थिर अक्षर *) mixes->तथ्य + (comb_offset + idx)*nbm0 + it*nbm1);
                स्थिर भग्न bv = *(स्थिर भग्न *) ((स्थिर अक्षर *) base->तथ्य  + (comb_offset + idx)*nbb0);
                स्थिर भग्न v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            भग्न योग_2 = 0.0f;
            क्रम (int64_t idst = 0; idst < hc; ++idst) {
                स्थिर int64_t idx = idst + hc*isrc;
                स्थिर भग्न v = expf(comb[idx] - max);
                comb[idx] = v;
                योग_2 += v;
            }

            स्थिर भग्न inv_योग = 1.0f / योग_2;
            क्रम (int64_t idst = 0; idst < hc; ++idst) {
                स्थिर int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_योग + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        क्रम (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
            क्रम (int64_t idst = 0; idst < hc; ++idst) {
                स्थिर int64_t idx = idst + hc*isrc;
                *(भग्न *) ((अक्षर *) dst->तथ्य + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

व्योम ggml_compute_forward_dsv4_hc_comb(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

अटल व्योम ggml_compute_forward_dsv4_hc_pre_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * x       = dst->src[0];
    स्थिर ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिर int64_t n_embd   = x->ne[0];
    स्थिर int64_t hc       = x->ne[1];
    स्थिर int64_t n_tokens = x->ne[2];

    स्थिर भग्न scale = ggml_get_op_params_f32(dst, 0);
    स्थिर bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    अगर (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } अन्यथा {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(माप_प्रकार, nbx, x,       nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbw, weights, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbd, dst,     nb);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t nr  = n_embd * n_tokens;
    स्थिर int64_t dr  = (nr + nth - 1) / nth;
    स्थिर int64_t ir0 = dr * ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i0 = ir % n_embd;
        स्थिर int64_t it = ir / n_embd;

        भग्न योग_2 = 0.0f;
        क्रम (int64_t ih = 0; ih < hc; ++ih) {
            स्थिर भग्न xv = *(स्थिर भग्न *) ((स्थिर अक्षर *) x->तथ्य + i0*nbx0 + ih*nbx1 + it*nbx2);
            भग्न wv;
            अगर (gated) {
                स्थिर भग्न gv = *(स्थिर भग्न *) ((स्थिर अक्षर *) weights->तथ्य + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } अन्यथा {
                wv = *(स्थिर भग्न *) ((स्थिर अक्षर *) weights->तथ्य + ih*nbw0 + it*nbw1);
            }
            योग_2 += xv * wv;
        }

        *(भग्न *) ((अक्षर *) dst->तथ्य + i0*nbd0 + it*nbd1) = scale * योग_2;
    }
}

व्योम ggml_compute_forward_dsv4_hc_pre(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

अटल व्योम ggml_compute_forward_dsv4_hc_post_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * x        = dst->src[0];
    स्थिर ggml_tensor * residual = dst->src[1];
    स्थिर ggml_tensor * post     = dst->src[2];
    स्थिर ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिर int64_t n_embd   = x->ne[0];
    स्थिर int64_t n_tokens = x->ne[1];
    स्थिर int64_t hc       = residual->ne[1];

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == hc);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == hc);
    GGML_ASSERT(post->ne[1] == n_tokens);

    // comb == NULL: identity mixing, each stream keeps its own residual
    माप_प्रकार nbc0 = 0;
    माप_प्रकार nbc1 = 0;
    माप_प्रकार nbc2 = 0;
    अगर (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        GGML_ASSERT(comb->ne[0] == hc);
        GGML_ASSERT(comb->ne[1] == hc);
        GGML_ASSERT(comb->ne[2] == n_tokens);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    GGML_TENSOR_LOCALS(माप_प्रकार, nbx, x,        nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbr, residual, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbp, post,     nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbd, dst,      nb);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t nr  = n_embd * hc * n_tokens;
    स्थिर int64_t dr  = (nr + nth - 1) / nth;
    स्थिर int64_t ir0 = dr * ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i0     = ir % n_embd;
        स्थिर int64_t idst   = (ir / n_embd) % hc;
        स्थिर int64_t it     = ir / (n_embd * hc);

        स्थिर भग्न xv = *(स्थिर भग्न *) ((स्थिर अक्षर *) x->तथ्य    + i0*nbx0 + it*nbx1);
        स्थिर भग्न pv = *(स्थिर भग्न *) ((स्थिर अक्षर *) post->तथ्य + idst*nbp0 + it*nbp1);

        भग्न योग_2 = xv * pv;
        अगर (comb) {
            क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
                स्थिर भग्न rv = *(स्थिर भग्न *) ((स्थिर अक्षर *) residual->तथ्य + i0*nbr0 + isrc*nbr1 + it*nbr2);
                स्थिर भग्न cv = *(स्थिर भग्न *) ((स्थिर अक्षर *) comb->तथ्य     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                योग_2 += rv * cv;
            }
        } अन्यथा {
            योग_2 += *(स्थिर भग्न *) ((स्थिर अक्षर *) residual->तथ्य + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(भग्न *) ((अक्षर *) dst->तथ्य + i0*nbd0 + idst*nbd1 + it*nbd2) = योग_2;
    }
}

व्योम ggml_compute_forward_dsv4_hc_post(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

व्योम ggml_compute_forward_शीर्ष_k_mask(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर ggml_tensor * mask   = dst->src[0];
    स्थिर ggml_tensor * शीर्ष_k  = dst->src[1];
    स्थिर ggml_tensor * score  = dst->src[2];
    स्थिर ggml_tensor * blocks = dst->src[3];
    स्थिर ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_आकार(dst->type));

    स्थिर int64_t n_kv    = mask->ne[0];
    स्थिर int64_t ne1     = dst->ne[1];
    स्थिर int64_t ne2     = dst->ne[2];
    स्थिर int64_t n_rows  = ggml_nrows(dst);
    स्थिर int64_t n_शीर्ष_2   = शीर्ष_k->ne[0];
    स्थिर int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    स्थिर int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    स्थिर int64_t n_extra = extra ? extra->ne[0] : 0;

    स्थिर माप_प्रकार ts = ggml_type_आकार(dst->type);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    क्रम (int64_t ir = ith; ir < n_rows; ir += nth) {
        स्थिर int64_t i1 = ir % ne1;
        स्थिर int64_t i2 = (ir / ne1) % ne2;
        स्थिर int64_t i3 = ir / (ne1*ne2);

        स्थिर अक्षर * mask_row = (स्थिर अक्षर *) mask->तथ्य
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        अक्षर * dst_row = (अक्षर *) dst->तथ्य + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        अगर (dst->type == GGML_TYPE_F16) {
            स्थिर ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-अनन्त);
            क्रम (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } अन्यथा {
            क्रम (int64_t c = 0; c < n_kv; ++c) {
                ((भग्न *) dst_row)[c] = -अनन्त;
            }
        }

        स्वतः keep = [&](int32_t c) {
            अगर (c >= 0 && c < n_kv) {
                स_नकल(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        स्थिर अक्षर * शीर्ष_k_row = (स्थिर अक्षर *) शीर्ष_k->तथ्य + i1*शीर्ष_k->nb[1] + i2*शीर्ष_k->nb[2] + i3*शीर्ष_k->nb[3];

        क्रम (int64_t j = 0; j < n_शीर्ष_2; ++j) {
            स्थिर int32_t p = *(स्थिर int32_t *) (शीर्ष_k_row + j*शीर्ष_k->nb[0]);
            अगर (p < 0 || p >= n_pick) {
                जारी;
            }
            अगर (score) {
                स्थिर भग्न s = *(स्थिर भग्न *) ((स्थिर अक्षर *) score->तथ्य
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                अगर (!(s > -अनन्त)) {
                    जारी;
                }
            }
            अगर (!blocks) {
                keep(p);
                जारी;
            }
            क्रम (int64_t k = 0; k < n_blk; ++k) {
                keep(*(स्थिर int32_t *) ((स्थिर अक्षर *) blocks->तथ्य + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        अगर (extra) {
            स्थिर अक्षर * extra_row = (स्थिर अक्षर *) extra->तथ्य + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            क्रम (int64_t j = 0; j < n_extra; ++j) {
                keep(*(स्थिर int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

अटल व्योम ggml_compute_forward_rwkv_wkv7_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिर int64_t T = dst->src[1]->ne[2];
    स्थिर int64_t C = dst->ne[0];
    स्थिर int64_t HEADS = dst->src[1]->ne[1];
    स्थिर int64_t n_seqs = dst->src[6]->ne[1];
    स्थिर int64_t शीर्ष_आकार = C / HEADS;

    भग्न * dst_तथ्य = (भग्न *) dst->तथ्य;
    भग्न * state = ((भग्न *) dst->तथ्य) + C * T;

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक h_start =  (HEADS * (ith    )) / nth;
    स्थिर पूर्णांक h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    भग्न * r = (भग्न *) dst->src[0]->तथ्य;
    भग्न * w = (भग्न *) dst->src[1]->तथ्य;
    भग्न * k = (भग्न *) dst->src[2]->तथ्य;
    भग्न * v = (भग्न *) dst->src[3]->तथ्य;
    भग्न * a = (भग्न *) dst->src[4]->तथ्य;
    भग्न * b = (भग्न *) dst->src[5]->तथ्य;

    int64_t t_stride = HEADS * शीर्ष_आकार; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = शीर्ष_आकार * शीर्ष_आकार;

    #अगर defined(GGML_SIMD)
        #अगर defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            क्रम (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
                भग्न * state_cur = state + state_offset;
                भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[6]->तथ्य + state_offset;

                क्रम (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    क्रम (int64_t i = 0; i < शीर्ष_आकार; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        भग्न v_val = v[t_h_i_offset];

                        भग्न sa = 0, परिणाम = 0;
                        क्रम (int64_t j = 0; j < शीर्ष_आकार; j++) {
                            sa += a[t_h_offset + j] * state_पूर्व[h_2d_i_offset + j];
                        }

                        क्रम (int64_t j = 0; j < शीर्ष_आकार; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            भग्न r_val = r[t_h_j_offset];
                            भग्न w_val = w[t_h_j_offset];
                            भग्न k_val = k[t_h_j_offset];
                            भग्न b_val = b[t_h_j_offset];
                            भग्न kv_val = v_val * k_val;
                            भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = पूर्व_state_val * w_val + kv_val + sa * b_val;
                            परिणाम += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_तथ्य[t_h_i_offset] = परिणाम;
                    }
                }
            }
        #अन्यथा
            क्रम (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
                भग्न * state_cur = state + state_offset;
                भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[6]->तथ्य + state_offset;

                क्रम (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    क्रम (int64_t ii = 0; ii < शीर्ष_आकार; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        भग्न sa = 0;
                        {
                            GGML_F32_VEC योग_2[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            क्रम (int64_t j = 0; j < शीर्ष_आकार; j += GGML_F32_STEP) {
                                क्रम (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                    ax[kk] = GGML_F32_VEC_LOAD(&a[t_h_offset + j + kk * GGML_F32_EPR]);
                                    ay[kk] = GGML_F32_VEC_LOAD(&state_पूर्व[h_2d_i_offset + j + kk * GGML_F32_EPR]);
                                    योग_2[kk] = GGML_F32_VEC_FMA(योग_2[kk], ax[kk], ay[kk]);
                                }
                            }
                            GGML_F32_VEC_REDUCE(sa, योग_2);
                        }

                        GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

                        int64_t j = 0;
                        GGML_F32_VEC परिणाम_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                        क्रम (; j < शीर्ष_आकार; j += GGML_F32_STEP) {
                            क्रम (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
                                int64_t t_h_j_offset = t_h_offset + j + kk * GGML_F32_EPR;
                                int64_t h_2d_i_j_offset = h_2d_i_offset + j + kk * GGML_F32_EPR;

                                GGML_F32_VEC r_vec = GGML_F32_VEC_LOAD(&r[t_h_j_offset]);
                                GGML_F32_VEC w_vec = GGML_F32_VEC_LOAD(&w[t_h_j_offset]);
                                GGML_F32_VEC k_vec = GGML_F32_VEC_LOAD(&k[t_h_j_offset]);
                                GGML_F32_VEC b_vec = GGML_F32_VEC_LOAD(&b[t_h_j_offset]);

                                k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

                                GGML_F32_VEC state_vec = GGML_F32_VEC_LOAD(&state_पूर्व[h_2d_i_j_offset]);
                                // kv + s * decay + sa * b
                                state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
                                state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
                                GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_offset], state_vec);

                                परिणाम_vec[kk] = GGML_F32_VEC_FMA(परिणाम_vec[kk], state_vec, r_vec);
                            }
                        }
                        GGML_F32_VEC_REDUCE(dst_तथ्य[t_h_i_offset], परिणाम_vec);

                        // There shouldn't be left-overs though.
                        क्रम (; j < शीर्ष_आकार; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            भग्न r_val = r[t_h_j_offset];
                            भग्न w_val = w[t_h_j_offset];
                            भग्न k_val = k[t_h_j_offset];
                            भग्न b_val = b[t_h_j_offset];
                            भग्न kv_val = v[t_h_i_offset] * k_val;

                            भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = पूर्व_state_val * w_val + kv_val + sa * b_val;
                            dst_तथ्य[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #पूर्ण_अगर
    #अन्यथा
        क्रम (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = शीर्ष_आकार * C * (t / (T / n_seqs));
            भग्न * state_cur = state + state_offset;
            भग्न * state_पूर्व = t % (T / n_seqs) ? state_cur : (भग्न*)dst->src[6]->तथ्य + state_offset;

            क्रम (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                क्रम (int64_t i = 0; i < शीर्ष_आकार; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    भग्न v_val = v[t_h_i_offset];

                    भग्न sa = 0, परिणाम = 0;
                    क्रम (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        sa += a[t_h_offset + j] * state_पूर्व[h_2d_i_offset + j];
                    }

                    क्रम (int64_t j = 0; j < शीर्ष_आकार; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        भग्न r_val = r[t_h_j_offset];
                        भग्न w_val = w[t_h_j_offset];
                        भग्न k_val = k[t_h_j_offset];
                        भग्न b_val = b[t_h_j_offset];
                        भग्न kv_val = v_val * k_val;
                        भग्न पूर्व_state_val = state_पूर्व[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = पूर्व_state_val * w_val + kv_val + sa * b_val;
                        परिणाम += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_तथ्य[t_h_i_offset] = परिणाम;
                }
            }
        }
    #पूर्ण_अगर
}


व्योम ggml_compute_forward_rwkv_wkv7(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

व्योम ggml_compute_forward_map_custom1(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * a = dst->src[0];

    काष्ठा ggml_map_custom1_op_params p;
    स_नकल(&p, dst->op_params, माप(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

व्योम ggml_compute_forward_map_custom2(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * a = dst->src[0];
    स्थिर ggml_tensor * b = dst->src[1];

    काष्ठा ggml_map_custom2_op_params p;
    स_नकल(&p, dst->op_params, माप(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

व्योम ggml_compute_forward_map_custom3(
        स्थिर ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिर ggml_tensor * a = dst->src[0];
    स्थिर ggml_tensor * b = dst->src[1];
    स्थिर ggml_tensor * c = dst->src[2];

    काष्ठा ggml_map_custom3_op_params p;
    स_नकल(&p, dst->op_params, माप(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

व्योम ggml_compute_forward_custom(
    स्थिर काष्ठा ggml_compute_params * params,
          काष्ठा ggml_tensor * dst) {

    काष्ठा ggml_custom_op_params p;
    स_नकल(&p, dst->op_params, माप(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

अटल व्योम ggml_compute_forward_cross_entropy_loss_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_आकार(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_आकार(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    स्थिर int64_t nc = src0->ne[0];
    स्थिर int64_t nr = ggml_nrows(src0);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    भग्न * sums =  (भग्न *) params->wdata;
    भग्न * st   = ((भग्न *) params->wdata) + nth + ith*nc;
    भग्न योग_thread = 0.0f;

    GGML_ASSERT(params->wsize >= माप(भग्न) * (nth + nth * nc));

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    क्रम (int64_t i1 = ir0; i1 < ir1; ++i1) {
        स्थिर भग्न * s0 = (स्थिर भग्न *)((स्थिर अक्षर *) src0->तथ्य + i1*src0->nb[1]);
        स्थिर भग्न * s1 = (स्थिर भग्न *)((स्थिर अक्षर *) src1->तथ्य + i1*src1->nb[1]);

#अगर_अघोषित न_संशोधन
        क्रम (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            निश्चित(!isnan(s0[i]));
            निश्चित(!isnan(s1[i]));
        }
#पूर्ण_अगर // NDEBUG

        भग्न max = -अनन्त;
        ggml_vec_max_f32(nc, &max, s0);
        स्थिर ggml_float योग_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        निश्चित(योग_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -योग_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        भग्न योग_st = 0.0f;
        ggml_vec_योग_f32(nc, &योग_st, st);
        योग_thread += योग_st;

#अगर_अघोषित न_संशोधन
        क्रम (int64_t i = 0; i < nc; ++i) {
            निश्चित(!isnan(st[i]));
            निश्चित(!isinf(st[i]));
        }
#पूर्ण_अगर // NDEBUG
    }
    sums[ith] = योग_thread;
    ggml_barrier(params->threadpool);

    अगर (ith == 0) {
        भग्न * dp = (भग्न *) dst->तथ्य;
        ggml_vec_योग_f32(nth, dp, sums);
        dp[0] *= -1.0f / (भग्न) nr;
    }
}

व्योम ggml_compute_forward_cross_entropy_loss(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

अटल व्योम ggml_compute_forward_cross_entropy_loss_back_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    स्थिर ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    स्थिर ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    स्थिर int64_t ith = params->ith;
    स्थिर int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    स्थिर int64_t nc = src0f->ne[0];
    स्थिर int64_t nr = ggml_nrows(src0f);

    // rows per thread
    स्थिर int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर int64_t ir0 = dr*ith;
    स्थिर int64_t ir1 = MIN(ir0 + dr, nr);

    स्थिर भग्न d_by_nr = ((स्थिर भग्न *) grad->तथ्य)[0] / (भग्न) nr;

    क्रम (int64_t i1 = ir0; i1 < ir1; i1++) {
        भग्न       * ds0 = (भग्न       *)((अक्षर       *) dst->तथ्य   + i1*dst->nb[1]);
        स्थिर भग्न * s0  = (स्थिर भग्न *)((स्थिर अक्षर *) src0f->तथ्य + i1*src0f->nb[1]);
        स्थिर भग्न * s1  = (स्थिर भग्न *)((स्थिर अक्षर *) src1f->तथ्य + i1*src1f->nb[1]);

#अगर_अघोषित न_संशोधन
        क्रम (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            निश्चित(!isnan(s0[i]));
            निश्चित(!isnan(s1[i]));
        }
#पूर्ण_अगर // NDEBUG

        // soft_max
        भग्न max = -अनन्त;
        ggml_vec_max_f32(nc, &max, s0);
        स्थिर ggml_float योग_2 = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        निश्चित(योग_2 > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/योग_2);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#अगर_अघोषित न_संशोधन
        क्रम (int64_t i = 0; i < nc; ++i) {
            निश्चित(!isnan(ds0[i]));
            निश्चित(!isinf(ds0[i]));
        }
#पूर्ण_अगर // NDEBUG
    }
}

व्योम ggml_compute_forward_cross_entropy_loss_back(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_compute_forward_opt_step_adamw_f32(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0         = dst->src[0];
    स्थिर ggml_tensor * src0_grad    = dst->src[1];
    स्थिर ggml_tensor * src0_grad_m  = dst->src[2];
    स्थिर ggml_tensor * src0_grad_v  = dst->src[3];
    स्थिर ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    स्थिर भग्न * adamw_params_ptr = ggml_get_तथ्य_f32(adamw_params);

    स्थिर भग्न alpha  = adamw_params_ptr[0];
    स्थिर भग्न beta1  = adamw_params_ptr[1];
    स्थिर भग्न beta2  = adamw_params_ptr[2];
    स्थिर भग्न eps    = adamw_params_ptr[3];
    स्थिर भग्न wd     = adamw_params_ptr[4];
    स्थिर भग्न beta1h = adamw_params_ptr[5];
    स्थिर भग्न beta2h = adamw_params_ptr[6];
    स्थिर भग्न keep   = 1.f - alpha * wd;
    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir/(ne02*ne01);
        स्थिर int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिर int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        स्थिर माप_प्रकार offset = i03*nb03 + i02*nb02 + i01*nb01;

        भग्न       * w = (भग्न       *) ((अक्षर       *) src0->तथ्य        + offset); // weight
        स्थिर भग्न * g = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_grad->तथ्य   + offset); // grad
        भग्न       * m = (भग्न       *) ((अक्षर       *) src0_grad_m->तथ्य + offset);
        भग्न       * v = (भग्न       *) ((अक्षर       *) src0_grad_v->तथ्य + offset);

        क्रम (पूर्णांक i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            स्थिर भग्न mh =       m[i00]*beta1h;
            स्थिर भग्न vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

व्योम ggml_compute_forward_opt_step_adamw(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error");
            }
    }
}

अटल व्योम ggml_compute_forward_opt_step_sgd_f32(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0       = dst->src[0];
    स्थिर ggml_tensor * src0_grad  = dst->src[1];
    स्थिर ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर पूर्णांक nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == माप(भग्न));

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1) / nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr * ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    स्थिर भग्न * sgd_params_ptr   = ggml_get_तथ्य_f32(sgd_params);
    स्थिर भग्न   alpha            = sgd_params_ptr[0];
    स्थिर भग्न   keep             = 1.f - alpha * sgd_params_ptr[1];

    क्रम (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिर int64_t i03 = ir / (ne02 * ne01);
        स्थिर int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        स्थिर int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        स्थिर माप_प्रकार offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        भग्न *       w = (भग्न *) ((अक्षर *) src0->तथ्य + offset);                   // weight
        स्थिर भग्न * g = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_grad->तथ्य + offset);  // grad

        क्रम (पूर्णांक i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

व्योम ggml_compute_forward_opt_step_sgd(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    चयन (src0->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

अटल अंतरभूत भग्न ggml_fwht_load(स्थिर भग्न value) {
    वापस value;
}

अटल अंतरभूत भग्न ggml_fwht_load(स्थिर ggml_fp16_t value) {
    वापस ggml_fp16_to_fp32(value);
}

ढाँचा<typename src_t>
अटल व्योम ggml_compute_forward_fwht_impl(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, भग्न> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    स्थिर int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    स्थिर int64_t nr = ne11 * ne12 * ne13;
    स्थिर int64_t rows_per_thread = (nr + nth - 1) / nth;
    स्थिर int64_t start_row = ith * rows_per_thread;
    स्थिर int64_t end_row = MIN(start_row + rows_per_thread, nr);

    स्थिर भग्न scale = 1.0f / sqrtf((भग्न)n);

#अगर defined(GGML_SIMD)
    स्थिर GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#पूर्ण_अगर

    क्रम (int64_t r = start_row; r < end_row; r++) {
        स्थिर int64_t i13 = r / (ne11 * ne12);
        स्थिर int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        स्थिर int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        स्थिर src_t * src_row = (स्थिर src_t *) ((स्थिर अक्षर *) src1->तथ्य + i11 * nb11 + i12 * nb12 + i13 * nb13);
        भग्न * dst_row = (भग्न *) ((अक्षर *) dst->तथ्य + i11 * nb1 + i12 * nb2 + i13 * nb3);

        क्रम (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#अगर defined(GGML_SIMD)
#अगर defined(__ARM_FEATURE_SVE)
        स्थिर पूर्णांक step = svcntw();
#अन्यथा
        स्थिर पूर्णांक step = GGML_F32_EPR;
#पूर्ण_अगर
#अन्यथा
        स्थिर पूर्णांक step = n;
#पूर्ण_अगर
        क्रम (int64_t len = 1; len < step && len < n; len <<= 1) {
            क्रम (int64_t i = 0; i < n; i += 2 * len) {
                क्रम (int64_t j = 0; j < len; j++) {
                    भग्न u = dst_row[i + j];
                    भग्न v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#अगर defined(GGML_SIMD)
        क्रम (int64_t len = step; len < n; len <<= 1) {
            क्रम (int64_t i = 0; i < n; i += 2 * len) {
                क्रम (int64_t j = 0; j < len; j += step) {
                    GGML_F32_VEC u = GGML_F32_VEC_LOAD(dst_row + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_LOAD(dst_row + i + len + j);

                    GGML_F32_VEC_STORE(dst_row + i + j,       GGML_F32_VEC_ADD(u, v));
                    GGML_F32_VEC_STORE(dst_row + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#पूर्ण_अगर
    }
}

व्योम ggml_compute_forward_fwht(स्थिर ggml_compute_params * params, ggml_tensor * dst) {
    स्थिर ggml_tensor * src1 = dst->src[1];

    चयन (src1->type) {
        हाल GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<भग्न>(params, dst);
            }
            अवरोध;
        हाल GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            अवरोध;
        शेष:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

व्योम ggml_compute_forward_lightning_indexer(
        स्थिर ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिर ggml_tensor * q = dst->src[0];
    स्थिर ggml_tensor * k = dst->src[1];
    स्थिर ggml_tensor * w = dst->src[2]; // weights
    स्थिर ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, नया,  w, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_आकार(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_आकार(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_आकार(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_आकार(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_आकार(  m->type));

    स्थिर पूर्णांक n_embd    = q->ne[0];
    स्थिर पूर्णांक n_शीर्ष    = q->ne[1];
    स्थिर पूर्णांक n_tokens  = q->ne[2];
    स्थिर पूर्णांक n_stream  = q->ne[3];
    स्थिर पूर्णांक n_kv      = k->ne[2];

    ggml_to_float_t स्थिर k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    स्थिर पूर्णांक nr  = n_kv;
    स्थिर पूर्णांक ith = params->ith;
    स्थिर पूर्णांक nth = params->nth;

    // (temporary) buffer for K converted to float
    भग्न * k_row_f32 = (भग्न *) params->wdata + ith*(1*n_embd + CACHE_पंक्ति_आकार_F32);

    // rows per thread
    स्थिर पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिर पूर्णांक ir0 = dr*ith;
    स्थिर पूर्णांक ir1 = MIN(ir0 + dr, nr);

    क्रम (पूर्णांक s = 0; s < n_stream; ++s) {
        क्रम (पूर्णांक t = 0; t < n_tokens; ++t) {
            स्थिर भग्न       *   w_row =       (भग्न *) ((अक्षर *)   w->तथ्य + t*nbw1 +        s*nbw3);
            स्थिर ggml_fp16_t *   m_row = (ggml_fp16_t *) ((अक्षर *)   m->तथ्य + t*nbm1 + (s%nem3)*nbm3);
            भग्न             * dst_row =       (भग्न *) ((अक्षर *) dst->तथ्य + t*nb1  +        s*nb3 );
            क्रम (पूर्णांक ik = ir0; ik < ir1; ++ik) {
                अक्षर * k_row = (अक्षर *) k->तथ्य + ik*nbk2 + s*nbk3;
                अगर (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } अन्यथा {
                    k_row_f32 = (भग्न *) k_row;
                }
                भग्न score = 0.0f;
                क्रम (पूर्णांक h = 0; h < n_शीर्ष; ++h) {
                    // dot product of q and k for head h
                    भग्न qk = 0.0f;
                    स्थिर भग्न * q_row = (भग्न *) ((अक्षर *) q->तथ्य + h*nbq1 + t*nbq2 + s*nbq3);
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
