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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    स्थिरांक माप_प्रकार nb0 = ggml_type_size(src0->type);

    स्थिरांक पूर्णांक ith = params->ith; // thread index
    स्थिरांक पूर्णांक nth = params->nth; // number of threads

    // parallelize by blocks
    स्थिरांक पूर्णांक nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    स्थिरांक पूर्णांक dr = (nk + nth - 1) / nth;
    स्थिरांक पूर्णांक k0 = dr * ith;
    स्थिरांक पूर्णांक k1 = MIN(k0 + dr, nk);

    जर (k0 < k1) {
        स_नकल(
            ((अक्षर *)  dst->data + k0*nb0),
            ((अक्षर *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

ढाँचा<typename src_t, typename dst_t>
स्थिर शून्य ggml_compute_forward_dup_flt(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith; // thread index
    स्थिरांक पूर्णांक nth = params->nth; // number of threads

    // parallelize by rows
    स्थिरांक पूर्णांक nr = ne01;
    // number of rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1) / nth;
    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr * ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    जर (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        स्थिरांक माप_प्रकार rs = ne00*nb00;
        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                    स_नकल(
                        ((अक्षर *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        परत;
    }

    // case: dst tensor is contiguous
    जर (ggml_is_contiguous(dst)) {
        जर (nb00 == आकार(src_t)) {
            जर constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                माप_प्रकार id = 0;
                स्थिरांक माप_प्रकार rs = ne00 * nb00;
                अक्षर * dst_ptr = (अक्षर *) dst->data;

                साठी (पूर्णांक i03 = 0; i03 < ne03; i03++) {
                    साठी (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        साठी (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                            स्थिरांक अक्षर * src0_ptr = (अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            स_नकल(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } अन्यथा {
                // casting between non-quantized types
                माप_प्रकार id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                साठी (पूर्णांक i03 = 0; i03 < ne03; i03++) {
                    साठी (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        साठी (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                            स्थिरांक src_t * src0_ptr = (src_t *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            साठी (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                                दशांश tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
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
            dst_t * dst_ptr = (dst_t *) dst->data;

            साठी (पूर्णांक i03 = 0; i03 < ne03; i03++) {
                साठी (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    साठी (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                        साठी (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                            स्थिरांक src_t * src0_ptr = (src_t *) ((अक्षर *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            दशांश tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        परत;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    जर constexpr (std::is_same_v<dst_t, src_t>) {
        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                पर्यंत (i10 >= ne0) {
                    i10 -= ne0;
                    जर (++i11 == ne1) {
                        i11 = 0;
                        जर (++i12 == ne2) {
                            i12 = 0;
                            जर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिरांक अक्षर * src0_ptr = ((अक्षर *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              अक्षर * dst_ptr  = ((अक्षर *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        स_नकल(dst_ptr, src0_ptr, आकार(dst_t));

                        जर (++i10 == ne00) {
                            i10 = 0;
                            जर (++i11 == ne01) {
                                i11 = 0;
                                जर (++i12 == ne02) {
                                    i12 = 0;
                                    जर (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                पर्यंत (i10 >= ne0) {
                    i10 -= ne0;
                    जर (++i11 == ne1) {
                        i11 = 0;
                        जर (++i12 == ne2) {
                            i12 = 0;
                            जर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } अन्यथा {
        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                पर्यंत (i10 >= ne0) {
                    i10 -= ne0;
                    जर (++i11 == ne1) {
                        i11 = 0;
                        जर (++i12 == ne2) {
                            i12 = 0;
                            जर (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिरांक अक्षर * src0_ptr = ((अक्षर *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              अक्षर * dst_ptr  = ((अक्षर *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        दशांश tmp = type_conversion_table<src_t>::to_f32(*(स्थिरांक src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        जर (++i10 == ne0) {
                            i10 = 0;
                            जर (++i11 == ne1) {
                                i11 = 0;
                                जर (++i12 == ne2) {
                                    i12 = 0;
                                    जर (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                पर्यंत (i10 >= ne0) {
                    i10 -= ne0;
                    जर (++i11 == ne1) {
                        i11 = 0;
                        जर (++i12 == ne2) {
                            i12 = 0;
                            जर (++i13 == ne3) {
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
स्थिर शून्य ggml_compute_forward_dup_to_q(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith; // thread index
    स्थिरांक पूर्णांक nth = params->nth; // number of threads

    // parallelize by rows
    स्थिरांक पूर्णांक nr = ne01;
    // number of rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1) / nth;
    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr * ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    जर (ggml_is_contiguous(dst) &&
            nb00 == आकार(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t स्थिरांक quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        दशांश * src0_f32 = (दशांश *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

        माप_प्रकार id = 0;
        माप_प्रकार rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        अक्षर * dst_ptr = (अक्षर *) dst->data;

        साठी (पूर्णांक i03 = 0; i03 < ne03; i03++) {
            साठी (पूर्णांक i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                साठी (पूर्णांक i01 = ir0; i01 < ir1; i01++) {
                    स्थिरांक src_t * src0_ptr = (src_t *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    साठी (पूर्णांक i00 = 0; i00 < ne00; i00++) {
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    जर (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        परत;
    }

    स्थिरांक माप_प्रकार type_size = ggml_type_size(src0->type);

    स्थिरांक पूर्णांक ith = params->ith; // thread index
    स्थिरांक पूर्णांक nth = params->nth; // number of threads

    // parallelize by rows
    स्थिरांक पूर्णांक nr = ne01;
    // number of rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1) / nth;
    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr * ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    जर (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        स्थिरांक माप_प्रकार rs = ggml_row_size(src0->type, ne00);
        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                    स_नकल(
                        ((अक्षर *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        परत;
    }

    जर (ggml_is_contiguous(dst)) {
        माप_प्रकार id = 0;
        अक्षर * dst_ptr = (अक्षर *) dst->data;
        स्थिरांक माप_प्रकार rs = ne00 * type_size;

        जर (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            साठी (int64_t i03 = 0; i03 < ne03; i03++) {
                साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                        स्थिरांक अक्षर * src0_ptr = (अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        स_नकल(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } अन्यथा {
            //printf("%s: this is not optimal - fix me\n", __func__);

            साठी (int64_t i03 = 0; i03 < ne03; i03++) {
                साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                        साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                            स्थिरांक अक्षर * src0_ptr = (अक्षर *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            स_नकल(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        परत;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    स्थिरांक int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    स्थिरांक int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            पर्यंत (k10 >= nk0) {
                k10 -= nk0;
                जर (++i11 == ne1) {
                    i11 = 0;
                    जर (++i12 == ne2) {
                        i12 = 0;
                        जर (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            साठी (int64_t i01 = ir0; i01 < ir1; i01++) {
                साठी (int64_t k00 = 0; k00 < nk00; k00++) {
                    स्थिरांक अक्षर * src0_ptr = ((अक्षर *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          अक्षर * dst_ptr  = ((अक्षर *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    स_नकल(dst_ptr, src0_ptr, type_size);

                    जर (++k10 == nk0) {
                        k10 = 0;
                        जर (++i11 == ne1) {
                            i11 = 0;
                            जर (++i12 == ne2) {
                                i12 = 0;
                                जर (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            पर्यंत (k10 >= nk0) {
                k10 -= nk0;
                जर (++i11 == ne1) {
                    i11 = 0;
                    जर (++i12 == ne2) {
                        i12 = 0;
                        जर (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_dup_from_q(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक ggml_type type = src0->type;
    ggml_to_float_t स्थिरांक dequantize_row_q = ggml_get_type_traits(type)->to_float;

    माप_प्रकार qk = ggml_blck_size(type);
    स्थिरांक int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        स्थिरांक int64_t i03 = i/(ne00 * ne01 * ne02);
        स्थिरांक int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        स्थिरांक int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        स्थिरांक int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        स्थिरांक int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        स्थिरांक int64_t i13 = i/(ne10 * ne11 * ne12);
        स्थिरांक int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        स्थिरांक int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        स्थिरांक int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        स्थिरांक int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (स्थिरांक शून्य *) ((अक्षर *) src0->data + x_offset),
                     (दशांश *) ((अक्षर *)  dst->data + dst_offset), qk);
    }
}

शून्य ggml_compute_forward_dup(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        परत;
    }

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
            {
                /**/ जर (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, दशांश      >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                /**/ जर (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, दशांश      >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                /**/ जर (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<दशांश, ggml_fp16_t>(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<दशांश, ggml_bf16_t>(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<दशांश, दशांश      >(params, dst);
                अन्यथा जर (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<दशांश, int32_t    >(params, dst);
                अन्यथा ggml_compute_forward_dup_to_q<दशांश>(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                जर (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, दशांश>(params, dst);
                अन्यथा GGML_ABORT("not implemented");
            } थांबा;
        पूर्वनिर्धारित:
            {
                जर (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    थांबा;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

स्थिर शून्य ggml_compute_forward_add_q_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक ggml_type type = src0->type;
    स्थिरांक ggml_type dtype = dst->type;
    ggml_to_float_t स्थिरांक dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t स्थिरांक quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == आकार(दशांश));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    दशांश * wdata = (दशांश *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        स्थिरांक पूर्णांक i03 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        स्थिरांक पूर्णांक i13 = i03;
        स्थिरांक पूर्णांक i12 = i02;
        स्थिरांक पूर्णांक i11 = i01;

        स्थिरांक पूर्णांक i3 = i03;
        स्थिरांक पूर्णांक i2 = i02;
        स्थिरांक पूर्णांक i1 = i01;

        शून्य  * src0_row = (शून्य *) ((अक्षर *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        दशांश * src1_row = (दशांश *)((अक्षर *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        शून्य  * dst_row  = (शून्य *) ((अक्षर *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        प्रतिज्ञा(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        जर (quantize_row_q != रिक्त) {
            quantize_row_q(wdata, dst_row, ne00);
        } अन्यथा {
            स_नकल(dst_row, wdata, ne0*nb0);
        }
    }
}

शून्य ggml_compute_forward_add(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

स्थिर शून्य ggml_compute_forward_add_id_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    स्थिरांक ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));
    GGML_ASSERT(src1->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == आकार(दशांश));
    GGML_ASSERT(nb10 == आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        स्थिरांक पूर्णांक i11 = *(int32_t *) ((अक्षर *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (दशांश *) ((अक्षर *) src1->data + i11*nb11));
    }
}

शून्य ggml_compute_forward_add_id(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

स्थिर शून्य ggml_compute_forward_add1_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == आकार(दशांश));
    GGML_ASSERT(nb00 == आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (दशांश *) ((अक्षर *) src1->data), 0,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#अन्यथा
        ggml_vec_add1_f32(ne0,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(दशांश *) src1->data);
#endif
    }
}

स्थिर शून्य ggml_compute_forward_add1_f16_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिरांक दशांश v = *(दशांश *) src1->data;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == आकार(ggml_fp16_t));
    GGML_ASSERT(nb00 == आकार(ggml_fp16_t));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        साठी (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_add1_f16_f16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिरांक दशांश v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == आकार(ggml_fp16_t));
    GGML_ASSERT(nb00 == आकार(ggml_fp16_t));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        साठी (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_add1_q_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिरांक दशांश v = *(दशांश *) src1->data;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक ggml_type type = src0->type;
    ggml_to_float_t स्थिरांक dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t स्थिरांक quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    दशांश * wdata = (दशांश *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32) * ith;

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        शून्य  * src0_row = (शून्य *) ((अक्षर *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        शून्य  * dst_row  = (शून्य *) ((अक्षर *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिरांक दशांश v = *(दशांश *) src1->data;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == आकार(ggml_bf16_t));
    GGML_ASSERT(nb00 == आकार(ggml_bf16_t));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        साठी (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_add1_bf16_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    स्थिरांक दशांश v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == आकार(ggml_bf16_t));
    GGML_ASSERT(nb00 == आकार(ggml_bf16_t));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne2*ne1);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        साठी (पूर्णांक i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

शून्य ggml_compute_forward_add1(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                जर (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                अन्यथा जर (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                अन्यथा {
                    GGML_ABORT("fatal error");
                }
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                जर (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                अन्यथा जर (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                अन्यथा {
                    GGML_ABORT("fatal error");
                }
            } थांबा;
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q8_1:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

स्थिर शून्य ggml_compute_forward_acc_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    जर (!inplace) {
        जर (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स_नकल(
                ((अक्षर *)  dst->data),
                ((अक्षर *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr = ggml_nrows(src1);
    स्थिरांक पूर्णांक nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    स्थिरांक माप_प्रकार nb0 = ggml_element_size(src0);

    स्थिरांक माप_प्रकार nb00 = nb0;
    स्थिरांक माप_प्रकार nb01 = nb1;
    स्थिरांक माप_प्रकार nb02 = nb2;
    स्थिरांक माप_प्रकार nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne12*ne11);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne12*ne11)/ne11;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (दशांश *) ((अक्षर *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#अन्यथा
        ggml_vec_add_f32(nc,
                (दशांश *) ((अक्षर *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (दशांश *) ((अक्षर *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

शून्य ggml_compute_forward_acc(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q8_1:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

स्थिर शून्य ggml_compute_forward_sum_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)

    ggml_float sum = 0;

    जर (nb00 == आकार(दशांश)) {
        ggml_float row_sum = 0;
        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (दशांश *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } अन्यथा {
        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिरांक अक्षर * row = (स्थिरांक अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(स्थिरांक दशांश *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((दशांश *) dst->data)[0] = sum;
}

स्थिर शून्य ggml_compute_forward_sum_f16(
    स्थिरांक ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    प्रतिज्ञा(src0->nb[0] == आकार(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)

    दशांश sum = 0;
    दशांश row_sum = 0;

    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((अक्षर *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

स्थिर शून्य ggml_compute_forward_sum_bf16(
    स्थिरांक ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(ggml_is_scalar(dst));

    प्रतिज्ञा(src0->nb[0] == आकार(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)

    दशांश sum = 0;
    दशांश row_sum = 0;

    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((अक्षर *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

शून्य ggml_compute_forward_sum(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

स्थिर शून्य ggml_compute_forward_cumsum_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));
    GGML_ASSERT(dst->nb[0] == आकार(दशांश));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    स्थिरांक स्वतः [ir0, ir1] = get_thread_range(params, src0);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne02*ne01);
        स्थिरांक int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिरांक int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        दशांश * src_row = (दशांश *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        दशांश * dst_row = (दशांश *) ((अक्षर *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

शून्य ggml_compute_forward_cumsum(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

स्थिर शून्य ggml_compute_forward_sum_rows_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));
    GGML_ASSERT(dst->nb[0] == आकार(दशांश));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    साठी (int64_t i3 = 0; i3 < ne03; i3++) {
        साठी (int64_t i2 = 0; i2 < ne02; i2++) {
            साठी (int64_t i1 = 0; i1 < ne01; i1++) {
                दशांश * src_row = (दशांश *) ((अक्षर *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                दशांश * dst_row = (दशांश *) ((अक्षर *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                दशांश row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

शून्य ggml_compute_forward_sum_rows(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

स्थिर शून्य ggml_compute_forward_mean_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(src0->nb[0] == आकार(दशांश));

    GGML_TENSOR_UNARY_OP_LOCALS

    प्रतिज्ञा(ne0 == 1);
    प्रतिज्ञा(ne1 == ne01);
    प्रतिज्ञा(ne2 == ne02);
    प्रतिज्ञा(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (दशांश *) ((अक्षर *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (दशांश *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(दशांश *) ((अक्षर *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (दशांश) ne00;
            }
        }
    }
}

शून्य ggml_compute_forward_mean(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

स्थिर शून्य ggml_compute_forward_argmax_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(src0->nb[0] == आकार(दशांश));
    प्रतिज्ञा(dst->nb[0] == आकार(दशांश));

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];

    स्थिरांक माप_प्रकार nb01 = src0->nb[1];
    स्थिरांक माप_प्रकार nb0 = dst->nb[0];

    साठी (int64_t i1 = 0; i1 < ne01; i1++) {
        दशांश * src = (दशांश *) ((अक्षर *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((अक्षर *)  dst->data + i1*nb0);
        पूर्णांक v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

शून्य ggml_compute_forward_argmax(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

स्थिर शून्य ggml_compute_forward_मोजणी_equal_i32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    स्थिरांक int64_t nr = ggml_nrows(src0);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 =  ir                        / (ne02*ne01);
        स्थिरांक int64_t i02 = (ir - i03*ne03)            /       ne01;
        स्थिरांक int64_t i01 =  ir - i03*ne03 - i02*ne02;

        स्थिरांक अक्षर * data0 = (स्थिरांक अक्षर *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        स्थिरांक अक्षर * data1 = (स्थिरांक अक्षर *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        साठी (int64_t i00 = 0; i00 < ne00; ++i00) {
            स्थिरांक int32_t val0 = *((स्थिरांक int32_t *) (data0 + i00*nb00));
            स्थिरांक int32_t val1 = *((स्थिरांक int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    जर (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    जर (ith != 0) {
        परत;
    }

    साठी (पूर्णांक ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

शून्य ggml_compute_forward_मोजणी_equal(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_I32:
            {
                ggml_compute_forward_मोजणी_equal_i32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

स्थिर शून्य ggml_compute_forward_repeat_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    स्थिरांक पूर्णांक nr0 = (पूर्णांक)(ne0/ne00);
    स्थिरांक पूर्णांक nr1 = (पूर्णांक)(ne1/ne01);
    स्थिरांक पूर्णांक nr2 = (पूर्णांक)(ne2/ne02);
    स्थिरांक पूर्णांक nr3 = (पूर्णांक)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == आकार(दशांश));
    GGML_ASSERT(nb00 == आकार(दशांश));

    // TODO: maybe this is not optimal?
    साठी                         (पूर्णांक i3 = 0; i3 < nr3;  i3++) {
        साठी                     (पूर्णांक k3 = 0; k3 < ne03; k3++) {
            साठी                 (पूर्णांक i2 = 0; i2 < nr2;  i2++) {
                साठी             (पूर्णांक k2 = 0; k2 < ne02; k2++) {
                    साठी         (पूर्णांक i1 = 0; i1 < nr1;  i1++) {
                        साठी     (पूर्णांक k1 = 0; k1 < ne01; k1++) {
                            साठी (पूर्णांक i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (दशांश *) ((अक्षर *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (दशांश *) ((अक्षर *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_repeat_f16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    स्थिरांक पूर्णांक nr0 = (पूर्णांक)(ne0/ne00);
    स्थिरांक पूर्णांक nr1 = (पूर्णांक)(ne1/ne01);
    स्थिरांक पूर्णांक nr2 = (पूर्णांक)(ne2/ne02);
    स्थिरांक पूर्णांक nr3 = (पूर्णांक)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == आकार(ggml_fp16_t));
    GGML_ASSERT(nb00 == आकार(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    साठी                         (पूर्णांक i3 = 0; i3 < nr3;  i3++) {
        साठी                     (पूर्णांक k3 = 0; k3 < ne03; k3++) {
            साठी                 (पूर्णांक i2 = 0; i2 < nr2;  i2++) {
                साठी             (पूर्णांक k2 = 0; k2 < ne02; k2++) {
                    साठी         (पूर्णांक i1 = 0; i1 < nr1;  i1++) {
                        साठी     (पूर्णांक k1 = 0; k1 < ne01; k1++) {
                            साठी (पूर्णांक i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((अक्षर *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((अक्षर *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                साठी (पूर्णांक i = 0; i < ne00; ++i) {
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
        स्थिती GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
        स्थिती GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } थांबा;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

स्थिर शून्य ggml_compute_forward_repeat_back_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    स्थिरांक पूर्णांक nr0 = (पूर्णांक)(ne00/ne0);
    स्थिरांक पूर्णांक nr1 = (पूर्णांक)(ne01/ne1);
    स्थिरांक पूर्णांक nr2 = (पूर्णांक)(ne02/ne2);
    स्थिरांक पूर्णांक nr3 = (पूर्णांक)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == आकार(दशांश));
    GGML_ASSERT(nb00 == आकार(दशांश));

    जर (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (दशांश *)dst->data, 0);
    } अन्यथा {
        साठी         (पूर्णांक k3 = 0; k3 < ne3; k3++) {
            साठी     (पूर्णांक k2 = 0; k2 < ne2; k2++) {
                साठी (पूर्णांक k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (दशांश *) ((अक्षर *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    साठी                         (पूर्णांक i3 = 0; i3 < nr3; i3++) {
        साठी                     (पूर्णांक k3 = 0; k3 < ne3; k3++) {
            साठी                 (पूर्णांक i2 = 0; i2 < nr2; i2++) {
                साठी             (पूर्णांक k2 = 0; k2 < ne2; k2++) {
                    साठी         (पूर्णांक i1 = 0; i1 < nr1; i1++) {
                        साठी     (पूर्णांक k1 = 0; k1 < ne1; k1++) {
                            साठी (पूर्णांक i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (दशांश *) ((अक्षर *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (दशांश *) ((अक्षर *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_repeat_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
स्थिर शून्य ggml_compute_forward_concat_any(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    जर (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } अन्यथा {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    साठी (पूर्णांक i3 = 0; i3 < ne03; i3++) {
        साठी (पूर्णांक i2 = ith; i2 < ne02; i2 += nth) {
            साठी (पूर्णांक i1 = 0; i1 < ne01; i1++) {
                स्थिरांक अक्षर * x = (स्थिरांक अक्षर *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      अक्षर * y = (      अक्षर *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                स_नकल(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    साठी (पूर्णांक i3 = 0; i3 < ne13; i3++) {
        साठी (पूर्णांक i2 = ith; i2 < ne12; i2 += nth) {
            साठी (पूर्णांक i1 = 0; i1 < ne11; i1++) {
                स्थिरांक अक्षर * x = (स्थिरांक अक्षर *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      अक्षर * y = (      अक्षर *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                स_नकल(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_i8(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == आकार(int8_t));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    स्थिरांक int8_t * x;

    // TODO: smarter multi-theading
    साठी (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        साठी (पूर्णांक i2 = ith; i2 < ne2; i2 += nth) {
            साठी (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                साठी (पूर्णांक i0 = 0; i0 < ne0; i0++) {
                    जर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (स्थिरांक int8_t *) ((स्थिरांक अक्षर *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (स्थिरांक int8_t *) ((स्थिरांक अक्षर *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((अक्षर *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == आकार(ggml_fp16_t));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    स्थिरांक ggml_fp16_t * x;

    // TODO: smarter multi-theading
    साठी (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        साठी (पूर्णांक i2 = ith; i2 < ne2; i2 += nth) {
            साठी (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                साठी (पूर्णांक i0 = 0; i0 < ne0; i0++) {
                    जर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (स्थिरांक ggml_fp16_t *) ((स्थिरांक अक्षर *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (स्थिरांक ggml_fp16_t *) ((स्थिरांक अक्षर *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((अक्षर *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_concat_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    स्थिरांक दशांश * x;

    // TODO: smarter multi-theading
    साठी (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        साठी (पूर्णांक i2 = ith; i2 < ne2; i2 += nth) {
            साठी (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                साठी (पूर्णांक i0 = 0; i0 < ne0; i0++) {
                    जर (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } अन्यथा {
                        x = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    दशांश * y = (दशांश *)((अक्षर *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_concat(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
        स्थिती GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
        स्थिती GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
ढाँचा <शून्य (*vec_op)(स्थिरांक पूर्णांक, ggml_bf16_t *, स्थिरांक ggml_bf16_t *)>
स्थिर शून्य ggml_compute_forward_unary_bf16_rows(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

स्थिर शून्य ggml_compute_forward_gelu_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

स्थिर शून्य ggml_compute_forward_fill_f32(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक दशांश c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb);

    स्थिरांक स्वतः [ir0, ir1] = get_thread_range(params, dst);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne2*ne1);
        स्थिरांक int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        स्थिरांक int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        दशांश * dst_ptr  = (दशांश *) ((अक्षर *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

स्थिर शून्य ggml_compute_forward_fill_f16(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb);

    स्थिरांक स्वतः [ir0, ir1] = get_thread_range(params, dst);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne2*ne1);
        स्थिरांक int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        स्थिरांक int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

शून्य ggml_compute_forward_fill(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

स्थिर शून्य ggml_compute_forward_tri_f32(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    स्थिरांक ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक स्वतः [ir0, ir1] = get_thread_range(params, src0);

    तार्किक (*bipred)(पूर्णांक, पूर्णांक);

    निवड (ttype) {
        स्थिती GGML_TRI_TYPE_LOWER:      bipred = [](पूर्णांक i, पूर्णांक r) { परत i <  r; }; थांबा;
        स्थिती GGML_TRI_TYPE_LOWER_DIAG: bipred = [](पूर्णांक i, पूर्णांक r) { परत i <= r; }; थांबा;
        स्थिती GGML_TRI_TYPE_UPPER:      bipred = [](पूर्णांक i, पूर्णांक r) { परत i >  r; }; थांबा;
        स्थिती GGML_TRI_TYPE_UPPER_DIAG: bipred = [](पूर्णांक i, पूर्णांक r) { परत i >= r; }; थांबा;
        पूर्वनिर्धारित: GGML_ABORT("invalid tri type");
    }

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne02*ne01);
        स्थिरांक int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिरांक int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        स्थिरांक दशांश * src_ptr = (स्थिरांक दशांश  *) ((स्थिरांक अक्षर *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              दशांश * dst_ptr = (      दशांश  *) ((      अक्षर *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        साठी (पूर्णांक i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

शून्य ggml_compute_forward_tri(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

स्थिर शून्य ggml_compute_forward_gelu_erf_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_erf_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_erf_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu_erf(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

स्थिर शून्य ggml_compute_forward_gelu_quick_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_quick_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_gelu_quick_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_gelu_quick(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

स्थिर शून्य ggml_compute_forward_silu_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(ggml_is_contiguous_rows(src0));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst,  nb)

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक पूर्णांक i3 = ir/(ne02*ne01);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne02*ne01)/ne01;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_silu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

स्थिर शून्य ggml_compute_forward_leaky_relu_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(ggml_is_contiguous_1(src0));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    स्थिरांक पूर्णांक n  = ggml_nrows(src0);
    स्थिरांक पूर्णांक nc = src0->ne[0];

    दशांश negative_slope;
    स_नकल(&negative_slope, dst->op_params, आकार(दशांश));

    प्रतिज्ञा(dst->nb[0]  == आकार(दशांश));
    प्रतिज्ञा(src0->nb[0] == आकार(दशांश));

    साठी (पूर्णांक i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i*( dst->nb[1])),
                (दशांश *) ((अक्षर *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

स्थिर शून्य ggml_compute_forward_leaky_relu_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    प्रतिज्ञा(ggml_is_contiguous_1(src0));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    स्थिरांक पूर्णांक n  = ggml_nrows(src0);
    स्थिरांक पूर्णांक nc = src0->ne[0];

    दशांश negative_slope;
    स_नकल(&negative_slope, dst->op_params, आकार(दशांश));

    प्रतिज्ञा(dst->nb[0]  == आकार(ggml_fp16_t));
    प्रतिज्ञा(src0->nb[0] == आकार(ggml_fp16_t));

    साठी (पूर्णांक i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((अक्षर *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

शून्य ggml_compute_forward_leaky_relu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

स्थिर शून्य ggml_compute_forward_silu_back_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * grad = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    प्रतिज्ञा(ggml_is_contiguous_1(grad));
    प्रतिज्ञा(ggml_is_contiguous_1(src1));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, grad));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i1*( dst->nb[1])),
                (दशांश *) ((अक्षर *) src1->data + i1*(src1->nb[1])),
                (दशांश *) ((अक्षर *) grad->data + i1*(grad->nb[1])));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_silu_back_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * grad = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    प्रतिज्ञा(ggml_is_contiguous_1(grad));
    प्रतिज्ञा(ggml_is_contiguous_1(src1));
    प्रतिज्ञा(ggml_is_contiguous_1(dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, dst));
    प्रतिज्ञा(ggml_are_same_shape(src1, grad));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((अक्षर *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((अक्षर *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((अक्षर *) grad->data + i1*(grad->nb[1])));

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

शून्य ggml_compute_forward_silu_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
ढाँचा <शून्य (*vec_op)(स्थिरांक पूर्णांक, ggml_bf16_t *, स्थिरांक ggml_bf16_t *, स्थिरांक ggml_bf16_t *)>
स्थिर शून्य ggml_compute_forward_glu_bf16_rows(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        स्थिरांक ggml_bf16_t * src0_p = (स्थिरांक ggml_bf16_t *) (src0_d + i1*src0_o);
        स्थिरांक ggml_bf16_t * src1_p = (स्थिरांक ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1]));

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

स्थिर शून्य ggml_compute_forward_reglu_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1*src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (दशांश *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_reglu_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_reglu_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_reglu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

स्थिर शून्य ggml_compute_forward_geglu_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1*src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (दशांश *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

स्थिर शून्य ggml_compute_forward_swiglu_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1*src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (दशांश *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_swiglu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

स्थिर शून्य ggml_compute_forward_swiglu_oai_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिरांक दशांश alpha = ggml_get_op_params_f32(dst, 2);
    स्थिरांक दशांश limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1*src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1*src1_o);
        दशांश * dst_p  = (दशांश *) ((अक्षर *) dst->data + i1*(dst->nb[1]));

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = std::min(src0_p[k], limit);
            स्थिरांक दशांश y = std::clamp(src1_p[k], -limit, limit);
            स्थिरांक दशांश out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = dst_p[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_oai_f16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिरांक दशांश alpha = ggml_get_op_params_f32(dst, 2);
    स्थिरांक दशांश limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1]));

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            स्थिरांक दशांश y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            स्थिरांक दशांश out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_oai(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

स्थिर शून्य ggml_compute_forward_swiglu_clamp_f32(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    अक्षर *              src0_d = (अक्षर *) src0->data;
    अक्षर *              src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार        src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिरांक दशांश   limit   = ggml_get_op_params_f32(dst, 3);

    स्थिरांक पूर्णांक dr  = (nr + nth - 1) / nth;
    स्थिरांक पूर्णांक ir0 = dr * ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1 * src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1 * src1_o);
        दशांश * dst_p  = (दशांश *) ((अक्षर *) dst->data + i1 * (dst->nb[1]));

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश gate = std::min(src0_p[k], limit);
            स्थिरांक दशांश up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = dst_p[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif  // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_clamp_f16(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0   = dst->src[0];
    स्थिरांक ggml_tensor * src1   = dst->src[1];
    अक्षर *              src0_d = (अक्षर *) src0->data;
    अक्षर *              src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार        src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);
    स्थिरांक दशांश   limit   = ggml_get_op_params_f32(dst, 3);

    स्थिरांक पूर्णांक dr  = (nr + nth - 1) / nth;
    स्थिरांक पूर्णांक ir0 = dr * ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((अक्षर *) dst->data + i1 * (dst->nb[1]));

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            स्थिरांक दशांश up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif  // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_swiglu_clamp(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    निवड (dst->src[0]->type) {
        स्थिती GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            थांबा;
        स्थिती GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

स्थिर शून्य ggml_compute_forward_geglu_erf_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1*src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (दशांश *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_erf_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_erf_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu_erf(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

स्थिर शून्य ggml_compute_forward_geglu_quick_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * src0_p = (दशांश *) (src0_d + i1*src0_o);
        दशांश * src1_p = (दशांश *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (दशांश *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक दशांश x = ((दशांश *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            प्रतिज्ञा(!isnan(x));
            प्रतिज्ञा(!isinf(x));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_quick_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    अक्षर * src0_d = (अक्षर *) src0->data;
    अक्षर * src1_d = (अक्षर *) (src1 ? src1->data : src0->data);
    स्थिरांक माप_प्रकार src0_o = src0->nb[1];
    स्थिरांक माप_प्रकार src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    जर (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    स्थिरांक int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        जर (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((अक्षर *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef न_संशोधन
        साठी (पूर्णांक k = 0; k < nc; k++) {
            स्थिरांक ggml_fp16_t x = ((ggml_fp16_t *) ((अक्षर *) dst->data + i1*( dst->nb[1])))[k];
            स्थिरांक दशांश v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            प्रतिज्ञा(!isnan(v));
            प्रतिज्ञा(!isinf(v));
        }
#endif // NDEBUG
    }
}

स्थिर शून्य ggml_compute_forward_geglu_quick_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

स्थिर शून्य ggml_compute_forward_geglu_quick(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

स्थिर शून्य ggml_compute_forward_norm_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    दशांश eps;
    स_नकल(&eps, dst->op_params, आकार(दशांश));

    GGML_ASSERT(eps >= 0.0f);

    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिरांक अक्षर * x = (स्थिरांक अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                अक्षर * y = (अक्षर *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                जर (nb00 == आकार(दशांश) && nb0 == आकार(दशांश)) {
                    स्थिरांक दशांश * xf = (स्थिरांक दशांश *) x;

                    दशांश sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    दशांश mean = sum/ne00;

                    दशांश * yf = (दशांश *) y;
                    दशांश variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#अन्यथा
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    स्थिरांक दशांश scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } अन्यथा {
                    दशांश sum = 0.0;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(स्थिरांक दशांश *) (x + i00*nb00);
                    }
                    स्थिरांक दशांश mean = sum/ne00;

                    दशांश variance = 0.0f;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिरांक दशांश v = *(स्थिरांक दशांश *) (x + i00*nb00) - mean;
                        *(दशांश *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    स्थिरांक दशांश scale = 1.0f/sqrtf(variance + eps);
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(दशांश *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_norm(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
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

ढाँचा <ggml_rms_norm_fuse_op FUSE_OP>
स्थिर शून्य ggml_compute_forward_rms_norm_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    स्थिरांक ggml_tensor * src0 = dst_rms_norm->src[0];
    स्थिरांक ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    जर constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    दशांश eps;
    स_नकल(&eps, dst_rms_norm->op_params, आकार(दशांश));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिरांक दशांश * x = (दशांश *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                स्थिरांक दशांश mean  = sum/ne00;
                स्थिरांक दशांश scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                प्रतिज्ञा(scale > 0.0f);

                दशांश * y = (दशांश *) ((अक्षर *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                जर constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    स्थिरांक int64_t i11 = i01 % ne11;
                    स्थिरांक int64_t i12 = i02 % ne12;
                    स्थिरांक int64_t i13 = i03 % ne13;
                    स्थिरांक दशांश * w = (दशांश *) ((अक्षर *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } अन्यथा {
                    स_नकल(y, x, ne00 * आकार(दशांश));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

शून्य ggml_compute_forward_rms_norm(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
शून्य ggml_compute_forward_rms_norm_mul_fused(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    स्थिरांक ggml_tensor * src0 = dst_rms_norm->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_rms_norm_back_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    स्थिरांक ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));
    GGML_ASSERT(src1->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_द्विमान_OP_LOCALS

    दशांश eps;
    स_नकल(&eps, dst->op_params, आकार(दशांश));

    // TODO: optimize
    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                स्थिरांक int64_t i11 = i01;
                स्थिरांक int64_t i12 = i02;
                स्थिरांक int64_t i13 = i03;

                स्थिरांक दशांश * dz = (दशांश *) ((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                स्थिरांक दशांश * x  = (दशांश *) ((अक्षर *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                स्थिरांक दशांश mean_eps = (दशांश)(sum_xx)/ne00 + eps;
                स्थिरांक दशांश sum_eps  = (दशांश)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                स्थिरांक दशांश rrms     = 1.0f / sqrtf(mean_eps);
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
                दशांश * dx = (दशांश *) ((अक्षर *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                स्थिरांक दशांश scale_x = (दशांश) (-sum_xdz) / sum_eps;
                साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_rms_norm_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

स्थिर शून्य ggml_compute_forward_group_norm_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    दशांश eps;
    स_नकल(&eps, dst->op_params + 1, आकार(दशांश));

    पूर्णांक n_channels = src0->ne[2];
    पूर्णांक n_groups = dst->op_params[0];
    पूर्णांक n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    साठी (पूर्णांक i = ith; i < n_groups; i += nth) {
        पूर्णांक start = i * n_channels_per_group;
        पूर्णांक end = start + n_channels_per_group;
        जर (end > n_channels) {
            end = n_channels;
        }
        पूर्णांक step = end - start;

        साठी (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            साठी (int64_t i02 = start; i02 < end; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिरांक दशांश * x = (दशांश *)((अक्षर *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            स्थिरांक दशांश mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            साठी (int64_t i02 = start; i02 < end; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिरांक दशांश * x = (दशांश *)((अक्षर *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    दशांश * y = (दशांश *)((अक्षर *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        दशांश v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            स्थिरांक दशांश variance = sum2 / (ne00 * ne01 * step);
            स्थिरांक दशांश scale = 1.0f / sqrtf(variance + eps);

            साठी (int64_t i02 = start; i02 < end; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    दशांश * y = (दशांश *)((अक्षर *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

शून्य ggml_compute_forward_group_norm(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

स्थिर शून्य ggml_compute_forward_l2_norm_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    दशांश eps;
    स_नकल(&eps, dst->op_params, आकार(दशांश));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिरांक अक्षर * x = (स्थिरांक अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                    स्थिरांक दशांश xi = *(स्थिरांक दशांश *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                स्थिरांक दशांश scale = 1.0f/fmaxf(sqrtf(sum), eps);

                अक्षर * y = (अक्षर *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                जर (nb00 == आकार(दशांश) && nb0 == आकार(दशांश)) {
                    स_नकल(y, x, ne00 * आकार(दशांश));
                    ggml_vec_scale_f32(ne00, (दशांश *) y, scale);
                } अन्यथा {
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        स्थिरांक दशांश xi = *(स्थिरांक दशांश *) (x + i00*nb00);
                        *(दशांश *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_l2_norm(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

स्थिर शून्य ggml_compute_forward_out_prod_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == आकार(दशांश));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    जर (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (दशांश *)dst->data, 0);
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
    स्थिरांक int64_t nr = ne1*ne2*ne3;

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    स्थिरांक int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    स्थिरांक int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    स्थिरांक int64_t dps2 = ne2 / ne02;
    स्थिरांक int64_t dps3 = ne3 / ne03;

    साठी (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        स्थिरांक int64_t bir1 = MIN(bir + blck_1, ir1);
        साठी (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            स्थिरांक int64_t bne01 = MIN(bi01 + blck_0, ne01);
            साठी (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                स्थिरांक int64_t i3 = ir/(ne2*ne1);
                स्थिरांक int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                स्थिरांक int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                स्थिरांक int64_t i02 = i2 / dps2;
                स्थिरांक int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                स्थिरांक int64_t i12 = i2;
                स्थिरांक int64_t i13 = i3;

#जर GGML_VEC_MAD_UNROLL > 2
                स्थिरांक int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                साठी (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    स्थिरांक int64_t i11 = i01;

                    दशांश * s0 = (दशांश *) ((अक्षर *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    दशांश * s1 = (दशांश *) ((अक्षर *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    दशांश * d  = (दशांश *) ((अक्षर *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                साठी (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    स्थिरांक int64_t i11 = i01;

                    दशांश * s0 = (दशांश *) ((अक्षर *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    दशांश * s1 = (दशांश *) ((अक्षर *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    दशांश * d  = (दशांश *) ((अक्षर *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#अन्यथा
                साठी (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    स्थिरांक int64_t i11 = i01;

                    दशांश * s0 = (दशांश *) ((अक्षर *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    दशांश * s1 = (दशांश *) ((अक्षर *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    दशांश * d  = (दशांश *) ((अक्षर *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_out_prod_q_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक ggml_type type = src0->type;
    ggml_to_float_t स्थिरांक dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    जर (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (दशांश *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    स्थिरांक int64_t nr = ne1*ne2*ne3;

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    दशांश * wdata = (दशांश *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32) * ith;

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        स्थिरांक int64_t i3 = ir/(ne2*ne1);
        स्थिरांक int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        स्थिरांक int64_t i02 = i2;
        स्थिरांक int64_t i03 = i3;

        //const int64_t i10 = i1;
        स्थिरांक int64_t i12 = i2;
        स्थिरांक int64_t i13 = i3;

        साठी (int64_t i01 = 0; i01 < ne01; ++i01) {
            स्थिरांक int64_t i11 = i01;

            दशांश * s0 = (दशांश *) ((अक्षर *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            दशांश * s1 = (दशांश *) ((अक्षर *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            दशांश * d  = (दशांश *) ((अक्षर *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

स्थिर शून्य ggml_compute_forward_out_prod_f16_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == आकार(ggml_fp16_t));
    GGML_ASSERT(nb0  == आकार(दशांश));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    जर (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (दशांश *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    स्थिरांक int64_t nr = ne1*ne2*ne3;
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    दशांश * wdata = (दशांश *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32) * ith;

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i3 = ir/(ne2*ne1);
        स्थिरांक int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        स्थिरांक int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        स्थिरांक int64_t i02 = i2;
        स्थिरांक int64_t i03 = i3;

        स्थिरांक int64_t i12 = i2;
        स्थिरांक int64_t i13 = i3;

        दशांश * d = (दशांश *) ((अक्षर *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        साठी (int64_t i01 = 0; i01 < ne01; ++i01) {
            स्थिरांक int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((अक्षर *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            दशांश * s1 = (दशांश *) ((अक्षर *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

शून्य ggml_compute_forward_out_prod(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

स्थिर शून्य ggml_compute_forward_scale_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    दशांश s; // scale factor
    दशांश b; // bias

    स_नकल(&s, (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&b, (दशांश *) dst->op_params + 1, आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    स्थिरांक माप_प्रकार nb01 = src0->nb[1];

    स्थिरांक माप_प्रकार nb1 = dst->nb[1];

    जर (b == 0.0f) {
        साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
            जर (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                स_नकल((अक्षर *)dst->data + i1*nb1, (अक्षर *)src0->data + i1*nb01, nc * आकार(दशांश));
            }
            ggml_vec_scale_f32(nc, (दशांश *) ((अक्षर *) dst->data + i1*nb1), s);
        }
    } अन्यथा {
        साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (दशांश *) ((अक्षर *) dst->data  + i1*nb1),
                (दशांश *) ((अक्षर *) src0->data + i1*nb1),
                s, b);
        }
    }
}

स्थिर शून्य ggml_compute_forward_scale_bf16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    दशांश s; // scale factor
    दशांश b; // bias

    स_नकल(&s, (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&b, (दशांश *) dst->op_params + 1, आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        स्थिरांक ggml_bf16_t * x = (स्थिरांक ggml_bf16_t *) ((स्थिरांक अक्षर *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((अक्षर *)       dst->data  + i1*dst->nb[1]);
        साठी (पूर्णांक k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

शून्य ggml_compute_forward_scale(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

स्थिर शून्य ggml_compute_forward_set_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    जर (!inplace) {
        जर (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स_नकल(
                ((अक्षर *)  dst->data),
                ((अक्षर *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr = ggml_nrows(src1);
    स्थिरांक पूर्णांक nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb1, src1, nb)

    // src0 and dst as viewed during set
    स्थिरांक माप_प्रकार nb0 = ggml_element_size(src0);

    स्थिरांक पूर्णांक im0 = (ne10 == 0 ? 0 : ne10-1);
    स्थिरांक पूर्णांक im1 = (ne11 == 0 ? 0 : ne11-1);
    स्थिरांक पूर्णांक im2 = (ne12 == 0 ? 0 : ne12-1);
    स्थिरांक पूर्णांक im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne12*ne11);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne12*ne11)/ne11;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (दशांश *) ((अक्षर *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (दशांश *) ((अक्षर *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

स्थिर शून्य ggml_compute_forward_set_i32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    माप_प्रकार nb1     = ((int32_t *) dst->op_params)[0];
    माप_प्रकार nb2     = ((int32_t *) dst->op_params)[1];
    माप_प्रकार nb3     = ((int32_t *) dst->op_params)[2];
    माप_प्रकार offset  = ((int32_t *) dst->op_params)[3];
    तार्किक   inplace = (तार्किक) ((int32_t *) dst->op_params)[4];

    जर (!inplace) {
        जर (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            स_नकल(
                ((अक्षर *)  dst->data),
                ((अक्षर *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr = ggml_nrows(src1);
    स्थिरांक पूर्णांक nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb1, src1, nb)

    // src0 and dst as viewed during set
    स्थिरांक माप_प्रकार nb0 = ggml_element_size(src0);

    स्थिरांक पूर्णांक im0 = (ne10 == 0 ? 0 : ne10-1);
    स्थिरांक पूर्णांक im1 = (ne11 == 0 ? 0 : ne11-1);
    स्थिरांक पूर्णांक im2 = (ne12 == 0 ? 0 : ne12-1);
    स्थिरांक पूर्णांक im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == आकार(int32_t));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        स्थिरांक पूर्णांक i3 = ir/(ne12*ne11);
        स्थिरांक पूर्णांक i2 = (ir - i3*ne12*ne11)/ne11;
        स्थिरांक पूर्णांक i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((अक्षर *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((अक्षर *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

शून्य ggml_compute_forward_set(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q8_1:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

शून्य ggml_compute_forward_cpy(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

शून्य ggml_compute_forward_cont(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

स्थिर शून्य ggml_compute_forward_get_rows_q(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int64_t nc = ne00;
    स्थिरांक int64_t nr = ggml_nelements(src1);

    स्थिरांक ggml_type type = src0->type;
    ggml_to_float_t स्थिरांक dequantize_row_q = ggml_get_type_traits(type)->to_float;

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == ggml_type_size(type));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t i = ir0; i < ir1; ++i) {
        स्थिरांक int64_t i12 = i/(ne11*ne10);
        स्थिरांक int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिरांक int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिरांक int64_t i01 = *(int32_t *) ((अक्षर *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (स्थिरांक शून्य *) ((अक्षर *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (दशांश *) ((अक्षर *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_f16(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int64_t nc = ne00;
    स्थिरांक int64_t nr = ggml_nelements(src1);

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == आकार(ggml_fp16_t));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t i = ir0; i < ir1; ++i) {
        स्थिरांक int64_t i12 = i/(ne11*ne10);
        स्थिरांक int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिरांक int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिरांक int64_t i01 = *(int32_t *) ((अक्षर *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (स्थिरांक ggml_fp16_t*) ((अक्षर *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (दशांश *) ((अक्षर *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_bf16(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int64_t nc = ne00;
    स्थिरांक int64_t nr = ggml_nelements(src1);

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == आकार(ggml_bf16_t));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t i = ir0; i < ir1; ++i) {
        स्थिरांक int64_t i12 = i/(ne11*ne10);
        स्थिरांक int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिरांक int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिरांक int64_t i01 = *(int32_t *) ((अक्षर *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (स्थिरांक ggml_bf16_t *) ((अक्षर *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (दशांश *) ((अक्षर *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int64_t nc = ne00;
    स्थिरांक int64_t nr = ggml_nelements(src1);

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne02 == ne11);
    प्रतिज्ञा(nb00 == आकार(दशांश));
    प्रतिज्ञा(ggml_nrows(dst) == nr);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t i = ir0; i < ir1; ++i) {
        स्थिरांक int64_t i12 = i/(ne11*ne10);
        स्थिरांक int64_t i11 = (i - i12*ne11*ne10)/ne10;
        स्थिरांक int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        स्थिरांक int64_t i01 = *(int32_t *) ((अक्षर *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (दशांश *) ((अक्षर *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (दशांश *) ((अक्षर *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

शून्य ggml_compute_forward_get_rows(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q8_1:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
        स्थिती GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
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
स्थिर शून्य ggml_compute_forward_set_rows_impl(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक int64_t nc = ne00;
    स्थिरांक int64_t nr = ne01;

    प्रतिज्ञा(ne0  == nc);
    प्रतिज्ञा(ne2  == ne02);
    प्रतिज्ञा(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    प्रतिज्ञा(ne02 % ne11 == 0);
    प्रतिज्ञा(ne03 % ne12 == 0);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = std::min(ir0 + dr, nr);

    स्थिरांक माप_प्रकार rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t स्थिरांक from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    साठी (int64_t i03 = 0; i03 < ne03; ++i03) {
        साठी (int64_t i02 = 0; i02 < ne02; ++i02) {
            साठी (int64_t i = ir0; i < ir1; ++i) {
                स्थिरांक int64_t i12 = i03%ne12;
                स्थिरांक int64_t i11 = i02%ne11;
                स्थिरांक int64_t i10 = i;

                स्थिरांक int64_t i1 = *(idx_t *) ((अक्षर *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                जर constexpr (std::is_same_v<src_t, दशांश>) {
                    from_float(
                            (स्थिरांक दशांश *) ((अक्षर *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((अक्षर *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } अन्यथा जर constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    जर (dst->type == GGML_TYPE_F16) {
                        स_नकल(
                                            ((अक्षर *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((अक्षर *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } अन्यथा {
                        दशांश * wdata = (दशांश *) params->wdata + (nc + CACHE_ओळ_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (स्थिरांक ggml_fp16_t *) ((अक्षर *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((अक्षर *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } अन्यथा {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_set_rows(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                जर (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<दशांश, int64_t>(params, dst);
                } अन्यथा जर (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<दशांश, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                जर (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } अन्यथा जर (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } अन्यथा {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

स्थिर शून्य ggml_compute_forward_get_rows_back_f32_f16(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    जर (params->ith != 0) {
        परत;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    स_रखो(dst->data, 0, ggml_nbytes(dst));

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == आकार(ggml_fp16_t));

    साठी (पूर्णांक i = 0; i < nr; ++i) {
        स्थिरांक पूर्णांक r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        साठी (पूर्णांक j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((अक्षर *) src0->data + i*src0->nb[1]))[j];
            ((दशांश *) ((अक्षर *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

स्थिर शून्य ggml_compute_forward_get_rows_back_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    जर (params->ith != 0) {
        परत;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    स_रखो(dst->data, 0, ggml_nbytes(dst));

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == आकार(दशांश));

    साठी (पूर्णांक i = 0; i < nr; ++i) {
        स्थिरांक पूर्णांक r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (दशांश *) ((अक्षर *)  dst->data + r*dst->nb[1]),
                (दशांश *) ((अक्षर *)  dst->data + r*dst->nb[1]),
                (दशांश *) ((अक्षर *) src0->data + i*src0->nb[1]));
    }
}

शून्य ggml_compute_forward_get_rows_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    जर (params->ith != 0) {
        परत;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == आकार(दशांश));
    GGML_ASSERT(nb0  == आकार(दशांश));

    साठी (पूर्णांक i3 = 0; i3 < ne3; i3++) {
        साठी (पूर्णांक i2 = 0; i2 < ne2; i2++) {
            साठी (पूर्णांक i1 = 0; i1 < ne1; i1++) {
                दशांश * d = (दशांश *)((अक्षर *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                दशांश * s = (दशांश *)((अक्षर *) src0->data + i3*nb03 + i2*nb02);
                साठी (पूर्णांक i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                साठी (पूर्णांक i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_diag(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

स्थिर शून्य ggml_compute_forward_diag_mask_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst,
        स्थिरांक दशांश value) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक  n_past  = ((int32_t *) dst->op_params)[0];
    स्थिरांक तार्किक inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    जर (!inplace) {
        जर (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            स_नकल(
                ((अक्षर *)  dst->data),
                ((अक्षर *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    स्थिरांक पूर्णांक n  = ggml_nrows(src0);
    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = src0->ne[1];
    स्थिरांक पूर्णांक nz = n/nr;

    GGML_ASSERT( dst->nb[0] == आकार(दशांश));
    GGML_ASSERT(src0->nb[0] == आकार(दशांश));

    साठी (पूर्णांक k = 0; k < nz; k++) {
        साठी (पूर्णांक j = ith; j < nr; j += nth) {
            साठी (पूर्णांक i = n_past; i < nc; i++) {
                जर (i > n_past + j) {
                    *(दशांश *)((अक्षर *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

शून्य ggml_compute_forward_diag_mask_inf(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -अनन्त);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

शून्य ggml_compute_forward_diag_mask_zero(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

स्थिर शून्य ggml_compute_forward_soft_max_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    स्थिरांक ggml_tensor * src2 = dst->src[2];

    प्रतिज्ञा(ggml_is_contiguous(dst));
    प्रतिज्ञा(ggml_are_same_shape(src0, dst));

    दशांश scale    = 1.0f;
    दशांश max_bias = 0.0f;

    स_नकल(&scale,    (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max_bias, (दशांश *) dst->op_params + 1, आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक int64_t nb11 = src1 ? src1->nb[1] : 1;
    स्थिरांक int64_t nb12 = src1 ? src1->nb[2] : 1;
    स्थिरांक int64_t nb13 = src1 ? src1->nb[3] : 1;

    स्थिरांक int64_t ne12 = src1 ? src1->ne[2] : 1;
    स्थिरांक int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    स्थिरांक uint32_t n_head      = ne02;
    स्थिरांक uint32_t n_head_log2 = 1u << (uint32_t) न्यूनमान(log2(n_head));

    स्थिरांक दशांश m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    स्थिरांक दशांश m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    दशांश * wp = (दशांश *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

    स्थिरांक तार्किक use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    स्थिरांक दशांश * sk = src2 ? (दशांश *)((अक्षर *) src2->data) : nullptr;

    साठी (int64_t i03 = 0; i03 < ne03; i03++) {
        साठी (int64_t i02 = 0; i02 < ne02; i02++) {
            साठी (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                स्थिरांक int64_t i11 = i01;
                स्थिरांक int64_t i12 = i02%ne12;
                स्थिरांक int64_t i13 = i03%ne13;

                // ALiBi
                स्थिरांक uint32_t h = i02; // head
                स्थिरांक दशांश slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                दशांश * sp = (दशांश *)((अक्षर *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                दशांश * dp = (दशांश *)((अक्षर *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((अक्षर *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : रिक्त;
                दशांश       * mp_f32 = src1 ? (दशांश       *)((अक्षर *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : रिक्त;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                जर (mp_f32) {
                    जर (use_f16) {
                        साठी (पूर्णांक i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } अन्यथा {
                        साठी (पूर्णांक i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef न_संशोधन
                साठी (पूर्णांक i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    प्रतिज्ञा(!isnan(wp[i]));
                }
#endif // NDEBUG

                दशांश max = -अनन्त;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                जर (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                प्रतिज्ञा(sum > 0.0);

                जर (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef न_संशोधन
                साठी (पूर्णांक i = 0; i < ne00; ++i) {
                    प्रतिज्ञा(!isnan(dp[i]));
                    प्रतिज्ञा(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

शून्य ggml_compute_forward_soft_max(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

स्थिर शून्य ggml_compute_forward_soft_max_ext_back_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    दशांश scale    = 1.0f;
    दशांश max_bias = 0.0f;

    स_नकल(&scale,    (स्थिरांक दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max_bias, (स्थिरांक दशांश *) dst->op_params + 1, आकार(दशांश));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc = src0->ne[0];
    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश *dy = (दशांश *)((अक्षर *) src0->data + i1*src0->nb[1]);
        दशांश *y  = (दशांश *)((अक्षर *) src1->data + i1*src1->nb[1]);
        दशांश *dx = (दशांश *)((अक्षर *) dst->data  + i1*dst->nb[1]);

#ifndef न_संशोधन
        साठी (पूर्णांक i = 0; i < nc; ++i) {
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
        दशांश dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        साठी (पूर्णांक i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ifndef न_संशोधन
        साठी (पूर्णांक i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(dx[i]));
            प्रतिज्ञा(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

शून्य ggml_compute_forward_soft_max_ext_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

स्थिर शून्य ggml_compute_forward_clamp_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    दशांश min;
    दशांश max;
    स_नकल(&min, (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max, (दशांश *) dst->op_params + 1, आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक n  = ggml_nrows(src0);
    स्थिरांक पूर्णांक nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == आकार(दशांश));
    GGML_ASSERT(nb00 == आकार(दशांश));

    साठी (पूर्णांक j = ith; j < n; j += nth) {
        स्थिरांक int64_t i1 = j % ne01;
        स्थिरांक int64_t i2 = (j / ne01) % ne02;
        स्थिरांक int64_t i3 = j / (ne01*ne02);

        दशांश * dst_ptr  = (दशांश *) ((अक्षर *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        दशांश * src0_ptr = (दशांश *) ((अक्षर *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        साठी (पूर्णांक i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

स्थिर शून्य ggml_compute_forward_clamp_f16(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    दशांश min;
    दशांश max;
    स_नकल(&min, (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max, (दशांश *) dst->op_params + 1, आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक n  = ggml_nrows(src0);
    स्थिरांक पूर्णांक nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == आकार(ggml_fp16_t));
    GGML_ASSERT(nb00 == आकार(ggml_fp16_t));

    साठी (पूर्णांक j = ith; j < n; j += nth) {
        स्थिरांक int64_t i1 = j % ne01;
        स्थिरांक int64_t i2 = (j / ne01) % ne02;
        स्थिरांक int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((अक्षर *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((अक्षर *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        साठी (पूर्णांक i = 0; i < nc; i++) {
            दशांश v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

शून्य ggml_compute_forward_clamp(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_BF16:
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q2_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q8_1:
        स्थिती GGML_TYPE_MXFP4:
        स्थिती GGML_TYPE_NVFP4:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_TQ1_0:
        स्थिती GGML_TYPE_TQ2_0:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ4_NL:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ2_S:
        स्थिती GGML_TYPE_Q8_K:
        स्थिती GGML_TYPE_I8:
        स्थिती GGML_TYPE_I16:
        स्थिती GGML_TYPE_I32:
        स्थिती GGML_TYPE_I64:
        स्थिती GGML_TYPE_F64:
        स्थिती GGML_TYPE_मोजणी:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

स्थिर दशांश rope_yarn_ramp(स्थिरांक दशांश खालचा, स्थिरांक दशांश वरचा, स्थिरांक पूर्णांक i0) {
    स्थिरांक दशांश y = (i0 / 2 - खालचा) / MAX(0.001f, वरचा - खालचा);
    परत 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
स्थिर शून्य rope_yarn(
    दशांश theta_extrap, दशांश freq_scale, दशांश corr_dims[2], int64_t i0, दशांश ext_factor, दशांश mscale,
    दशांश * cos_theta, दशांश * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    दशांश theta_interp = freq_scale * theta_extrap;
    दशांश theta = theta_interp;
    जर (ext_factor != 0.0f) {
        दशांश ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

स्थिर शून्य ggml_rope_cache_init(
     दशांश theta_base, दशांश freq_scale, स्थिरांक दशांश * freq_factors, दशांश corr_dims[2], int64_t ne0, दशांश ext_factor, दशांश mscale,
     दशांश * cache, दशांश sin_sign, दशांश theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    दशांश theta = theta_base;
    साठी (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        स्थिरांक दशांश ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

स्थिर शून्य ggml_mrope_cache_init(
     दशांश theta_base_t, दशांश theta_base_h, दशांश theta_base_w, दशांश theta_base_e, पूर्णांक sections[4], तार्किक is_imrope, तार्किक indep_sects,
     दशांश freq_scale, स्थिरांक दशांश * freq_factors, दशांश corr_dims[2], int64_t ne0, दशांश ext_factor, दशांश mscale,
     दशांश * cache, दशांश sin_sign, दशांश theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    दशांश theta_t = theta_base_t;
    दशांश theta_h = theta_base_h;
    दशांश theta_w = theta_base_w;
    दशांश theta_e = theta_base_e;  // extra position id for vision encoder
    पूर्णांक sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    पूर्णांक sec_w = sections[1] + sections[0];
    पूर्णांक sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    साठी (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        स्थिरांक दशांश ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        पूर्णांक sector = (i0 / 2) % sect_dims;
        जर (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            जर (sector == 0) {
                theta_t = theta_base_t;
            }
            अन्यथा जर (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            अन्यथा जर (sector == sec_w) {
                theta_w = theta_base_w;
            }
            अन्यथा जर (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        दशांश theta = theta_t;
        जर (is_imrope) { // qwen3vl apply interleaved mrope
            जर (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } अन्यथा जर (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } अन्यथा जर (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } अन्यथा {
                theta = theta_e;
            }
        } अन्यथा {
            जर (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            अन्यथा जर (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            अन्यथा जर (sector >= sec_w + sections[2]) {
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
स्थिर शून्य rotate_pairs(स्थिरांक int64_t n, स्थिरांक int64_t n_offset, स्थिरांक दशांश * cache, स्थिरांक T * src_data, T * dst_data, स्थिरांक पूर्णांक scale = 2) {
  साठी (int64_t i0 = 0; i0 < n; i0 += 2) {
    स्थिरांक int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    स्थिरांक दशांश cos_theta = cache[i0 + 0];
    स्थिरांक दशांश sin_theta = cache[i0 + 1];

    स्थिरांक T * स्थिरांक src = src_data + ic;
    T * dst             = dst_data + ic;

    स्थिरांक दशांश x0 = type_conversion_table<T>::to_f32(src[0]);
    स्थिरांक दशांश x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

ढाँचा<typename T> //float or ggml_fp16_t
स्थिर शून्य ggml_compute_forward_rope_flt(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst,
        स्थिरांक तार्किक forward) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    स्थिरांक ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    दशांश freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    पूर्णांक sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    स्थिरांक पूर्णांक n_dims     = ((int32_t *) dst->op_params)[1];
    स्थिरांक पूर्णांक mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    स्थिरांक पूर्णांक n_ctx_orig = ((int32_t *) dst->op_params)[4];

    स_नकल(&freq_base,   (int32_t *) dst->op_params +  5, आकार(दशांश));
    स_नकल(&freq_scale,  (int32_t *) dst->op_params +  6, आकार(दशांश));
    स_नकल(&ext_factor,  (int32_t *) dst->op_params +  7, आकार(दशांश));
    स_नकल(&attn_factor, (int32_t *) dst->op_params +  8, आकार(दशांश));
    स_नकल(&beta_fast,   (int32_t *) dst->op_params +  9, आकार(दशांश));
    स_नकल(&beta_slow,   (int32_t *) dst->op_params + 10, आकार(दशांश));
    स_नकल(&sections,    (int32_t *) dst->op_params + 11, आकार(पूर्णांक)*4);

    स्थिरांक पूर्णांक n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == आकार(T));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    पूर्णांक ir = 0;

    स्थिरांक दशांश theta_scale = powf(freq_base, -2.0f/n_dims);

    दशांश corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    स्थिरांक तार्किक is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    स्थिरांक तार्किक mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    स्थिरांक तार्किक is_vision = mode == GGML_ROPE_TYPE_VISION;

    जर (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    जर (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    स्थिरांक दशांश * freq_factors = रिक्त;
    जर (src2 != रिक्त) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (स्थिरांक दशांश *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    स्थिरांक दशांश sin_sign = forward ? 1.0f : -1.0f;

    स्थिरांक int32_t * pos = (स्थिरांक int32_t *) src1->data;

    int64_t last_i2 = -1;

    साठी (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        साठी (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            साठी (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                जर (ir++ < ir0) सुरूठेवा; // skip rows mapped to other threads
                जर (ir   > ir1) थांबा;

                दशांश * cache = (दशांश *) params->wdata + (ne0 + CACHE_ओळ_SIZE_F32)*ith;
                जर (last_i2 != i2) {
                    जर (!mrope_used) {
                        स्थिरांक int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    अन्यथा {
                        स्थिरांक int64_t p_t = pos[i2];
                        स्थिरांक int64_t p_h = pos[i2 + ne2];
                        स्थिरांक int64_t p_w = pos[i2 + ne2 * 2];
                        स्थिरांक int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((अक्षर *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                निवड (mode) {
                    स्थिती GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        थांबा;
                    स्थिती GGML_ROPE_TYPE_NEOX:
                    स्थिती GGML_ROPE_TYPE_MROPE:
                    स्थिती GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        थांबा;
                    स्थिती GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        थांबा;
                    पूर्वनिर्धारित:
                        GGML_ABORT("rope type not supported");
                }

                जर (!is_vision) {
                    // fill the remain channels with data from src tensor
                    साठी (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        जर (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            सुरूठेवा;
                        }
                        स्थिरांक T * स्थिरांक src = (T *)((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((अक्षर *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

शून्य ggml_compute_forward_rope(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, सत्य);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<दशांश>(params, dst, सत्य);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

शून्य ggml_compute_forward_rope_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, असत्य);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<दशांश>(params, dst, असत्य);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

स्थिर शून्य ggml_compute_forward_conv_transpose_1d_f16_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == आकार(ggml_fp16_t));
    GGML_ASSERT(nb10 == आकार(दशांश));

    जर (ith == 0) {
        स_रखो(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * स्थिरांक wdata = (ggml_fp16_t *) params->wdata + 0;

            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिरांक ggml_fp16_t * स्थिरांक src = (ggml_fp16_t *)((अक्षर *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * स्थिरांक wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            साठी (int64_t i11 = 0; i11 < ne11; i11++) {
                स्थिरांक दशांश * स्थिरांक src = (दशांश *)((अक्षर *) src1->data + i11*nb11);
                साठी (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        स_रखो(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t*)(dst->op_params))[0];

    // total rows in dst
    स्थिरांक पूर्णांक nr = ne1;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * स्थिरांक wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * स्थिरांक wdata_src = wdata + nk;

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * dst_data = (दशांश *)((अक्षर *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        साठी (पूर्णांक i10 = 0; i10 < ne10; i10++) {
            स्थिरांक पूर्णांक i1n = i10*ne11;
            साठी (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                दशांश v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

स्थिर शून्य ggml_compute_forward_conv_transpose_1d_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == आकार(दशांश));
    GGML_ASSERT(nb10 == आकार(दशांश));

    जर (ith == 0) {
        स_रखो(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            दशांश * स्थिरांक wdata = (दशांश *) params->wdata + 0;

            साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                    स्थिरांक दशांश * स्थिरांक src = (दशांश *)((अक्षर *) src0->data + i02*nb02 + i01*nb01);
                    दशांश * dst_data = wdata + i01*ne00*ne02;
                    साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            दशांश * स्थिरांक wdata = (दशांश *) params->wdata + nk;
            दशांश * dst_data = wdata;

            साठी (int64_t i11 = 0; i11 < ne11; i11++) {
                स्थिरांक दशांश * स्थिरांक src = (दशांश *)((अक्षर *) src1->data + i11*nb11);
                साठी (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        स_रखो(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t*)(dst->op_params))[0];

    // total rows in dst
    स्थिरांक पूर्णांक nr = ne1;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    दशांश * स्थिरांक wdata     = (दशांश *) params->wdata + 0;
    दशांश * स्थिरांक wdata_src = wdata + nk;

    साठी (पूर्णांक i1 = ir0; i1 < ir1; i1++) {
        दशांश * dst_data = (दशांश *)((अक्षर *) dst->data + i1*nb1);
        दशांश * wdata_kernel = wdata + i1*ne02*ne00;
        साठी (पूर्णांक i10 = 0; i10 < ne10; i10++) {
            स्थिरांक पूर्णांक i1n = i10*ne11;
            साठी (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                दशांश v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_transpose_1d(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t s1 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t p0 = ((स्थिरांक int32_t *)(dst->op_params))[2];
    स्थिरांक int32_t p1 = ((स्थिरांक int32_t *)(dst->op_params))[3];
    स्थिरांक int32_t d0 = ((स्थिरांक int32_t *)(dst->op_params))[4];
    स्थिरांक int32_t d1 = ((स्थिरांक int32_t *)(dst->op_params))[5];
    स्थिरांक तार्किक is_2D = ((स्थिरांक int32_t *)(dst->op_params))[6] == 1;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t N  = is_2D ? ne13 : ne12;
    स्थिरांक int64_t IC = is_2D ? ne12 : ne11;
    स्थिरांक int64_t IH = is_2D ? ne11 : 1;
    स्थिरांक int64_t IW = ne10;

    स्थिरांक int64_t KH = is_2D ? ne01 : 1;
    स्थिरांक int64_t KW = ne00;

    स्थिरांक int64_t OH = is_2D ? ne2 : 1;
    स्थिरांक int64_t OW = ne1;

    पूर्णांक ofs0 = is_2D ? nb13 : nb12;
    पूर्णांक ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == आकार(दशांश));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        दशांश * स्थिरांक wdata = (दशांश *) dst->data;

        साठी (int64_t in = 0; in < N; in++) {
            साठी (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                साठी (int64_t iow = 0; iow < OW; iow++) {
                    साठी (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        दशांश * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        स्थिरांक दशांश * स्थिरांक src_data = (दशांश *)((अक्षर *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        साठी (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            साठी (int64_t ikw = 0; ikw < KW; ikw++) {
                                स्थिरांक int64_t iiw = iow*s0 + ikw*d0 - p0;
                                स्थिरांक int64_t iih = ioh*s1 + ikh*d1 - p1;

                                जर (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t s1 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t p0 = ((स्थिरांक int32_t *)(dst->op_params))[2];
    स्थिरांक int32_t p1 = ((स्थिरांक int32_t *)(dst->op_params))[3];
    स्थिरांक int32_t d0 = ((स्थिरांक int32_t *)(dst->op_params))[4];
    स्थिरांक int32_t d1 = ((स्थिरांक int32_t *)(dst->op_params))[5];
    स्थिरांक तार्किक is_2D = ((स्थिरांक int32_t *)(dst->op_params))[6] == 1;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t N  = is_2D ? ne13 : ne12;
    स्थिरांक int64_t IC = is_2D ? ne12 : ne11;
    स्थिरांक int64_t IH = is_2D ? ne11 : 1;
    स्थिरांक int64_t IW = ne10;

    स्थिरांक int64_t KH = is_2D ? ne01 : 1;
    स्थिरांक int64_t KW = ne00;

    स्थिरांक int64_t OH = is_2D ? ne2 : 1;
    स्थिरांक int64_t OW = ne1;

    पूर्णांक ofs0 = is_2D ? nb13 : nb12;
    पूर्णांक ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * स्थिरांक wdata = (ggml_fp16_t *) dst->data;

        साठी (int64_t in = 0; in < N; in++) {
            साठी (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                साठी (int64_t iow = 0; iow < OW; iow++) {
                    साठी (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        स्थिरांक दशांश * स्थिरांक src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (स्थिरांक दशांश *)((स्थिरांक अक्षर *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        स्थिरांक ggml_fp16_t * स्थिरांक src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (स्थिरांक ggml_fp16_t *)((स्थिरांक अक्षर *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        साठी (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            साठी (int64_t ikw = 0; ikw < KW; ikw++) {
                                स्थिरांक int64_t iiw = iow*s0 + ikw*d0 - p0;
                                स्थिरांक int64_t iih = ioh*s1 + ikh*d1 - p1;

                                जर (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } अन्यथा {
                                    जर (src_data_f32 != nullptr) {
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {
    निवड (dst->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

शून्य ggml_compute_forward_im2col_back_f32(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    स्थिरांक ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t s1 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t p0 = ((स्थिरांक int32_t *)(dst->op_params))[2];
    स्थिरांक int32_t p1 = ((स्थिरांक int32_t *)(dst->op_params))[3];
    स्थिरांक int32_t d0 = ((स्थिरांक int32_t *)(dst->op_params))[4];
    स्थिरांक int32_t d1 = ((स्थिरांक int32_t *)(dst->op_params))[5];
    स्थिरांक तार्किक is_2D = ((स्थिरांक int32_t *)(dst->op_params))[6] == 1;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t N  = is_2D ? ne3 : ne2;
    स्थिरांक int64_t IC = is_2D ? ne2 : ne1;
    स्थिरांक int64_t IH = is_2D ? ne1 : 1;
    स्थिरांक int64_t IW = ne0;

    स्थिरांक int64_t KH = is_2D ? ne11 : 1;
    स्थिरांक int64_t KW = ne10;

    स्थिरांक int64_t OH = is_2D ? ne02 : 1;
    स्थिरांक int64_t OW = ne01;

    पूर्णांक ofs0 = is_2D ? nb3 : nb2;
    पूर्णांक ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == आकार(दशांश));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        दशांश * स्थिरांक wdata = (दशांश *) dst->data;

        साठी (int64_t in = 0; in < N; in++) {
            साठी (int64_t iic = ith; iic < IC; iic += nth) {
                साठी (int64_t iih = 0; iih < IH; iih++) {
                    साठी (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        दशांश grad = 0.0f;
                        साठी (int64_t ikh = 0; ikh < KH; ikh++) {
                            साठी (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                स्थिरांक int64_t tmpw = (iiw + p0 - ikw*d0);
                                जर (tmpw % s0 != 0) {
                                    सुरूठेवा;
                                }
                                स्थिरांक int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                जर (is_2D) {
                                    स्थिरांक int64_t tmph = iih + p1 - ikh*d1;

                                    जर (tmph % s1 != 0) {
                                        सुरूठेवा;
                                    }

                                    ioh = tmph / s1;
                                } अन्यथा {
                                    ioh = 0;
                                }

                                जर (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    सुरूठेवा;
                                }

                                स्थिरांक दशांश * स्थिरांक grad_in = (स्थिरांक दशांश *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        दशांश * dst_data = (दशांश *)((अक्षर *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t s1 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t s2 = ((स्थिरांक int32_t *)(dst->op_params))[2];
    स्थिरांक int32_t p0 = ((स्थिरांक int32_t *)(dst->op_params))[3];
    स्थिरांक int32_t p1 = ((स्थिरांक int32_t *)(dst->op_params))[4];
    स्थिरांक int32_t p2 = ((स्थिरांक int32_t *)(dst->op_params))[5];
    स्थिरांक int32_t d0 = ((स्थिरांक int32_t *)(dst->op_params))[6];
    स्थिरांक int32_t d1 = ((स्थिरांक int32_t *)(dst->op_params))[7];
    स्थिरांक int32_t d2 = ((स्थिरांक int32_t *)(dst->op_params))[8];
    स्थिरांक int32_t IC = ((स्थिरांक int32_t *)(dst->op_params))[9];


    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t N  = ne13 / IC;
    स्थिरांक int64_t ID = ne12;
    स्थिरांक int64_t IH = ne11;
    स्थिरांक int64_t IW = ne10;

    स्थिरांक int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    स्थिरांक int64_t KD = ne02;
    स्थिरांक int64_t KH = ne01;
    स्थिरांक int64_t KW = ne00;

    स्थिरांक int64_t OD = ne3 / N;
    स्थिरांक int64_t OH = ne2;
    स्थिरांक int64_t OW = ne1;
    स्थिरांक int64_t OH_OW = OH*OW;
    स्थिरांक int64_t KD_KH_KW = KD*KH*KW;
    स्थिरांक int64_t KH_KW = KH*KW;
    स्थिरांक int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == आकार(दशांश));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * स्थिरांक wdata = (ggml_fp16_t *) dst->data;

        साठी (int64_t in = 0; in < N; in++) {
            साठी (int64_t iod = 0; iod < OD; iod++) {
                साठी (int64_t ioh = 0; ioh < OH; ioh++) {
                    साठी (int64_t iow = 0; iow < OW; iow++) {
                        साठी (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            स्थिरांक दशांश * स्थिरांक src_data = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            साठी (int64_t ikd = 0; ikd < KD; ikd++) {
                                साठी (int64_t ikh = 0; ikh < KH; ikh++) {
                                    साठी (int64_t ikw = 0; ikw < KW; ikw++) {
                                        स्थिरांक int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        स्थिरांक int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        स्थिरांक int64_t iid = iod*s2 + ikd*d2 - p2;

                                        जर (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            स्थिरांक दशांश * स्थिरांक s = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS;

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t s1 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t s2 = ((स्थिरांक int32_t *)(dst->op_params))[2];
    स्थिरांक int32_t p0 = ((स्थिरांक int32_t *)(dst->op_params))[3];
    स्थिरांक int32_t p1 = ((स्थिरांक int32_t *)(dst->op_params))[4];
    स्थिरांक int32_t p2 = ((स्थिरांक int32_t *)(dst->op_params))[5];
    स्थिरांक int32_t d0 = ((स्थिरांक int32_t *)(dst->op_params))[6];
    स्थिरांक int32_t d1 = ((स्थिरांक int32_t *)(dst->op_params))[7];
    स्थिरांक int32_t d2 = ((स्थिरांक int32_t *)(dst->op_params))[8];
    स्थिरांक int32_t IC = ((स्थिरांक int32_t *)(dst->op_params))[9];


    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t N  = ne13 / IC;
    स्थिरांक int64_t ID = ne12;
    स्थिरांक int64_t IH = ne11;
    स्थिरांक int64_t IW = ne10;

    स्थिरांक int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    स्थिरांक int64_t KD = ne02;
    स्थिरांक int64_t KH = ne01;
    स्थिरांक int64_t KW = ne00;

    स्थिरांक int64_t OD = ne3 / N;
    स्थिरांक int64_t OH = ne2;
    स्थिरांक int64_t OW = ne1;

    स्थिरांक int64_t OH_OW = OH*OW;
    स्थिरांक int64_t KD_KH_KW = KD*KH*KW;
    स्थिरांक int64_t KH_KW = KH*KW;
    स्थिरांक int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == आकार(दशांश));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        दशांश * स्थिरांक wdata = (दशांश *) dst->data;

        साठी (int64_t in = 0; in < N; in++) {
            साठी (int64_t iod = 0; iod < OD; iod++) {
                साठी (int64_t ioh = 0; ioh < OH; ioh++) {
                    साठी (int64_t iow = 0; iow < OW; iow++) {
                        साठी (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            दशांश * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            स्थिरांक दशांश * स्थिरांक src_data = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            साठी (int64_t ikd = 0; ikd < KD; ikd++) {
                                साठी (int64_t ikh = 0; ikh < KH; ikh++) {
                                    साठी (int64_t ikw = 0; ikw < KW; ikw++) {
                                        स्थिरांक int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        स्थिरांक int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        स्थिरांक int64_t iid = iod*s2 + ikd*d2 - p2;

                                        जर (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } अन्यथा {
                                            स्थिरांक दशांश * स्थिरांक s = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {
    निवड (dst->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_call_mul_mat(ggml_type type, स्थिरांक ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              शून्य * a, शून्य * b, दशांश * c) {
    स्थिरांक ggml_type_traits * traits = ggml_get_type_traits(type);
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
    dst.nb[0] = आकार(दशांश);
    dst.nb[1] = n * आकार(दशांश);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

स्थिर ओळीत int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    परत (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

ढाँचा <typename elem_t>
स्थिर शून्य ggml_compute_forward_col2im_1d_impl(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    स्थिरांक int32_t s0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t OC = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t p0 = ((स्थिरांक int32_t *)(dst->op_params))[2];

    स्थिरांक int64_t K_OC = src->ne[0];
    स्थिरांक int64_t T_in = src->ne[1];
    स्थिरांक int64_t K    = K_OC / OC;
    स्थिरांक int64_t T_out = dst->ne[0];

    स्थिरांक elem_t * col_data = (स्थिरांक elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    स्थिरांक int64_t dr = (T_out + nth - 1) / nth;
    स्थिरांक int64_t it0 = dr * ith;
    स्थिरांक int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    साठी (int64_t oc = 0; oc < OC; oc++) {
        साठी (int64_t t_out = it0; t_out < it1; t_out++) {
            स्थिरांक int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            जर (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            जर (t_in_max >= T_in) t_in_max = T_in - 1;

            दशांश sum = 0.0f;
            साठी (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                जर (k >= 0 && k < K) {
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    निवड (dst->src[0]->type) {
        स्थिती GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<दशांश>      (params, dst); थांबा;
        स्थिती GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); थांबा;
        स्थिती GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); थांबा;
        पूर्वनिर्धारित: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


स्थिर शून्य ggml_compute_forward_conv_2d_impl(स्थिरांक ggml_compute_params * params,
                                              स्थिरांक ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              स्थिरांक ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    स्थिरांक ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    स्थिरांक int32_t stride_x   = dst->op_params[0];
    स्थिरांक int32_t stride_y   = dst->op_params[1];
    स्थिरांक int32_t pad_x      = dst->op_params[2];
    स्थिरांक int32_t pad_y      = dst->op_params[3];
    स्थिरांक int32_t dilation_x = dst->op_params[4];
    स्थिरांक int32_t dilation_y = dst->op_params[5];

    स्थिरांक int64_t c_in  = src->ne[2];
    स्थिरांक int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    स्थिरांक int64_t src_w = src->ne[0];
    स्थिरांक int64_t src_h = src->ne[1];
    स्थिरांक int64_t knl_w = kernel->ne[0];
    स्थिरांक int64_t knl_h = kernel->ne[1];
    स्थिरांक int64_t dst_w = dst->ne[0];
    स्थिरांक int64_t dst_h = dst->ne[1];

    स्थिरांक दशांश * src_data = (दशांश *) src->data;
    शून्य  * knl_data       = kernel->data;
    दशांश * dst_data       = (दशांश *) dst->data;

    स्थिरांक int64_t knl_n           = knl_w * knl_h * c_in;
    स्थिरांक int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    स्थिरांक int64_t space_per_patch   = knl_n * traits->type_size + c_out * आकार(दशांश);
    स्थिरांक int64_t batch_size        = params->wsize / space_per_patch;
    स्थिरांक int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    स्थिरांक int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    शून्य * tmp = params->wdata;

    साठी (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        स्थिरांक int64_t patch_start_batch = batch_i * patches_per_batch;
        स्थिरांक int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        स्थिरांक int64_t patch_n           = patch_end_batch - patch_start_batch;

        स्थिरांक int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        स्थिरांक int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        स्थिरांक int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        साठी (int64_t p = patch_start; p < patch_end; ++p) {
            स्थिरांक int64_t  batch_n     =  p / (dst_w * dst_h);
            स्थिरांक int64_t  src_x       = (p / dst_w) % dst_h;
            स्थिरांक int64_t  src_y       =  p % dst_w;

            स्थिरांक दशांश * src_base = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_data + batch_n * src->nb[3]);
            अक्षर *        dst_row  = (अक्षर *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            साठी (int64_t ic = 0; ic < c_in; ++ic) {
                साठी (int64_t ky = 0; ky < knl_h; ++ky) {
                    साठी (int64_t kx = 0; kx < knl_w; ++kx) {
                        स्थिरांक int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        स्थिरांक int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        दशांश src_val;
                        जर (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } अन्यथा {
                            स्थिरांक दशांश * src_ptr = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        अक्षर * element_ptr = dst_row + dst_idx * traits->type_size;
                        जर (kernel_type == GGML_TYPE_F32) {
                            *(दशांश *) element_ptr = src_val;
                        } अन्यथा जर (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        दशांश * gemm_output = (दशांश *) ((अक्षर *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (दशांश*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        स्थिरांक int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        स्थिरांक int64_t permute_start = params->ith * permute_per_thread;
        स्थिरांक int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        साठी (int64_t i = permute_start; i < permute_end; ++i) {
            स्थिरांक int64_t p       = patch_start_batch + i;
            स्थिरांक int64_t batch_n = p / (dst_w * dst_h);
            स्थिरांक int64_t dst_y   = (p / dst_w) % dst_h;
            स्थिरांक int64_t dst_x   = p % dst_w;

            साठी (int64_t oc = 0; oc < c_out; ++oc) {
                स्थिरांक दशांश value = gemm_output[i * c_out + oc];
                दशांश * dst_ptr = (दशांश *)((अक्षर *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_2d(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

स्थिर शून्य ggml_compute_forward_conv_3d_impl(स्थिरांक ggml_compute_params * params,
                                              स्थिरांक ggml_tensor *         kernel,
                                              स्थिरांक ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    स्थिरांक ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    स्थिरांक int32_t s0 = dst->op_params[0];
    स्थिरांक int32_t s1 = dst->op_params[1];
    स्थिरांक int32_t s2 = dst->op_params[2];
    स्थिरांक int32_t p0 = dst->op_params[3];
    स्थिरांक int32_t p1 = dst->op_params[4];
    स्थिरांक int32_t p2 = dst->op_params[5];
    स्थिरांक int32_t d0 = dst->op_params[6];
    स्थिरांक int32_t d1 = dst->op_params[7];
    स्थिरांक int32_t d2 = dst->op_params[8];
    स्थिरांक int32_t c  = dst->op_params[9];
    स्थिरांक int32_t n  = dst->op_params[10];
    स्थिरांक int32_t oc = dst->op_params[11];

    स्थिरांक int64_t src_w = src->ne[0];
    स्थिरांक int64_t src_h = src->ne[1];
    स्थिरांक int64_t src_d = src->ne[2];
    स्थिरांक int64_t knl_w = kernel->ne[0];
    स्थिरांक int64_t knl_h = kernel->ne[1];
    स्थिरांक int64_t knl_d = kernel->ne[2];
    स्थिरांक int64_t dst_w = dst->ne[0];
    स्थिरांक int64_t dst_h = dst->ne[1];
    स्थिरांक int64_t dst_d = dst->ne[2];

    स्थिरांक दशांश * src_data = (दशांश *) src->data;
    शून्य  * knl_data       = kernel->data;
    दशांश * dst_data       = (दशांश *) dst->data;

    स्थिरांक int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    स्थिरांक int64_t knl_n_total       = knl_n_per_channel * c;
    स्थिरांक int64_t patch_total       = n * dst_w * dst_h * dst_d;

    स्थिरांक int64_t space_per_patch   = knl_n_total * traits->type_size + oc * आकार(दशांश);
    स्थिरांक int64_t batch_size        = params->wsize / space_per_patch;
    स्थिरांक int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    स्थिरांक int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    शून्य * tmp = params->wdata;

    साठी (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        स्थिरांक int64_t patch_start_batch = batch_i * patches_per_batch;
        स्थिरांक int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        स्थिरांक int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        स्थिरांक int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        स्थिरांक int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        स्थिरांक int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        साठी (int64_t p = patch_start; p < patch_end; ++p) {
            स्थिरांक int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            स्थिरांक int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            स्थिरांक int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            स्थिरांक int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            स्थिरांक int64_t dst_y      = p_in_depth / dst_w;
            स्थिरांक int64_t dst_x      = p_in_depth % dst_w;

            अक्षर * dst_row = (अक्षर *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            साठी (int64_t ic = 0; ic < c; ++ic) {
                साठी (int64_t kz = 0; kz < knl_d; ++kz) {
                    साठी (int64_t ky = 0; ky < knl_h; ++ky) {
                        साठी (int64_t kx = 0; kx < knl_w; ++kx) {
                            स्थिरांक int64_t sz = dst_z * s2 + kz * d2 - p2;
                            स्थिरांक int64_t sy = dst_y * s1 + ky * d1 - p1;
                            स्थिरांक int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            दशांश src_val;
                            जर (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } अन्यथा {
                                स्थिरांक int64_t cn_idx = batch_idx * c + ic;
                                स्थिरांक दशांश * src_ptr = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            अक्षर * element_ptr = dst_row + dst_idx * traits->type_size;
                            जर (kernel_type == GGML_TYPE_F32) {
                                *(दशांश *)element_ptr = src_val;
                            } अन्यथा जर (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        दशांश * gemm_output = (दशांश *) ((अक्षर *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        स्थिरांक int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        स्थिरांक int64_t permute_start = params->ith * permute_per_thread;
        स्थिरांक int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        साठी (int64_t i = permute_start; i < permute_end; ++i) {
            स्थिरांक int64_t p = patch_start_batch + i;
            स्थिरांक int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            स्थिरांक int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            स्थिरांक int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            स्थिरांक int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            स्थिरांक int64_t dst_y      = p_in_depth / dst_w;
            स्थिरांक int64_t dst_x      = p_in_depth % dst_w;

            साठी (int64_t ioc = 0; ioc < oc; ++ioc) {
                स्थिरांक दशांश value = gemm_output[i * oc + ioc];
                स्थिरांक int64_t ocn_idx = batch_idx * oc + ioc;
                दशांश * dst_ptr = (दशांश *)((अक्षर *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

शून्य ggml_compute_forward_conv_3d(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

ढाँचा <typename kernel_t>
स्थिर शून्य ggml_compute_forward_conv_transpose_2d_impl(
    स्थिरांक ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == आकार(दशांश));

    जर (ith == 0) {
        स_रखो(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * स्थिरांक wdata = (kernel_t *) params->wdata + 0;

            साठी (int64_t i03 = 0; i03 < ne03; i03++) {
                साठी (int64_t i02 = 0; i02 < ne02; i02++) {
                    स्थिरांक kernel_t * स्थिरांक src = (kernel_t *)((अक्षर *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    साठी (int64_t i01 = 0; i01 < ne01; i01++) {
                        साठी (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * स्थिरांक wdata = (kernel_t *) params->wdata + nk;
            साठी (पूर्णांक i13 = 0; i13 < ne13; i13++) {
                kernel_t * स्थिरांक wdata_b = wdata + i13*ne10*ne11*ne12;
                साठी (पूर्णांक i12 = 0; i12 < ne12; i12++) {
                    साठी (पूर्णांक i11 = 0; i11 < ne11; i11++) {
                        स्थिरांक दशांश * स्थिरांक src = (दशांश *)((अक्षर *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        साठी (पूर्णांक i10 = 0; i10 < ne10; i10++) {
                            जर constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } अन्यथा {
                                dst_data[i10*ne12 + i12] = src[i10];
                            }
                        }
                    }
                }
            }
        }

        स_रखो(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    स्थिरांक int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    स्थिरांक पूर्णांक np = ne2;

    // patches per thread
    स्थिरांक पूर्णांक dp = (np + nth - 1)/nth;

    // patch range for this thread
    स्थिरांक पूर्णांक ip0 = dp*ith;
    स्थिरांक पूर्णांक ip1 = MIN(ip0 + dp, np);

    kernel_t * स्थिरांक wdata = (kernel_t *) params->wdata + 0;
    kernel_t * स्थिरांक wdata_src = wdata + nk;

    साठी (पूर्णांक i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        साठी (पूर्णांक i3 = 0; i3 < ne3; i3++) { // batch
            दशांश * dst_data = (दशांश *)((अक्षर *) dst->data + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            साठी (पूर्णांक i11 = 0; i11 < ne11; i11++) {
                साठी (पूर्णांक i10 = 0; i10 < ne10; i10++) {
                    स्थिरांक पूर्णांक i1n = i11*ne10*ne12 + i10*ne12;
                    साठी (पूर्णांक i01 = 0; i01 < ne01; i01++) {
                        साठी (पूर्णांक i00 = 0; i00 < ne00; i00++) {
                            दशांश v = 0;
                            जर constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
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
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } थांबा;
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<दशांश>(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
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
    पूर्णांक stride_x;
    पूर्णांक stride_y;
    पूर्णांक pad_x;
    पूर्णांक pad_y;
    पूर्णांक dilation_x;
    पूर्णांक dilation_y;
};

स्थिर ओळीत दशांश ggml_conv_2d_dw_knl_f32(स्थिरांक अक्षर * data, int64_t i, ggml_type type) {
    जर (type == GGML_TYPE_F16) {
        परत GGML_FP16_TO_FP32(((स्थिरांक ggml_fp16_t *)data)[i]);
    }
    परत ((स्थिरांक दशांश *)data)[i];
}

स्थिर शून्य ggml_compute_forward_conv_2d_dw_cwhn(
        स्थिरांक ggml_compute_params * params,
        स्थिरांक ggml_tensor * src,
        स्थिरांक ggml_tensor * kernel,
        ggml_tensor * dst,
        स्थिरांक ggml_conv_2d_dw_params & p) {

    स्थिरांक int64_t c = p.channels;
    स्थिरांक अक्षर * knl_data = (स्थिरांक अक्षर *)kernel->data;
    स्थिरांक ggml_type knl_type = kernel->type;

    स्थिरांक int64_t rows_total = p.dst_h * p.batch;
    स्थिरांक int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    स्थिरांक int64_t row_start = params->ith * rows_per_thread;
    स्थिरांक int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    जर (knl_type == GGML_TYPE_F32) {
    #जर defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #अन्यथा
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#अन्यथा
    स्थिरांक int64_t c_pkg_end = 0;
#endif

    साठी (int64_t row = row_start; row < row_end; ++row) {
        स्थिरांक int64_t dst_y = row % p.dst_h;
        स्थिरांक दशांश * src_data = (स्थिरांक दशांश *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        साठी (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            दशांश * dst_data = (दशांश *)dst->data + (row * p.dst_w + dst_x) * c;
            स्थिरांक int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            स्थिरांक int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            साठी (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                साठी (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    स्थिरांक int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    जर (src_y < 0 || src_y >= p.src_h) {
                        सुरूठेवा;
                    }
                    साठी (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        स्थिरांक int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        जर (src_x < 0 || src_x >= p.src_w) {
                            सुरूठेवा;
                        }
                        स्थिरांक दशांश * kp = (स्थिरांक दशांश *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            साठी (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                दशांश sum = 0.0f;
                साठी (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    स्थिरांक int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    जर (src_y < 0 || src_y >= p.src_h) {
                        सुरूठेवा;
                    }
                    साठी (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        स्थिरांक int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        जर (src_x < 0 || src_x >= p.src_w) {
                            सुरूठेवा;
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
        स्थिरांक ggml_compute_params * params,
        स्थिरांक ggml_tensor * src,
        स्थिरांक ggml_tensor * kernel,
        ggml_tensor * dst,
        स्थिरांक ggml_conv_2d_dw_params & p) {

    स्थिरांक int64_t n = p.channels * p.batch;
    स्थिरांक int64_t per_thread = (n + params->nth - 1) / params->nth;
    स्थिरांक int64_t start = params->ith * per_thread;
    स्थिरांक int64_t end = MIN(start + per_thread, n);
    स्थिरांक अक्षर * knl_base = (स्थिरांक अक्षर *)kernel->data;
    स्थिरांक ggml_type knl_type = kernel->type;

    साठी (int64_t i = start; i < end; ++i) {
        स्थिरांक int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        स्थिरांक दशांश * src_data = (स्थिरांक दशांश *)src->data + i * p.src_w * p.src_h;
        दशांश * dst_data = (दशांश *)dst->data + i * p.dst_w * p.dst_h;

        साठी (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            साठी (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                दशांश sum = 0.0f;
                साठी (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    स्थिरांक int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    जर (src_y < 0 || src_y >= p.src_h) {
                        सुरूठेवा;
                    }
                    साठी (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        स्थिरांक int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        जर (src_x < 0 || src_x >= p.src_w) {
                            सुरूठेवा;
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * kernel = dst->src[0];
    स्थिरांक ggml_tensor * src = dst->src[1];
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

    जर (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } अन्यथा जर (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } अन्यथा {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
स्थिर शून्य ggml_compute_forward_pool_1d_ksp(
        स्थिरांक ggml_compute_params * params,
        स्थिरांक ggml_op_pool op,
        स्थिरांक पूर्णांक k,
        स्थिरांक पूर्णांक s,
        स्थिरांक पूर्णांक p,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src = dst->src[0];

    प्रतिज्ञा(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    जर (params->ith != 0) {
        परत;
    }

    स्थिरांक int64_t IW = src->ne[0];
    स्थिरांक int64_t OW = dst->ne[0];

    स्थिरांक int64_t nr = ggml_nrows(src);

    साठी (int64_t ir = 0; ir < nr; ++ir) {
        स्थिरांक अक्षर * srow_bytes =            (स्थिरांक अक्षर *) src->data + ir * src->nb[1];
        दशांश      * drow       = (दशांश *) ((      अक्षर *) dst->data + ir * dst->nb[1]);

        साठी (int64_t ow = 0; ow < OW; ++ow) {
            दशांश res = 0;
            निवड (op) {
                स्थिती GGML_OP_POOL_AVG: res = 0.0f;     थांबा;
                स्थिती GGML_OP_POOL_MAX: res = -भग्न_उच्च; थांबा;
                स्थिती GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
            }

            पूर्णांक count = 0;
            स्थिरांक पूर्णांक base = (पूर्णांक) ow * s - p;

            साठी (पूर्णांक ki = 0; ki < k; ++ki) {
                स्थिरांक पूर्णांक j = base + ki;
                जर (j < 0 || j >= (पूर्णांक) IW) {
                    सुरूठेवा;
                }

                दशांश v;
                जर (src->type == GGML_TYPE_F32) {
                    v = ((स्थिरांक दशांश *) srow_bytes)[j];
                } अन्यथा {
                    v = GGML_CPU_FP16_TO_FP32(((स्थिरांक ggml_fp16_t *) srow_bytes)[j]);
                }

                निवड (op) {
                    स्थिती GGML_OP_POOL_AVG: res += v;                थांबा;
                    स्थिती GGML_OP_POOL_MAX: res =  std::max(v, res); थांबा;
                    स्थिती GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
                }

                ++count;
            }

            निवड (op) {
                स्थिती GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; थांबा;
                स्थिती GGML_OP_POOL_MAX:                                           थांबा;
                स्थिती GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

शून्य ggml_compute_forward_pool_1d(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक int32_t * opts = (स्थिरांक int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिरांक पूर्णांक k0 = opts[1];
    स्थिरांक पूर्णांक s0 = opts[2];
    स्थिरांक पूर्णांक p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

शून्य ggml_compute_forward_pool_2d(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src = dst->src[0];

    प्रतिज्ञा(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    जर (params->ith != 0) {
        परत;
    }

    स्थिरांक int32_t * opts = (स्थिरांक int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिरांक पूर्णांक k0 = opts[1];
    स्थिरांक पूर्णांक k1 = opts[2];
    स्थिरांक पूर्णांक s0 = opts[3];
    स्थिरांक पूर्णांक s1 = opts[4];
    स्थिरांक पूर्णांक p0 = opts[5];
    स्थिरांक पूर्णांक p1 = opts[6];
    स्थिरांक अक्षर * cdata = (स्थिरांक अक्षर*)src->data;
    स्थिरांक अक्षर * स्थिरांक data_end = cdata + ggml_nbytes(src);

    स्थिरांक int64_t px = dst->ne[0];
    स्थिरांक int64_t py = dst->ne[1];
    स्थिरांक int64_t pa = px * py;

    दशांश * dplane = (दशांश *)dst->data;

    स्थिरांक पूर्णांक ka = k0 * k1;
    स्थिरांक पूर्णांक offset0 = -p0;
    स्थिरांक पूर्णांक offset1 = -p1;

    पर्यंत (cdata < data_end) {
        साठी (पूर्णांक oy = 0; oy < py; ++oy) {
            दशांश * स्थिरांक drow = dplane + oy * px;
            दशांश * स्थिरांक out  = drow;

            साठी (पूर्णांक ox = 0; ox < px; ++ox) {
                दशांश res = 0;
                निवड (op) {
                    स्थिती GGML_OP_POOL_AVG: res = 0;        थांबा;
                    स्थिती GGML_OP_POOL_MAX: res = -भग्न_उच्च; थांबा;
                    स्थिती GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
                }

                स्थिरांक पूर्णांक ix = offset0 + ox * s0;
                स्थिरांक पूर्णांक iy = offset1 + oy * s1;

                साठी (पूर्णांक ky = 0; ky < k1; ++ky) {
                    जर (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        सुरूठेवा;
                    }

                    स्थिरांक शून्य * srow = (स्थिरांक शून्य *)(cdata + src->nb[1] * (iy + ky));
                    साठी (पूर्णांक kx = 0; kx < k0; ++kx) {
                        पूर्णांक j = ix + kx;
                        जर (j < 0 || j >= src->ne[0]) {
                            सुरूठेवा;
                        }

                        स्थिरांक दशांश srow_j = (src->type == GGML_TYPE_F32) ? ((स्थिरांक दशांश*)srow)[j] : GGML_CPU_FP16_TO_FP32(((स्थिरांक ggml_fp16_t*)srow)[j]);
                        निवड (op) {
                            स्थिती GGML_OP_POOL_AVG: res += srow_j;                थांबा;
                            स्थिती GGML_OP_POOL_MAX: res =  std::max(srow_j, res); थांबा;
                            स्थिती GGML_OP_POOL_मोजणी:               GGML_ABORT("fatal error");
                        }
                    }
                }
                निवड (op) {
                    स्थिती GGML_OP_POOL_AVG:           res /= ka; थांबा;
                    स्थिती GGML_OP_POOL_MAX:                      थांबा;
                    स्थिती GGML_OP_POOL_मोजणी: GGML_ABORT("fatal error");
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src  = dst->src[0];
    स्थिरांक ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    प्रतिज्ञा(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    जर (params->ith != 0) {
        परत;
    }

    स्थिरांक int32_t * opts = (स्थिरांक int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    स्थिरांक पूर्णांक k0 = opts[1];
    स्थिरांक पूर्णांक k1 = opts[2];
    स्थिरांक पूर्णांक s0 = opts[3];
    स्थिरांक पूर्णांक s1 = opts[4];
    स्थिरांक पूर्णांक p0 = opts[5];
    स्थिरांक पूर्णांक p1 = opts[6];

    अक्षर       * cdata  = (अक्षर       *) dst->data;
    स्थिरांक अक्षर * cdataf = (स्थिरांक अक्षर *) dstf->data;
    स्थिरांक अक्षर * स्थिरांक data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    स_रखो(cdata, 0, ggml_nbytes(dst));

    स्थिरांक int64_t px = src->ne[0];
    स्थिरांक int64_t py = src->ne[1];
    स्थिरांक int64_t pa = px * py;

    स्थिरांक दशांश * splane = (स्थिरांक दशांश *) src->data;

    स्थिरांक पूर्णांक ka = k0 * k1;
    स्थिरांक पूर्णांक offset0 = -p0;
    स्थिरांक पूर्णांक offset1 = -p1;

    पर्यंत (cdata < data_end) {
        साठी (पूर्णांक oy = 0; oy < py; ++oy) {
            स्थिरांक दशांश * स्थिरांक srow = splane + oy * px;
            साठी (पूर्णांक ox = 0; ox < px; ++ox) {
                स्थिरांक दशांश grad0 = srow[ox];

                स्थिरांक पूर्णांक ix = offset0 + ox * s0;
                स्थिरांक पूर्णांक iy = offset1 + oy * s1;

                जर (op == GGML_OP_POOL_MAX) {
                    दशांश maxval = -भग्न_उच्च;
                    पूर्णांक kxmax = -1;
                    पूर्णांक kymax = -1;

                    साठी (पूर्णांक ky = 0; ky < k1; ++ky) {
                        जर (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            सुरूठेवा;
                        }
                        स्थिरांक शून्य * drowf = (स्थिरांक शून्य *)(cdataf + dst->nb[1] * (iy + ky));
                        साठी (पूर्णांक kx = 0; kx < k0; ++kx) {
                            पूर्णांक j = ix + kx;
                            जर (j < 0 || j >= dst->ne[0]) {
                                सुरूठेवा;
                            }

                            स्थिरांक दशांश val = dst->type == GGML_TYPE_F32 ?
                                ((स्थिरांक दशांश *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((स्थिरांक ggml_fp16_t *) drowf)[j]);
                            जर (val <= maxval) {
                                सुरूठेवा;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    जर (kxmax == -1 || kymax == -1) {
                        सुरूठेवा;
                    }

                    शून्य * drow = (शून्य *)(cdata + dst->nb[1] * (iy + kymax));
                    स्थिरांक पूर्णांक j = ix + kxmax;
                    जर (dst->type == GGML_TYPE_F32) {
                        ((दशांश *) drow)[j] += grad0;
                    } अन्यथा {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((स्थिरांक ggml_fp16_t *) drow)[j]));
                    }
                } अन्यथा जर (op == GGML_OP_POOL_AVG) {
                    स्थिरांक दशांश grad = grad0 / ka;

                    साठी (पूर्णांक ky = 0; ky < k1; ++ky) {
                        जर (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            सुरूठेवा;
                        }
                        शून्य * drow = (शून्य *)(cdata + dst->nb[1] * (iy + ky));
                        साठी (पूर्णांक kx = 0; kx < k0; ++kx) {
                            पूर्णांक j = ix + kx;
                            जर (j < 0 || j >= dst->ne[0]) {
                                सुरूठेवा;
                            }

                            जर (dst->type == GGML_TYPE_F32) {
                                ((दशांश *) drow)[j] += grad;
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
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    दशांश sf0 = (दशांश)ne0/src0->ne[0];
    दशांश sf1 = (दशांश)ne1/src0->ne[1];
    दशांश sf2 = (दशांश)ne2/src0->ne[2];
    दशांश sf3 = (दशांश)ne3/src0->ne[3];
    दशांश pixel_offset = 0.5f;

    स्थिरांक int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    स्थिरांक ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    जर (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (दशांश)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (दशांश)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    जर (mode == GGML_SCALE_MODE_NEAREST) {
        साठी (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिरांक int64_t i03 = i3 / sf3;
            साठी (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिरांक int64_t i02 = i2 / sf2;
                साठी (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिरांक int64_t i01 = i1 / sf1;
                    साठी (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिरांक int64_t i00 = i0 / sf0;

                        स्थिरांक दशांश * x = (दशांश *)((अक्षर *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              दशांश * y = (दशांश *)((अक्षर *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } अन्यथा जर (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        स्वतः triangle_filter = [](दशांश x) -> दशांश {
            परत std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        स्थिरांक दशांश support1  = std::max(1.0f, 1.0f / sf1);
        स्थिरांक दशांश invscale1 = 1.0f / support1;
        स्थिरांक दशांश support0  = std::max(1.0f, 1.0f / sf0);
        स्थिरांक दशांश invscale0 = 1.0f / support0;

        साठी (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिरांक int64_t i03 = i3 / sf3;
            साठी (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिरांक int64_t i02 = i2 / sf2;
                साठी (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिरांक दशांश y = ((दशांश) i1 + pixel_offset) / sf1;
                    साठी (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिरांक दशांश x = ((दशांश) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        स्थिरांक int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        स्थिरांक int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        स्थिरांक int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        स्थिरांक int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        दशांश val = 0.0f;
                        दशांश total_weight = 0.0f;

                        साठी (int64_t sy = y_min; sy < y_max; sy++) {
                            स्थिरांक दशांश weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            साठी (int64_t sx = x_min; sx < x_max; sx++) {
                                स्थिरांक दशांश weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                स्थिरांक दशांश weight = weight_x * weight_y;

                                जर (weight <= 0.0f) {
                                    सुरूठेवा;
                                }

                                स्थिरांक दशांश pixel = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        जर (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        दशांश * dst_ptr = (दशांश *)((अक्षर *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } अन्यथा जर (mode == GGML_SCALE_MODE_BILINEAR) {
        साठी (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिरांक int64_t i03 = i3 / sf3;
            साठी (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिरांक int64_t i02 = i2 / sf2;
                साठी (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिरांक दशांश y = ((दशांश)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    दशांश dy = y - (दशांश)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    साठी (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिरांक दशांश x = ((दशांश)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        दशांश dx = x - (दशांश)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        स्थिरांक दशांश a = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        स्थिरांक दशांश b = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        स्थिरांक दशांश c = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        स्थिरांक दशांश d = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        स्थिरांक दशांश val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        दशांश * y_dst = (दशांश *)((अक्षर *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } अन्यथा जर (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        स्थिरांक दशांश a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        स्वतः weight1 = [a](दशांश x) { परत ((a + 2) * x - (a + 3)) * x * x + 1; };
        स्वतः weight2 = [a](दशांश x) { परत ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        स्वतः bicubic = [=](दशांश p0, दशांश p1, दशांश p2, दशांश p3, दशांश x) {
            स्थिरांक दशांश w0 = weight2(x + 1);
            स्थिरांक दशांश w1 = weight1(x + 0);
            स्थिरांक दशांश w2 = weight1(1 - x);
            स्थिरांक दशांश w3 = weight2(2 - x);
            परत p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        साठी (int64_t i3 = 0; i3 < ne3; i3++) {
            स्थिरांक int64_t i03 = i3 / sf3;
            साठी (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                स्थिरांक int64_t i02 = i2 / sf2;
                साठी (int64_t i1 = 0; i1 < ne1; i1++) {
                    स्थिरांक दशांश y = ((दशांश)i1 + pixel_offset) / sf1 - pixel_offset;
                    स्थिरांक int64_t y0 = (int64_t)floorf(y);
                    स्थिरांक दशांश dy = y - (दशांश)y0;

                    साठी (int64_t i0 = 0; i0 < ne0; i0++) {
                        स्थिरांक दशांश x = ((दशांश)i0 + pixel_offset) / sf0 - pixel_offset;
                        स्थिरांक int64_t x0 = (int64_t)floorf(x);
                        स्थिरांक दशांश dx = x - (दशांश)x0;

                        स्वतः p = [=](int64_t x_off, int64_t y_off) -> दशांश {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            परत *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        स्थिरांक दशांश val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        दशांश * y_dst = (दशांश *)((अक्षर *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
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
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

ढाँचा<तार्किक circular_t>
स्थिर शून्य ggml_compute_forward_pad_f32(
    स्थिरांक ggml_compute_params * params,
          ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    प्रतिज्ञा(dst->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    दशांश * dst_ptr = (दशांश *) dst->data;
    स्थिरांक int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    स्थिरांक int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    स्थिरांक int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    स्थिरांक int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    स्थिरांक int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    स्थिरांक int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    स्थिरांक int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    स्थिरांक int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    स्थिरांक int64_t nrows = ne1 * ne2 * ne3;
    स्थिरांक तार्किक src_row_contig = src0->nb[0] == आकार(दशांश);

    साठी (int64_t row = ith; row < nrows; row += nth) {
        स्थिरांक int64_t i1 = row % ne1;
        स्थिरांक int64_t i2 = (row / ne1) % ne2;
        स्थिरांक int64_t i3 = row / (ne1 * ne2);

        दशांश * dst_row = dst_ptr + row * ne0;

        जर constexpr (circular_t) {
            // circular padding wraps around on a torus
            स्थिरांक int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            स्थिरांक int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            स्थिरांक int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                स्थिरांक int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } अन्यथा {
            स्थिरांक तार्किक interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            जर (!interior) {
                स_रखो(dst_row, 0, ne0 * आकार(दशांश));
                सुरूठेवा;
            }

            स्थिरांक अक्षर * src_row = (स्थिरांक अक्षर *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            जर (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                स_रखो(dst_row, 0, lp0 * आकार(दशांश));
                स_नकल(dst_row + lp0, src_row, ne00 * आकार(दशांश));
                स_रखो(dst_row + lp0 + ne00, 0, rp0 * आकार(दशांश));
            } अन्यथा {
                साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                    जर (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(स्थिरांक दशांश *)(src_row + (i0 - lp0)*nb00);
                    } अन्यथा {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


शून्य ggml_compute_forward_pad(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक तार्किक circular = (तार्किक) ggml_get_op_params_i32(dst, 8);
    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                जर (circular) {
                    ggml_compute_forward_pad_f32<सत्य>(params, dst);
                } अन्यथा {
                    ggml_compute_forward_pad_f32<असत्य>(params, dst);
                }
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

शून्य ggml_compute_forward_pad_reflect_1d(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int32_t * opts = (स्थिरांक int32_t *) dst->op_params;
    स्थिरांक पूर्णांक p0 = opts[0];
    स्थिरांक पूर्णांक p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    साठी (int64_t i3 = 0; i3 < ne3; i3++) {
        साठी (int64_t i2 = 0; i2 < ne2; i2++) {
            साठी (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                दशांश * left  = (दशांश *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                दशांश * right = (दशांश *) ((अक्षर *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (दशांश *) ((अक्षर *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                साठी (पूर्णांक i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                साठी (पूर्णांक i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

स्थिर int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    जर (i < 0) {
        परत i + ne;
    } अन्यथा जर (i >= ne) {
        परत i - ne;
    }
    परत i;
}

स्थिर शून्य ggml_compute_forward_roll_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक दशांश * src_data = (स्थिरांक दशांश *) src0->data;
    दशांश * dst_data = (दशांश *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक पूर्णांक s0 = ggml_get_op_params_i32(dst, 0);
    स्थिरांक पूर्णांक s1 = ggml_get_op_params_i32(dst, 1);
    स्थिरांक पूर्णांक s2 = ggml_get_op_params_i32(dst, 2);
    स्थिरांक पूर्णांक s3 = ggml_get_op_params_i32(dst, 3);

    स्थिरांक int64_t total = ne1 * ne2 * ne3;
    स्थिरांक int64_t per_thread = (total + params->nth) / params->nth;
    स्थिरांक int64_t start = params->ith * per_thread;
    स्थिरांक int64_t end   = std::min(start + per_thread, total);

    साठी (int64_t i = start; i < end; ++i) {
        स्थिरांक int64_t i1 = i % ne1;
        स्थिरांक int64_t i2 = (i / ne1) % ne2;
        स्थिरांक int64_t i3 = i / (ne2 * ne1);
        दशांश * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / आकार(दशांश);

        स्थिरांक int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        स्थिरांक int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        स्थिरांक int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        स्थिरांक दशांश * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / आकार(दशांश);

        स्थिरांक int64_t s = ggml_wrap_index(-s0, ne00);
        स्थिरांक int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

शून्य ggml_compute_forward_roll(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

स्थिर शून्य ggml_compute_forward_arange_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक दशांश start = ggml_get_op_params_f32(dst, 0);
    स्थिरांक दशांश stop  = ggml_get_op_params_f32(dst, 1);
    स्थिरांक दशांश step  = ggml_get_op_params_f32(dst, 2);

    स्थिरांक int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    साठी (int64_t i = ith; i < steps; i+= nth) {
        दशांश value = start + step * i;
        ((दशांश *)dst->data)[i] = value;
    }
}

शून्य ggml_compute_forward_arange(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {
    निवड (dst->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_timestep_embedding_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक पूर्णांक dim = ggml_get_op_params_i32(dst, 0);
    स्थिरांक पूर्णांक max_period = ggml_get_op_params_i32(dst, 1);

    पूर्णांक half = dim / 2;

    साठी (int64_t i = 0; i < ne00; i++) {
        दशांश * embed_data = (दशांश *)((अक्षर *)  dst->data +  i*nb1);
        साठी (int64_t j = ith; j < half; j += nth) {
            दशांश timestep = ((दशांश *)src0->data)[i];
            दशांश freq = (दशांश)expf(-logf(max_period) * j / half);
            दशांश arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        जर (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

शून्य ggml_compute_forward_timestep_embedding(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

ढाँचा<गणना ggml_sort_order order>
संरचना cmp_argsort {
    स्थिरांक दशांश * data;
    तार्किक चालक()(int32_t a, int32_t b) स्थिरांक {
        जर constexpr (order == GGML_SORT_ORDER_ASC) {
            परत data[a] < data[b];
        } अन्यथा {
            परत data[a] > data[b];
        }
    }
};

स्थिर शून्य ggml_compute_forward_argsort_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    साठी (int64_t i = ith; i < nr; i += nth) {
        स्थिरांक दशांश * src_data = (दशांश *)((अक्षर *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((अक्षर *) dst->data + i*nb1);

        साठी (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        निवड (order) {
            स्थिती GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                थांबा;

            स्थिती GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                थांबा;

            पूर्वनिर्धारित:
                GGML_ABORT("invalid sort order");
        }
    }
}

शून्य ggml_compute_forward_argsort(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

संरचना cmp_शीर्ष_k {
    स्थिरांक दशांश * data;
    तार्किक चालक()(int32_t a, int32_t b) स्थिरांक {
        परत data[a] > data[b];
    }
};

स्थिर शून्य ggml_compute_forward_शीर्ष_k_f32(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == आकार(दशांश));

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t nr = ggml_nrows(src0);

    स्थिरांक पूर्णांक शीर्ष_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_ओळ_SIZE_F32) * ith;

    साठी (int64_t i = ith; i < nr; i += nth) {
        स्थिरांक दशांश * src_data = (दशांश *)((अक्षर *) src0->data + i*nb01);

        साठी (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + शीर्ष_k, tmp + ne00, cmp_शीर्ष_k{src_data});

        int32_t * dst_data = (int32_t *)((अक्षर *) dst->data + i*nb1);

        std::copy(tmp, tmp + शीर्ष_k, dst_data);

        // emphasize that the order is not important
        जर (शीर्ष_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

शून्य ggml_compute_forward_शीर्ष_k(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_शीर्ष_k_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst,
        पूर्णांक ir0, पूर्णांक ir1,
        int64_t ic_start, int64_t ic_end,
        दशांश * partials, int64_t partial_stride) {

    स्थिरांक तार्किक write_partials = (partials != nullptr);
    स्थिरांक ggml_tensor * q     = dst->src[0];
    स्थिरांक ggml_tensor * k     = dst->src[1];
    स्थिरांक ggml_tensor * v     = dst->src[2];
    स्थिरांक ggml_tensor * mask  = dst->src[3];
    स्थिरांक ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिरांक int64_t DK = nek0;
    स्थिरांक int64_t DV = nev0;
    स्थिरांक int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    स्थिरांक int64_t rk2 = neq2/nek2;
    स्थिरांक int64_t rk3 = neq3/nek3;

    स्थिरांक int64_t rv2 = neq2/nev2;
    स्थिरांक int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    दशांश scale         = 1.0f;
    दशांश max_bias      = 0.0f;
    दशांश logit_softcap = 0.0f;

    स_नकल(&scale,         (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max_bias,      (दशांश *) dst->op_params + 1, आकार(दशांश));
    स_नकल(&logit_softcap, (दशांश *) dst->op_params + 2, आकार(दशांश));

    जर (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    स्थिरांक uint32_t n_head      = neq2;
    स्थिरांक uint32_t n_head_log2 = 1u << (uint32_t) न्यूनमान(log2(n_head));

    स्थिरांक दशांश m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    स्थिरांक दशांश m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         स्थिरांक k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t स्थिरांक q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    स्थिरांक kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   स्थिरांक v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    पूर्णांक ith = params->ith;

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // q indices
        स्थिरांक पूर्णांक iq3 = ir/(neq2*neq1);
        स्थिरांक पूर्णांक iq2 = (ir - iq3*neq2*neq1)/neq1;
        स्थिरांक पूर्णांक iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        स्थिरांक uint32_t h = iq2; // head index
        स्थिरांक दशांश slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        दशांश S = 0.0f;      // sum
        दशांश M = -अनन्त; // maximum KQ value

        दशांश       * VKQ32 = (दशांश       *) params->wdata + ith*(1*DK + 2*DV + CACHE_ओळ_SIZE_F32); // FP32 VKQ accumulator
        दशांश       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        जर (v->type == GGML_TYPE_F16) {
            स_रखो(VKQ16, 0, DV*आकार(ggml_fp16_t));
        } अन्यथा {
            स_रखो(VKQ32, 0, DV*आकार(दशांश));
        }

        स्थिरांक ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((अक्षर *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : रिक्त;

        // k indices
        स्थिरांक पूर्णांक ik3 = iq3 / rk3;
        स्थिरांक पूर्णांक ik2 = iq2 / rk2;

        // v indices
        स्थिरांक पूर्णांक iv3 = iq3 / rv3;
        स्थिरांक पूर्णांक iv2 = iq2 / rv2;

        स्थिरांक दशांश * pq = (स्थिरांक दशांश *) ((अक्षर *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        साठी (int64_t ic = ic_start; ic < ic_end; ++ic) {
            स्थिरांक दशांश mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            जर (mv == -अनन्त) {
                सुरूठेवा;
            }

            दशांश s; // KQ value

            स्थिरांक अक्षर * k_data = (स्थिरांक अक्षर *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            जर (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            स्थिरांक दशांश Mold = M;

            दशांश ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            दशांश vs = 1.0f; // post-softmax KQ value, expf(s - M)

            स्थिरांक अक्षर * v_data = ((स्थिरांक अक्षर *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            जर (v->type == GGML_TYPE_F16) {
                जर (s > M) {
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
                ggml_vec_mad_f16(DV, VKQ16, (स्थिरांक ggml_fp16_t *) v_data, vs);
            } अन्यथा {
                जर (s > M) {
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
                जर (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } अन्यथा {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (स्थिरांक दशांश *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        जर (v->type == GGML_TYPE_F16) {
            साठी (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        जर (sinks && ic_start == 0) {
            स्थिरांक दशांश s = ((दशांश *)((अक्षर *) sinks->data))[h];

            दशांश ms = 1.0f;
            दशांश vs = 1.0f;

            जर (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } अन्यथा {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        जर (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            दशांश * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            स_नकल(partial + 2, VKQ32, DV * आकार(दशांश));
        } अन्यथा {
            // V /= S
            स्थिरांक दशांश S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            स्थिरांक पूर्णांक i1 = iq1;
            स्थिरांक पूर्णांक i2 = iq2;
            स्थिरांक पूर्णांक i3 = iq3;

            // permute(0, 2, 1, 3)
            स_नकल((अक्षर *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

स्थिर शून्य ggml_compute_forward_flash_attn_ext_tiled(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst,
        पूर्णांक ir0, पूर्णांक ir1) {
    स्थिरांक ggml_tensor * q     = dst->src[0];
    स्थिरांक ggml_tensor * k     = dst->src[1];
    स्थिरांक ggml_tensor * v     = dst->src[2];
    स्थिरांक ggml_tensor * mask  = dst->src[3];
    स्थिरांक ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिरांक int64_t DK = nek0;
    स्थिरांक int64_t DV = nev0;
    स्थिरांक int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    स्थिरांक ggml_type kv_type = k->type;


    // broadcast factors
    स्थिरांक int64_t rk2 = neq2/nek2;
    स्थिरांक int64_t rk3 = neq3/nek3;

    स्थिरांक int64_t rv2 = neq2/nev2;
    स्थिरांक int64_t rv3 = neq3/nev3;

    दशांश scale         = 1.0f;
    दशांश max_bias      = 0.0f;
    दशांश logit_softcap = 0.0f;

    स_नकल(&scale,         (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&max_bias,      (दशांश *) dst->op_params + 1, आकार(दशांश));
    स_नकल(&logit_softcap, (दशांश *) dst->op_params + 2, आकार(दशांश));

    जर (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    स्थिरांक uint32_t n_head      = neq2;
    स्थिरांक uint32_t n_head_log2 = 1u << (uint32_t) न्यूनमान(log2(n_head));

    स्थिरांक दशांश m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    स्थिरांक दशांश m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    पूर्णांक ith = params->ith;

    स्थिर constexpr पूर्णांक Q_TILE_SZ  = ggml_fa_tile_config::Q;
    स्थिर constexpr पूर्णांक KV_TILE_SZ = ggml_fa_tile_config::KV;

    पूर्णांक ir = ir0;
    पर्यंत (ir < ir1) {
        // q indices for the start of this tile
        स्थिरांक पूर्णांक iq3 = ir/(neq2*neq1);
        स्थिरांक पूर्णांक iq2 = (ir - iq3*neq2*neq1)/neq1;
        स्थिरांक पूर्णांक iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        स्थिरांक पूर्णांक tile_rows = MIN(Q_TILE_SZ, MIN((पूर्णांक)(ir1 - ir), (पूर्णांक)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        स्थिरांक uint32_t h = iq2; // head index
        स्थिरांक दशांश slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        दशांश S[Q_TILE_SZ];
        दशांश M[Q_TILE_SZ];

        साठी (पूर्णांक i = 0 ; i < Q_TILE_SZ; ++i) {
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
        दशांश * base  = (दशांश *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_ओळ_SIZE_F32);

        शून्य  * Q_q    = base;
        दशांश * KQ     = (दशांश *)((अक्षर *)base + Q_TILE_SZ * DK * आकार(दशांश));
        दशांश * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        दशांश * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        दशांश * V32    = VKQ32 + Q_TILE_SZ * DV;
        दशांश * K_f32  = V32 + KV_TILE_SZ * DV;

        स_रखो(VKQ32, 0, Q_TILE_SZ * DV * आकार(दशांश));
        स_रखो(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * आकार(दशांश));

        // k indices
        स्थिरांक पूर्णांक ik3 = iq3 / rk3;
        स्थिरांक पूर्णांक ik2 = iq2 / rk2;

        // v indices
        स्थिरांक पूर्णांक iv3 = iq3 / rv3;
        स्थिरांक पूर्णांक iv2 = iq2 / rv2;

        {
            दशांश * Q_f32 = (दशांश *)Q_q;
            साठी (पूर्णांक tq = 0; tq < tile_rows; tq++) {
                स्थिरांक दशांश * pq = (स्थिरांक दशांश *) ((अक्षर *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                स_नकल(Q_f32 + tq * DK, pq, DK * आकार(दशांश));
            }
            साठी (पूर्णांक tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                स_रखो(Q_f32 + tq * DK, 0, DK * आकार(दशांश));
            }
        }

        स_रखो(K_f32, 0, DK * KV_TILE_SZ * आकार(दशांश));
        स_रखो(V32,   0, KV_TILE_SZ * DV * आकार(दशांश));

        साठी (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            स्थिरांक पूर्णांक kv_tile = (पूर्णांक)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            जर (mask) {
                तार्किक can_skip = सत्य;
                साठी (पूर्णांक tq = 0; tq < tile_rows; tq++) {
                    स्थिरांक ggml_fp16_t * mp_row = (स्थिरांक ggml_fp16_t *)((स्थिरांक अक्षर *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    साठी (पूर्णांक tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        जर (mask32[tq * KV_TILE_SZ + tk] != -अनन्त) {
                            can_skip = असत्य;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    साठी (पूर्णांक tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -अनन्त;
                    }
                }

                जर (can_skip) {
                    सुरूठेवा;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            साठी (पूर्णांक tk = 0; tk < kv_tile; tk++) {
                स्थिरांक अक्षर * k_data = (स्थिरांक अक्षर *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                जर (kv_type == GGML_TYPE_F16) {
                    स्थिरांक ggml_fp16_t * k_f16 = (स्थिरांक ggml_fp16_t *)k_data;
                    साठी (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } अन्यथा {
                    स्थिरांक दशांश * k_f32_src = (स्थिरांक दशांश *)k_data;
                    साठी (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            स_रखो(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * आकार(दशांश));
            simd_gemm(KQ, (स्थिरांक दशांश *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            जर (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            जर (kv_tile < KV_TILE_SZ) {
                साठी (पूर्णांक tq = 0; tq < Q_TILE_SZ; tq++) {
                    साठी (पूर्णांक tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -अनन्त;
                    }
                }
            }

            जर (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            तार्किक skip[Q_TILE_SZ] = {};

            साठी (पूर्णांक tq = 0; tq < Q_TILE_SZ; tq++) {
                दशांश * kq_row = KQ + tq * KV_TILE_SZ;

                दशांश tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                जर (tile_max == -अनन्त) {
                    skip[tq] = सत्य;
                    सुरूठेवा;
                }

                स्थिरांक दशांश Mold = M[tq];
                स्थिरांक दशांश Mnew = fmaxf(Mold, tile_max);

                जर (Mnew > Mold) {
                    स्थिरांक दशांश ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            साठी (पूर्णांक tk = 0; tk < kv_tile; tk++) {
                स्थिरांक अक्षर * v_data = (स्थिरांक अक्षर *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                जर (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((स्थिरांक ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } अन्यथा {
                    स_नकल(V32 + tk * DV, v_data, DV * आकार(दशांश));
                }
            }
            साठी (पूर्णांक tq = 0; tq < Q_TILE_SZ; tq++) {
                जर (skip[tq]) {
                    स_रखो(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * आकार(दशांश));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        जर (sinks) {
            स्थिरांक दशांश s = ((दशांश *)((अक्षर *) sinks->data))[h];

            साठी (पूर्णांक tq = 0; tq < tile_rows; tq++) {
                दशांश ms = 1.0f;
                दशांश vs = 1.0f;

                जर (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } अन्यथा {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        साठी (पूर्णांक tq = 0; tq < tile_rows; tq++) {
            // V /= S
            स्थिरांक दशांश S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            स्थिरांक पूर्णांक i1 = iq1 + tq;
            स्थिरांक पूर्णांक i2 = iq2;
            स्थिरांक पूर्णांक i3 = iq3;

            // permute(0, 2, 1, 3)
            स_नकल((अक्षर *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
स्थिर शून्य ggml_flash_attn_ext_reduce_partials(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst,
        स्थिरांक int64_t n_chunks,
        स्थिरांक int64_t chunk_size) {

    स्थिरांक ggml_tensor * q = dst->src[0];
    स्थिरांक ggml_tensor * k = dst->src[1];
    स्थिरांक ggml_tensor * v = dst->src[2];

    स्थिरांक int64_t DK        = k->ne[0];
    स्थिरांक int64_t DV        = v->ne[0];
    स्थिरांक int64_t nek1      = k->ne[1];
    स्थिरांक int64_t n_q_heads = q->ne[2];

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t wdata_per_thread = DK + 2*DV + CACHE_ओळ_SIZE_F32;
    दशांश *       thread_wdata     = (दशांश *) params->wdata + ith * wdata_per_thread;

    स्थिरांक int64_t partials_offset  = nth * (DK + 2*DV + CACHE_ओळ_SIZE_F32);
    स्थिरांक int64_t partial_size     = 2 + DV;
    स्थिरांक दशांश * partials_base    = (स्थिरांक दशांश *) params->wdata + partials_offset;

    // Output layout
    स्थिरांक int64_t ne1 = dst->ne[1];
    स्थिरांक int64_t ne2 = dst->ne[2];
    स्थिरांक माप_प्रकार  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    साठी (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        दशांश   M_final   = -अनन्त;
        दशांश   S_final   = 0.0f;
        दशांश * VKQ_final = thread_wdata;
        स_रखो(VKQ_final, 0, DV * आकार(दशांश));

        // Combine partials from all chunks
        साठी (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            स्थिरांक int64_t ic_start = chunk_idx * chunk_size;
            जर (ic_start >= nek1) सुरूठेवा;

            स्थिरांक दशांश * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            स्थिरांक दशांश   M_chunk   = partial[0];
            स्थिरांक दशांश   S_chunk   = partial[1];
            स्थिरांक दशांश * VKQ_chunk = partial + 2;

            जर (S_chunk == 0.0f) सुरूठेवा;

            स्थिरांक दशांश M_new     = fmaxf(M_final, M_chunk);
            स्थिरांक दशांश scale_old = expf(M_final - M_new);
            स्थिरांक दशांश scale_new = expf(M_chunk - M_new);

            साठी (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        जर (S_final != 0.0f) {
            स्थिरांक दशांश S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        स_नकल((अक्षर *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

स्थिर शून्य ggml_compute_forward_flash_attn_ext_f16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * q     = dst->src[0];
    स्थिरांक ggml_tensor * k     = dst->src[1];
    स्थिरांक ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb,  dst, nb)

    स्थिरांक int64_t DK = nek0;
    स्थिरांक int64_t DV = nev0;
    स्थिरांक int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    स्थिरांक तार्किक use_ref = params->use_ref;

    स्थिरांक तार्किक kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    स्थिरांक तार्किक use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    जर (use_split_kv_path) {
        स्थिरांक int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        स्थिरांक int64_t partial_size  = 2 + DV;
        दशांश *       partials_base = (दशांश *) params->wdata + nth * (DK + 2*DV + CACHE_ओळ_SIZE_F32);

        स्थिरांक int64_t ic_start = ith * chunk_size;
        स्थिरांक int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        स्थिरांक int64_t partial_stride = nth * partial_size;
        दशांश *       chunk_partials = partials_base + ith * partial_size;

        जर (ic_start < nek1) {
            साठी (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } अन्यथा {
            साठी (int64_t q_head = 0; q_head < neq2; q_head++) {
                दशांश * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -अनन्त;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } अन्यथा {

        // total rows in q
        स्थिरांक int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        स्थिरांक तार्किक disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        पूर्णांक nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        जर (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        जर (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        स्थिरांक int64_t dr = (nr + nchunk - 1) / nchunk;

        स्थिर constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        तार्किक use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#जर defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#जर defined(__ARM_FEATURE_SVE)
        स्थिरांक int64_t f32_epr = svcntw();
#अन्यथा
        स्थिरांक int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        पूर्णांक current_chunk = ith;

        पर्यंत (current_chunk < nchunk) {
            स्थिरांक int64_t ir0 = dr * current_chunk;
            स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

            जर (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } अन्यथा {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

शून्य ggml_compute_forward_flash_attn_ext(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    निवड (dst->op_params[3]) {
        स्थिती GGML_PREC_DEFAULT:
        स्थिती GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

स्थिर शून्य ggml_compute_forward_flash_attn_back_f32(
        स्थिरांक ggml_compute_params * params,
        स्थिरांक तार्किक masked,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * q = dst->src[0];
    स्थिरांक ggml_tensor * k = dst->src[1];
    स्थिरांक ggml_tensor * v = dst->src[2];
    स्थिरांक ggml_tensor * d = dst->src[3];

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

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t D = neq0;
    स्थिरांक int64_t N = neq1;
    स्थिरांक int64_t P = nek1 - N;
    स्थिरांक int64_t M = P + N;

    स्थिरांक पूर्णांक Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    स्थिरांक पूर्णांक mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == आकार(दशांश));
    GGML_ASSERT(nbk0 == आकार(दशांश));
    GGML_ASSERT(nbv0 == आकार(दशांश));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    जर (ith == 0) {
        स_रखो(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    स्थिरांक int64_t elem_q = ggml_nelements(q);
    स्थिरांक int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    स्थिरांक माप_प्रकार tsize = ggml_type_size(result_type);

    स्थिरांक माप_प्रकार offs_q = 0;
    स्थिरांक माप_प्रकार offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    स्थिरांक माप_प्रकार offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    शून्य * grad_q = (अक्षर *) dst->data;
    शून्य * grad_k = (अक्षर *) dst->data + offs_k;
    शून्य * grad_v = (अक्षर *) dst->data + offs_v;

    स्थिरांक माप_प्रकार nbgq1 = nb0*neq0;
    स्थिरांक माप_प्रकार nbgq2 = nb0*neq0*neq1;
    स्थिरांक माप_प्रकार nbgq3 = nb0*neq0*neq1*neq2;

    स्थिरांक माप_प्रकार nbgk1 = nb0*nek0;
    स्थिरांक माप_प्रकार nbgk2 = nb0*nek0*nek1;
    स्थिरांक माप_प्रकार nbgk3 = nb0*nek0*nek1*neq2;

    स्थिरांक माप_प्रकार nbgv1 = nb0*nev0;
    स्थिरांक माप_प्रकार nbgv2 = nb0*nev0*nev1;
    स्थिरांक माप_प्रकार nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    स्थिरांक पूर्णांक nr = nek2*nek3;

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    स्थिरांक दशांश scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    पूर्णांक nrep = neq2/nek2;

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        // q indices
        स्थिरांक पूर्णांक ik3 = ir/(nek2);
        स्थिरांक पूर्णांक ik2 = ir - ik3*nek2;

        स्थिरांक पूर्णांक iq3 = ik3;
        स्थिरांक पूर्णांक id3 = ik3;
        स्थिरांक पूर्णांक iv3 = ik3;
        स्थिरांक पूर्णांक iv2 = ik2;

        साठी (पूर्णांक irep = 0; irep < nrep; ++irep) {
            स्थिरांक पूर्णांक iq2 = ik2 + irep*nek2;
            स्थिरांक पूर्णांक id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            साठी (पूर्णांक iq1 = 0; iq1 < neq1; ++iq1) {
                स्थिरांक पूर्णांक id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                दशांश * S  = (दशांश *) params->wdata + ith*2*(mxDM + CACHE_ओळ_SIZE_F32) + 0*(mxDM+CACHE_ओळ_SIZE_F32);
                दशांश * SM = (दशांश *) params->wdata + ith*2*(mxDM + CACHE_ओळ_SIZE_F32) + 1*(mxDM+CACHE_ओळ_SIZE_F32);

                साठी (पूर्णांक i = M; i < Mup; ++i) {
                    S[i] = -अनन्त;
                }

                स्थिरांक int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                साठी (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    स्थिरांक पूर्णांक ik1 = ic;

                    // S indices
                    स्थिरांक पूर्णांक i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (दशांश *) ((अक्षर *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (दशांश *) ((अक्षर *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                साठी (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -अनन्त;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    दशांश max = -अनन्त;
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
                साठी (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (दशांश *) ((अक्षर *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(दशांश *) ((अक्षर *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                दशांश dot_SM_gradSM = 0;
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
                साठी (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (दशांश *) ((अक्षर *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (दशांश *) ((अक्षर *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                साठी (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (दशांश *) ((अक्षर *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (दशांश *) ((अक्षर *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                साठी (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (दशांश *) ((अक्षर *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(दशांश *) ((अक्षर *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

शून्य ggml_compute_forward_flash_attn_back(
        स्थिरांक ggml_compute_params * params,
        स्थिरांक तार्किक masked,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * q = dst->src[0];

    निवड (q->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

स्थिर शून्य ggml_compute_forward_ssm_conv_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0]; // conv_x
    स्थिरांक ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nc  = src1->ne[0]; // d_conv
    स्थिरांक पूर्णांक ncs = src0->ne[0]; // d_conv - 1 + n_t
    स्थिरांक पूर्णांक nr  = src0->ne[1]; // d_inner
    स्थिरांक पूर्णांक n_t =  dst->ne[1]; // tokens per sequence
    स्थिरांक पूर्णांक n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == आकार(दशांश));
    GGML_ASSERT(src1->nb[0] == आकार(दशांश));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);
    स्थिरांक पूर्णांक ir  = ir1 - ir0;

    साठी (पूर्णांक i3 = 0; i3 < n_s; ++i3) {
        साठी (पूर्णांक i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            स्थिरांक दशांश * s = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            स्थिरांक दशांश * c = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            दशांश * x = (दशांश *) ((अक्षर *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            साठी (पूर्णांक i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                दशांश sumf = 0.0f;

                // d_conv
                साठी (पूर्णांक i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

शून्य ggml_compute_forward_ssm_conv(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    निवड (dst->src[0]->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

स्थिर शून्य ggml_compute_forward_ssm_scan_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    स्थिरांक ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    स्थिरांक ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    स्थिरांक ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    स्थिरांक ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    स्थिरांक ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    स्थिरांक ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t nc = src0->ne[0]; // d_state
    स्थिरांक int64_t nr = src0->ne[1]; // dim
    स्थिरांक int64_t nh = src1->ne[1]; // n_head
    स्थिरांक int64_t ng = src4->ne[1];
    स्थिरांक int64_t nt = src1->ne[2]; // number of tokens per sequence
    स्थिरांक int64_t ns = src1->ne[3]; // number of sequences in the batch
    स्थिरांक int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    स्थिरांक int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == आकार(दशांश));
    GGML_ASSERT(src1->nb[0] == आकार(दशांश));
    GGML_ASSERT(src2->nb[0] == आकार(दशांश));
    GGML_ASSERT(src3->nb[0] == आकार(दशांश));
    GGML_ASSERT(src4->nb[0] == आकार(दशांश));
    GGML_ASSERT(src5->nb[0] == आकार(दशांश));
    GGML_ASSERT(src6->nb[0] == आकार(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    स्थिरांक पूर्णांक dh = (nh + nth - 1)/nth;

    // head range for this thread
    स्थिरांक पूर्णांक ih0 = dh*ith;
    स्थिरांक पूर्णांक ih1 = MIN(ih0 + dh, nh);

    स्थिरांक int32_t * ids = (स्थिरांक int32_t *) src6->data;

    साठी (पूर्णांक i3 = 0; i3 < ns; ++i3) {
        स्थिरांक दशांश * s0 = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              दशांश * s  = (      दशांश *) ((      अक्षर *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        साठी (पूर्णांक i2 = 0; i2 < nt; ++i2) {
            स्थिरांक दशांश * x  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            स्थिरांक दशांश * dt = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            स्थिरांक दशांश * A  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src3->data); // {d_state, nh} or {1, nh}
            स्थिरांक दशांश * B  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            स्थिरांक दशांश * C  = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  दशांश * y  = (      दशांश *) ((      अक्षर *) dst->data + i2*(nh*nr*आकार(दशांश)) + i3*(nt*nh*nr*आकार(दशांश))); // {dim, nh, nt, ns}

            जर (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                साठी (पूर्णांक h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    स्थिरांक दशांश dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    स्थिरांक दशांश dA = expf(dt_soft_plus * A[h]);
                    स्थिरांक पूर्णांक g = h / (nh / ng); // repeat_interleave

                    // dim
                    साठी (पूर्णांक i1 = 0; i1 < nr; ++i1) {
                        स्थिरांक पूर्णांक ii = i1 + h*nr;
                        स्थिरांक दशांश x_dt = x[ii] * dt_soft_plus;
                        दशांश sumf = 0.0f;
#जर defined(GGML_SIMD)
    #जर defined(__ARM_FEATURE_SVE)
                        स्थिरांक पूर्णांक ggml_f32_epr = svcntw();
                        स्थिरांक पूर्णांक ggml_f32_step = 1 * ggml_f32_epr;

                        स्थिरांक पूर्णांक np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        साठी (पूर्णांक i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            साठी (पूर्णांक j = 0; j < 1; j++) {
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
    #किंवाजर defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        स्थिरांक पूर्णांक np = 0;
    #अन्यथा
                        स्थिरांक पूर्णांक np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        साठी (पूर्णांक i = 0; i < np; i += GGML_F32_STEP) {
                            साठी (पूर्णांक j = 0; j < GGML_F32_ARR; j++) {
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
                        स्थिरांक पूर्णांक np = 0;
#endif
                        // d_state
                        साठी (पूर्णांक i0 = np; i0 < nc; ++i0) {
                            स्थिरांक पूर्णांक i = i0 + ii*nc;
                            स्थिरांक पूर्णांक ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            स्थिरांक दशांश state = (s0[i] * dA) + (B[ig] * x_dt);
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
                साठी (पूर्णांक h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    स्थिरांक दशांश dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    स्थिरांक पूर्णांक g = h / (nh / ng); // repeat_interleave

                    // dim
                    साठी (पूर्णांक i1 = 0; i1 < nr; ++i1) {
                        स्थिरांक पूर्णांक ii = i1 + h*nr;
                        स्थिरांक दशांश x_dt = x[ii] * dt_soft_plus;
#जर defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        साठी (int64_t k = 0; k < nc; k += svcntw()) {
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
                        दशांश sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        साठी (पूर्णांक i0 = 0; i0 < nc; ++i0) {
                            स्थिरांक पूर्णांक i = i0 + ii*nc;
                            स्थिरांक पूर्णांक ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            स्थिरांक दशांश state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            स्थिरांक int64_t slot = nt - 1 - i2;
            जर (K > 1 && slot > 0 && slot < K) {
                दशांश * s_snapshot = (दशांश *) ((अक्षर *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                साठी (पूर्णांक h = ih0; h < ih1; ++h) {
                    स_नकल((अक्षर *) s_snapshot + h*src0->nb[2], (अक्षर *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

शून्य ggml_compute_forward_ssm_scan(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    निवड (dst->src[0]->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

स्थिर शून्य ggml_compute_forward_win_part_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    स्थिरांक int32_t nep0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t nep1 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t w    = ((स्थिरांक int32_t *)(dst->op_params))[2];

    प्रतिज्ञा(ne00 == ne0);
    प्रतिज्ञा(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    साठी (पूर्णांक py = 0; py < nep1; ++py) {
        साठी (पूर्णांक px = 0; px < nep0; ++px) {
            स्थिरांक int64_t i3 = py*nep0 + px;
            साठी (int64_t i2 = 0; i2 < ne2; ++i2) {
                साठी (int64_t i1 = 0; i1 < ne1; ++i1) {
                    साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                        स्थिरांक int64_t i02 = py*w + i2;
                        स्थिरांक int64_t i01 = px*w + i1;
                        स्थिरांक int64_t i00 = i0;

                        स्थिरांक int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        स्थिरांक int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        जर (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((दशांश *) dst->data)[i] = 0.0f;
                        } अन्यथा {
                            ((दशांश *) dst->data)[i] = ((दशांश *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_win_part(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

स्थिर शून्य ggml_compute_forward_win_unpart_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    स्थिरांक int32_t w = ((स्थिरांक int32_t *)(dst->op_params))[0];

    // padding
    स्थिरांक पूर्णांक px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    स्थिरांक पूर्णांक npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    प्रतिज्ञा(ne0 == ne00);

    // TODO: optimize / multi-thread
    साठी (int64_t i2 = 0; i2 < ne2; ++i2) {
        साठी (int64_t i1 = 0; i1 < ne1; ++i1) {
            साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                स्थिरांक पूर्णांक ip2 = i2/w;
                स्थिरांक पूर्णांक ip1 = i1/w;

                स्थिरांक int64_t i02 = i2%w;
                स्थिरांक int64_t i01 = i1%w;
                स्थिरांक int64_t i00 = i0;

                स्थिरांक int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                स्थिरांक int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((दशांश *) dst->data)[j] = ((दशांश *) src0->data)[i];
            }
        }
    }
}

शून्य ggml_compute_forward_win_unpart(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

शून्य ggml_compute_forward_unary(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_unary_op op = ggml_get_unary_op(dst);

    निवड (op) {
        स्थिती GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } थांबा;
        स्थिती GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

शून्य ggml_compute_forward_glu(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_glu_op op = ggml_get_glu_op(dst);

    निवड (op) {
        स्थिती GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } थांबा;
        स्थिती GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } थांबा;
        स्थिती GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } थांबा;
        स्थिती GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } थांबा;
        स्थिती GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } थांबा;
        स्थिती GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } थांबा;
        स्थिती GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

स्थिर शून्य ggml_compute_forward_get_rel_pos_f16(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    स्थिरांक int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    साठी (int64_t i2 = 0; i2 < ne2; ++i2) {
        साठी (int64_t i1 = 0; i1 < ne1; ++i1) {
            स्थिरांक int64_t pos = (w - i1 - 1) + i2;
            साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

शून्य ggml_compute_forward_get_rel_pos(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

स्थिर शून्य ggml_compute_forward_add_rel_pos_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];
    स्थिरांक ggml_tensor * src2 = dst->src[2];

    स्थिरांक तार्किक inplace = (तार्किक) ((int32_t *) dst->op_params)[0];
    जर (!inplace) {
        जर (params->ith == 0) {
            स_नकल((अक्षर *) dst->data, (अक्षर *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    दशांश * src1_data = (दशांश *) src1->data;
    दशांश * src2_data = (दशांश *) src2->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    स्थिरांक int64_t ne10 = src1->ne[0];
    स्थिरांक int64_t ne11 = src1->ne[1];
    स्थिरांक int64_t ne12 = src1->ne[2];
    स्थिरांक int64_t ne13 = src1->ne[3];

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // total patches in dst
    स्थिरांक पूर्णांक np = ne13;

    // patches per thread
    स्थिरांक पूर्णांक dp = (np + nth - 1)/nth;

    // patch range for this thread
    स्थिरांक पूर्णांक ip0 = dp*ith;
    स्थिरांक पूर्णांक ip1 = MIN(ip0 + dp, np);

    साठी (int64_t i13 = ip0; i13 < ip1; ++i13) {
        साठी (int64_t i12 = 0; i12 < ne12; ++i12) {
            साठी (int64_t i11 = 0; i11 < ne11; ++i11) {
                स्थिरांक int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                साठी (int64_t i10 = 0; i10 < ne10; ++i10) {
                    स्थिरांक int64_t jp0  = jp1 + i10;
                    स्थिरांक दशांश src1_e = src1_data[jp0];
                    स्थिरांक दशांश src2_e = src2_data[jp0];

                    स्थिरांक int64_t jdh = jp0 * ne10;
                    स्थिरांक int64_t jdw = jdh - (ne10 - 1) * i10;

                    साठी (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

शून्य ggml_compute_forward_add_rel_pos(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

स्थिर शून्य ggml_compute_forward_rwkv_wkv6_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक int64_t T = dst->src[1]->ne[2];
    स्थिरांक int64_t C = dst->ne[0];
    स्थिरांक int64_t HEADS = dst->src[1]->ne[1];
    स्थिरांक int64_t n_seqs = dst->src[5]->ne[1];
    स्थिरांक int64_t head_size = C / HEADS;

    दशांश * dst_data = (दशांश *) dst->data;
    दशांश * state = ((दशांश *) dst->data) + C * T;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक h_start =  (HEADS * (ith    )) / nth;
    स्थिरांक पूर्णांक h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    दशांश * k =          (दशांश *) dst->src[0]->data;
    दशांश * v =          (दशांश *) dst->src[1]->data;
    दशांश * r =          (दशांश *) dst->src[2]->data;
    दशांश * time_faaaa = (दशांश *) dst->src[3]->data;
    दशांश * time_decay = (दशांश *) dst->src[4]->data;

    माप_प्रकार t_stride = HEADS * head_size; // Same to C

    माप_प्रकार h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    माप_प्रकार h_stride_2d = head_size * head_size;

    जर (ith == 0) {
        स_रखो(dst_data, 0, T * C * आकार(दशांश));
    }
    ggml_barrier(params->threadpool);


    #जर defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #किंवाजर defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #किंवाजर defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #किंवाजर defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        पूर्णांक wkv_vector_size;
        #जर defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #अन्यथा
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        स्थिरांक int64_t vec_मोजणी = head_size / wkv_vector_size;

        साठी (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = head_size * C * (t / (T / n_seqs));
            दशांश * state_cur = state + state_offset;
            दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[5]->data + state_offset;

            साठी (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                साठी (int64_t i = 0; i < head_size; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_i_offset = h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    दशांश k_val = k[t_h_i_offset];
                    दशांश r_val = r[t_h_i_offset];
                    दशांश time_faaaa_val = time_faaaa[h_i_offset];
                    दशांश time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    साठी (int64_t j = 0; j < vec_मोजणी; j++) {
                        माप_प्रकार base_j = j * wkv_vector_size;
                        माप_प्रकार t_h_j_offset = t_h_offset + base_j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + base_j;

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
                    साठी (int64_t j = vec_मोजणी * wkv_vector_size; j < head_size; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;
                        दशांश v_val = v[t_h_j_offset];
                        दशांश kv_val = v_val * k_val;
                        दशांश prev_state_val = state_prev[h_2d_i_j_offset];
                        दशांश temp_val = kv_val * time_faaaa_val + prev_state_val;
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
        साठी (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = head_size * C * (t / (T / n_seqs));
            दशांश * state_cur = state + state_offset;
            दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[5]->data + state_offset;

            साठी (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                साठी (int64_t i = 0; i < head_size; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_i_offset = h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    दशांश k_val = k[t_h_i_offset];
                    दशांश r_val = r[t_h_i_offset];
                    दशांश time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    दशांश time_decay_val = time_decay[t_h_i_offset];

                    साठी (int64_t j = 0; j < head_size; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;

                        दशांश v_val = v[t_h_j_offset];
                        दशांश kv_val = v_val * k_val;
                        दशांश prev_state_val = state_prev[h_2d_i_j_offset];
                        दशांश temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


शून्य ggml_compute_forward_rwkv_wkv6(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

स्थिर शून्य ggml_compute_forward_gla_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक int64_t T = dst->src[1]->ne[2];
    स्थिरांक int64_t C = dst->ne[0];
    स्थिरांक int64_t HEADS = dst->src[1]->ne[1];
    स्थिरांक int64_t n_seqs = dst->src[4]->ne[1];
    स्थिरांक int64_t head_size = C / HEADS;
    स्थिरांक दशांश scale = ggml_get_op_params_f32(dst, 0);

    दशांश * dst_data = (दशांश *) dst->data;
    दशांश * state = ((दशांश *) dst->data) + C * T;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक h_start =  (HEADS * (ith    )) / nth;
    स्थिरांक पूर्णांक h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    दशांश * k = (दशांश *) dst->src[0]->data;
    दशांश * v = (दशांश *) dst->src[1]->data;
    दशांश * q = (दशांश *) dst->src[2]->data;
    दशांश * g = (दशांश *) dst->src[3]->data;

    माप_प्रकार t_stride = HEADS * head_size; // Same to C

    माप_प्रकार h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    माप_प्रकार h_stride_2d = head_size * head_size;

    जर (ith == 0) {
        स_रखो(dst_data, 0, T * C * आकार(दशांश));
    }
    ggml_barrier(params->threadpool);


    #जर defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #किंवाजर defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #किंवाजर defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #किंवाजर defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        पूर्णांक gla_vector_size;
        #जर defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #अन्यथा
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        स्थिरांक int64_t vec_मोजणी = head_size / gla_vector_size;

        साठी (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = head_size * C * (t / (T / n_seqs));
            दशांश * state_cur = state + state_offset;
            दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[4]->data + state_offset;

            साठी (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                साठी (int64_t i = 0; i < head_size; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    दशांश k_val = k[t_h_i_offset];
                    दशांश q_val = q[t_h_i_offset] * scale;
                    दशांश g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    साठी (int64_t j = 0; j < vec_मोजणी; j++) {
                        माप_प्रकार base_j = j * gla_vector_size;
                        माप_प्रकार t_h_j_offset = t_h_offset + base_j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + base_j;

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
                    साठी (int64_t j = vec_मोजणी * gla_vector_size; j < head_size; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;
                        दशांश v_val = v[t_h_j_offset];
                        दशांश kv_val = v_val * k_val;
                        दशांश prev_state_val = state_prev[h_2d_i_j_offset];
                        दशांश temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #अन्यथा
        साठी (int64_t t = 0; t < T; t++) {
            माप_प्रकार t_offset = t * t_stride;
            माप_प्रकार state_offset = head_size * C * (t / (T / n_seqs));
            दशांश * state_cur = state + state_offset;
            दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[4]->data + state_offset;

            साठी (int64_t h = h_start; h < h_end; h++) {
                माप_प्रकार h_offset = h * h_stride;
                माप_प्रकार t_h_offset = t_offset + h_offset;
                माप_प्रकार h_2d_offset = h * h_stride_2d;

                साठी (int64_t i = 0; i < head_size; i++) {
                    माप_प्रकार t_h_i_offset = t_h_offset + i;
                    माप_प्रकार h_2d_i_offset = h_2d_offset + i * h_stride;

                    दशांश k_val = k[t_h_i_offset];
                    दशांश q_val = q[t_h_i_offset] * scale;
                    दशांश g_val = g[t_h_i_offset];

                    साठी (int64_t j = 0; j < head_size; j++) {
                        माप_प्रकार t_h_j_offset = t_h_offset + j;
                        माप_प्रकार h_2d_i_j_offset = h_2d_i_offset + j;

                        दशांश v_val = v[t_h_j_offset];
                        दशांश kv_val = v_val * k_val;
                        दशांश prev_state_val = state_prev[h_2d_i_j_offset];
                        दशांश temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


शून्य ggml_compute_forward_gla(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_solve_tri_f32(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) {
    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_द्विमान_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t k = ne10;   // number of RHS columns
    स्थिरांक int64_t n = ne11;   // A is n×n
    स्थिरांक int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    स्थिरांक दशांश * A = (स्थिरांक दशांश *) src0->data;  // [n, n, B1, B2]
    स्थिरांक दशांश * B = (स्थिरांक दशांश *) src1->data;  // [n, k, B1, B2]
          दशांश * X = (      दशांश *) dst->data;   // [n, k, B1, B2]

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne02*k);
        स्थिरांक int64_t i02 = (ir - i03*ne02*k)/k;
        स्थिरांक int64_t i01 = (ir - i03*ne02*k - i02*k);

        स्थिरांक दशांश * A_batch = A + i02 * nb02 / आकार(दशांश) + i03 * nb03 / आकार(दशांश);
        स्थिरांक दशांश * B_batch = B + i02 * nb12 / आकार(दशांश) + i03 * nb13 / आकार(दशांश);

        दशांश * X_batch = X + i02 * nb2 / आकार(दशांश) + i03 * nb3 / आकार(दशांश);

        साठी (int64_t i00 = 0; i00 < n; ++i00) {
            दशांश sum = 0.0f;
            साठी (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            स्थिरांक दशांश diag = A_batch[i00 * n + i00];
            प्रतिज्ञा(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

शून्य ggml_compute_forward_solve_tri(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    जर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } अन्यथा {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
स्थिर शून्य ggml_compute_forward_gated_delta_net_one_chunk(
    स्थिरांक ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    स्थिरांक int64_t S_v      = src_v->ne[0];
    स्थिरांक int64_t H        = src_v->ne[1];
    स्थिरांक int64_t n_tokens = src_v->ne[2];
    स्थिरांक int64_t n_seqs   = src_v->ne[3];

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

    स्थिरांक तार्किक kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    स्थिरांक int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    स्थिरांक int64_t state_seq_stride = src_state->nb[3] / आकार(दशांश);

    स्थिरांक int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    स्थिरांक पूर्णांक ith = params->ith;

    दशांश * delta       = (दशांश *)params->wdata + ith * per_thread + CACHE_ओळ_SIZE_F32;
    दशांश * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    स्थिरांक int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    स्थिरांक int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    दशांश * attn_out_base  = (दशांश *)dst->data;
    दशांश * state_out_base = (दशांश *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    स्थिरांक दशांश * state_in_base = (स्थिरांक दशांश *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    स्थिरांक int64_t rq3 = nev3 / neq3;
    स्थिरांक int64_t rk3 = nev3 / nek3;

    स्थिरांक दशांश scale = 1.0f / sqrtf((दशांश) S_v);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t iv1 = ir % H; // head_index
        स्थिरांक int64_t iv3 = ir / H; // sequence

        स्थिरांक int64_t iq1 = iv1 % neq1;
        स्थिरांक int64_t ik1 = iv1 % nek1;

        स्थिरांक int64_t iq3 = iv3 / rq3;
        स्थिरांक int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        दशांश * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        स्थिरांक दशांश * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        स_नकल(s_out, s_in, S_v * S_v * आकार(दशांश));

        // attn output pointer for first token of this (head, seq)
        दशांश * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        साठी (int64_t t = 0; t < n_tokens; t++) {
            स्थिरांक दशांश * q_d = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            स्थिरांक दशांश * k_d = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            स्थिरांक दशांश * v_d = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            स्थिरांक दशांश beta_val = *(स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            स्थिरांक दशांश * g_d    =  (स्थिरांक दशांश *)((स्थिरांक अक्षर *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            जर (kda) {
                // precompute exp(g) into delta scratch (reused below)
                साठी (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                साठी (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } अन्यथा {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            साठी (int64_t j = 0; j < S_v; ++j) {
                दशांश sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            साठी (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            साठी (int64_t j = 0; j < S_v; ++j) {
                दशांश sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            जर (K > 1) {
                स्थिरांक int64_t target_slot = n_tokens - 1 - t;
                जर (target_slot >= 0 && target_slot < K) {
                    दशांश * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    स_नकल(curr_state_o, s_out, S_v * S_v * आकार(दशांश));
                }
            }
        }
    }
}


स्थिर शून्य ggml_compute_forward_gated_delta_net_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    स्थिरांक तार्किक disable_chunking = ggml_is_numa();

    पूर्णांक nth = params->nth;
    पूर्णांक ith = params->ith;

    // 4x chunks per thread
    पूर्णांक nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    जर (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    जर (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    स्थिरांक int64_t dr = (nr + nchunk - 1) / nchunk;

    पूर्णांक current_chunk = ith;

    पर्यंत (current_chunk < nchunk) {
        स्थिरांक int64_t ir0 = dr * current_chunk;
        स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

शून्य ggml_compute_forward_gated_delta_net(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

स्थिर शून्य ggml_dsv4_hc_comb_norm_cols(दशांश * comb, दशांश eps) {
    constexpr int64_t hc = 4;

    साठी (int64_t idst = 0; idst < hc; ++idst) {
        दशांश sum = eps;
        साठी (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        स्थिरांक दशांश inv_sum = 1.0f / sum;
        साठी (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

स्थिर शून्य ggml_dsv4_hc_comb_norm_rows(दशांश * comb, दशांश eps) {
    constexpr int64_t hc = 4;

    साठी (int64_t isrc = 0; isrc < hc; ++isrc) {
        दशांश sum = eps;
        साठी (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        स्थिरांक दशांश inv_sum = 1.0f / sum;
        साठी (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

स्थिर शून्य ggml_compute_forward_dsv4_hc_comb_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * mixes = dst->src[0];
    स्थिरांक ggml_tensor * scale = dst->src[1];
    स्थिरांक ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    स्थिरांक int64_t n_tokens = mixes->ne[1];

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

    स्थिरांक दशांश eps = ggml_get_op_params_f32(dst, 0);
    स्थिरांक int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t dr  = (n_tokens + nth - 1) / nth;
    स्थिरांक int64_t it0 = dr * ith;
    स्थिरांक int64_t it1 = MIN(it0 + dr, n_tokens);

    स्थिरांक दशांश scale_comb = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) scale->data + 2*nbs0);

    साठी (int64_t it = it0; it < it1; ++it) {
        दशांश comb[hc*hc];

        साठी (int64_t isrc = 0; isrc < hc; ++isrc) {
            दशांश max = -अनन्त;
            साठी (int64_t idst = 0; idst < hc; ++idst) {
                स्थिरांक int64_t idx = idst + hc*isrc;
                स्थिरांक दशांश xv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                स्थिरांक दशांश bv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) base->data  + (comb_offset + idx)*nbb0);
                स्थिरांक दशांश v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            दशांश sum = 0.0f;
            साठी (int64_t idst = 0; idst < hc; ++idst) {
                स्थिरांक int64_t idx = idst + hc*isrc;
                स्थिरांक दशांश v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            स्थिरांक दशांश inv_sum = 1.0f / sum;
            साठी (int64_t idst = 0; idst < hc; ++idst) {
                स्थिरांक int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        साठी (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        साठी (int64_t isrc = 0; isrc < hc; ++isrc) {
            साठी (int64_t idst = 0; idst < hc; ++idst) {
                स्थिरांक int64_t idx = idst + hc*isrc;
                *(दशांश *) ((अक्षर *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

शून्य ggml_compute_forward_dsv4_hc_comb(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

स्थिर शून्य ggml_compute_forward_dsv4_hc_pre_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * x       = dst->src[0];
    स्थिरांक ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिरांक int64_t n_embd   = x->ne[0];
    स्थिरांक int64_t hc       = x->ne[1];
    स्थिरांक int64_t n_tokens = x->ne[2];

    स्थिरांक दशांश scale = ggml_get_op_params_f32(dst, 0);
    स्थिरांक तार्किक  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    जर (gated) {
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

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t nr  = n_embd * n_tokens;
    स्थिरांक int64_t dr  = (nr + nth - 1) / nth;
    स्थिरांक int64_t ir0 = dr * ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i0 = ir % n_embd;
        स्थिरांक int64_t it = ir / n_embd;

        दशांश sum = 0.0f;
        साठी (int64_t ih = 0; ih < hc; ++ih) {
            स्थिरांक दशांश xv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            दशांश wv;
            जर (gated) {
                स्थिरांक दशांश gv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } अन्यथा {
                wv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(दशांश *) ((अक्षर *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
    }
}

शून्य ggml_compute_forward_dsv4_hc_pre(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

स्थिर शून्य ggml_compute_forward_dsv4_hc_post_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * x        = dst->src[0];
    स्थिरांक ggml_tensor * residual = dst->src[1];
    स्थिरांक ggml_tensor * post     = dst->src[2];
    स्थिरांक ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिरांक int64_t n_embd   = x->ne[0];
    स्थिरांक int64_t n_tokens = x->ne[1];
    स्थिरांक int64_t hc       = residual->ne[1];

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
    जर (comb) {
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

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t nr  = n_embd * hc * n_tokens;
    स्थिरांक int64_t dr  = (nr + nth - 1) / nth;
    स्थिरांक int64_t ir0 = dr * ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i0     = ir % n_embd;
        स्थिरांक int64_t idst   = (ir / n_embd) % hc;
        स्थिरांक int64_t it     = ir / (n_embd * hc);

        स्थिरांक दशांश xv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) x->data    + i0*nbx0 + it*nbx1);
        स्थिरांक दशांश pv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) post->data + idst*nbp0 + it*nbp1);

        दशांश sum = xv * pv;
        जर (comb) {
            साठी (int64_t isrc = 0; isrc < hc; ++isrc) {
                स्थिरांक दशांश rv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                स्थिरांक दशांश cv = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } अन्यथा {
            sum += *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(दशांश *) ((अक्षर *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

शून्य ggml_compute_forward_dsv4_hc_post(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

शून्य ggml_compute_forward_शीर्ष_k_mask(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक ggml_tensor * mask   = dst->src[0];
    स्थिरांक ggml_tensor * शीर्ष_k  = dst->src[1];
    स्थिरांक ggml_tensor * score  = dst->src[2];
    स्थिरांक ggml_tensor * blocks = dst->src[3];
    स्थिरांक ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    स्थिरांक int64_t n_kv    = mask->ne[0];
    स्थिरांक int64_t ne1     = dst->ne[1];
    स्थिरांक int64_t ne2     = dst->ne[2];
    स्थिरांक int64_t n_rows  = ggml_nrows(dst);
    स्थिरांक int64_t n_शीर्ष   = शीर्ष_k->ne[0];
    स्थिरांक int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    स्थिरांक int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    स्थिरांक int64_t n_extra = extra ? extra->ne[0] : 0;

    स्थिरांक माप_प्रकार ts = ggml_type_size(dst->type);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    साठी (int64_t ir = ith; ir < n_rows; ir += nth) {
        स्थिरांक int64_t i1 = ir % ne1;
        स्थिरांक int64_t i2 = (ir / ne1) % ne2;
        स्थिरांक int64_t i3 = ir / (ne1*ne2);

        स्थिरांक अक्षर * mask_row = (स्थिरांक अक्षर *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        अक्षर * dst_row = (अक्षर *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        जर (dst->type == GGML_TYPE_F16) {
            स्थिरांक ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-अनन्त);
            साठी (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } अन्यथा {
            साठी (int64_t c = 0; c < n_kv; ++c) {
                ((दशांश *) dst_row)[c] = -अनन्त;
            }
        }

        स्वतः keep = [&](int32_t c) {
            जर (c >= 0 && c < n_kv) {
                स_नकल(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        स्थिरांक अक्षर * शीर्ष_k_row = (स्थिरांक अक्षर *) शीर्ष_k->data + i1*शीर्ष_k->nb[1] + i2*शीर्ष_k->nb[2] + i3*शीर्ष_k->nb[3];

        साठी (int64_t j = 0; j < n_शीर्ष; ++j) {
            स्थिरांक int32_t p = *(स्थिरांक int32_t *) (शीर्ष_k_row + j*शीर्ष_k->nb[0]);
            जर (p < 0 || p >= n_pick) {
                सुरूठेवा;
            }
            जर (score) {
                स्थिरांक दशांश s = *(स्थिरांक दशांश *) ((स्थिरांक अक्षर *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                जर (!(s > -अनन्त)) {
                    सुरूठेवा;
                }
            }
            जर (!blocks) {
                keep(p);
                सुरूठेवा;
            }
            साठी (int64_t k = 0; k < n_blk; ++k) {
                keep(*(स्थिरांक int32_t *) ((स्थिरांक अक्षर *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        जर (extra) {
            स्थिरांक अक्षर * extra_row = (स्थिरांक अक्षर *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            साठी (int64_t j = 0; j < n_extra; ++j) {
                keep(*(स्थिरांक int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

स्थिर शून्य ggml_compute_forward_rwkv_wkv7_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {
    स्थिरांक int64_t T = dst->src[1]->ne[2];
    स्थिरांक int64_t C = dst->ne[0];
    स्थिरांक int64_t HEADS = dst->src[1]->ne[1];
    स्थिरांक int64_t n_seqs = dst->src[6]->ne[1];
    स्थिरांक int64_t head_size = C / HEADS;

    दशांश * dst_data = (दशांश *) dst->data;
    दशांश * state = ((दशांश *) dst->data) + C * T;

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक h_start =  (HEADS * (ith    )) / nth;
    स्थिरांक पूर्णांक h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    दशांश * r = (दशांश *) dst->src[0]->data;
    दशांश * w = (दशांश *) dst->src[1]->data;
    दशांश * k = (दशांश *) dst->src[2]->data;
    दशांश * v = (दशांश *) dst->src[3]->data;
    दशांश * a = (दशांश *) dst->src[4]->data;
    दशांश * b = (दशांश *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #जर defined(GGML_SIMD)
        #जर defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            साठी (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                दशांश * state_cur = state + state_offset;
                दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[6]->data + state_offset;

                साठी (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    साठी (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        दशांश v_val = v[t_h_i_offset];

                        दशांश sa = 0, result = 0;
                        साठी (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        साठी (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            दशांश r_val = r[t_h_j_offset];
                            दशांश w_val = w[t_h_j_offset];
                            दशांश k_val = k[t_h_j_offset];
                            दशांश b_val = b[t_h_j_offset];
                            दशांश kv_val = v_val * k_val;
                            दशांश prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #अन्यथा
            साठी (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                दशांश * state_cur = state + state_offset;
                दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[6]->data + state_offset;

                साठी (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    साठी (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        दशांश sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            साठी (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                साठी (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        साठी (; j < head_size; j += GGML_F32_STEP) {
                            साठी (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        साठी (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            दशांश r_val = r[t_h_j_offset];
                            दशांश w_val = w[t_h_j_offset];
                            दशांश k_val = k[t_h_j_offset];
                            दशांश b_val = b[t_h_j_offset];
                            दशांश kv_val = v[t_h_i_offset] * k_val;

                            दशांश prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #अन्यथा
        साठी (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            दशांश * state_cur = state + state_offset;
            दशांश * state_prev = t % (T / n_seqs) ? state_cur : (दशांश*)dst->src[6]->data + state_offset;

            साठी (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                साठी (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    दशांश v_val = v[t_h_i_offset];

                    दशांश sa = 0, result = 0;
                    साठी (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    साठी (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        दशांश r_val = r[t_h_j_offset];
                        दशांश w_val = w[t_h_j_offset];
                        दशांश k_val = k[t_h_j_offset];
                        दशांश b_val = b[t_h_j_offset];
                        दशांश kv_val = v_val * k_val;
                        दशांश prev_state_val = state_prev[h_2d_i_j_offset];
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
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

शून्य ggml_compute_forward_map_custom1(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * a = dst->src[0];

    संरचना ggml_map_custom1_op_params p;
    स_नकल(&p, dst->op_params, आकार(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

शून्य ggml_compute_forward_map_custom2(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * a = dst->src[0];
    स्थिरांक ggml_tensor * b = dst->src[1];

    संरचना ggml_map_custom2_op_params p;
    स_नकल(&p, dst->op_params, आकार(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

शून्य ggml_compute_forward_map_custom3(
        स्थिरांक ggml_compute_params * params,
              ggml_tensor * dst) {

    स्थिरांक ggml_tensor * a = dst->src[0];
    स्थिरांक ggml_tensor * b = dst->src[1];
    स्थिरांक ggml_tensor * c = dst->src[2];

    संरचना ggml_map_custom3_op_params p;
    स_नकल(&p, dst->op_params, आकार(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

शून्य ggml_compute_forward_custom(
    स्थिरांक संरचना ggml_compute_params * params,
          संरचना ggml_tensor * dst) {

    संरचना ggml_custom_op_params p;
    स_नकल(&p, dst->op_params, आकार(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

स्थिर शून्य ggml_compute_forward_cross_entropy_loss_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    स्थिरांक int64_t nc = src0->ne[0];
    स्थिरांक int64_t nr = ggml_nrows(src0);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    दशांश * sums =  (दशांश *) params->wdata;
    दशांश * st   = ((दशांश *) params->wdata) + nth + ith*nc;
    दशांश sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= आकार(दशांश) * (nth + nth * nc));

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    साठी (int64_t i1 = ir0; i1 < ir1; ++i1) {
        स्थिरांक दशांश * s0 = (स्थिरांक दशांश *)((स्थिरांक अक्षर *) src0->data + i1*src0->nb[1]);
        स्थिरांक दशांश * s1 = (स्थिरांक दशांश *)((स्थिरांक अक्षर *) src1->data + i1*src1->nb[1]);

#ifndef न_संशोधन
        साठी (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(s0[i]));
            प्रतिज्ञा(!isnan(s1[i]));
        }
#endif // NDEBUG

        दशांश max = -अनन्त;
        ggml_vec_max_f32(nc, &max, s0);
        स्थिरांक ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        प्रतिज्ञा(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        दशांश sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef न_संशोधन
        साठी (int64_t i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(st[i]));
            प्रतिज्ञा(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    जर (ith == 0) {
        दशांश * dp = (दशांश *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (दशांश) nr;
    }
}

शून्य ggml_compute_forward_cross_entropy_loss(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

स्थिर शून्य ggml_compute_forward_cross_entropy_loss_back_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    स्थिरांक ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    स्थिरांक ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    स्थिरांक int64_t ith = params->ith;
    स्थिरांक int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    स्थिरांक int64_t nc = src0f->ne[0];
    स्थिरांक int64_t nr = ggml_nrows(src0f);

    // rows per thread
    स्थिरांक int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक int64_t ir0 = dr*ith;
    स्थिरांक int64_t ir1 = MIN(ir0 + dr, nr);

    स्थिरांक दशांश d_by_nr = ((स्थिरांक दशांश *) grad->data)[0] / (दशांश) nr;

    साठी (int64_t i1 = ir0; i1 < ir1; i1++) {
        दशांश       * ds0 = (दशांश       *)((अक्षर       *) dst->data   + i1*dst->nb[1]);
        स्थिरांक दशांश * s0  = (स्थिरांक दशांश *)((स्थिरांक अक्षर *) src0f->data + i1*src0f->nb[1]);
        स्थिरांक दशांश * s1  = (स्थिरांक दशांश *)((स्थिरांक अक्षर *) src1f->data + i1*src1f->nb[1]);

#ifndef न_संशोधन
        साठी (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            प्रतिज्ञा(!isnan(s0[i]));
            प्रतिज्ञा(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        दशांश max = -अनन्त;
        ggml_vec_max_f32(nc, &max, s0);
        स्थिरांक ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        प्रतिज्ञा(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef न_संशोधन
        साठी (int64_t i = 0; i < nc; ++i) {
            प्रतिज्ञा(!isnan(ds0[i]));
            प्रतिज्ञा(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

शून्य ggml_compute_forward_cross_entropy_loss_back(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_opt_step_adamw_f32(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0         = dst->src[0];
    स्थिरांक ggml_tensor * src0_grad    = dst->src[1];
    स्थिरांक ggml_tensor * src0_grad_m  = dst->src[2];
    स्थिरांक ggml_tensor * src0_grad_v  = dst->src[3];
    स्थिरांक ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    स्थिरांक दशांश * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    स्थिरांक दशांश alpha  = adamw_params_ptr[0];
    स्थिरांक दशांश beta1  = adamw_params_ptr[1];
    स्थिरांक दशांश beta2  = adamw_params_ptr[2];
    स्थिरांक दशांश eps    = adamw_params_ptr[3];
    स्थिरांक दशांश wd     = adamw_params_ptr[4];
    स्थिरांक दशांश beta1h = adamw_params_ptr[5];
    स्थिरांक दशांश beta2h = adamw_params_ptr[6];
    स्थिरांक दशांश keep   = 1.f - alpha * wd;
    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir/(ne02*ne01);
        स्थिरांक int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        स्थिरांक int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        स्थिरांक माप_प्रकार offset = i03*nb03 + i02*nb02 + i01*nb01;

        दशांश       * w = (दशांश       *) ((अक्षर       *) src0->data        + offset); // weight
        स्थिरांक दशांश * g = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_grad->data   + offset); // grad
        दशांश       * m = (दशांश       *) ((अक्षर       *) src0_grad_m->data + offset);
        दशांश       * v = (दशांश       *) ((अक्षर       *) src0_grad_v->data + offset);

        साठी (पूर्णांक i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            स्थिरांक दशांश mh =       m[i00]*beta1h;
            स्थिरांक दशांश vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

शून्य ggml_compute_forward_opt_step_adamw(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error");
            }
    }
}

स्थिर शून्य ggml_compute_forward_opt_step_sgd_f32(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0       = dst->src[0];
    स्थिरांक ggml_tensor * src0_grad  = dst->src[1];
    स्थिरांक ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक पूर्णांक nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == आकार(दशांश));

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1) / nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr * ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    स्थिरांक दशांश * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    स्थिरांक दशांश   alpha            = sgd_params_ptr[0];
    स्थिरांक दशांश   keep             = 1.f - alpha * sgd_params_ptr[1];

    साठी (पूर्णांक ir = ir0; ir < ir1; ++ir) {
        स्थिरांक int64_t i03 = ir / (ne02 * ne01);
        स्थिरांक int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        स्थिरांक int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        स्थिरांक माप_प्रकार offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        दशांश *       w = (दशांश *) ((अक्षर *) src0->data + offset);                   // weight
        स्थिरांक दशांश * g = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_grad->data + offset);  // grad

        साठी (पूर्णांक i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

शून्य ggml_compute_forward_opt_step_sgd(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    निवड (src0->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

स्थिर ओळीत दशांश ggml_fwht_load(स्थिरांक दशांश value) {
    परत value;
}

स्थिर ओळीत दशांश ggml_fwht_load(स्थिरांक ggml_fp16_t value) {
    परत ggml_fp16_to_fp32(value);
}

ढाँचा<typename src_t>
स्थिर शून्य ggml_compute_forward_fwht_impl(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, दशांश> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    स्थिरांक int64_t nr = ne11 * ne12 * ne13;
    स्थिरांक int64_t rows_per_thread = (nr + nth - 1) / nth;
    स्थिरांक int64_t start_row = ith * rows_per_thread;
    स्थिरांक int64_t end_row = MIN(start_row + rows_per_thread, nr);

    स्थिरांक दशांश scale = 1.0f / sqrtf((दशांश)n);

#जर defined(GGML_SIMD)
    स्थिरांक GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    साठी (int64_t r = start_row; r < end_row; r++) {
        स्थिरांक int64_t i13 = r / (ne11 * ne12);
        स्थिरांक int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        स्थिरांक int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        स्थिरांक src_t * src_row = (स्थिरांक src_t *) ((स्थिरांक अक्षर *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        दशांश * dst_row = (दशांश *) ((अक्षर *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        साठी (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#जर defined(GGML_SIMD)
#जर defined(__ARM_FEATURE_SVE)
        स्थिरांक पूर्णांक step = svcntw();
#अन्यथा
        स्थिरांक पूर्णांक step = GGML_F32_EPR;
#endif
#अन्यथा
        स्थिरांक पूर्णांक step = n;
#endif
        साठी (int64_t len = 1; len < step && len < n; len <<= 1) {
            साठी (int64_t i = 0; i < n; i += 2 * len) {
                साठी (int64_t j = 0; j < len; j++) {
                    दशांश u = dst_row[i + j];
                    दशांश v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#जर defined(GGML_SIMD)
        साठी (int64_t len = step; len < n; len <<= 1) {
            साठी (int64_t i = 0; i < n; i += 2 * len) {
                साठी (int64_t j = 0; j < len; j += step) {
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

शून्य ggml_compute_forward_fwht(स्थिरांक ggml_compute_params * params, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    निवड (src1->type) {
        स्थिती GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<दशांश>(params, dst);
            }
            थांबा;
        स्थिती GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            थांबा;
        पूर्वनिर्धारित:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

शून्य ggml_compute_forward_lightning_indexer(
        स्थिरांक ggml_compute_params * params,
        ggml_tensor * dst) {

    स्थिरांक ggml_tensor * q = dst->src[0];
    स्थिरांक ggml_tensor * k = dst->src[1];
    स्थिरांक ggml_tensor * w = dst->src[2]; // weights
    स्थिरांक ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, नवीन,  w, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(माप_प्रकार,  nb, dst, nb)

    GGML_ASSERT( nb0 == ggml_type_size(dst->type));
    GGML_ASSERT(nbq0 == ggml_type_size(  q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(  k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(  w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(  m->type));

    स्थिरांक पूर्णांक n_embd    = q->ne[0];
    स्थिरांक पूर्णांक n_head    = q->ne[1];
    स्थिरांक पूर्णांक n_tokens  = q->ne[2];
    स्थिरांक पूर्णांक n_stream  = q->ne[3];
    स्थिरांक पूर्णांक n_kv      = k->ne[2];

    ggml_to_float_t स्थिरांक k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    स्थिरांक पूर्णांक nr  = n_kv;
    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    // (temporary) buffer for K converted to float
    दशांश * k_row_f32 = (दशांश *) params->wdata + ith*(1*n_embd + CACHE_ओळ_SIZE_F32);

    // rows per thread
    स्थिरांक पूर्णांक dr = (nr + nth - 1)/nth;

    // row range for this thread
    स्थिरांक पूर्णांक ir0 = dr*ith;
    स्थिरांक पूर्णांक ir1 = MIN(ir0 + dr, nr);

    साठी (पूर्णांक s = 0; s < n_stream; ++s) {
        साठी (पूर्णांक t = 0; t < n_tokens; ++t) {
            स्थिरांक दशांश       *   w_row =       (दशांश *) ((अक्षर *)   w->data + t*nbw1 +        s*nbw3);
            स्थिरांक ggml_fp16_t *   m_row = (ggml_fp16_t *) ((अक्षर *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            दशांश             * dst_row =       (दशांश *) ((अक्षर *) dst->data + t*nb1  +        s*nb3 );
            साठी (पूर्णांक ik = ir0; ik < ir1; ++ik) {
                अक्षर * k_row = (अक्षर *) k->data + ik*nbk2 + s*nbk3;
                जर (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } अन्यथा {
                    k_row_f32 = (दशांश *) k_row;
                }
                दशांश score = 0.0f;
                साठी (पूर्णांक h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    दशांश qk = 0.0f;
                    स्थिरांक दशांश * q_row = (दशांश *) ((अक्षर *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
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
