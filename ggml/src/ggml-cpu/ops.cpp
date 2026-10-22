#அந்தர்பாவ "ops.h"

#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "binary-ops.h"
#அந்தர்பாவ "simd-gemm.h"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "unary-ops.h"
#அந்தர்பாவ "vec.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cfloat>
#அந்தர்பாவ <cmath>

// ggml_compute_forward_dup

நிலையான வெறுமை ggml_compute_forward_dup_same_cont(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    மாறிலி size_t nb0 = ggml_type_size(src0->type);

    மாறிலி முழு ith = params->ith; // thread index
    மாறிலி முழு nth = params->nth; // number of threads

    // parallelize by blocks
    மாறிலி முழு nk = ggml_nelements(src0)/ggml_blck_size(src0->type);
    மாறிலி முழு dr = (nk + nth - 1) / nth;
    மாறிலி முழு k0 = dr * ith;
    மாறிலி முழு k1 = MIN(k0 + dr, nk);

    எனில் (k0 < k1) {
        memcpy(
            ((எழுத்து *)  dst->data + k0*nb0),
            ((எழுத்து *) src0->data + k0*nb0),
            (k1 - k0) * nb0);
    }
}

template<typename src_t, typename dst_t>
நிலையான வெறுமை ggml_compute_forward_dup_flt(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி முழு ith = params->ith; // thread index
    மாறிலி முழு nth = params->nth; // number of threads

    // parallelize by rows
    மாறிலி முழு nr = ne01;
    // number of rows per thread
    மாறிலி முழு dr = (nr + nth - 1) / nth;
    // row range for this thread
    மாறிலி முழு ir0 = dr * ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    // case: type & row size equal
    எனில் (src0->type == dst->type &&
        ne00 == ne0 &&
        nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
        // copy by rows
        மாறிலி size_t rs = ne00*nb00;
        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((எழுத்து *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        திரும்பு;
    }

    // case: dst tensor is contiguous
    எனில் (ggml_is_contiguous(dst)) {
        எனில் (nb00 == அளவு(src_t)) {
            எனில் constexpr (std::is_same_v<dst_t, src_t>) {
                // same type
                size_t id = 0;
                மாறிலி size_t rs = ne00 * nb00;
                எழுத்து * dst_ptr = (எழுத்து *) dst->data;

                க்கு (முழு i03 = 0; i03 < ne03; i03++) {
                    க்கு (முழு i02 = 0; i02 < ne02; i02++) {
                        id += rs * ir0;
                        க்கு (முழு i01 = ir0; i01 < ir1; i01++) {
                            மாறிலி எழுத்து * src0_ptr = (எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, rs);
                            id += rs;
                        }
                        id += rs * (ne01 - ir1);
                    }
                }
            } இல்லையெனில் {
                // casting between non-quantized types
                size_t id = 0;
                dst_t * dst_ptr = (dst_t *) dst->data;

                க்கு (முழு i03 = 0; i03 < ne03; i03++) {
                    க்கு (முழு i02 = 0; i02 < ne02; i02++) {
                        id += ne00 * ir0;
                        க்கு (முழு i01 = ir0; i01 < ir1; i01++) {
                            மாறிலி src_t * src0_ptr = (src_t *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                            க்கு (முழு i00 = 0; i00 < ne00; i00++) {
                                தசம tmp = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                                dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                                id++;
                            }
                        }
                        id += ne00 * (ne01 - ir1);
                    }
                }
            }
        } இல்லையெனில் {
            //printf("%s: this is not optimal - fix me\n", __func__);

            size_t id = 0;
            dst_t * dst_ptr = (dst_t *) dst->data;

            க்கு (முழு i03 = 0; i03 < ne03; i03++) {
                க்கு (முழு i02 = 0; i02 < ne02; i02++) {
                    id += ne00 * ir0;
                    க்கு (முழு i01 = ir0; i01 < ir1; i01++) {
                        க்கு (முழு i00 = 0; i00 < ne00; i00++) {
                            மாறிலி src_t * src0_ptr = (src_t *) ((எழுத்து *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            தசம tmp = type_conversion_table<src_t>::to_f32(*src0_ptr);
                            dst_ptr[id] = type_conversion_table<dst_t>::from_f32(tmp);
                            id++;
                        }
                    }
                    id += ne00 * (ne01 - ir1);
                }
            }
        }
        திரும்பு;
    }

    // dst counters
    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    எனில் constexpr (std::is_same_v<dst_t, src_t>) {
        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                போது (i10 >= ne0) {
                    i10 -= ne0;
                    எனில் (++i11 == ne1) {
                        i11 = 0;
                        எனில் (++i12 == ne2) {
                            i12 = 0;
                            எனில் (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        மாறிலி எழுத்து * src0_ptr = ((எழுத்து *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              எழுத்து * dst_ptr  = ((எழுத்து *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        memcpy(dst_ptr, src0_ptr, அளவு(dst_t));

                        எனில் (++i10 == ne00) {
                            i10 = 0;
                            எனில் (++i11 == ne01) {
                                i11 = 0;
                                எனில் (++i12 == ne02) {
                                    i12 = 0;
                                    எனில் (++i13 == ne03) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                போது (i10 >= ne0) {
                    i10 -= ne0;
                    எனில் (++i11 == ne1) {
                        i11 = 0;
                        எனில் (++i12 == ne2) {
                            i12 = 0;
                            எனில் (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
            }
        }

    } இல்லையெனில் {
        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                i10 += ne00 * ir0;
                போது (i10 >= ne0) {
                    i10 -= ne0;
                    எனில் (++i11 == ne1) {
                        i11 = 0;
                        எனில் (++i12 == ne2) {
                            i12 = 0;
                            எனில் (++i13 == ne3) {
                                i13 = 0;
                            }
                        }
                    }
                }
                க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        மாறிலி எழுத்து * src0_ptr = ((எழுத்து *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              எழுத்து * dst_ptr  = ((எழுத்து *)  dst->data + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                        தசம tmp = type_conversion_table<src_t>::to_f32(*(மாறிலி src_t *) src0_ptr);
                        *(dst_t *) dst_ptr = type_conversion_table<dst_t>::from_f32(tmp);

                        எனில் (++i10 == ne0) {
                            i10 = 0;
                            எனில் (++i11 == ne1) {
                                i11 = 0;
                                எனில் (++i12 == ne2) {
                                    i12 = 0;
                                    எனில் (++i13 == ne3) {
                                        i13 = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                i10 += ne00 * (ne01 - ir1);
                போது (i10 >= ne0) {
                    i10 -= ne0;
                    எனில் (++i11 == ne1) {
                        i11 = 0;
                        எனில் (++i12 == ne2) {
                            i12 = 0;
                            எனில் (++i13 == ne3) {
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
நிலையான வெறுமை ggml_compute_forward_dup_to_q(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(!ggml_is_quantized(src0->type));

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி முழு ith = params->ith; // thread index
    மாறிலி முழு nth = params->nth; // number of threads

    // parallelize by rows
    மாறிலி முழு nr = ne01;
    // number of rows per thread
    மாறிலி முழு dr = (nr + nth - 1) / nth;
    // row range for this thread
    மாறிலி முழு ir0 = dr * ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    எனில் (ggml_is_contiguous(dst) &&
            nb00 == அளவு(src_t) &&
            ggml_get_type_traits_cpu(dst->type)->from_float) {
        // casting non-quantized types --> intermediate f32 --> quantized
        ggml_from_float_t மாறிலி quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
        தசம * src0_f32 = (தசம *) params->wdata + (ne00 + CACHE_வரி_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
        எழுத்து * dst_ptr = (எழுத்து *) dst->data;

        க்கு (முழு i03 = 0; i03 < ne03; i03++) {
            க்கு (முழு i02 = 0; i02 < ne02; i02++) {
                id += rs * ir0;
                க்கு (முழு i01 = ir0; i01 < ir1; i01++) {
                    மாறிலி src_t * src0_ptr = (src_t *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                    க்கு (முழு i00 = 0; i00 < ne00; i00++) {
                        src0_f32[i00] = type_conversion_table<src_t>::to_f32(src0_ptr[i00]);
                    }

                    quantize_row_q(src0_f32, dst_ptr + id, ne00);
                    id += rs;
                }
                id += rs * (ne01 - ir1);
            }
        }
    } இல்லையெனில் {
        // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
        GGML_ABORT("not implemented");
    }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
நிலையான வெறுமை ggml_compute_forward_dup_bytes(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(src0->type == dst->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    எனில் (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
        ggml_compute_forward_dup_same_cont(params, dst);
        திரும்பு;
    }

    மாறிலி size_t type_size = ggml_type_size(src0->type);

    மாறிலி முழு ith = params->ith; // thread index
    மாறிலி முழு nth = params->nth; // number of threads

    // parallelize by rows
    மாறிலி முழு nr = ne01;
    // number of rows per thread
    மாறிலி முழு dr = (nr + nth - 1) / nth;
    // row range for this thread
    மாறிலி முழு ir0 = dr * ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    எனில் (src0->type == dst->type &&
        ggml_are_same_shape(src0, dst) &&
        nb00 == type_size && nb0 == type_size) {
        // copy by rows
        மாறிலி size_t rs = ggml_row_size(src0->type, ne00);
        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                    memcpy(
                        ((எழுத்து *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03),
                        rs);
                }
            }
        }
        திரும்பு;
    }

    எனில் (ggml_is_contiguous(dst)) {
        size_t id = 0;
        எழுத்து * dst_ptr = (எழுத்து *) dst->data;
        மாறிலி size_t rs = ne00 * type_size;

        எனில் (nb00 == type_size) {
            // src0 is contiguous on first dimension, copy by rows
            க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
                க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                        மாறிலி எழுத்து * src0_ptr = (எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        memcpy(dst_ptr + id, src0_ptr, rs);
                        id += rs;
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        } இல்லையெனில் {
            //printf("%s: this is not optimal - fix me\n", __func__);

            க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
                க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                    id += rs * ir0;
                    க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                        க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                            மாறிலி எழுத்து * src0_ptr = (எழுத்து *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;
                            memcpy(dst_ptr + id, src0_ptr, type_size);

                            id += type_size;
                        }
                    }
                    id += rs * (ne01 - ir1);
                }
            }
        }

        திரும்பு;
    }

    // dst counters
    int64_t k10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    // number of blocks in a row
    மாறிலி int64_t nk00 = ne00 / ggml_blck_size(src0->type);
    மாறிலி int64_t nk0  = ne0  / ggml_blck_size(dst->type);

    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            k10 += nk00 * ir0;
            போது (k10 >= nk0) {
                k10 -= nk0;
                எனில் (++i11 == ne1) {
                    i11 = 0;
                    எனில் (++i12 == ne2) {
                        i12 = 0;
                        எனில் (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
            க்கு (int64_t i01 = ir0; i01 < ir1; i01++) {
                க்கு (int64_t k00 = 0; k00 < nk00; k00++) {
                    மாறிலி எழுத்து * src0_ptr = ((எழுத்து *) src0->data + k00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                          எழுத்து * dst_ptr  = ((எழுத்து *)  dst->data + k10*nb0  + i11*nb1  + i12*nb2  + i13*nb3);

                    memcpy(dst_ptr, src0_ptr, type_size);

                    எனில் (++k10 == nk0) {
                        k10 = 0;
                        எனில் (++i11 == ne1) {
                            i11 = 0;
                            எனில் (++i12 == ne2) {
                                i12 = 0;
                                எனில் (++i13 == ne3) {
                                    i13 = 0;
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (ne01 - ir1);
            போது (k10 >= nk0) {
                k10 -= nk0;
                எனில் (++i11 == ne1) {
                    i11 = 0;
                    எனில் (++i12 == ne2) {
                        i12 = 0;
                        எனில் (++i13 == ne3) {
                            i13 = 0;
                        }
                    }
                }
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_dup_from_q(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி ggml_type type = src0->type;
    ggml_to_float_t மாறிலி dequantize_row_q = ggml_get_type_traits(type)->to_float;

    size_t qk = ggml_blck_size(type);
    மாறிலி int64_t nr = ggml_nelements(src1) / qk;

    // destination must be contiguous in the first dimension
    GGML_ASSERT(nb10 == ggml_type_size(dst->type));
    // must either have first dimension large enough to hold a row, or fully contiguous
    GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {

        uint32_t i = ir * qk;

        மாறிலி int64_t i03 = i/(ne00 * ne01 * ne02);
        மாறிலி int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
        மாறிலி int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
        மாறிலி int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
        மாறிலி int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

        மாறிலி int64_t i13 = i/(ne10 * ne11 * ne12);
        மாறிலி int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
        மாறிலி int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
        மாறிலி int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
        மாறிலி int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

        dequantize_row_q(
                (மாறிலி வெறுமை *) ((எழுத்து *) src0->data + x_offset),
                     (தசம *) ((எழுத்து *)  dst->data + dst_offset), qk);
    }
}

வெறுமை ggml_compute_forward_dup(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (src0->type == dst->type) {
        ggml_compute_forward_dup_bytes(params, dst);
        திரும்பு;
    }

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
            {
                /**/ எனில் (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_fp16_t>(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, தசம      >(params, dst);
                இல்லையெனில் ggml_compute_forward_dup_to_q<ggml_fp16_t>(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                /**/ எனில் (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_bf16_t>(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, தசம      >(params, dst);
                இல்லையெனில் ggml_compute_forward_dup_to_q<ggml_bf16_t>(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                /**/ எனில் (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<தசம, ggml_fp16_t>(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<தசம, ggml_bf16_t>(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<தசம, தசம      >(params, dst);
                இல்லையெனில் எனில் (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<தசம, int32_t    >(params, dst);
                இல்லையெனில் ggml_compute_forward_dup_to_q<தசம>(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                எனில் (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, தசம>(params, dst);
                இல்லையெனில் GGML_ABORT("not implemented");
            } நிறுத்து;
        இயல்பு:
            {
                எனில் (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
                    ggml_compute_forward_dup_from_q(params, dst);
                    நிறுத்து;
                }
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add

நிலையான வெறுமை ggml_compute_forward_add_q_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி ggml_type type = src0->type;
    மாறிலி ggml_type dtype = dst->type;
    ggml_to_float_t மாறிலி dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t மாறிலி quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == அளவு(தசம));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ggml_is_quantized(src0->type));
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    தசம * wdata = (தசம *) params->wdata + (ne00 + CACHE_வரி_SIZE_F32) * ith;

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        மாறிலி முழு i03 = ir/(ne02*ne01);
        மாறிலி முழு i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி முழு i01 = (ir - i03*ne02*ne01 - i02*ne01);

        // src1 and dst are same shape as src0 => same indices
        மாறிலி முழு i13 = i03;
        மாறிலி முழு i12 = i02;
        மாறிலி முழு i11 = i01;

        மாறிலி முழு i3 = i03;
        மாறிலி முழு i2 = i02;
        மாறிலி முழு i1 = i01;

        வெறுமை  * src0_row = (வெறுமை *) ((எழுத்து *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
        தசம * src1_row = (தசம *)((எழுத்து *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13));
        வெறுமை  * dst_row  = (வெறுமை *) ((எழுத்து *)  dst->data + ( i1*nb1  +  i2*nb2  +  i3*nb3));

        ப்ரதிஜ்ஞா(ne00 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne00);
        // add src1
        ggml_vec_acc_f32(ne00, wdata, src1_row);
        // quantize row to dst
        எனில் (quantize_row_q != வெற்று) {
            quantize_row_q(wdata, dst_row, ne00);
        } இல்லையெனில் {
            memcpy(dst_row, wdata, ne0*nb0);
        }
    }
}

வெறுமை ggml_compute_forward_add(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_add_non_quantized(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add_q_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_id

நிலையான வெறுமை ggml_compute_forward_add_id_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(src1->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_TERNARY_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(தசம));
    GGML_ASSERT(nb10 == அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

        // src1 indices
        மாறிலி முழு i11 = *(int32_t *) ((எழுத்து *) src2->data + i1*nb20 + i2*nb21);

        GGML_ASSERT(i11 >= 0 && i11 < ne11);

        ggml_vec_add_f32(ne0,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
                (தசம *) ((எழுத்து *) src1->data + i11*nb11));
    }
}

வெறுமை ggml_compute_forward_add_id(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_add_id_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_add1

நிலையான வெறுமை ggml_compute_forward_add1_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(தசம));
    GGML_ASSERT(nb00 == அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

#ifdef GGML_USE_ACCELERATE
        GGML_UNUSED(ggml_vec_add1_f32);

        vDSP_vadd(
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01), 1,
                (தசம *) ((எழுத்து *) src1->data), 0,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ), 1,
                ne0);
#இல்லையெனில்
        ggml_vec_add1_f32(ne0,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 ),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01),
               *(தசம *) src1->data);
#endif
    }
}

நிலையான வெறுமை ggml_compute_forward_add1_f16_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    மாறிலி தசம v = *(தசம *) src1->data;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == அளவு(ggml_fp16_t));
    GGML_ASSERT(nb00 == அளவு(ggml_fp16_t));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        க்கு (முழு i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_add1_f16_f16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    மாறிலி தசம v = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *) src1->data);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type  == GGML_TYPE_F16);

    GGML_ASSERT( nb0 == அளவு(ggml_fp16_t));
    GGML_ASSERT(nb00 == அளவு(ggml_fp16_t));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        க்கு (முழு i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_add1_q_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    மாறிலி தசம v = *(தசம *) src1->data;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி ggml_type type = src0->type;
    ggml_to_float_t மாறிலி dequantize_row_q = ggml_get_type_traits(type)->to_float;
    ggml_from_float_t மாறிலி quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

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
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    தசம * wdata = (தசம *) params->wdata + (ne0 + CACHE_வரி_SIZE_F32) * ith;

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

        வெறுமை  * src0_row = (வெறுமை *) ((எழுத்து *) src0->data + (i1*nb01 + i2*nb02 + i3*nb03));
        வெறுமை  * dst_row  = (வெறுமை *) ((எழுத்து *)  dst->data + (i1*nb1  + i2*nb2  + i3*nb0 ));

        ப்ரதிஜ்ஞா(ne0 % 32 == 0);

        // unquantize row from src0 to temp buffer
        dequantize_row_q(src0_row, wdata, ne0);
        // add src1
        ggml_vec_acc1_f32(ne0, wdata, v);
        // quantize row to dst
        quantize_row_q(wdata, dst_row, ne0);
    }
}

நிலையான வெறுமை ggml_compute_forward_add1_bf16_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    மாறிலி தசம v = *(தசம *) src1->data;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == அளவு(ggml_bf16_t));
    GGML_ASSERT(nb00 == அளவு(ggml_bf16_t));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        க்கு (முழு i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_add1_bf16_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_scalar(src1));

    // scalar to add
    மாறிலி தசம v = GGML_BF16_TO_FP32(*(ggml_bf16_t *) src1->data);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type  == GGML_TYPE_BF16);

    GGML_ASSERT( nb0 == அளவு(ggml_bf16_t));
    GGML_ASSERT(nb00 == அளவு(ggml_bf16_t));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are same shape => same indices
        மாறிலி முழு i3 = ir/(ne2*ne1);
        மாறிலி முழு i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி முழு i1 = (ir - i3*ne2*ne1 - i2*ne1);

        ggml_bf16_t * dst_ptr  = (ggml_bf16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1 );
        ggml_bf16_t * src0_ptr = (ggml_bf16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
        க்கு (முழு i = 0; i < ne0; i++) {
            dst_ptr[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src0_ptr[i]) + v);
        }
    }
}

வெறுமை ggml_compute_forward_add1(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_add1_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                எனில் (src1->type == GGML_TYPE_F16) {
                    ggml_compute_forward_add1_f16_f16(params, dst);
                }
                இல்லையெனில் எனில் (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_f16_f32(params, dst);
                }
                இல்லையெனில் {
                    GGML_ABORT("fatal error");
                }
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                எனில் (src1->type == GGML_TYPE_BF16) {
                    ggml_compute_forward_add1_bf16_bf16(params, dst);
                }
                இல்லையெனில் எனில் (src1->type == GGML_TYPE_F32) {
                    ggml_compute_forward_add1_bf16_f32(params, dst);
                }
                இல்லையெனில் {
                    GGML_ABORT("fatal error");
                }
            } நிறுத்து;
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_Q8_1:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_add1_q_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_acc

நிலையான வெறுமை ggml_compute_forward_acc_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    தருக்க   inplace = (தருக்க) ((int32_t *) dst->op_params)[4];

    எனில் (!inplace) {
        எனில் (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((எழுத்து *)  dst->data),
                ((எழுத்து *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr = ggml_nrows(src1);
    மாறிலி முழு nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during acc
    மாறிலி size_t nb0 = ggml_element_size(src0);

    மாறிலி size_t nb00 = nb0;
    மாறிலி size_t nb01 = nb1;
    மாறிலி size_t nb02 = nb2;
    மாறிலி size_t nb03 = nb3;

    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb0  + (ne11 == 0 ? 0 : ne11-1)*nb1  + (ne12 == 0 ? 0 : ne12-1)*nb2  + (ne13 == 0 ? 0 : ne13-1)*nb3  < ggml_nbytes(dst));
    GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10-1)*nb00 + (ne11 == 0 ? 0 : ne11-1)*nb01 + (ne12 == 0 ? 0 : ne12-1)*nb02 + (ne13 == 0 ? 0 : ne13-1)*nb03 < ggml_nbytes(src0));

    GGML_ASSERT(nb10 == அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        மாறிலி முழு i3 = ir/(ne12*ne11);
        மாறிலி முழு i2 = (ir - i3*ne12*ne11)/ne11;
        மாறிலி முழு i1 = (ir - i3*ne12*ne11 - i2*ne11);

#ifdef GGML_USE_ACCELERATE
        vDSP_vadd(
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset), 1,
                (தசம *) ((எழுத்து *) src1->data + i3*nb13 + i2*nb12 + i1*nb11), 1,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1  + offset), 1, nc);
#இல்லையெனில்
        ggml_vec_add_f32(nc,
                (தசம *) ((எழுத்து *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + offset),
                (தசம *) ((எழுத்து *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
#endif
    }
}

வெறுமை ggml_compute_forward_acc(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_acc_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_BF16:
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_Q8_1:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum

நிலையான வெறுமை ggml_compute_forward_sum_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(ggml_is_scalar(dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    ggml_float sum = 0;

    எனில் (nb00 == அளவு(தசம)) {
        ggml_float row_sum = 0;
        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    ggml_vec_sum_f32_ggf(ne00,
                            &row_sum,
                            (தசம *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
                    sum += row_sum;
                }
            }
        }
    } இல்லையெனில் {
        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    மாறிலி எழுத்து * row = (மாறிலி எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float) *(மாறிலி தசம *) (row + i00*nb00);
                    }
                }
            }
        }
    }

    ((தசம *) dst->data)[0] = sum;
}

நிலையான வெறுமை ggml_compute_forward_sum_f16(
    மாறிலி ggml_compute_params * params,
          ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(ggml_is_scalar(dst));

    ப்ரதிஜ்ஞா(src0->nb[0] == அளவு(ggml_fp16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    தசம sum = 0;
    தசம row_sum = 0;

    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f16_ggf(ne00,
                    &row_sum,
                    (ggml_fp16_t *) ((எழுத்து *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_fp16_t *) dst->data)[0] = GGML_CPU_FP32_TO_FP16(sum);
}

நிலையான வெறுமை ggml_compute_forward_sum_bf16(
    மாறிலி ggml_compute_params * params,
          ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(ggml_is_scalar(dst));

    ப்ரதிஜ்ஞா(src0->nb[0] == அளவு(ggml_bf16_t));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)

    தசம sum = 0;
    தசம row_sum = 0;

    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_bf16_ggf(ne00,
                    &row_sum,
                    (ggml_bf16_t *) ((எழுத்து *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
                sum += row_sum;
            }
        }
    }
    ((ggml_bf16_t *) dst->data)[0] = GGML_FP32_TO_BF16(sum);
}

வெறுமை ggml_compute_forward_sum(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_sum_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_sum_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cumsum

நிலையான வெறுமை ggml_compute_forward_cumsum_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(dst->nb[0] == அளவு(தசம));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    மாறிலி auto [ir0, ir1] = get_thread_range(params, src0);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        தசம * src_row = (தசம *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
        தசம * dst_row = (தசம *) ((எழுத்து *) dst->data  + i01*nb1  + i02*nb2  + i03*nb3);

        ggml_vec_cumsum_f32(ne00, dst_row, src_row);
    }
}

வெறுமை ggml_compute_forward_cumsum(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_cumsum_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_sum_rows

நிலையான வெறுமை ggml_compute_forward_sum_rows_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(dst->nb[0] == அளவு(தசம));

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne0 == 1);
    GGML_ASSERT(ne1 == ne01);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    க்கு (int64_t i3 = 0; i3 < ne03; i3++) {
        க்கு (int64_t i2 = 0; i2 < ne02; i2++) {
            க்கு (int64_t i1 = 0; i1 < ne01; i1++) {
                தசம * src_row = (தசம *) ((எழுத்து *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);
                தசம * dst_row = (தசம *) ((எழுத்து *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3);
                தசம row_sum = 0;
                ggml_vec_sum_f32(ne00, &row_sum, src_row);
                dst_row[0] = row_sum;
            }
        }
    }
}

வெறுமை ggml_compute_forward_sum_rows(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_sum_rows_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_mean

நிலையான வெறுமை ggml_compute_forward_mean_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(src0->nb[0] == அளவு(தசம));

    GGML_TENSOR_UNARY_OP_LOCALS

    ப்ரதிஜ்ஞா(ne0 == 1);
    ப்ரதிஜ்ஞா(ne1 == ne01);
    ப்ரதிஜ்ஞா(ne2 == ne02);
    ப்ரதிஜ்ஞா(ne3 == ne03);

    GGML_UNUSED(ne0);
    GGML_UNUSED(ne1);
    GGML_UNUSED(ne2);
    GGML_UNUSED(ne3);

    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (தசம *) ((எழுத்து *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (தசம *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(தசம *) ((எழுத்து *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (தசம) ne00;
            }
        }
    }
}

வெறுமை ggml_compute_forward_mean(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argmax

நிலையான வெறுமை ggml_compute_forward_argmax_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(src0->nb[0] == அளவு(தசம));
    ப்ரதிஜ்ஞா(dst->nb[0] == அளவு(தசம));

    மாறிலி int64_t ne00 = src0->ne[0];
    மாறிலி int64_t ne01 = src0->ne[1];

    மாறிலி size_t nb01 = src0->nb[1];
    மாறிலி size_t nb0 = dst->nb[0];

    க்கு (int64_t i1 = 0; i1 < ne01; i1++) {
        தசம * src = (தசம *) ((எழுத்து *) src0->data + i1*nb01);
        int32_t * dst_ = (int32_t *) ((எழுத்து *)  dst->data + i1*nb0);
        முழு v = 0;
        ggml_vec_argmax_f32(ne00, &v, src);
        dst_[0] = v;
    }
}

வெறுமை ggml_compute_forward_argmax(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_argmax_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_count_equal

நிலையான வெறுமை ggml_compute_forward_மொத்தம்_equal_i32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);

    மாறிலி int64_t nr = ggml_nrows(src0);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    int64_t * sums = (int64_t *) params->wdata;
    int64_t sum_thread = 0;

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 =  ir                        / (ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne03)            /       ne01;
        மாறிலி int64_t i01 =  ir - i03*ne03 - i02*ne02;

        மாறிலி எழுத்து * data0 = (மாறிலி எழுத்து *) src0->data + i03*nb03 + i02*nb02 + i01*nb01;
        மாறிலி எழுத்து * data1 = (மாறிலி எழுத்து *) src1->data + i03*nb13 + i02*nb12 + i01*nb11;

        க்கு (int64_t i00 = 0; i00 < ne00; ++i00) {
            மாறிலி int32_t val0 = *((மாறிலி int32_t *) (data0 + i00*nb00));
            மாறிலி int32_t val1 = *((மாறிலி int32_t *) (data1 + i00*nb10));

            sum_thread += val0 == val1;
        }
    }
    எனில் (ith != 0) {
        sums[ith] = sum_thread;
    }
    ggml_barrier(params->threadpool);

    எனில் (ith != 0) {
        திரும்பு;
    }

    க்கு (முழு ith_other = 1; ith_other < nth; ++ith_other) {
        sum_thread += sums[ith_other];
    }
    *((int64_t *) dst->data) = sum_thread;
}

வெறுமை ggml_compute_forward_மொத்தம்_equal(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_I32:
            {
                ggml_compute_forward_மொத்தம்_equal_i32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat

நிலையான வெறுமை ggml_compute_forward_repeat_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    மாறிலி முழு nr0 = (முழு)(ne0/ne00);
    மாறிலி முழு nr1 = (முழு)(ne1/ne01);
    மாறிலி முழு nr2 = (முழு)(ne2/ne02);
    மாறிலி முழு nr3 = (முழு)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == அளவு(தசம));
    GGML_ASSERT(nb00 == அளவு(தசம));

    // TODO: maybe this is not optimal?
    க்கு                         (முழு i3 = 0; i3 < nr3;  i3++) {
        க்கு                     (முழு k3 = 0; k3 < ne03; k3++) {
            க்கு                 (முழு i2 = 0; i2 < nr2;  i2++) {
                க்கு             (முழு k2 = 0; k2 < ne02; k2++) {
                    க்கு         (முழு i1 = 0; i1 < nr1;  i1++) {
                        க்கு     (முழு k1 = 0; k1 < ne01; k1++) {
                            க்கு (முழு i0 = 0; i0 < nr0;  i0++) {
                                ggml_vec_cpy_f32(ne00,
                                        (தசம *) ((எழுத்து *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0),
                                        (தசம *) ((எழுத்து *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01));
                            }
                        }
                    }
                }
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_repeat_f16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    GGML_ASSERT(ggml_can_repeat(src0, dst));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    மாறிலி முழு nr0 = (முழு)(ne0/ne00);
    மாறிலி முழு nr1 = (முழு)(ne1/ne01);
    மாறிலி முழு nr2 = (முழு)(ne2/ne02);
    மாறிலி முழு nr3 = (முழு)(ne3/ne03);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == அளவு(ggml_fp16_t));
    GGML_ASSERT(nb00 == அளவு(ggml_fp16_t));

    // TODO: maybe this is not optimal?
    க்கு                         (முழு i3 = 0; i3 < nr3;  i3++) {
        க்கு                     (முழு k3 = 0; k3 < ne03; k3++) {
            க்கு                 (முழு i2 = 0; i2 < nr2;  i2++) {
                க்கு             (முழு k2 = 0; k2 < ne02; k2++) {
                    க்கு         (முழு i1 = 0; i1 < nr1;  i1++) {
                        க்கு     (முழு k1 = 0; k1 < ne01; k1++) {
                            க்கு (முழு i0 = 0; i0 < nr0;  i0++) {
                                ggml_fp16_t * y = (ggml_fp16_t *) ((எழுத்து *)  dst->data + (i3*ne03 + k3)*nb3  + (i2*ne02 + k2)*nb2  + (i1*ne01 + k1)*nb1  + (i0*ne00)*nb0);
                                ggml_fp16_t * x = (ggml_fp16_t *) ((எழுத்து *) src0->data + (          k3)*nb03 + (          k2)*nb02 + (          k1)*nb01);
                                // ggml_vec_cpy_f16(ne00, y, x)
                                க்கு (முழு i = 0; i < ne00; ++i) {
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

வெறுமை ggml_compute_forward_repeat(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_BF16:
        நிலை GGML_TYPE_I16:
            {
                ggml_compute_forward_repeat_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
        நிலை GGML_TYPE_I32:
            {
                ggml_compute_forward_repeat_f32(params, dst);
            } நிறுத்து;
        // TODO: templateify the implementation and support for I64
        //       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
        //case GGML_TYPE_I64:
        //    {
        //        ggml_compute_forward_repeat_i64(params, dst);
        //    } break;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_repeat_back

நிலையான வெறுமை ggml_compute_forward_repeat_back_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    GGML_ASSERT(ggml_can_repeat(dst, src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    // guaranteed to be an integer due to the check in ggml_can_repeat
    மாறிலி முழு nr0 = (முழு)(ne00/ne0);
    மாறிலி முழு nr1 = (முழு)(ne01/ne1);
    மாறிலி முழு nr2 = (முழு)(ne02/ne2);
    மாறிலி முழு nr3 = (முழு)(ne03/ne3);

    // TODO: support for transposed / permuted tensors
    GGML_ASSERT(nb0  == அளவு(தசம));
    GGML_ASSERT(nb00 == அளவு(தசம));

    எனில் (ggml_is_contiguous(dst)) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (தசம *)dst->data, 0);
    } இல்லையெனில் {
        க்கு         (முழு k3 = 0; k3 < ne3; k3++) {
            க்கு     (முழு k2 = 0; k2 < ne2; k2++) {
                க்கு (முழு k1 = 0; k1 < ne1; k1++) {
                    ggml_vec_set_f32(ne0,
                        (தசம *) ((எழுத்து *) dst->data + k1*nb1 + k2*nb2 + k3*nb3),
                        0);
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    க்கு                         (முழு i3 = 0; i3 < nr3; i3++) {
        க்கு                     (முழு k3 = 0; k3 < ne3; k3++) {
            க்கு                 (முழு i2 = 0; i2 < nr2; i2++) {
                க்கு             (முழு k2 = 0; k2 < ne2; k2++) {
                    க்கு         (முழு i1 = 0; i1 < nr1; i1++) {
                        க்கு     (முழு k1 = 0; k1 < ne1; k1++) {
                            க்கு (முழு i0 = 0; i0 < nr0; i0++) {
                                ggml_vec_acc_f32(ne0,
                                        (தசம *) ((எழுத்து *)  dst->data + (         k3)*nb3  + (         k2)*nb2  + (         k1)*nb1),
                                        (தசம *) ((எழுத்து *) src0->data + (i3*ne3 + k3)*nb03 + (i2*ne2 + k2)*nb02 + (i1*ne1 + k1)*nb01 + (i0*ne0)*nb00));
                            }
                        }
                    }
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_repeat_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_repeat_back_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_concat
நிலையான வெறுமை ggml_compute_forward_concat_any(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous_rows(src1));

    int64_t o[4] = {0, 0, 0, 0};

    எனில் (dim == 0) {
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        o[dim] = src0->ne[dim]/ggml_blck_size(src0->type);
    } இல்லையெனில் {
        o[dim] = src0->ne[dim];
    }

    // Region 1: copy rows from src0
    க்கு (முழு i3 = 0; i3 < ne03; i3++) {
        க்கு (முழு i2 = ith; i2 < ne02; i2 += nth) {
            க்கு (முழு i1 = 0; i1 < ne01; i1++) {
                மாறிலி எழுத்து * x = (மாறிலி எழுத்து *) src0->data + i1*nb01 + i2*nb02 + i3*nb03;
                      எழுத்து * y = (      எழுத்து *) dst->data  + i1*nb1  + i2*nb2  + i3*nb3;
                memcpy(y, x, ggml_row_size(src0->type, ne00));
            }
        }
    }

    // Region 2: copy rows from src1, offset into dst by o[]
    க்கு (முழு i3 = 0; i3 < ne13; i3++) {
        க்கு (முழு i2 = ith; i2 < ne12; i2 += nth) {
            க்கு (முழு i1 = 0; i1 < ne11; i1++) {
                மாறிலி எழுத்து * x = (மாறிலி எழுத்து *) src1->data + i1*nb11         + i2*nb12         + i3*nb13;
                      எழுத்து * y = (      எழுத்து *)  dst->data + (i1 + o[1])*nb1 + (i2 + o[2])*nb2 + (i3 + o[3])*nb3 + o[0]*nb0;
                memcpy(y, x, ggml_row_size(src1->type, ne10));
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_concat_i8(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == அளவு(int8_t));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    மாறிலி int8_t * x;

    // TODO: smarter multi-theading
    க்கு (முழு i3 = 0; i3 < ne3; i3++) {
        க்கு (முழு i2 = ith; i2 < ne2; i2 += nth) {
            க்கு (முழு i1 = 0; i1 < ne1; i1++) {
                க்கு (முழு i0 = 0; i0 < ne0; i0++) {
                    எனில் (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (மாறிலி int8_t *) ((மாறிலி எழுத்து *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } இல்லையெனில் {
                        x = (மாறிலி int8_t *) ((மாறிலி எழுத்து *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    int8_t * y = (int8_t *)((எழுத்து *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_concat_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == அளவு(ggml_fp16_t));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    மாறிலி ggml_fp16_t * x;

    // TODO: smarter multi-theading
    க்கு (முழு i3 = 0; i3 < ne3; i3++) {
        க்கு (முழு i2 = ith; i2 < ne2; i2 += nth) {
            க்கு (முழு i1 = 0; i1 < ne1; i1++) {
                க்கு (முழு i0 = 0; i0 < ne0; i0++) {
                    எனில் (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (மாறிலி ggml_fp16_t *) ((மாறிலி எழுத்து *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } இல்லையெனில் {
                        x = (மாறிலி ggml_fp16_t *) ((மாறிலி எழுத்து *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    ggml_fp16_t * y = (ggml_fp16_t *)((எழுத்து *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_concat_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int32_t dim = ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = {0, 0, 0, 0};
    o[dim] = src0->ne[dim];

    மாறிலி தசம * x;

    // TODO: smarter multi-theading
    க்கு (முழு i3 = 0; i3 < ne3; i3++) {
        க்கு (முழு i2 = ith; i2 < ne2; i2 += nth) {
            க்கு (முழு i1 = 0; i1 < ne1; i1++) {
                க்கு (முழு i0 = 0; i0 < ne0; i0++) {
                    எனில் (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        x = (மாறிலி தசம *) ((மாறிலி எழுத்து *)src0->data + (i0       )*nb00 + (i1       )*nb01 + (i2       )*nb02 + (i3       )*nb03);
                    } இல்லையெனில் {
                        x = (மாறிலி தசம *) ((மாறிலி எழுத்து *)src1->data + (i0 - o[0])*nb10 + (i1 - o[1])*nb11 + (i2 - o[2])*nb12 + (i3 - o[3])*nb13);
                    }

                    தசம * y = (தசம *)((எழுத்து *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);

                    *y = *x;
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_concat(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_BF16:
        நிலை GGML_TYPE_I16:
            {
                ggml_compute_forward_concat_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_I8:
            {
                ggml_compute_forward_concat_i8(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
        நிலை GGML_TYPE_I32:
            {
                ggml_compute_forward_concat_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                ggml_compute_forward_concat_any(params, dst);
            }
    }
}

// row loop shared by the BF16 unary ops
template <வெறுமை (*vec_op)(மாறிலி முழு, ggml_bf16_t *, மாறிலி ggml_bf16_t *)>
நிலையான வெறுமை ggml_compute_forward_unary_bf16_rows(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        vec_op(nc,
                (ggml_bf16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_bf16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));
    }
}

// ggml_compute_forward_gelu

நிலையான வெறுமை ggml_compute_forward_gelu_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_gelu_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_f16(nc,
                (ggml_fp16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_gelu_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_gelu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_fill

நிலையான வெறுமை ggml_compute_forward_fill_f32(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி தசம c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    மாறிலி auto [ir0, ir1] = get_thread_range(params, dst);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne2*ne1);
        மாறிலி int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        மாறிலி int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        தசம * dst_ptr  = (தசம *) ((எழுத்து *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

நிலையான வெறுமை ggml_compute_forward_fill_f16(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    மாறிலி auto [ir0, ir1] = get_thread_range(params, dst);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne2*ne1);
        மாறிலி int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        மாறிலி int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((எழுத்து *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

வெறுமை ggml_compute_forward_fill(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_tri

நிலையான வெறுமை ggml_compute_forward_tri_f32(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    மாறிலி ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி auto [ir0, ir1] = get_thread_range(params, src0);

    தருக்க (*bipred)(முழு, முழு);

    தேர்வு (ttype) {
        நிலை GGML_TRI_TYPE_LOWER:      bipred = [](முழு i, முழு r) { திரும்பு i <  r; }; நிறுத்து;
        நிலை GGML_TRI_TYPE_LOWER_DIAG: bipred = [](முழு i, முழு r) { திரும்பு i <= r; }; நிறுத்து;
        நிலை GGML_TRI_TYPE_UPPER:      bipred = [](முழு i, முழு r) { திரும்பு i >  r; }; நிறுத்து;
        நிலை GGML_TRI_TYPE_UPPER_DIAG: bipred = [](முழு i, முழு r) { திரும்பு i >= r; }; நிறுத்து;
        இயல்பு: GGML_ABORT("invalid tri type");
    }

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        மாறிலி தசம * src_ptr = (மாறிலி தசம  *) ((மாறிலி எழுத்து *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
              தசம * dst_ptr = (      தசம  *) ((      எழுத்து *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1);

        க்கு (முழு i0 = 0; i0 < ne0; ++i0) {
            dst_ptr[i0] = bipred(i0, i01) ? src_ptr[i0] : 0.0f;
        }
    }
}

வெறுமை ggml_compute_forward_tri(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_tri_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_erf

நிலையான வெறுமை ggml_compute_forward_gelu_erf_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_gelu_erf_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_erf_f16(nc,
                (ggml_fp16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_gelu_erf_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_erf_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_gelu_erf(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_erf_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_erf_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_erf_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gelu_quick

நிலையான வெறுமை ggml_compute_forward_gelu_quick_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_gelu_quick_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_gelu_quick_f16(nc,
                (ggml_fp16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_gelu_quick_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_gelu_quick_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_gelu_quick(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_gelu_quick_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_gelu_quick_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_gelu_quick_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu

நிலையான வெறுமை ggml_compute_forward_silu_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*(dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_silu_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_rows(src0));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி முழு i3 = ir/(ne02*ne01);
        மாறிலி முழு i2 = (ir - i3*ne02*ne01)/ne01;
        மாறிலி முழு i1 = (ir - i3*ne02*ne01 - i2*ne01);

        ggml_vec_silu_f16(nc,
                (ggml_fp16_t *) ((எழுத்து *) dst->data  + i3*nb3  + i2*nb2  + i1*nb1),
                (ggml_fp16_t *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_silu_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_unary_bf16_rows<ggml_vec_silu_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_silu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_silu_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}
// ggml_compute_forward_leaky_relu

நிலையான வெறுமை ggml_compute_forward_leaky_relu_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(src0));
    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    மாறிலி முழு n  = ggml_nrows(src0);
    மாறிலி முழு nc = src0->ne[0];

    தசம negative_slope;
    memcpy(&negative_slope, dst->op_params, அளவு(தசம));

    ப்ரதிஜ்ஞா(dst->nb[0]  == அளவு(தசம));
    ப்ரதிஜ்ஞா(src0->nb[0] == அளவு(தசம));

    க்கு (முழு i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i*( dst->nb[1])),
                (தசம *) ((எழுத்து *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

நிலையான வெறுமை ggml_compute_forward_leaky_relu_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(src0));
    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    மாறிலி முழு n  = ggml_nrows(src0);
    மாறிலி முழு nc = src0->ne[0];

    தசம negative_slope;
    memcpy(&negative_slope, dst->op_params, அளவு(தசம));

    ப்ரதிஜ்ஞா(dst->nb[0]  == அளவு(ggml_fp16_t));
    ப்ரதிஜ்ஞா(src0->nb[0] == அளவு(ggml_fp16_t));

    க்கு (முழு i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f16(nc,
                (ggml_fp16_t *) ((எழுத்து *) dst->data  + i*( dst->nb[1])),
                (ggml_fp16_t *) ((எழுத்து *) src0->data + i*(src0->nb[1])), negative_slope);
    }
}

வெறுமை ggml_compute_forward_leaky_relu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_leaky_relu_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_leaky_relu_f16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_silu_back

நிலையான வெறுமை ggml_compute_forward_silu_back_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * grad = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(grad));
    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(src1));
    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src1, dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src1, grad));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1->ne[0];
    மாறிலி முழு nr = ggml_nrows(src1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i1*( dst->nb[1])),
                (தசம *) ((எழுத்து *) src1->data + i1*(src1->nb[1])),
                (தசம *) ((எழுத்து *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_silu_back_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * grad = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(grad));
    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(src1));
    ப்ரதிஜ்ஞா(ggml_is_contiguous_1(dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src1, dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src1, grad));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1->ne[0];
    மாறிலி முழு nr = ggml_nrows(src1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_backward_f16(nc,
                (ggml_fp16_t *) ((எழுத்து *) dst->data  + i1*( dst->nb[1])),
                (ggml_fp16_t *) ((எழுத்து *) src1->data + i1*(src1->nb[1])),
                (ggml_fp16_t *) ((எழுத்து *) grad->data + i1*(grad->nb[1])));

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_CPU_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

வெறுமை ggml_compute_forward_silu_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_silu_back_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_silu_back_f16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_reglu

// row loop shared by the BF16 GLU ops
template <வெறுமை (*vec_op)(மாறிலி முழு, ggml_bf16_t *, மாறிலி ggml_bf16_t *, மாறிலி ggml_bf16_t *)>
நிலையான வெறுமை ggml_compute_forward_glu_bf16_rows(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        மாறிலி ggml_bf16_t * src0_p = (மாறிலி ggml_bf16_t *) (src0_d + i1*src0_o);
        மாறிலி ggml_bf16_t * src1_p = (மாறிலி ggml_bf16_t *) (src1_d + i1*src1_o);
        ggml_bf16_t       * dst_p  = (ggml_bf16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1]));

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        vec_op(nc, dst_p, src0_p, src1_p);
    }
}

நிலையான வெறுமை ggml_compute_forward_reglu_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1*src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f32(nc, (தசம *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_reglu_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_reglu_f16(nc, (ggml_fp16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_reglu_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_reglu_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_reglu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_reglu_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_reglu_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_reglu_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu

நிலையான வெறுமை ggml_compute_forward_geglu_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1*src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f32(nc, (தசம *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_geglu_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_f16(nc, (ggml_fp16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_geglu_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_geglu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu

நிலையான வெறுமை ggml_compute_forward_swiglu_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1*src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f32(nc, (தசம *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_swiglu_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_swiglu_f16(nc, (ggml_fp16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_swiglu_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_swiglu_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_swiglu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_swiglu_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_oai

நிலையான வெறுமை ggml_compute_forward_swiglu_oai_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);
    மாறிலி தசம alpha = ggml_get_op_params_f32(dst, 2);
    மாறிலி தசம limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1*src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1*src1_o);
        தசம * dst_p  = (தசம *) ((எழுத்து *) dst->data + i1*(dst->nb[1]));

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = std::min(src0_p[k], limit);
            மாறிலி தசம y = std::clamp(src1_p[k], -limit, limit);
            மாறிலி தசம out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = out_glu * (y + 1.f);
        }

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = dst_p[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_swiglu_oai_f16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);
    மாறிலி தசம alpha = ggml_get_op_params_f32(dst, 2);
    மாறிலி தசம limit = ggml_get_op_params_f32(dst, 3);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1]));

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = std::min(GGML_CPU_FP16_TO_FP32(src0_p[k]), limit);
            மாறிலி தசம y = std::clamp(GGML_CPU_FP16_TO_FP32(src1_p[k]), -limit, limit);
            மாறிலி தசம out_glu = x / (1.f + expf(alpha * (-x)));
            dst_p[k] = GGML_CPU_FP32_TO_FP16(out_glu * (y + 1.f));
        }

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = GGML_CPU_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_swiglu_oai(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_swiglu_oai_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_swiglu_oai_f16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_swiglu_clamp

நிலையான வெறுமை ggml_compute_forward_swiglu_clamp_f32(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0   = dst->src[0];
    மாறிலி ggml_tensor * src1   = dst->src[1];
    எழுத்து *              src0_d = (எழுத்து *) src0->data;
    எழுத்து *              src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t        src0_o = src0->nb[1];
    மாறிலி size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);
    மாறிலி தசம   limit   = ggml_get_op_params_f32(dst, 3);

    மாறிலி முழு dr  = (nr + nth - 1) / nth;
    மாறிலி முழு ir0 = dr * ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1 * src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1 * src1_o);
        தசம * dst_p  = (தசம *) ((எழுத்து *) dst->data + i1 * (dst->nb[1]));

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம gate = std::min(src0_p[k], limit);
            மாறிலி தசம up   = std::clamp(src1_p[k], -limit, limit);
            dst_p[k]         = gate / (1.f + expf(-gate)) * up;
        }

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = dst_p[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif  // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_swiglu_clamp_f16(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0   = dst->src[0];
    மாறிலி ggml_tensor * src1   = dst->src[1];
    எழுத்து *              src0_d = (எழுத்து *) src0->data;
    எழுத்து *              src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t        src0_o = src0->nb[1];
    மாறிலி size_t        src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);
    மாறிலி தசம   limit   = ggml_get_op_params_f32(dst, 3);

    மாறிலி முழு dr  = (nr + nth - 1) / nth;
    மாறிலி முழு ir0 = dr * ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1 * src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1 * src1_o);
        ggml_fp16_t * dst_p  = (ggml_fp16_t *) ((எழுத்து *) dst->data + i1 * (dst->nb[1]));

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம gate = std::min(GGML_FP16_TO_FP32(src0_p[k]), limit);
            மாறிலி தசம up   = std::clamp(GGML_FP16_TO_FP32(src1_p[k]), -limit, limit);
            dst_p[k]         = GGML_FP32_TO_FP16(gate / (1.f + expf(-gate)) * up);
        }

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = GGML_FP16_TO_FP32(dst_p[k]);
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif  // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_swiglu_clamp(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    தேர்வு (dst->src[0]->type) {
        நிலை GGML_TYPE_F32:
            ggml_compute_forward_swiglu_clamp_f32(params, dst);
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            ggml_compute_forward_swiglu_clamp_f16(params, dst);
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_geglu_erf

நிலையான வெறுமை ggml_compute_forward_geglu_erf_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1*src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f32(nc, (தசம *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_geglu_erf_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_erf_f16(nc, (ggml_fp16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_geglu_erf_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_erf_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_geglu_erf(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_erf_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_erf_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_erf_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_geglu_quick

நிலையான வெறுமை ggml_compute_forward_geglu_quick_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * src0_p = (தசம *) (src0_d + i1*src0_o);
        தசம * src1_p = (தசம *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f32(nc, (தசம *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி தசம x = ((தசம *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            GGML_UNUSED(x);
            ப்ரதிஜ்ஞா(!isnan(x));
            ப்ரதிஜ்ஞா(!isinf(x));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_geglu_quick_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    எழுத்து * src0_d = (எழுத்து *) src0->data;
    எழுத்து * src1_d = (எழுத்து *) (src1 ? src1->data : src0->data);
    மாறிலி size_t src0_o = src0->nb[1];
    மாறிலி size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));

    எனில் (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src0->type == src1->type);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == nr);

    மாறிலி int32_t swapped = ggml_get_op_params_i32(dst, 1);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        ggml_fp16_t * src0_p = (ggml_fp16_t *) (src0_d + i1*src0_o);
        ggml_fp16_t * src1_p = (ggml_fp16_t *) (src1_d + i1*src1_o);

        எனில் (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        ggml_vec_geglu_quick_f16(nc, (ggml_fp16_t *) ((எழுத்து *) dst->data + i1*(dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
        க்கு (முழு k = 0; k < nc; k++) {
            மாறிலி ggml_fp16_t x = ((ggml_fp16_t *) ((எழுத்து *) dst->data + i1*( dst->nb[1])))[k];
            மாறிலி தசம v = GGML_FP16_TO_FP32(x);
            GGML_UNUSED(v);
            ப்ரதிஜ்ஞா(!isnan(v));
            ப்ரதிஜ்ஞா(!isinf(v));
        }
#endif // NDEBUG
    }
}

நிலையான வெறுமை ggml_compute_forward_geglu_quick_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_glu_bf16_rows<ggml_vec_geglu_quick_bf16>(params, dst);
}

நிலையான வெறுமை ggml_compute_forward_geglu_quick(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_geglu_quick_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_geglu_quick_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_geglu_quick_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_norm

நிலையான வெறுமை ggml_compute_forward_norm_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));

    GGML_ASSERT(eps >= 0.0f);

    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                மாறிலி எழுத்து * x = (மாறிலி எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                எழுத்து * y = (எழுத்து *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                எனில் (nb00 == அளவு(தசம) && nb0 == அளவு(தசம)) {
                    மாறிலி தசம * xf = (மாறிலி தசம *) x;

                    தசம sum = 0.0;
                    ggml_vec_sum_f32(ne00, &sum, xf);
                    தசம mean = sum/ne00;

                    தசம * yf = (தசம *) y;
                    தசம variance = 0;

#ifdef GGML_USE_ACCELERATE
                    mean = -mean;
                    vDSP_vsadd(xf, 1, &mean, yf, 1, ne00);
                    vDSP_measqv(yf, 1, &variance, ne00);
#இல்லையெனில்
                    variance = ggml_vec_cvar_f32(ne00, yf, xf, mean);
#endif //GGML_USE_ACCELERATE

                    மாறிலி தசம scale = 1.0f/sqrtf(variance + eps);
                    ggml_vec_scale_f32(ne00, yf, scale);
                } இல்லையெனில் {
                    தசம sum = 0.0;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += *(மாறிலி தசம *) (x + i00*nb00);
                    }
                    மாறிலி தசம mean = sum/ne00;

                    தசம variance = 0.0f;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        மாறிலி தசம v = *(மாறிலி தசம *) (x + i00*nb00) - mean;
                        *(தசம *) (y + i00*nb0) = v;
                        variance += v * v;
                    }
                    variance /= ne00;

                    மாறிலி தசம scale = 1.0f/sqrtf(variance + eps);
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        *(தசம *) (y + i00*nb0) *= scale;
                    }
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_norm(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
எண்ணிக்கை ggml_rms_norm_fuse_op {
    GGML_RMS_NORM_FUSE_OP_NONE,
    GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
நிலையான வெறுமை ggml_compute_forward_rms_norm_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    மாறிலி ggml_tensor * src0 = dst_rms_norm->src[0];
    மாறிலி ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    எனில் constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_இருமம்_OP_LOCALS

    தசம eps;
    memcpy(&eps, dst_rms_norm->op_params, அளவு(தசம));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                மாறிலி தசம * x = (தசம *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                மாறிலி தசம mean  = sum/ne00;
                மாறிலி தசம scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                ப்ரதிஜ்ஞா(scale > 0.0f);

                தசம * y = (தசம *) ((எழுத்து *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                எனில் constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    மாறிலி int64_t i11 = i01 % ne11;
                    மாறிலி int64_t i12 = i02 % ne12;
                    மாறிலி int64_t i13 = i03 % ne13;
                    மாறிலி தசம * w = (தசம *) ((எழுத்து *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } இல்லையெனில் {
                    memcpy(y, x, ne00 * அளவு(தசம));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_rms_norm(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_NONE>(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
வெறுமை ggml_compute_forward_rms_norm_mul_fused(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_mul) {

    GGML_ASSERT(dst_mul != nullptr);
    GGML_ASSERT(dst_mul->src[0] == dst_rms_norm || dst_mul->src[1] == dst_rms_norm);

    மாறிலி ggml_tensor * src0 = dst_rms_norm->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32<GGML_RMS_NORM_FUSE_OP_MUL>(params, dst_rms_norm, dst_mul);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_compute_forward_rms_norm_back_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0]; // gradients from forward pass output
    மாறிலி ggml_tensor * src1 = dst->src[1]; // src1 from forward pass

    GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(src1->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_இருமம்_OP_LOCALS

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));

    // TODO: optimize
    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                // src1 is same shape as src0 => same indices
                மாறிலி int64_t i11 = i01;
                மாறிலி int64_t i12 = i02;
                மாறிலி int64_t i13 = i03;

                மாறிலி தசம * dz = (தசம *) ((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                மாறிலி தசம * x  = (தசம *) ((எழுத்து *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                ggml_float sum_xx  = 0.0;
                ggml_float sum_xdz = 0.0;

                க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum_xx  += (ggml_float)(x[i00] * x[i00]);
                    sum_xdz += (ggml_float)(x[i00] * dz[i00]);
                }

                //const float mean     = (float)(sum_xx)/ne00;
                மாறிலி தசம mean_eps = (தசம)(sum_xx)/ne00 + eps;
                மாறிலி தசம sum_eps  = (தசம)(sum_xx) + eps*ne00;
                //const float mean_xdz = (float)(sum_xdz)/ne00;
                // we could cache rms from forward pass to improve performance.
                // to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
                //const float rms      = sqrtf(mean_eps);
                மாறிலி தசம rrms     = 1.0f / sqrtf(mean_eps);
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
                தசம * dx = (தசம *) ((எழுத்து *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
                // note: https://github.com/ggml-org/ggml/issues/1491
                மாறிலி தசம scale_x = (தசம) (-sum_xdz) / sum_eps;
                க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                    dx[i00] = (dz[i00] + x[i00] * scale_x) * rrms;
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_rms_norm_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rms_norm_back_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_group_norm

நிலையான வெறுமை ggml_compute_forward_group_norm_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    // TODO: optimize

    தசம eps;
    memcpy(&eps, dst->op_params + 1, அளவு(தசம));

    முழு n_channels = src0->ne[2];
    முழு n_groups = dst->op_params[0];
    முழு n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
    க்கு (முழு i = ith; i < n_groups; i += nth) {
        முழு start = i * n_channels_per_group;
        முழு end = start + n_channels_per_group;
        எனில் (end > n_channels) {
            end = n_channels;
        }
        முழு step = end - start;

        க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
            ggml_float sum = 0.0;
            க்கு (int64_t i02 = start; i02 < end; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    மாறிலி தசம * x = (தசம *)((எழுத்து *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    ggml_float sumr = 0.0;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        sumr += (ggml_float)x[i00];
                    }
                    sum += sumr;
                }
            }
            மாறிலி தசம mean = sum / (ne00 * ne01 * step);

            ggml_float sum2 = 0.0;
            க்கு (int64_t i02 = start; i02 < end; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    மாறிலி தசம * x = (தசம *)((எழுத்து *) src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

                    தசம * y = (தசம *)((எழுத்து *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

                    ggml_float sumr = 0.0;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        தசம v = x[i00] - mean;
                        y[i00] = v;
                        sumr += (ggml_float)(v * v);
                    }
                    sum2 += sumr;
                }
            }
            மாறிலி தசம variance = sum2 / (ne00 * ne01 * step);
            மாறிலி தசம scale = 1.0f / sqrtf(variance + eps);

            க்கு (int64_t i02 = start; i02 < end; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    தசம * y = (தசம *)((எழுத்து *) dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_group_norm(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_group_norm_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_l2_norm

நிலையான வெறுமை ggml_compute_forward_l2_norm_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    தசம eps;
    memcpy(&eps, dst->op_params, அளவு(தசம));

    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                மாறிலி எழுத்து * x = (மாறிலி எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;

                ggml_float sum = 0.0;
                க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                    மாறிலி தசம xi = *(மாறிலி தசம *) (x + i00*nb00);
                    sum += (ggml_float)(xi * xi);
                }

                மாறிலி தசம scale = 1.0f/fmaxf(sqrtf(sum), eps);

                எழுத்து * y = (எழுத்து *) dst->data + i01*nb1 + i02*nb2 + i03*nb3;

                எனில் (nb00 == அளவு(தசம) && nb0 == அளவு(தசம)) {
                    memcpy(y, x, ne00 * அளவு(தசம));
                    ggml_vec_scale_f32(ne00, (தசம *) y, scale);
                } இல்லையெனில் {
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        மாறிலி தசம xi = *(மாறிலி தசம *) (x + i00*nb00);
                        *(தசம *) (y + i00*nb0) = xi * scale;
                    }
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_l2_norm(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_l2_norm_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_out_prod

நிலையான வெறுமை ggml_compute_forward_out_prod_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(ne2 % ne02 == 0);
    GGML_ASSERT(ne3 % ne03 == 0);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == அளவு(தசம));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    எனில் (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (தசம *)dst->data, 0);
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
    மாறிலி int64_t nr = ne1*ne2*ne3;

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    // block-tiling attempt
    மாறிலி int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
    மாறிலி int64_t blck_1 = 16;

    // dps == dst per src0, used for group query attention
    மாறிலி int64_t dps2 = ne2 / ne02;
    மாறிலி int64_t dps3 = ne3 / ne03;

    க்கு (int64_t bir = ir0; bir < ir1; bir += blck_1) {
        மாறிலி int64_t bir1 = MIN(bir + blck_1, ir1);
        க்கு (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
            மாறிலி int64_t bne01 = MIN(bi01 + blck_0, ne01);
            க்கு (int64_t ir = bir; ir < bir1; ++ir) {
                // dst indices
                மாறிலி int64_t i3 = ir/(ne2*ne1);
                மாறிலி int64_t i2 = (ir - i3*ne2*ne1)/ne1;
                மாறிலி int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

                மாறிலி int64_t i02 = i2 / dps2;
                மாறிலி int64_t i03 = i3 / dps3;

                //const int64_t i10 = i1;
                மாறிலி int64_t i12 = i2;
                மாறிலி int64_t i13 = i3;

#எனில் GGML_VEC_MAD_UNROLL > 2
                மாறிலி int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
                க்கு (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
                    மாறிலி int64_t i11 = i01;

                    தசம * s0 = (தசம *) ((எழுத்து *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    தசம * s1 = (தசம *) ((எழுத்து *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    தசம * d  = (தசம *) ((எழுத்து *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
                }
                க்கு (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
                    மாறிலி int64_t i11 = i01;

                    தசம * s0 = (தசம *) ((எழுத்து *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    தசம * s1 = (தசம *) ((எழுத்து *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    தசம * d  = (தசம *) ((எழுத்து *)  dst->data + (          i1*nb1   + i2*nb2   + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#இல்லையெனில்
                க்கு (int64_t i01 = bi01; i01 < bne01; ++i01) {
                    மாறிலி int64_t i11 = i01;

                    தசம * s0 = (தசம *) ((எழுத்து *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
                    தசம * s1 = (தசம *) ((எழுத்து *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
                    தசம * d  = (தசம *) ((எழுத்து *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

                    ggml_vec_mad_f32(ne0, d, s0, *s1);
                }
#endif
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_out_prod_q_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி ggml_type type = src0->type;
    ggml_to_float_t மாறிலி dequantize_row_q = ggml_get_type_traits(type)->to_float;

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    // we don't support permuted src0 dim0
    GGML_ASSERT(nb00 == ggml_type_size(type));

    // dst dim0 cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    // GGML_ASSERT(nb0 <= nb1);
    // GGML_ASSERT(nb1 <= nb2);
    // GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    எனில் (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (தசம *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    // parallelize by last three dimensions

    // total rows in dst
    மாறிலி int64_t nr = ne1*ne2*ne3;

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    // dst[:,:,:,:] = 0
    // for i2,i3:
    //   for i1:
    //     for i01:
    //       for i0:
    //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

    தசம * wdata = (தசம *) params->wdata + (ne0 + CACHE_வரி_SIZE_F32) * ith;

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        // dst indices
        மாறிலி int64_t i3 = ir/(ne2*ne1);
        மாறிலி int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        மாறிலி int64_t i02 = i2;
        மாறிலி int64_t i03 = i3;

        //const int64_t i10 = i1;
        மாறிலி int64_t i12 = i2;
        மாறிலி int64_t i13 = i3;

        க்கு (int64_t i01 = 0; i01 < ne01; ++i01) {
            மாறிலி int64_t i11 = i01;

            தசம * s0 = (தசம *) ((எழுத்து *) src0->data + (          i01*nb01 + i02*nb02 + i03*nb03));
            தசம * s1 = (தசம *) ((எழுத்து *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            தசம * d  = (தசம *) ((எழுத்து *)  dst->data + (          i1*nb1 + i2*nb2 + i3*nb3));

            dequantize_row_q(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_out_prod_f16_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);
    GGML_ASSERT(ne2  == ne12);
    GGML_ASSERT(ne3  == ne13);

    GGML_ASSERT(nb00 == அளவு(ggml_fp16_t));
    GGML_ASSERT(nb0  == அளவு(தசம));

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne1 == ne10);
    GGML_ASSERT(ne2 == ne02);
    GGML_ASSERT(ne3 == ne03);

    எனில் (ith == 0) {
        ggml_vec_set_f32(ne0*ne1*ne2*ne3, (தசம *)dst->data, 0);
    }
    ggml_barrier(params->threadpool);

    மாறிலி int64_t nr = ne1*ne2*ne3;
    மாறிலி int64_t dr = (nr + nth - 1)/nth;
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    தசம * wdata = (தசம *) params->wdata + (ne0 + CACHE_வரி_SIZE_F32) * ith;

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i3 = ir/(ne2*ne1);
        மாறிலி int64_t i2 = (ir - i3*ne2*ne1)/ne1;
        மாறிலி int64_t i1 = (ir - i3*ne2*ne1 - i2*ne1);

        மாறிலி int64_t i02 = i2;
        மாறிலி int64_t i03 = i3;

        மாறிலி int64_t i12 = i2;
        மாறிலி int64_t i13 = i3;

        தசம * d = (தசம *) ((எழுத்து *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3));

        க்கு (int64_t i01 = 0; i01 < ne01; ++i01) {
            மாறிலி int64_t i11 = i01;
            ggml_fp16_t * s0 = (ggml_fp16_t *) ((எழுத்து *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            தசம * s1 = (தசம *) ((எழுத்து *) src1->data + (i1*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
            ggml_fp16_to_fp32_row(s0, wdata, ne0);
            ggml_vec_mad_f32(ne0, d, wdata, *s1);
        }
    }
}

வெறுமை ggml_compute_forward_out_prod(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_out_prod_q_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_out_prod_f16_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_out_prod_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_scale

நிலையான வெறுமை ggml_compute_forward_scale_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    தசம s; // scale factor
    தசம b; // bias

    memcpy(&s, (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&b, (தசம *) dst->op_params + 1, அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    மாறிலி size_t nb01 = src0->nb[1];

    மாறிலி size_t nb1 = dst->nb[1];

    எனில் (b == 0.0f) {
        க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
            எனில் (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((எழுத்து *)dst->data + i1*nb1, (எழுத்து *)src0->data + i1*nb01, nc * அளவு(தசம));
            }
            ggml_vec_scale_f32(nc, (தசம *) ((எழுத்து *) dst->data + i1*nb1), s);
        }
    } இல்லையெனில் {
        க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (தசம *) ((எழுத்து *) dst->data  + i1*nb1),
                (தசம *) ((எழுத்து *) src0->data + i1*nb1),
                s, b);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_scale_bf16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    தசம s; // scale factor
    தசம b; // bias

    memcpy(&s, (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&b, (தசம *) dst->op_params + 1, அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        மாறிலி ggml_bf16_t * x = (மாறிலி ggml_bf16_t *) ((மாறிலி எழுத்து *) src0->data + i1*src0->nb[1]);
        ggml_bf16_t       * y = (ggml_bf16_t *)       ((எழுத்து *)       dst->data  + i1*dst->nb[1]);
        க்கு (முழு k = 0; k < nc; k++) {
            y[k] = GGML_FP32_TO_BF16(s*GGML_BF16_TO_FP32(x[k]) + b);
        }
    }
}

வெறுமை ggml_compute_forward_scale(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_scale_bf16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_set

நிலையான வெறுமை ggml_compute_forward_set_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    தருக்க   inplace = (தருக்க) ((int32_t *) dst->op_params)[4];

    எனில் (!inplace) {
        எனில் (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((எழுத்து *)  dst->data),
                ((எழுத்து *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr = ggml_nrows(src1);
    மாறிலி முழு nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    மாறிலி size_t nb0 = ggml_element_size(src0);

    மாறிலி முழு im0 = (ne10 == 0 ? 0 : ne10-1);
    மாறிலி முழு im1 = (ne11 == 0 ? 0 : ne11-1);
    மாறிலி முழு im2 = (ne12 == 0 ? 0 : ne12-1);
    மாறிலி முழு im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        மாறிலி முழு i3 = ir/(ne12*ne11);
        மாறிலி முழு i2 = (ir - i3*ne12*ne11)/ne11;
        மாறிலி முழு i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_f32(nc,
                (தசம *) ((எழுத்து *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (தசம *) ((எழுத்து *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

நிலையான வெறுமை ggml_compute_forward_set_i32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // view src0 and dst with these strides and data offset inbytes during set
    // nb0 is implicitly element_size because src0 and dst are contiguous
    size_t nb1     = ((int32_t *) dst->op_params)[0];
    size_t nb2     = ((int32_t *) dst->op_params)[1];
    size_t nb3     = ((int32_t *) dst->op_params)[2];
    size_t offset  = ((int32_t *) dst->op_params)[3];
    தருக்க   inplace = (தருக்க) ((int32_t *) dst->op_params)[4];

    எனில் (!inplace) {
        எனில் (params->ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            memcpy(
                ((எழுத்து *)  dst->data),
                ((எழுத்து *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr = ggml_nrows(src1);
    மாறிலி முழு nc = src1->ne[0];

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

    // src0 and dst as viewed during set
    மாறிலி size_t nb0 = ggml_element_size(src0);

    மாறிலி முழு im0 = (ne10 == 0 ? 0 : ne10-1);
    மாறிலி முழு im1 = (ne11 == 0 ? 0 : ne11-1);
    மாறிலி முழு im2 = (ne12 == 0 ? 0 : ne12-1);
    மாறிலி முழு im3 = (ne13 == 0 ? 0 : ne13-1);

    GGML_ASSERT(offset + im0*nb0  + im1*nb1  + im2*nb2  + im3*nb3  <= ggml_nbytes(dst));

    GGML_ASSERT(nb10 == அளவு(int32_t));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        மாறிலி முழு i3 = ir/(ne12*ne11);
        மாறிலி முழு i2 = (ir - i3*ne12*ne11)/ne11;
        மாறிலி முழு i1 = (ir - i3*ne12*ne11 - i2*ne11);

        ggml_vec_cpy_i32(nc,
                (int32_t *) ((எழுத்து *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + offset),
                (int32_t *) ((எழுத்து *) src1->data + i3*nb13 + i2*nb12 + i1*nb11));
    }
}

வெறுமை ggml_compute_forward_set(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_set_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_I32:
            {
                ggml_compute_forward_set_i32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_BF16:
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_Q8_1:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cpy

வெறுமை ggml_compute_forward_cpy(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_cont

வெறுமை ggml_compute_forward_cont(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_get_rows

நிலையான வெறுமை ggml_compute_forward_get_rows_q(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int64_t nc = ne00;
    மாறிலி int64_t nr = ggml_nelements(src1);

    மாறிலி ggml_type type = src0->type;
    ggml_to_float_t மாறிலி dequantize_row_q = ggml_get_type_traits(type)->to_float;

    ப்ரதிஜ்ஞா(ne0  == nc);
    ப்ரதிஜ்ஞா(ne02 == ne11);
    ப்ரதிஜ்ஞா(nb00 == ggml_type_size(type));
    ப்ரதிஜ்ஞா(ggml_nrows(dst) == nr);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t i = ir0; i < ir1; ++i) {
        மாறிலி int64_t i12 = i/(ne11*ne10);
        மாறிலி int64_t i11 = (i - i12*ne11*ne10)/ne10;
        மாறிலி int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        மாறிலி int64_t i01 = *(int32_t *) ((எழுத்து *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        dequantize_row_q(
                (மாறிலி வெறுமை *) ((எழுத்து *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                     (தசம *) ((எழுத்து *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

நிலையான வெறுமை ggml_compute_forward_get_rows_f16(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int64_t nc = ne00;
    மாறிலி int64_t nr = ggml_nelements(src1);

    ப்ரதிஜ்ஞா(ne0  == nc);
    ப்ரதிஜ்ஞா(ne02 == ne11);
    ப்ரதிஜ்ஞா(nb00 == அளவு(ggml_fp16_t));
    ப்ரதிஜ்ஞா(ggml_nrows(dst) == nr);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t i = ir0; i < ir1; ++i) {
        மாறிலி int64_t i12 = i/(ne11*ne10);
        மாறிலி int64_t i11 = (i - i12*ne11*ne10)/ne10;
        மாறிலி int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        மாறிலி int64_t i01 = *(int32_t *) ((எழுத்து *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_fp16_to_fp32(
            (மாறிலி ggml_fp16_t*) ((எழுத்து *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                       (தசம *) ((எழுத்து *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

நிலையான வெறுமை ggml_compute_forward_get_rows_bf16(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int64_t nc = ne00;
    மாறிலி int64_t nr = ggml_nelements(src1);

    ப்ரதிஜ்ஞா(ne0  == nc);
    ப்ரதிஜ்ஞா(ne02 == ne11);
    ப்ரதிஜ்ஞா(nb00 == அளவு(ggml_bf16_t));
    ப்ரதிஜ்ஞா(ggml_nrows(dst) == nr);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t i = ir0; i < ir1; ++i) {
        மாறிலி int64_t i12 = i/(ne11*ne10);
        மாறிலி int64_t i11 = (i - i12*ne11*ne10)/ne10;
        மாறிலி int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        மாறிலி int64_t i01 = *(int32_t *) ((எழுத்து *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_cpu_bf16_to_fp32(
            (மாறிலி ggml_bf16_t *) ((எழுத்து *) src0->data + i01*nb01 + i11*nb02 + i12*nb03),
                        (தசம *) ((எழுத்து *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3), nc);
    }
}

நிலையான வெறுமை ggml_compute_forward_get_rows_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int64_t nc = ne00;
    மாறிலி int64_t nr = ggml_nelements(src1);

    ப்ரதிஜ்ஞா(ne0  == nc);
    ப்ரதிஜ்ஞா(ne02 == ne11);
    ப்ரதிஜ்ஞா(nb00 == அளவு(தசம));
    ப்ரதிஜ்ஞா(ggml_nrows(dst) == nr);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t i = ir0; i < ir1; ++i) {
        மாறிலி int64_t i12 = i/(ne11*ne10);
        மாறிலி int64_t i11 = (i - i12*ne11*ne10)/ne10;
        மாறிலி int64_t i10 = (i - i12*ne11*ne10 - i11*ne10);
        மாறிலி int64_t i01 = *(int32_t *) ((எழுத்து *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

        GGML_ASSERT(i01 >= 0 && i01 < ne01);

        ggml_vec_cpy_f32(nc,
                (தசம *) ((எழுத்து *)  dst->data + i10*nb1  + i11*nb2  + i12*nb3),
                (தசம *) ((எழுத்து *) src0->data + i01*nb01 + i11*nb02 + i12*nb03));
    }
}

வெறுமை ggml_compute_forward_get_rows(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_Q8_1:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
            {
                ggml_compute_forward_get_rows_q(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rows_bf16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
        நிலை GGML_TYPE_I32:
            {
                ggml_compute_forward_get_rows_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
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
நிலையான வெறுமை ggml_compute_forward_set_rows_impl(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி int64_t nc = ne00;
    மாறிலி int64_t nr = ne01;

    ப்ரதிஜ்ஞா(ne0  == nc);
    ப்ரதிஜ்ஞா(ne2  == ne02);
    ப்ரதிஜ்ஞா(ne3  == ne03);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    ப்ரதிஜ்ஞா(ne02 % ne11 == 0);
    ப்ரதிஜ்ஞா(ne03 % ne12 == 0);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = std::min(ir0 + dr, nr);

    மாறிலி size_t rs = ggml_row_size(src0->type, nc);

    ggml_from_float_t மாறிலி from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

    க்கு (int64_t i03 = 0; i03 < ne03; ++i03) {
        க்கு (int64_t i02 = 0; i02 < ne02; ++i02) {
            க்கு (int64_t i = ir0; i < ir1; ++i) {
                மாறிலி int64_t i12 = i03%ne12;
                மாறிலி int64_t i11 = i02%ne11;
                மாறிலி int64_t i10 = i;

                மாறிலி int64_t i1 = *(idx_t *) ((எழுத்து *) src1->data + i10*nb10 + i11*nb11 + i12*nb12);

                GGML_ASSERT(i1 >= 0 && i1 < ne1);

                எனில் constexpr (std::is_same_v<src_t, தசம>) {
                    from_float(
                            (மாறிலி தசம *) ((எழுத்து *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            ((எழுத்து *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3), nc);
                } இல்லையெனில் எனில் constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
                    எனில் (dst->type == GGML_TYPE_F16) {
                        memcpy(
                                            ((எழுத்து *)  dst->data + i1*nb1  + i02*nb2  + i03*nb3),
                                            ((எழுத்து *) src0->data +  i*nb01 + i02*nb02 + i03*nb03),
                                            rs);
                    } இல்லையெனில் {
                        தசம * wdata = (தசம *) params->wdata + (nc + CACHE_வரி_SIZE_F32) * ith;
                        ggml_fp16_to_fp32_row(
                                (மாறிலி ggml_fp16_t *) ((எழுத்து *) src0->data + i*nb01 + i02*nb02 + i03*nb03),
                                wdata, nc);
                        from_float(wdata,
                                ((எழுத்து *) dst->data + i1*nb1 + i02*nb2 + i03*nb3), nc);
                    }
                } இல்லையெனில் {
                    GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_set_rows(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                எனில் (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<தசம, int64_t>(params, dst);
                } இல்லையெனில் எனில் (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<தசம, int32_t>(params, dst);
                } இல்லையெனில் {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                எனில் (src1->type == GGML_TYPE_I64) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int64_t>(params, dst);
                } இல்லையெனில் எனில் (src1->type == GGML_TYPE_I32) {
                    ggml_compute_forward_set_rows_impl<ggml_fp16_t, int32_t>(params, dst);
                } இல்லையெனில் {
                    GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
                }
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
            }
    }
}

// ggml_compute_forward_get_rows_back

நிலையான வெறுமை ggml_compute_forward_get_rows_back_f32_f16(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == அளவு(ggml_fp16_t));

    க்கு (முழு i = 0; i < nr; ++i) {
        மாறிலி முழு r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        க்கு (முழு j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((எழுத்து *) src0->data + i*src0->nb[1]))[j];
            ((தசம *) ((எழுத்து *) dst->data + r*dst->nb[1]))[j] += GGML_CPU_FP16_TO_FP32(v);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_get_rows_back_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    GGML_ASSERT(ggml_is_contiguous(dst));

    // ggml_compute_forward_dup_same_cont(params, opt0, dst);

    memset(dst->data, 0, ggml_nbytes(dst));

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nelements(src1);

    GGML_ASSERT( dst->ne[0] == nc);
    GGML_ASSERT(src0->nb[0] == அளவு(தசம));

    க்கு (முழு i = 0; i < nr; ++i) {
        மாறிலி முழு r = ((int32_t *) src1->data)[i];

        GGML_ASSERT(r >= 0 && r < dst->ne[1]);

        ggml_vec_add_f32(nc,
                (தசம *) ((எழுத்து *)  dst->data + r*dst->nb[1]),
                (தசம *) ((எழுத்து *)  dst->data + r*dst->nb[1]),
                (தசம *) ((எழுத்து *) src0->data + i*src0->nb[1]));
    }
}

வெறுமை ggml_compute_forward_get_rows_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_get_rows_back_f32_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_get_rows_back_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
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

நிலையான வெறுமை ggml_compute_forward_diag_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    // TODO: handle transposed/permuted matrices

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(ne00 == ne0);
    GGML_ASSERT(ne00 == ne1);
    GGML_ASSERT(ne01 == 1);
    GGML_ASSERT(ne02 == ne2);
    GGML_ASSERT(ne03 == ne3);

    GGML_ASSERT(nb00 == அளவு(தசம));
    GGML_ASSERT(nb0  == அளவு(தசம));

    க்கு (முழு i3 = 0; i3 < ne3; i3++) {
        க்கு (முழு i2 = 0; i2 < ne2; i2++) {
            க்கு (முழு i1 = 0; i1 < ne1; i1++) {
                தசம * d = (தசம *)((எழுத்து *)  dst->data + i3*nb3  + i2*nb2 + i1*nb1);
                தசம * s = (தசம *)((எழுத்து *) src0->data + i3*nb03 + i2*nb02);
                க்கு (முழு i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                க்கு (முழு i0 = i1+1; i0 < ne0; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_diag(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_diag_mask_inf

நிலையான வெறுமை ggml_compute_forward_diag_mask_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst,
        மாறிலி தசம value) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு  n_past  = ((int32_t *) dst->op_params)[0];
    மாறிலி தருக்க inplace = src0->data == dst->data;

    GGML_ASSERT(n_past >= 0);

    எனில் (!inplace) {
        எனில் (ith == 0) {
            // memcpy needs to be synchronized across threads to avoid race conditions.
            // => do it in INIT phase
            GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
            GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
            memcpy(
                ((எழுத்து *)  dst->data),
                ((எழுத்து *) src0->data),
                ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }

    // TODO: handle transposed/permuted matrices

    மாறிலி முழு n  = ggml_nrows(src0);
    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = src0->ne[1];
    மாறிலி முழு nz = n/nr;

    GGML_ASSERT( dst->nb[0] == அளவு(தசம));
    GGML_ASSERT(src0->nb[0] == அளவு(தசம));

    க்கு (முழு k = 0; k < nz; k++) {
        க்கு (முழு j = ith; j < nr; j += nth) {
            க்கு (முழு i = n_past; i < nc; i++) {
                எனில் (i > n_past + j) {
                    *(தசம *)((எழுத்து *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = value;
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_diag_mask_inf(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

வெறுமை ggml_compute_forward_diag_mask_zero(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_diag_mask_f32(params, dst, 0);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_soft_max

நிலையான வெறுமை ggml_compute_forward_soft_max_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி ggml_tensor * src2 = dst->src[2];

    ப்ரதிஜ்ஞா(ggml_is_contiguous(dst));
    ப்ரதிஜ்ஞா(ggml_are_same_shape(src0, dst));

    தசம scale    = 1.0f;
    தசம max_bias = 0.0f;

    memcpy(&scale,    (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max_bias, (தசம *) dst->op_params + 1, அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி int64_t nb11 = src1 ? src1->nb[1] : 1;
    மாறிலி int64_t nb12 = src1 ? src1->nb[2] : 1;
    மாறிலி int64_t nb13 = src1 ? src1->nb[3] : 1;

    மாறிலி int64_t ne12 = src1 ? src1->ne[2] : 1;
    மாறிலி int64_t ne13 = src1 ? src1->ne[3] : 1;

    // TODO: is this supposed to be ceil instead of floor?
    //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
    மாறிலி uint32_t n_head      = ne02;
    மாறிலி uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    மாறிலி தசம m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    மாறிலி தசம m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    தசம * wp = (தசம *) params->wdata + (ne00 + CACHE_வரி_SIZE_F32) * ith;

    மாறிலி தருக்க use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    // sinks
    மாறிலி தசம * sk = src2 ? (தசம *)((எழுத்து *) src2->data) : nullptr;

    க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
        க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
            க்கு (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                மாறிலி int64_t i11 = i01;
                மாறிலி int64_t i12 = i02%ne12;
                மாறிலி int64_t i13 = i03%ne13;

                // ALiBi
                மாறிலி uint32_t h = i02; // head
                மாறிலி தசம slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

                தசம * sp = (தசம *)((எழுத்து *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
                தசம * dp = (தசம *)((எழுத்து *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3);

                // broadcast the mask across rows
                ggml_fp16_t * mp_f16 = src1 ? (ggml_fp16_t *)((எழுத்து *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : வெற்று;
                தசம       * mp_f32 = src1 ? (தசம       *)((எழுத்து *) src1->data + i11*nb11 + i12*nb12 + i13*nb13) : வெற்று;

                ggml_vec_cpy_f32  (ne00, wp, sp);
                ggml_vec_scale_f32(ne00, wp, scale);
                எனில் (mp_f32) {
                    எனில் (use_f16) {
                        க்கு (முழு i = 0; i < ne00; ++i) {
                            wp[i] += slope*GGML_CPU_FP16_TO_FP32(mp_f16[i]);
                        }
                    } இல்லையெனில் {
                        க்கு (முழு i = 0; i < ne00; ++i) {
                            wp[i] += slope*mp_f32[i];
                        }
                    }
                }

#ifndef NDEBUG
                க்கு (முழு i = 0; i < ne00; ++i) {
                    //printf("p[%d] = %f\n", i, p[i]);
                    ப்ரதிஜ்ஞா(!isnan(wp[i]));
                }
#endif // NDEBUG

                தசம max = -INFINITY;
                ggml_vec_max_f32(ne00, &max, wp);

                // if we have sinks, make a correction as if they were included in the softmax
                எனில் (sk) {
                    max = MAX(max, sk[i02]);
                }

                ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp, max);
                ப்ரதிஜ்ஞா(sum > 0.0);

                எனில் (sk) {
                    sum += (ggml_float) expf(sk[i02] - max);
                }

                sum = 1.0/sum;
                ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
                க்கு (முழு i = 0; i < ne00; ++i) {
                    ப்ரதிஜ்ஞா(!isnan(dp[i]));
                    ப்ரதிஜ்ஞா(!isinf(dp[i]));
                }
#endif // NDEBUG
            }
        }
    }
}

வெறுமை ggml_compute_forward_soft_max(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_soft_max_ext_back

நிலையான வெறுமை ggml_compute_forward_soft_max_ext_back_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_are_same_shape(src1, dst));

    தசம scale    = 1.0f;
    தசம max_bias = 0.0f;

    memcpy(&scale,    (மாறிலி தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max_bias, (மாறிலி தசம *) dst->op_params + 1, அளவு(தசம));

    GGML_ASSERT(max_bias == 0.0f);

    // TODO: handle transposed/permuted matrices

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc = src0->ne[0];
    மாறிலி முழு nr = ggml_nrows(src0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம *dy = (தசம *)((எழுத்து *) src0->data + i1*src0->nb[1]);
        தசம *y  = (தசம *)((எழுத்து *) src1->data + i1*src1->nb[1]);
        தசம *dx = (தசம *)((எழுத்து *) dst->data  + i1*dst->nb[1]);

#ifndef NDEBUG
        க்கு (முழு i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ப்ரதிஜ்ஞா(!isnan(dy[i]));
            ப்ரதிஜ்ஞா(!isnan(y[i]));
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
        தசம dot_y_dy = 0;
        ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
        க்கு (முழு i = 0; i < nc; i++) {
            dx[i] = scale * (dy[i] - dot_y_dy) * y[i];
        }

#ifndef NDEBUG
        க்கு (முழு i = 0; i < nc; ++i) {
            ப்ரதிஜ்ஞா(!isnan(dx[i]));
            ப்ரதிஜ்ஞா(!isinf(dx[i]));
        }
#endif // NDEBUG
    }
}

வெறுமை ggml_compute_forward_soft_max_ext_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_soft_max_ext_back_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_clamp

நிலையான வெறுமை ggml_compute_forward_clamp_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தசம min;
    தசம max;
    memcpy(&min, (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max, (தசம *) dst->op_params + 1, அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு n  = ggml_nrows(src0);
    மாறிலி முழு nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(தசம));
    GGML_ASSERT(nb00 == அளவு(தசம));

    க்கு (முழு j = ith; j < n; j += nth) {
        மாறிலி int64_t i1 = j % ne01;
        மாறிலி int64_t i2 = (j / ne01) % ne02;
        மாறிலி int64_t i3 = j / (ne01*ne02);

        தசம * dst_ptr  = (தசம *) ((எழுத்து *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        தசம * src0_ptr = (தசம *) ((எழுத்து *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        க்கு (முழு i = 0; i < nc; i++) {
            dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_clamp_f16(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தசம min;
    தசம max;
    memcpy(&min, (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max, (தசம *) dst->op_params + 1, அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு n  = ggml_nrows(src0);
    மாறிலி முழு nc = src0->ne[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT( nb0 == அளவு(ggml_fp16_t));
    GGML_ASSERT(nb00 == அளவு(ggml_fp16_t));

    க்கு (முழு j = ith; j < n; j += nth) {
        மாறிலி int64_t i1 = j % ne01;
        மாறிலி int64_t i2 = (j / ne01) % ne02;
        மாறிலி int64_t i3 = j / (ne01*ne02);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((எழுத்து *)  dst->data + i1*nb1  + i2*nb2  + i3*nb3);
        ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((எழுத்து *) src0->data + i1*nb01 + i2*nb02 + i3*nb03);

        க்கு (முழு i = 0; i < nc; i++) {
            தசம v = GGML_CPU_FP16_TO_FP32(src0_ptr[i]);
            dst_ptr[i] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v, max), min));
        }
    }
}

வெறுமை ggml_compute_forward_clamp(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_clamp_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_clamp_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_BF16:
        நிலை GGML_TYPE_Q1_0:
        நிலை GGML_TYPE_Q2_0:
        நிலை GGML_TYPE_Q4_0:
        நிலை GGML_TYPE_Q4_1:
        நிலை GGML_TYPE_Q5_0:
        நிலை GGML_TYPE_Q5_1:
        நிலை GGML_TYPE_Q8_0:
        நிலை GGML_TYPE_Q8_1:
        நிலை GGML_TYPE_MXFP4:
        நிலை GGML_TYPE_NVFP4:
        நிலை GGML_TYPE_Q2_K:
        நிலை GGML_TYPE_Q3_K:
        நிலை GGML_TYPE_Q4_K:
        நிலை GGML_TYPE_Q5_K:
        நிலை GGML_TYPE_Q6_K:
        நிலை GGML_TYPE_TQ1_0:
        நிலை GGML_TYPE_TQ2_0:
        நிலை GGML_TYPE_IQ2_XXS:
        நிலை GGML_TYPE_IQ2_XS:
        நிலை GGML_TYPE_IQ3_XXS:
        நிலை GGML_TYPE_IQ1_S:
        நிலை GGML_TYPE_IQ1_M:
        நிலை GGML_TYPE_IQ4_NL:
        நிலை GGML_TYPE_IQ4_XS:
        நிலை GGML_TYPE_IQ3_S:
        நிலை GGML_TYPE_IQ2_S:
        நிலை GGML_TYPE_Q8_K:
        நிலை GGML_TYPE_I8:
        நிலை GGML_TYPE_I16:
        நிலை GGML_TYPE_I32:
        நிலை GGML_TYPE_I64:
        நிலை GGML_TYPE_F64:
        நிலை GGML_TYPE_மொத்தம்:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope

நிலையான தசம rope_yarn_ramp(மாறிலி தசம கீழ், மாறிலி தசம உயர்வு, மாறிலி முழு i0) {
    மாறிலி தசம y = (i0 / 2 - கீழ்) / MAX(0.001f, உயர்வு - கீழ்);
    திரும்பு 1 - MIN(1, MAX(0, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
நிலையான வெறுமை rope_yarn(
    தசம theta_extrap, தசம freq_scale, தசம corr_dims[2], int64_t i0, தசம ext_factor, தசம mscale,
    தசம * cos_theta, தசம * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    தசம theta_interp = freq_scale * theta_extrap;
    தசம theta = theta_interp;
    எனில் (ext_factor != 0.0f) {
        தசம ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

நிலையான வெறுமை ggml_rope_cache_init(
     தசம theta_base, தசம freq_scale, மாறிலி தசம * freq_factors, தசம corr_dims[2], int64_t ne0, தசம ext_factor, தசம mscale,
     தசம * cache, தசம sin_sign, தசம theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    தசம theta = theta_base;
    க்கு (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        மாறிலி தசம ff = freq_factors ? freq_factors[i0/2] : 1.0f;
        rope_yarn(
            theta/ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
        );
        cache[i0 + 1] *= sin_sign;

        theta *= theta_scale;
    }
}

நிலையான வெறுமை ggml_mrope_cache_init(
     தசம theta_base_t, தசம theta_base_h, தசம theta_base_w, தசம theta_base_e, முழு sections[4], தருக்க is_imrope, தருக்க indep_sects,
     தசம freq_scale, மாறிலி தசம * freq_factors, தசம corr_dims[2], int64_t ne0, தசம ext_factor, தசம mscale,
     தசம * cache, தசம sin_sign, தசம theta_scale) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    தசம theta_t = theta_base_t;
    தசம theta_h = theta_base_h;
    தசம theta_w = theta_base_w;
    தசம theta_e = theta_base_e;  // extra position id for vision encoder
    முழு sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    முழு sec_w = sections[1] + sections[0];
    முழு sec_e = sections[2] + sec_w;
    GGML_ASSERT(sect_dims <= ne0);

    க்கு (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        மாறிலி தசம ff = freq_factors ? freq_factors[i0/2] : 1.0f;

        முழு sector = (i0 / 2) % sect_dims;
        எனில் (indep_sects) {
            // compute theta independently for each dim sections
            // (i.e. reset corresponding theta when `i0` go from one section to another)
            எனில் (sector == 0) {
                theta_t = theta_base_t;
            }
            இல்லையெனில் எனில் (sector == sections[0]) {
                theta_h = theta_base_h;;
            }
            இல்லையெனில் எனில் (sector == sec_w) {
                theta_w = theta_base_w;
            }
            இல்லையெனில் எனில் (sector == sec_e) {
                theta_e = theta_base_e;
            }
        }

        தசம theta = theta_t;
        எனில் (is_imrope) { // qwen3vl apply interleaved mrope
            எனில் (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } இல்லையெனில் எனில் (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } இல்லையெனில் எனில் (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } இல்லையெனில் {
                theta = theta_e;
            }
        } இல்லையெனில் {
            எனில் (sector >= sections[0] && sector < sec_w) {
                theta = theta_h;
            }
            இல்லையெனில் எனில் (sector >= sec_w && sector < sec_w + sections[2]) {
                theta = theta_w;
            }
            இல்லையெனில் எனில் (sector >= sec_w + sections[2]) {
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
நிலையான வெறுமை rotate_pairs(மாறிலி int64_t n, மாறிலி int64_t n_offset, மாறிலி தசம * cache, மாறிலி T * src_data, T * dst_data, மாறிலி முழு scale = 2) {
  க்கு (int64_t i0 = 0; i0 < n; i0 += 2) {
    மாறிலி int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    மாறிலி தசம cos_theta = cache[i0 + 0];
    மாறிலி தசம sin_theta = cache[i0 + 1];

    மாறிலி T * மாறிலி src = src_data + ic;
    T * dst             = dst_data + ic;

    மாறிலி தசம x0 = type_conversion_table<T>::to_f32(src[0]);
    மாறிலி தசம x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}

template<typename T> //float or ggml_fp16_t
நிலையான வெறுமை ggml_compute_forward_rope_flt(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst,
        மாறிலி தருக்க forward) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    தசம freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    முழு sections[4];

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    மாறிலி முழு n_dims     = ((int32_t *) dst->op_params)[1];
    மாறிலி முழு mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    மாறிலி முழு n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, அளவு(தசம));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, அளவு(தசம));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, அளவு(தசம));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, அளவு(தசம));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, அளவு(தசம));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, அளவு(தசம));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, அளவு(முழு)*4);

    மாறிலி முழு n_offs = ((int32_t *) dst->op_params)[15];

    GGML_TENSOR_UNARY_OP_LOCALS

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    GGML_ASSERT(nb0 == nb00);
    GGML_ASSERT(nb0 == அளவு(T));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr = ggml_nrows(dst);

    GGML_ASSERT(n_dims <= ne0);
    GGML_ASSERT(n_dims % 2 == 0);

    GGML_ASSERT(n_offs >= 0);
    GGML_ASSERT(n_offs % 2 == 0);
    GGML_ASSERT(n_offs + n_dims <= ne0);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    // row index used to determine which thread to use
    முழு ir = 0;

    மாறிலி தசம theta_scale = powf(freq_base, -2.0f/n_dims);

    தசம corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    மாறிலி தருக்க is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
    மாறிலி தருக்க mrope_used = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
    மாறிலி தருக்க is_vision = mode == GGML_ROPE_TYPE_VISION;

    எனில் (mrope_used) {
        GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
    }

    எனில் (is_vision) {
        GGML_ASSERT(n_dims == ne0/2);
        GGML_ASSERT(n_offs == 0);
    }

    மாறிலி தசம * freq_factors = வெற்று;
    எனில் (src2 != வெற்று) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors = (மாறிலி தசம *) src2->data;
    }

    // backward process uses inverse rotation by cos and sin.
    // cos and sin build a rotation matrix, where the inverse is the transpose.
    // this essentially just switches the sign of sin.
    மாறிலி தசம sin_sign = forward ? 1.0f : -1.0f;

    மாறிலி int32_t * pos = (மாறிலி int32_t *) src1->data;

    int64_t last_i2 = -1;

    க்கு (int64_t i3 = 0; i3 < ne3; i3++) { // batch
        க்கு (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len
            க்கு (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
                எனில் (ir++ < ir0) தொடர்; // skip rows mapped to other threads
                எனில் (ir   > ir1) நிறுத்து;

                தசம * cache = (தசம *) params->wdata + (ne0 + CACHE_வரி_SIZE_F32)*ith;
                எனில் (last_i2 != i2) {
                    எனில் (!mrope_used) {
                        மாறிலி int64_t p = pos[i2];
                        ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }
                    இல்லையெனில் {
                        மாறிலி int64_t p_t = pos[i2];
                        மாறிலி int64_t p_h = pos[i2 + ne2];
                        மாறிலி int64_t p_w = pos[i2 + ne2 * 2];
                        மாறிலி int64_t p_e = pos[i2 + ne2 * 3];
                        ggml_mrope_cache_init(
                            p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
                            freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
                    }

                    last_i2 = i2;
                }

                T * src = (T *)((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01);
                T * dst_data  = (T *)((எழுத்து *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1);

                தேர்வு (mode) {
                    நிலை GGML_ROPE_TYPE_NORMAL:
                        rotate_pairs<T>(n_dims, 1, cache, src + n_offs, dst_data + n_offs, 1);
                        நிறுத்து;
                    நிலை GGML_ROPE_TYPE_NEOX:
                    நிலை GGML_ROPE_TYPE_MROPE:
                    நிலை GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
                        நிறுத்து;
                    நிலை GGML_ROPE_TYPE_VISION:
                        rotate_pairs<T>(ne0, n_dims, cache, src, dst_data);
                        நிறுத்து;
                    இயல்பு:
                        GGML_ABORT("rope type not supported");
                }

                எனில் (!is_vision) {
                    // fill the remain channels with data from src tensor
                    க்கு (int64_t i0 = 0; i0 < ne0; i0 += 2) {
                        எனில் (i0 == n_offs) {
                            i0 += n_dims - 2; // skip the rotated channels
                            தொடர்;
                        }
                        மாறிலி T * மாறிலி src = (T *)((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        T * dst_data  = (T *)((எழுத்து *)  dst->data + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

                        dst_data[0] = src[0];
                        dst_data[1] = src[1];
                    }
                }
            } //attn-heads
        }
    }
}

வெறுமை ggml_compute_forward_rope(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, உண்மை);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<தசம>(params, dst, உண்மை);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rope_back

வெறுமை ggml_compute_forward_rope_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_rope_flt<ggml_fp16_t>(params, dst, பொய்);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rope_flt<தசம>(params, dst, பொய்);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_transpose_1d

நிலையான வெறுமை ggml_compute_forward_conv_transpose_1d_f16_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == அளவு(ggml_fp16_t));
    GGML_ASSERT(nb10 == அளவு(தசம));

    எனில் (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            ggml_fp16_t * மாறிலி wdata = (ggml_fp16_t *) params->wdata + 0;

            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    மாறிலி ggml_fp16_t * மாறிலி src = (ggml_fp16_t *)((எழுத்து *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i01*ne00*ne02;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        {
            ggml_fp16_t * மாறிலி wdata = (ggml_fp16_t *) params->wdata + nk;
            ggml_fp16_t * dst_data = wdata;

            க்கு (int64_t i11 = 0; i11 < ne11; i11++) {
                மாறிலி தசம * மாறிலி src = (தசம *)((எழுத்து *) src1->data + i11*nb11);
                க்கு (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = GGML_CPU_FP32_TO_FP16(src[i10]);
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    மாறிலி int32_t s0 = ((மாறிலி int32_t*)(dst->op_params))[0];

    // total rows in dst
    மாறிலி முழு nr = ne1;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    ggml_fp16_t * மாறிலி wdata     = (ggml_fp16_t *) params->wdata + 0;
    ggml_fp16_t * மாறிலி wdata_src = wdata + nk;

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * dst_data = (தசம *)((எழுத்து *) dst->data + i1*nb1);
        ggml_fp16_t * wdata_kernel = wdata + i1*ne02*ne00;
        க்கு (முழு i10 = 0; i10 < ne10; i10++) {
            மாறிலி முழு i1n = i10*ne11;
            க்கு (முழு i00 = 0; i00 < ne00; i00++) {
                தசம v = 0;
                ggml_vec_dot_f16(ne02, &v, 0,
                        (ggml_fp16_t *)    wdata_src + i1n, 0,
                        (ggml_fp16_t *) wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_conv_transpose_1d_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nk = ne00*ne01*ne02;

    GGML_ASSERT(nb00 == அளவு(தசம));
    GGML_ASSERT(nb10 == அளவு(தசம));

    எனில் (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        {
            தசம * மாறிலி wdata = (தசம *) params->wdata + 0;

            க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                    மாறிலி தசம * மாறிலி src = (தசம *)((எழுத்து *) src0->data + i02*nb02 + i01*nb01);
                    தசம * dst_data = wdata + i01*ne00*ne02;
                    க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ne02 + i02] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            தசம * மாறிலி wdata = (தசம *) params->wdata + nk;
            தசம * dst_data = wdata;

            க்கு (int64_t i11 = 0; i11 < ne11; i11++) {
                மாறிலி தசம * மாறிலி src = (தசம *)((எழுத்து *) src1->data + i11*nb11);
                க்கு (int64_t i10 = 0; i10 < ne10; i10++) {
                    dst_data[i10*ne11 + i11] = src[i10];
                }
            }
        }

        // need to zero dst since we are accumulating into it
        memset(dst->data, 0, ggml_nbytes(dst));
    }
    ggml_barrier(params->threadpool);

    மாறிலி int32_t s0 = ((மாறிலி int32_t*)(dst->op_params))[0];

    // total rows in dst
    மாறிலி முழு nr = ne1;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    தசம * மாறிலி wdata     = (தசம *) params->wdata + 0;
    தசம * மாறிலி wdata_src = wdata + nk;

    க்கு (முழு i1 = ir0; i1 < ir1; i1++) {
        தசம * dst_data = (தசம *)((எழுத்து *) dst->data + i1*nb1);
        தசம * wdata_kernel = wdata + i1*ne02*ne00;
        க்கு (முழு i10 = 0; i10 < ne10; i10++) {
            மாறிலி முழு i1n = i10*ne11;
            க்கு (முழு i00 = 0; i00 < ne00; i00++) {
                தசம v = 0;
                ggml_vec_dot_f32(ne02, &v, 0,
                        wdata_src + i1n, 0,
                        wdata_kernel + i00*ne02, 0, 1);
                dst_data[i10*s0 + i00] += v;
            }
        }
    }
}

வெறுமை ggml_compute_forward_conv_transpose_1d(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_1d_f16_f32(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_1d_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
நிலையான வெறுமை ggml_compute_forward_im2col_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி int32_t s0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t s1 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *)(dst->op_params))[2];
    மாறிலி int32_t p1 = ((மாறிலி int32_t *)(dst->op_params))[3];
    மாறிலி int32_t d0 = ((மாறிலி int32_t *)(dst->op_params))[4];
    மாறிலி int32_t d1 = ((மாறிலி int32_t *)(dst->op_params))[5];
    மாறிலி தருக்க is_2D = ((மாறிலி int32_t *)(dst->op_params))[6] == 1;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t N  = is_2D ? ne13 : ne12;
    மாறிலி int64_t IC = is_2D ? ne12 : ne11;
    மாறிலி int64_t IH = is_2D ? ne11 : 1;
    மாறிலி int64_t IW = ne10;

    மாறிலி int64_t KH = is_2D ? ne01 : 1;
    மாறிலி int64_t KW = ne00;

    மாறிலி int64_t OH = is_2D ? ne2 : 1;
    மாறிலி int64_t OW = ne1;

    முழு ofs0 = is_2D ? nb13 : nb12;
    முழு ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == அளவு(தசம));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        தசம * மாறிலி wdata = (தசம *) dst->data;

        க்கு (int64_t in = 0; in < N; in++) {
            க்கு (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                க்கு (int64_t iow = 0; iow < OW; iow++) {
                    க்கு (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        தசம * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        மாறிலி தசம * மாறிலி src_data = (தசம *)((எழுத்து *) src1->data + in*ofs0 + iic*ofs1); // [IH, IW]

                        க்கு (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            க்கு (int64_t ikw = 0; ikw < KW; ikw++) {
                                மாறிலி int64_t iiw = iow*s0 + ikw*d0 - p0;
                                மாறிலி int64_t iih = ioh*s1 + ikh*d1 - p1;

                                எனில் (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } இல்லையெனில் {
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
நிலையான வெறுமை ggml_compute_forward_im2col_f16(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி int32_t s0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t s1 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *)(dst->op_params))[2];
    மாறிலி int32_t p1 = ((மாறிலி int32_t *)(dst->op_params))[3];
    மாறிலி int32_t d0 = ((மாறிலி int32_t *)(dst->op_params))[4];
    மாறிலி int32_t d1 = ((மாறிலி int32_t *)(dst->op_params))[5];
    மாறிலி தருக்க is_2D = ((மாறிலி int32_t *)(dst->op_params))[6] == 1;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t N  = is_2D ? ne13 : ne12;
    மாறிலி int64_t IC = is_2D ? ne12 : ne11;
    மாறிலி int64_t IH = is_2D ? ne11 : 1;
    மாறிலி int64_t IW = ne10;

    மாறிலி int64_t KH = is_2D ? ne01 : 1;
    மாறிலி int64_t KW = ne00;

    மாறிலி int64_t OH = is_2D ? ne2 : 1;
    மாறிலி int64_t OW = ne1;

    முழு ofs0 = is_2D ? nb13 : nb12;
    முழு ofs1 = is_2D ? nb12 : nb11;

    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        ggml_fp16_t * மாறிலி wdata = (ggml_fp16_t *) dst->data;

        க்கு (int64_t in = 0; in < N; in++) {
            க்கு (int64_t ioh = 0; ioh < OH; ioh++) { // 1
                க்கு (int64_t iow = 0; iow < OW; iow++) {
                    க்கு (int64_t iic = ith; iic < IC; iic += nth) {

                        // micro kernel
                        ggml_fp16_t * dst_data = wdata + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                        மாறிலி தசம * மாறிலி src_data_f32 = src1->type == GGML_TYPE_F32
                            ? (மாறிலி தசம *)((மாறிலி எழுத்து *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]
                        மாறிலி ggml_fp16_t * மாறிலி src_data_f16 = src1->type == GGML_TYPE_F16
                            ? (மாறிலி ggml_fp16_t *)((மாறிலி எழுத்து *) src1->data + in*ofs0 + iic*ofs1)
                            : nullptr; // [IH, IW]

                        க்கு (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                            க்கு (int64_t ikw = 0; ikw < KW; ikw++) {
                                மாறிலி int64_t iiw = iow*s0 + ikw*d0 - p0;
                                மாறிலி int64_t iih = ioh*s1 + ikh*d1 - p1;

                                எனில் (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                                } இல்லையெனில் {
                                    எனில் (src_data_f32 != nullptr) {
                                        dst_data[iic*(KH*KW) + ikh*KW + ikw] = GGML_CPU_FP32_TO_FP16(src_data_f32[iih*IW + iiw]);
                                    } இல்லையெனில் {
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

வெறுமை ggml_compute_forward_im2col(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {
    தேர்வு (dst->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_im2col_back_f32

வெறுமை ggml_compute_forward_im2col_back_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    மாறிலி ggml_tensor * src1 = dst->src[1]; // convolution kernel

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி int32_t s0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t s1 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *)(dst->op_params))[2];
    மாறிலி int32_t p1 = ((மாறிலி int32_t *)(dst->op_params))[3];
    மாறிலி int32_t d0 = ((மாறிலி int32_t *)(dst->op_params))[4];
    மாறிலி int32_t d1 = ((மாறிலி int32_t *)(dst->op_params))[5];
    மாறிலி தருக்க is_2D = ((மாறிலி int32_t *)(dst->op_params))[6] == 1;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t N  = is_2D ? ne3 : ne2;
    மாறிலி int64_t IC = is_2D ? ne2 : ne1;
    மாறிலி int64_t IH = is_2D ? ne1 : 1;
    மாறிலி int64_t IW = ne0;

    மாறிலி int64_t KH = is_2D ? ne11 : 1;
    மாறிலி int64_t KW = ne10;

    மாறிலி int64_t OH = is_2D ? ne02 : 1;
    மாறிலி int64_t OW = ne01;

    முழு ofs0 = is_2D ? nb3 : nb2;
    முழு ofs1 = is_2D ? nb2 : nb1;

    GGML_ASSERT(nb0  == அளவு(தசம));

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    {
        தசம * மாறிலி wdata = (தசம *) dst->data;

        க்கு (int64_t in = 0; in < N; in++) {
            க்கு (int64_t iic = ith; iic < IC; iic += nth) {
                க்கு (int64_t iih = 0; iih < IH; iih++) {
                    க்கு (int64_t iiw = 0; iiw < IW; iiw++) {

                        // micro kernel
                        தசம grad = 0.0f;
                        க்கு (int64_t ikh = 0; ikh < KH; ikh++) {
                            க்கு (int64_t ikw = 0; ikw < KW; ikw++) {
                                // For s0 > 1 some values were skipped over in the forward pass.
                                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                                மாறிலி int64_t tmpw = (iiw + p0 - ikw*d0);
                                எனில் (tmpw % s0 != 0) {
                                    தொடர்;
                                }
                                மாறிலி int64_t iow = tmpw / s0;

                                // Equivalent logic as above except for s1.
                                int64_t ioh;
                                எனில் (is_2D) {
                                    மாறிலி int64_t tmph = iih + p1 - ikh*d1;

                                    எனில் (tmph % s1 != 0) {
                                        தொடர்;
                                    }

                                    ioh = tmph / s1;
                                } இல்லையெனில் {
                                    ioh = 0;
                                }

                                எனில் (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
                                    தொடர்;
                                }

                                மாறிலி தசம * மாறிலி grad_in = (மாறிலி தசம *) src0->data
                                    + (in*OH*OW + ioh*OW + iow)*(IC*KH*KW); // [IC, KH, KW]
                                grad += grad_in[iic*(KH*KW) + ikh*KW + ikw];
                            }
                        }
                        தசம * dst_data = (தசம *)((எழுத்து *) wdata + (in*ofs0 + iic*ofs1)); // [IH, IW]
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
நிலையான வெறுமை ggml_compute_forward_im2col_3d_f16(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16);

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி int32_t s0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t s1 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t s2 = ((மாறிலி int32_t *)(dst->op_params))[2];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *)(dst->op_params))[3];
    மாறிலி int32_t p1 = ((மாறிலி int32_t *)(dst->op_params))[4];
    மாறிலி int32_t p2 = ((மாறிலி int32_t *)(dst->op_params))[5];
    மாறிலி int32_t d0 = ((மாறிலி int32_t *)(dst->op_params))[6];
    மாறிலி int32_t d1 = ((மாறிலி int32_t *)(dst->op_params))[7];
    மாறிலி int32_t d2 = ((மாறிலி int32_t *)(dst->op_params))[8];
    மாறிலி int32_t IC = ((மாறிலி int32_t *)(dst->op_params))[9];


    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t N  = ne13 / IC;
    மாறிலி int64_t ID = ne12;
    மாறிலி int64_t IH = ne11;
    மாறிலி int64_t IW = ne10;

    மாறிலி int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    மாறிலி int64_t KD = ne02;
    மாறிலி int64_t KH = ne01;
    மாறிலி int64_t KW = ne00;

    மாறிலி int64_t OD = ne3 / N;
    மாறிலி int64_t OH = ne2;
    மாறிலி int64_t OW = ne1;
    மாறிலி int64_t OH_OW = OH*OW;
    மாறிலி int64_t KD_KH_KW = KD*KH*KW;
    மாறிலி int64_t KH_KW = KH*KW;
    மாறிலி int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == அளவு(தசம));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        ggml_fp16_t * மாறிலி wdata = (ggml_fp16_t *) dst->data;

        க்கு (int64_t in = 0; in < N; in++) {
            க்கு (int64_t iod = 0; iod < OD; iod++) {
                க்கு (int64_t ioh = 0; ioh < OH; ioh++) {
                    க்கு (int64_t iow = 0; iow < OW; iow++) {
                        க்கு (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            ggml_fp16_t * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            மாறிலி தசம * மாறிலி src_data = (மாறிலி தசம *) ((மாறிலி எழுத்து *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            க்கு (int64_t ikd = 0; ikd < KD; ikd++) {
                                க்கு (int64_t ikh = 0; ikh < KH; ikh++) {
                                    க்கு (int64_t ikw = 0; ikw < KW; ikw++) {
                                        மாறிலி int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        மாறிலி int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        மாறிலி int64_t iid = iod*s2 + ikd*d2 - p2;

                                        எனில் (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } இல்லையெனில் {
                                            மாறிலி தசம * மாறிலி s = (மாறிலி தசம *) ((மாறிலி எழுத்து *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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
நிலையான வெறுமை ggml_compute_forward_im2col_3d_f32(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS;

    மாறிலி int32_t s0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t s1 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t s2 = ((மாறிலி int32_t *)(dst->op_params))[2];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *)(dst->op_params))[3];
    மாறிலி int32_t p1 = ((மாறிலி int32_t *)(dst->op_params))[4];
    மாறிலி int32_t p2 = ((மாறிலி int32_t *)(dst->op_params))[5];
    மாறிலி int32_t d0 = ((மாறிலி int32_t *)(dst->op_params))[6];
    மாறிலி int32_t d1 = ((மாறிலி int32_t *)(dst->op_params))[7];
    மாறிலி int32_t d2 = ((மாறிலி int32_t *)(dst->op_params))[8];
    மாறிலி int32_t IC = ((மாறிலி int32_t *)(dst->op_params))[9];


    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t N  = ne13 / IC;
    மாறிலி int64_t ID = ne12;
    மாறிலி int64_t IH = ne11;
    மாறிலி int64_t IW = ne10;

    மாறிலி int64_t OC = ne03 / IC;
    GGML_UNUSED(OC);
    மாறிலி int64_t KD = ne02;
    மாறிலி int64_t KH = ne01;
    மாறிலி int64_t KW = ne00;

    மாறிலி int64_t OD = ne3 / N;
    மாறிலி int64_t OH = ne2;
    மாறிலி int64_t OW = ne1;

    மாறிலி int64_t OH_OW = OH*OW;
    மாறிலி int64_t KD_KH_KW = KD*KH*KW;
    மாறிலி int64_t KH_KW = KH*KW;
    மாறிலி int64_t IC_KD_KH_KW = IC*KD*KH*KW;

    GGML_ASSERT(nb10 == அளவு(தசம));

    // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
    {
        தசம * மாறிலி wdata = (தசம *) dst->data;

        க்கு (int64_t in = 0; in < N; in++) {
            க்கு (int64_t iod = 0; iod < OD; iod++) {
                க்கு (int64_t ioh = 0; ioh < OH; ioh++) {
                    க்கு (int64_t iow = 0; iow < OW; iow++) {
                        க்கு (int64_t iic = ith; iic < IC; iic += nth) {

                            // micro kernel
                            தசம * dst_data = wdata + (in*OD*OH_OW + iod*OH_OW + ioh*OW + iow)*IC_KD_KH_KW; // [IC, KD, KH, KW]
                            மாறிலி தசம * மாறிலி src_data = (மாறிலி தசம *) ((மாறிலி எழுத்து *)src1->data + (in*IC + iic)*nb13); // [ID, IH, IW]

                            க்கு (int64_t ikd = 0; ikd < KD; ikd++) {
                                க்கு (int64_t ikh = 0; ikh < KH; ikh++) {
                                    க்கு (int64_t ikw = 0; ikw < KW; ikw++) {
                                        மாறிலி int64_t iiw = iow*s0 + ikw*d0 - p0;
                                        மாறிலி int64_t iih = ioh*s1 + ikh*d1 - p1;
                                        மாறிலி int64_t iid = iod*s2 + ikd*d2 - p2;

                                        எனில் (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
                                            dst_data[iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw] = 0;
                                        } இல்லையெனில் {
                                            மாறிலி தசம * மாறிலி s = (மாறிலி தசம *) ((மாறிலி எழுத்து *)src_data + iid*nb12 + iih*nb11 + iiw*nb10); // [ID, IH, IW]
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


வெறுமை ggml_compute_forward_im2col_3d(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {
    தேர்வு (dst->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_im2col_3d_f16(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_im2col_3d_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_call_mul_mat(ggml_type type, மாறிலி ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                              வெறுமை * a, வெறுமை * b, தசம * c) {
    மாறிலி ggml_type_traits * traits = ggml_get_type_traits(type);
    கட்டமைப்பு ggml_tensor src1 = {};
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

    கட்டமைப்பு ggml_tensor src0 = {};
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

    கட்டமைப்பு ggml_tensor dst = {};
    dst.ne[0] = n;
    dst.ne[1] = m;
    dst.ne[2] = 1;
    dst.ne[3] = 1;
    dst.nb[0] = அளவு(தசம);
    dst.nb[1] = n * அளவு(தசம);
    dst.nb[2] = dst.nb[1];
    dst.nb[3] = dst.nb[2];
    dst.data  = c;
    dst.src[0] = &src0;
    dst.src[1] = &src1;

    ggml_compute_forward_mul_mat(params, &dst);
}

நிலையான உள்வரி int64_t ggml_wrap_around(int64_t coord, int64_t size) {
    திரும்பு (coord  + size) % size; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
நிலையான வெறுமை ggml_compute_forward_col2im_1d_impl(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src = dst->src[0];  // [K*OC, T_in]

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    மாறிலி int32_t s0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t OC = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t p0 = ((மாறிலி int32_t *)(dst->op_params))[2];

    மாறிலி int64_t K_OC = src->ne[0];
    மாறிலி int64_t T_in = src->ne[1];
    மாறிலி int64_t K    = K_OC / OC;
    மாறிலி int64_t T_out = dst->ne[0];

    மாறிலி elem_t * col_data = (மாறிலி elem_t *) src->data;
    elem_t       * dst_data = (elem_t *) dst->data;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // Parallelize over the time axis: the split stays balanced whatever OC is,
    // down to OC = 1 for mono audio, and threads read disjoint column bands
    மாறிலி int64_t dr = (T_out + nth - 1) / nth;
    மாறிலி int64_t it0 = dr * ith;
    மாறிலி int64_t it1 = it0 + dr < T_out ? it0 + dr : T_out;

    க்கு (int64_t oc = 0; oc < OC; oc++) {
        க்கு (int64_t t_out = it0; t_out < it1; t_out++) {
            மாறிலி int64_t t_abs = t_out + p0;  // absolute position in uncropped signal
            // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
            int64_t t_in_min = (t_abs - K + 1 + s0 - 1) / s0;  // ceil((t_abs-K+1)/s)
            எனில் (t_in_min < 0) t_in_min = 0;
            int64_t t_in_max = t_abs / s0;
            எனில் (t_in_max >= T_in) t_in_max = T_in - 1;

            தசம sum = 0.0f;
            க்கு (int64_t t_in = t_in_min; t_in <= t_in_max; t_in++) {
                int64_t k = t_abs - t_in * s0;
                எனில் (k >= 0 && k < K) {
                    // col layout: [K*OC, T_in], element (oc*K+k, t_in)
                    sum += type_conversion_table<elem_t>::to_f32(col_data[(oc * K + k) + t_in * K_OC]);
                }
            }
            // dst layout: [T_out, OC], element (t_out, oc)
            dst_data[t_out + oc * T_out] = type_conversion_table<elem_t>::from_f32(sum);
        }
    }
}

வெறுமை ggml_compute_forward_col2im_1d(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    தேர்வு (dst->src[0]->type) {
        நிலை GGML_TYPE_F32:  ggml_compute_forward_col2im_1d_impl<தசம>      (params, dst); நிறுத்து;
        நிலை GGML_TYPE_F16:  ggml_compute_forward_col2im_1d_impl<ggml_fp16_t>(params, dst); நிறுத்து;
        நிலை GGML_TYPE_BF16: ggml_compute_forward_col2im_1d_impl<ggml_bf16_t>(params, dst); நிறுத்து;
        இயல்பு: GGML_ABORT("col2im_1d: unsupported type %d", dst->src[0]->type);
    }
}

// ggml_compute_forward_conv_2d


நிலையான வெறுமை ggml_compute_forward_conv_2d_impl(மாறிலி ggml_compute_params * params,
                                              மாறிலி ggml_tensor *         kernel,  // [KW, KH, IC, OC]
                                              மாறிலி ggml_tensor *         src,     // [W, H, C, N]
                                              ggml_tensor *               dst,     // [OW, OH, OC, N]
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    மாறிலி ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    மாறிலி int32_t stride_x   = dst->op_params[0];
    மாறிலி int32_t stride_y   = dst->op_params[1];
    மாறிலி int32_t pad_x      = dst->op_params[2];
    மாறிலி int32_t pad_y      = dst->op_params[3];
    மாறிலி int32_t dilation_x = dst->op_params[4];
    மாறிலி int32_t dilation_y = dst->op_params[5];

    மாறிலி int64_t c_in  = src->ne[2];
    மாறிலி int64_t c_out = kernel->ne[3];
    GGML_ASSERT(c_in == kernel->ne[2]);

    மாறிலி int64_t src_w = src->ne[0];
    மாறிலி int64_t src_h = src->ne[1];
    மாறிலி int64_t knl_w = kernel->ne[0];
    மாறிலி int64_t knl_h = kernel->ne[1];
    மாறிலி int64_t dst_w = dst->ne[0];
    மாறிலி int64_t dst_h = dst->ne[1];

    மாறிலி தசம * src_data = (தசம *) src->data;
    வெறுமை  * knl_data       = kernel->data;
    தசம * dst_data       = (தசம *) dst->data;

    மாறிலி int64_t knl_n           = knl_w * knl_h * c_in;
    மாறிலி int64_t patch_total     = dst->ne[3] * dst_w * dst_h;

    மாறிலி int64_t space_per_patch   = knl_n * traits->type_size + c_out * அளவு(தசம);
    மாறிலி int64_t batch_size        = params->wsize / space_per_patch;
    மாறிலி int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    மாறிலி int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    வெறுமை * tmp = params->wdata;

    க்கு (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

        மாறிலி int64_t patch_start_batch = batch_i * patches_per_batch;
        மாறிலி int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch,
                                              patch_total);
        மாறிலி int64_t patch_n           = patch_end_batch - patch_start_batch;

        மாறிலி int64_t patch_per_thread  = (patch_n + params->nth - 1) / params->nth;
        மாறிலி int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        மாறிலி int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        //im2col for a patch
        க்கு (int64_t p = patch_start; p < patch_end; ++p) {
            மாறிலி int64_t  batch_n     =  p / (dst_w * dst_h);
            மாறிலி int64_t  src_x       = (p / dst_w) % dst_h;
            மாறிலி int64_t  src_y       =  p % dst_w;

            மாறிலி தசம * src_base = (மாறிலி தசம *)((மாறிலி எழுத்து *)src_data + batch_n * src->nb[3]);
            எழுத்து *        dst_row  = (எழுத்து *) tmp + (p % patches_per_batch) * knl_n * traits->type_size;

            க்கு (int64_t ic = 0; ic < c_in; ++ic) {
                க்கு (int64_t ky = 0; ky < knl_h; ++ky) {
                    க்கு (int64_t kx = 0; kx < knl_w; ++kx) {
                        மாறிலி int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
                        மாறிலி int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

                        int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

                        தசம src_val;
                        எனில் (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                            src_val = 0.0f;
                        } இல்லையெனில் {
                            மாறிலி தசம * src_ptr = (மாறிலி தசம *)((மாறிலி எழுத்து *)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
                            src_val               = *src_ptr;
                        }

                        எழுத்து * element_ptr = dst_row + dst_idx * traits->type_size;
                        எனில் (kernel_type == GGML_TYPE_F32) {
                            *(தசம *) element_ptr = src_val;
                        } இல்லையெனில் எனில் (kernel_type == GGML_TYPE_F16) {
                            *(ggml_fp16_t *) element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                        }
                    }
                }
            }
        }   // patches handled by this thread

        ggml_barrier(params->threadpool);

        தசம * gemm_output = (தசம *) ((எழுத்து *) tmp + patches_per_batch * knl_n * traits->type_size);

        GGML_ASSERT(gemm_output + patch_n * c_out <= (தசம*)tmp + params->wsize);

        // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
        ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);


        //permute back [OC, N, OH, OW] to [N, OC, OH, OW]
        மாறிலி int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
        மாறிலி int64_t permute_start = params->ith * permute_per_thread;
        மாறிலி int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

        க்கு (int64_t i = permute_start; i < permute_end; ++i) {
            மாறிலி int64_t p       = patch_start_batch + i;
            மாறிலி int64_t batch_n = p / (dst_w * dst_h);
            மாறிலி int64_t dst_y   = (p / dst_w) % dst_h;
            மாறிலி int64_t dst_x   = p % dst_w;

            க்கு (int64_t oc = 0; oc < c_out; ++oc) {
                மாறிலி தசம value = gemm_output[i * c_out + oc];
                தசம * dst_ptr = (தசம *)((எழுத்து *)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

வெறுமை ggml_compute_forward_conv_2d(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
}

// ggml_compute_forward_conv_3d

நிலையான வெறுமை ggml_compute_forward_conv_3d_impl(மாறிலி ggml_compute_params * params,
                                              மாறிலி ggml_tensor *         kernel,
                                              மாறிலி ggml_tensor *         src,
                                              ggml_tensor *               dst,
                                              ggml_type                   kernel_type) {

    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
    GGML_ASSERT(kernel->type == kernel_type);

    மாறிலி ggml_type_traits * traits = ggml_get_type_traits(kernel_type);

    மாறிலி int32_t s0 = dst->op_params[0];
    மாறிலி int32_t s1 = dst->op_params[1];
    மாறிலி int32_t s2 = dst->op_params[2];
    மாறிலி int32_t p0 = dst->op_params[3];
    மாறிலி int32_t p1 = dst->op_params[4];
    மாறிலி int32_t p2 = dst->op_params[5];
    மாறிலி int32_t d0 = dst->op_params[6];
    மாறிலி int32_t d1 = dst->op_params[7];
    மாறிலி int32_t d2 = dst->op_params[8];
    மாறிலி int32_t c  = dst->op_params[9];
    மாறிலி int32_t n  = dst->op_params[10];
    மாறிலி int32_t oc = dst->op_params[11];

    மாறிலி int64_t src_w = src->ne[0];
    மாறிலி int64_t src_h = src->ne[1];
    மாறிலி int64_t src_d = src->ne[2];
    மாறிலி int64_t knl_w = kernel->ne[0];
    மாறிலி int64_t knl_h = kernel->ne[1];
    மாறிலி int64_t knl_d = kernel->ne[2];
    மாறிலி int64_t dst_w = dst->ne[0];
    மாறிலி int64_t dst_h = dst->ne[1];
    மாறிலி int64_t dst_d = dst->ne[2];

    மாறிலி தசம * src_data = (தசம *) src->data;
    வெறுமை  * knl_data       = kernel->data;
    தசம * dst_data       = (தசம *) dst->data;

    மாறிலி int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
    மாறிலி int64_t knl_n_total       = knl_n_per_channel * c;
    மாறிலி int64_t patch_total       = n * dst_w * dst_h * dst_d;

    மாறிலி int64_t space_per_patch   = knl_n_total * traits->type_size + oc * அளவு(தசம);
    மாறிலி int64_t batch_size        = params->wsize / space_per_patch;
    மாறிலி int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
    மாறிலி int64_t batch_n           = (patch_total + patches_per_batch - 1) / patches_per_batch;

    GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

    வெறுமை * tmp = params->wdata;

    க்கு (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {
        மாறிலி int64_t patch_start_batch = batch_i * patches_per_batch;
        மாறிலி int64_t patch_end_batch   = std::min(patch_start_batch + patches_per_batch, patch_total);
        மாறிலி int64_t patch_n_in_batch  = patch_end_batch - patch_start_batch;

        மாறிலி int64_t patch_per_thread  = (patch_n_in_batch + params->nth - 1) / params->nth;
        மாறிலி int64_t patch_start       = patch_start_batch + params->ith * patch_per_thread;
        மாறிலி int64_t patch_end         = std::min(patch_start + patch_per_thread, patch_end_batch);

        க்கு (int64_t p = patch_start; p < patch_end; ++p) {
            மாறிலி int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            மாறிலி int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            மாறிலி int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            மாறிலி int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            மாறிலி int64_t dst_y      = p_in_depth / dst_w;
            மாறிலி int64_t dst_x      = p_in_depth % dst_w;

            எழுத்து * dst_row = (எழுத்து *) tmp + (p % patches_per_batch) * knl_n_total * traits->type_size;

            க்கு (int64_t ic = 0; ic < c; ++ic) {
                க்கு (int64_t kz = 0; kz < knl_d; ++kz) {
                    க்கு (int64_t ky = 0; ky < knl_h; ++ky) {
                        க்கு (int64_t kx = 0; kx < knl_w; ++kx) {
                            மாறிலி int64_t sz = dst_z * s2 + kz * d2 - p2;
                            மாறிலி int64_t sy = dst_y * s1 + ky * d1 - p1;
                            மாறிலி int64_t sx = dst_x * s0 + kx * d0 - p0;

                            int64_t dst_idx = ic * knl_n_per_channel + kz * (knl_h * knl_w) + ky * knl_w + kx;

                            தசம src_val;
                            எனில் (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
                                src_val = 0.0f;
                            } இல்லையெனில் {
                                மாறிலி int64_t cn_idx = batch_idx * c + ic;
                                மாறிலி தசம * src_ptr = (மாறிலி தசம *)((மாறிலி எழுத்து *)src_data + sx*src->nb[0] + sy*src->nb[1] + sz*src->nb[2] + cn_idx*src->nb[3]);
                                src_val = *src_ptr;
                            }

                            எழுத்து * element_ptr = dst_row + dst_idx * traits->type_size;
                            எனில் (kernel_type == GGML_TYPE_F32) {
                                *(தசம *)element_ptr = src_val;
                            } இல்லையெனில் எனில் (kernel_type == GGML_TYPE_F16) {
                                *(ggml_fp16_t *)element_ptr = GGML_CPU_FP32_TO_FP16(src_val);
                            }
                        }
                    }
                }
            }
        }

        ggml_barrier(params->threadpool);

        தசம * gemm_output = (தசம *) ((எழுத்து *) tmp + patches_per_batch * knl_n_total * traits->type_size);
        ggml_call_mul_mat(kernel_type, params, patch_n_in_batch, oc, knl_n_total, tmp, knl_data, gemm_output);

        ggml_barrier(params->threadpool);

        மாறிலி int64_t permute_per_thread = (patch_n_in_batch + params->nth - 1) / params->nth;
        மாறிலி int64_t permute_start = params->ith * permute_per_thread;
        மாறிலி int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n_in_batch);

        க்கு (int64_t i = permute_start; i < permute_end; ++i) {
            மாறிலி int64_t p = patch_start_batch + i;
            மாறிலி int64_t p_in_batch = p % (dst_w * dst_h * dst_d);
            மாறிலி int64_t p_in_depth = p_in_batch % (dst_w * dst_h);
            மாறிலி int64_t batch_idx  = p / (dst_w * dst_h * dst_d);
            மாறிலி int64_t dst_z      = p_in_batch / (dst_w * dst_h);
            மாறிலி int64_t dst_y      = p_in_depth / dst_w;
            மாறிலி int64_t dst_x      = p_in_depth % dst_w;

            க்கு (int64_t ioc = 0; ioc < oc; ++ioc) {
                மாறிலி தசம value = gemm_output[i * oc + ioc];
                மாறிலி int64_t ocn_idx = batch_idx * oc + ioc;
                தசம * dst_ptr = (தசம *)((எழுத்து *)dst_data + dst_x*dst->nb[0] + dst_y*dst->nb[1] + dst_z*dst->nb[2] + ocn_idx*dst->nb[3]);
                *dst_ptr = value;
            }
        }
    }
}

வெறுமை ggml_compute_forward_conv_3d(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    ggml_compute_forward_conv_3d_impl(params, src0, src1, dst, src0->type);
}

template <typename kernel_t>
நிலையான வெறுமை ggml_compute_forward_conv_transpose_2d_impl(
    மாறிலி ggml_compute_params * params,
          ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nk = ne00*ne01*ne02*ne03;

    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == அளவு(தசம));

    எனில் (ith == 0) {
        memset(params->wdata, 0, params->wsize);

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        {
            kernel_t * மாறிலி wdata = (kernel_t *) params->wdata + 0;

            க்கு (int64_t i03 = 0; i03 < ne03; i03++) {
                க்கு (int64_t i02 = 0; i02 < ne02; i02++) {
                    மாறிலி kernel_t * மாறிலி src = (kernel_t *)((எழுத்து *) src0->data + i03*nb03 + i02*nb02);
                    kernel_t * dst_data = wdata + i02*ne01*ne00*ne03;
                    க்கு (int64_t i01 = 0; i01 < ne01; i01++) {
                        க்கு (int64_t i00 = 0; i00 < ne00; i00++) {
                            dst_data[i01*ne00*ne03 + i00*ne03 + i03] = src[i01 * ne00 + i00];
                        }
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh), for all batches
        {
            kernel_t * மாறிலி wdata = (kernel_t *) params->wdata + nk;
            க்கு (முழு i13 = 0; i13 < ne13; i13++) {
                kernel_t * மாறிலி wdata_b = wdata + i13*ne10*ne11*ne12;
                க்கு (முழு i12 = 0; i12 < ne12; i12++) {
                    க்கு (முழு i11 = 0; i11 < ne11; i11++) {
                        மாறிலி தசம * மாறிலி src = (தசம *)((எழுத்து *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
                        kernel_t * dst_data = wdata_b + i11*ne10*ne12;
                        க்கு (முழு i10 = 0; i10 < ne10; i10++) {
                            எனில் constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                dst_data[i10*ne12 + i12] = GGML_CPU_FP32_TO_FP16(src[i10]);
                            } இல்லையெனில் {
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

    மாறிலி int32_t stride = ggml_get_op_params_i32(dst, 0);

    // total patches in dst
    மாறிலி முழு np = ne2;

    // patches per thread
    மாறிலி முழு dp = (np + nth - 1)/nth;

    // patch range for this thread
    மாறிலி முழு ip0 = dp*ith;
    மாறிலி முழு ip1 = MIN(ip0 + dp, np);

    kernel_t * மாறிலி wdata = (kernel_t *) params->wdata + 0;
    kernel_t * மாறிலி wdata_src = wdata + nk;

    க்கு (முழு i2 = ip0; i2 < ip1; i2++) { // Cout
        kernel_t * wdata_kernel = wdata + i2*ne01*ne00*ne03;
        க்கு (முழு i3 = 0; i3 < ne3; i3++) { // batch
            தசம * dst_data = (தசம *)((எழுத்து *) dst->data + i3*nb3 + i2*nb2);
            kernel_t * wdata_src_b = wdata_src + i3*ne10*ne11*ne12;
            க்கு (முழு i11 = 0; i11 < ne11; i11++) {
                க்கு (முழு i10 = 0; i10 < ne10; i10++) {
                    மாறிலி முழு i1n = i11*ne10*ne12 + i10*ne12;
                    க்கு (முழு i01 = 0; i01 < ne01; i01++) {
                        க்கு (முழு i00 = 0; i00 < ne00; i00++) {
                            தசம v = 0;
                            எனில் constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
                                ggml_vec_dot_f16(ne03, &v, 0,
                                        wdata_src_b + i1n, 0,
                                        wdata_kernel + i01*ne00*ne03 + i00*ne03, 0, 1);
                            } இல்லையெனில் {
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

வெறுமை ggml_compute_forward_conv_transpose_2d(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_conv_transpose_2d_impl<ggml_fp16_t>(params, dst);
            } நிறுத்து;
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_conv_transpose_2d_impl<தசம>(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_conv_2d_dw

கட்டமைப்பு ggml_conv_2d_dw_params {
    int64_t channels;
    int64_t batch;
    int64_t src_w;
    int64_t src_h;
    int64_t dst_w;
    int64_t dst_h;
    int64_t knl_w;
    int64_t knl_h;
    முழு stride_x;
    முழு stride_y;
    முழு pad_x;
    முழு pad_y;
    முழு dilation_x;
    முழு dilation_y;
};

நிலையான உள்வரி தசம ggml_conv_2d_dw_knl_f32(மாறிலி எழுத்து * data, int64_t i, ggml_type type) {
    எனில் (type == GGML_TYPE_F16) {
        திரும்பு GGML_FP16_TO_FP32(((மாறிலி ggml_fp16_t *)data)[i]);
    }
    திரும்பு ((மாறிலி தசம *)data)[i];
}

நிலையான வெறுமை ggml_compute_forward_conv_2d_dw_cwhn(
        மாறிலி ggml_compute_params * params,
        மாறிலி ggml_tensor * src,
        மாறிலி ggml_tensor * kernel,
        ggml_tensor * dst,
        மாறிலி ggml_conv_2d_dw_params & p) {

    மாறிலி int64_t c = p.channels;
    மாறிலி எழுத்து * knl_data = (மாறிலி எழுத்து *)kernel->data;
    மாறிலி ggml_type knl_type = kernel->type;

    மாறிலி int64_t rows_total = p.dst_h * p.batch;
    மாறிலி int64_t rows_per_thread = (rows_total + params->nth - 1) / params->nth;
    மாறிலி int64_t row_start = params->ith * rows_per_thread;
    மாறிலி int64_t row_end = MIN(row_start + rows_per_thread, rows_total);

#ifdef GGML_SIMD
    int64_t c_pkg_end = 0;
    int64_t pkg_size = GGML_F32_EPR;
    எனில் (knl_type == GGML_TYPE_F32) {
    #எனில் defined(__ARM_FEATURE_SVE)
        pkg_size = svcntw();
    #இல்லையெனில்
        pkg_size = GGML_F32_EPR;
    #endif
        c_pkg_end = (c / pkg_size) * pkg_size;
    }
#இல்லையெனில்
    மாறிலி int64_t c_pkg_end = 0;
#endif

    க்கு (int64_t row = row_start; row < row_end; ++row) {
        மாறிலி int64_t dst_y = row % p.dst_h;
        மாறிலி தசம * src_data = (மாறிலி தசம *)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
        க்கு (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
            தசம * dst_data = (தசம *)dst->data + (row * p.dst_w + dst_x) * c;
            மாறிலி int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
            மாறிலி int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
            க்கு (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
                GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
                க்கு (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    மாறிலி int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    எனில் (src_y < 0 || src_y >= p.src_h) {
                        தொடர்;
                    }
                    க்கு (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        மாறிலி int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        எனில் (src_x < 0 || src_x >= p.src_w) {
                            தொடர்;
                        }
                        மாறிலி தசம * kp = (மாறிலி தசம *)knl_data + (knl_y * p.knl_w + knl_x) * c + c_i;
                        GGML_F32_VEC k = GGML_F32_VEC_LOAD(kp);
                        GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
                        sum = GGML_F32_VEC_FMA(sum, k, s);
                    }
                }
                GGML_F32_VEC_STORE(dst_data + c_i, sum);
            }
#endif
            க்கு (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
                தசம sum = 0.0f;
                க்கு (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    மாறிலி int64_t src_y = src_y_base + knl_y * p.dilation_y;
                    எனில் (src_y < 0 || src_y >= p.src_h) {
                        தொடர்;
                    }
                    க்கு (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        மாறிலி int64_t src_x = src_x_base + knl_x * p.dilation_x;
                        எனில் (src_x < 0 || src_x >= p.src_w) {
                            தொடர்;
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

நிலையான வெறுமை ggml_compute_forward_conv_2d_dw_whcn(
        மாறிலி ggml_compute_params * params,
        மாறிலி ggml_tensor * src,
        மாறிலி ggml_tensor * kernel,
        ggml_tensor * dst,
        மாறிலி ggml_conv_2d_dw_params & p) {

    மாறிலி int64_t n = p.channels * p.batch;
    மாறிலி int64_t per_thread = (n + params->nth - 1) / params->nth;
    மாறிலி int64_t start = params->ith * per_thread;
    மாறிலி int64_t end = MIN(start + per_thread, n);
    மாறிலி எழுத்து * knl_base = (மாறிலி எழுத்து *)kernel->data;
    மாறிலி ggml_type knl_type = kernel->type;

    க்கு (int64_t i = start; i < end; ++i) {
        மாறிலி int64_t knl_offset = (i % p.channels) * p.knl_w * p.knl_h;
        மாறிலி தசம * src_data = (மாறிலி தசம *)src->data + i * p.src_w * p.src_h;
        தசம * dst_data = (தசம *)dst->data + i * p.dst_w * p.dst_h;

        க்கு (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
            க்கு (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {

                தசம sum = 0.0f;
                க்கு (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
                    மாறிலி int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
                    எனில் (src_y < 0 || src_y >= p.src_h) {
                        தொடர்;
                    }
                    க்கு (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
                        மாறிலி int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
                        எனில் (src_x < 0 || src_x >= p.src_w) {
                            தொடர்;
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

வெறுமை ggml_compute_forward_conv_2d_dw(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * kernel = dst->src[0];
    மாறிலி ggml_tensor * src = dst->src[1];
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

    எனில் (ggml_is_contiguous(src)) {
        ggml_compute_forward_conv_2d_dw_whcn(params, src, kernel, dst, p);
    } இல்லையெனில் எனில் (ggml_is_contiguous_channels(src)) {
        GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
        ggml_compute_forward_conv_2d_dw_cwhn(params, src, kernel, dst, p);
    } இல்லையெனில் {
        GGML_ABORT("non-contiguous memory layout not supported");
    }
}

// ggml_compute_forward_pool_1d_ksp
நிலையான வெறுமை ggml_compute_forward_pool_1d_ksp(
        மாறிலி ggml_compute_params * params,
        மாறிலி ggml_op_pool op,
        மாறிலி முழு k,
        மாறிலி முழு s,
        மாறிலி முழு p,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src = dst->src[0];

    ப்ரதிஜ்ஞா(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    மாறிலி int64_t IW = src->ne[0];
    மாறிலி int64_t OW = dst->ne[0];

    மாறிலி int64_t nr = ggml_nrows(src);

    க்கு (int64_t ir = 0; ir < nr; ++ir) {
        மாறிலி எழுத்து * srow_bytes =            (மாறிலி எழுத்து *) src->data + ir * src->nb[1];
        தசம      * drow       = (தசம *) ((      எழுத்து *) dst->data + ir * dst->nb[1]);

        க்கு (int64_t ow = 0; ow < OW; ++ow) {
            தசம res = 0;
            தேர்வு (op) {
                நிலை GGML_OP_POOL_AVG: res = 0.0f;     நிறுத்து;
                நிலை GGML_OP_POOL_MAX: res = -FLT_MAX; நிறுத்து;
                நிலை GGML_OP_POOL_மொத்தம்: GGML_ABORT("fatal error");
            }

            முழு count = 0;
            மாறிலி முழு base = (முழு) ow * s - p;

            க்கு (முழு ki = 0; ki < k; ++ki) {
                மாறிலி முழு j = base + ki;
                எனில் (j < 0 || j >= (முழு) IW) {
                    தொடர்;
                }

                தசம v;
                எனில் (src->type == GGML_TYPE_F32) {
                    v = ((மாறிலி தசம *) srow_bytes)[j];
                } இல்லையெனில் {
                    v = GGML_CPU_FP16_TO_FP32(((மாறிலி ggml_fp16_t *) srow_bytes)[j]);
                }

                தேர்வு (op) {
                    நிலை GGML_OP_POOL_AVG: res += v;                நிறுத்து;
                    நிலை GGML_OP_POOL_MAX: res =  std::max(v, res); நிறுத்து;
                    நிலை GGML_OP_POOL_மொத்தம்: GGML_ABORT("fatal error");
                }

                ++count;
            }

            தேர்வு (op) {
                நிலை GGML_OP_POOL_AVG: res = (count > 0) ? (res / count) : 0.0f; நிறுத்து;
                நிலை GGML_OP_POOL_MAX:                                           நிறுத்து;
                நிலை GGML_OP_POOL_மொத்தம்: GGML_ABORT("fatal error");
            }

            drow[ow] = res;
        }
    }
}

// ggml_compute_forward_pool_1d

வெறுமை ggml_compute_forward_pool_1d(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி int32_t * opts = (மாறிலி int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    மாறிலி முழு k0 = opts[1];
    மாறிலி முழு s0 = opts[2];
    மாறிலி முழு p0 = opts[3];

    ggml_compute_forward_pool_1d_ksp(params, op, k0, s0, p0, dst);
}

// ggml_compute_forward_pool_2d

வெறுமை ggml_compute_forward_pool_2d(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src = dst->src[0];

    ப்ரதிஜ்ஞா(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    மாறிலி int32_t * opts = (மாறிலி int32_t *)dst->op_params;

    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    மாறிலி முழு k0 = opts[1];
    மாறிலி முழு k1 = opts[2];
    மாறிலி முழு s0 = opts[3];
    மாறிலி முழு s1 = opts[4];
    மாறிலி முழு p0 = opts[5];
    மாறிலி முழு p1 = opts[6];
    மாறிலி எழுத்து * cdata = (மாறிலி எழுத்து*)src->data;
    மாறிலி எழுத்து * மாறிலி data_end = cdata + ggml_nbytes(src);

    மாறிலி int64_t px = dst->ne[0];
    மாறிலி int64_t py = dst->ne[1];
    மாறிலி int64_t pa = px * py;

    தசம * dplane = (தசம *)dst->data;

    மாறிலி முழு ka = k0 * k1;
    மாறிலி முழு offset0 = -p0;
    மாறிலி முழு offset1 = -p1;

    போது (cdata < data_end) {
        க்கு (முழு oy = 0; oy < py; ++oy) {
            தசம * மாறிலி drow = dplane + oy * px;
            தசம * மாறிலி out  = drow;

            க்கு (முழு ox = 0; ox < px; ++ox) {
                தசம res = 0;
                தேர்வு (op) {
                    நிலை GGML_OP_POOL_AVG: res = 0;        நிறுத்து;
                    நிலை GGML_OP_POOL_MAX: res = -FLT_MAX; நிறுத்து;
                    நிலை GGML_OP_POOL_மொத்தம்: GGML_ABORT("fatal error");
                }

                மாறிலி முழு ix = offset0 + ox * s0;
                மாறிலி முழு iy = offset1 + oy * s1;

                க்கு (முழு ky = 0; ky < k1; ++ky) {
                    எனில் (iy + ky < 0 || iy + ky >= src->ne[1]) {
                        தொடர்;
                    }

                    மாறிலி வெறுமை * srow = (மாறிலி வெறுமை *)(cdata + src->nb[1] * (iy + ky));
                    க்கு (முழு kx = 0; kx < k0; ++kx) {
                        முழு j = ix + kx;
                        எனில் (j < 0 || j >= src->ne[0]) {
                            தொடர்;
                        }

                        மாறிலி தசம srow_j = (src->type == GGML_TYPE_F32) ? ((மாறிலி தசம*)srow)[j] : GGML_CPU_FP16_TO_FP32(((மாறிலி ggml_fp16_t*)srow)[j]);
                        தேர்வு (op) {
                            நிலை GGML_OP_POOL_AVG: res += srow_j;                நிறுத்து;
                            நிலை GGML_OP_POOL_MAX: res =  std::max(srow_j, res); நிறுத்து;
                            நிலை GGML_OP_POOL_மொத்தம்:               GGML_ABORT("fatal error");
                        }
                    }
                }
                தேர்வு (op) {
                    நிலை GGML_OP_POOL_AVG:           res /= ka; நிறுத்து;
                    நிலை GGML_OP_POOL_MAX:                      நிறுத்து;
                    நிலை GGML_OP_POOL_மொத்தம்: GGML_ABORT("fatal error");
                }

                out[ox] = res;
            }
        }

        cdata  += src->nb[2];
        dplane += pa;
    }
}

// ggml_compute_forward_pool_2d_back

வெறுமை ggml_compute_forward_pool_2d_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src  = dst->src[0];
    மாறிலி ggml_tensor * dstf = dst->src[1]; // forward tensor of dst

    ப்ரதிஜ்ஞா(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    எனில் (params->ith != 0) {
        திரும்பு;
    }

    மாறிலி int32_t * opts = (மாறிலி int32_t *)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    மாறிலி முழு k0 = opts[1];
    மாறிலி முழு k1 = opts[2];
    மாறிலி முழு s0 = opts[3];
    மாறிலி முழு s1 = opts[4];
    மாறிலி முழு p0 = opts[5];
    மாறிலி முழு p1 = opts[6];

    எழுத்து       * cdata  = (எழுத்து       *) dst->data;
    மாறிலி எழுத்து * cdataf = (மாறிலி எழுத்து *) dstf->data;
    மாறிலி எழுத்து * மாறிலி data_end = cdata + ggml_nbytes(dst);

    GGML_ASSERT(params->ith == 0);
    memset(cdata, 0, ggml_nbytes(dst));

    மாறிலி int64_t px = src->ne[0];
    மாறிலி int64_t py = src->ne[1];
    மாறிலி int64_t pa = px * py;

    மாறிலி தசம * splane = (மாறிலி தசம *) src->data;

    மாறிலி முழு ka = k0 * k1;
    மாறிலி முழு offset0 = -p0;
    மாறிலி முழு offset1 = -p1;

    போது (cdata < data_end) {
        க்கு (முழு oy = 0; oy < py; ++oy) {
            மாறிலி தசம * மாறிலி srow = splane + oy * px;
            க்கு (முழு ox = 0; ox < px; ++ox) {
                மாறிலி தசம grad0 = srow[ox];

                மாறிலி முழு ix = offset0 + ox * s0;
                மாறிலி முழு iy = offset1 + oy * s1;

                எனில் (op == GGML_OP_POOL_MAX) {
                    தசம maxval = -FLT_MAX;
                    முழு kxmax = -1;
                    முழு kymax = -1;

                    க்கு (முழு ky = 0; ky < k1; ++ky) {
                        எனில் (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            தொடர்;
                        }
                        மாறிலி வெறுமை * drowf = (மாறிலி வெறுமை *)(cdataf + dst->nb[1] * (iy + ky));
                        க்கு (முழு kx = 0; kx < k0; ++kx) {
                            முழு j = ix + kx;
                            எனில் (j < 0 || j >= dst->ne[0]) {
                                தொடர்;
                            }

                            மாறிலி தசம val = dst->type == GGML_TYPE_F32 ?
                                ((மாறிலி தசம *) drowf)[j] : GGML_CPU_FP16_TO_FP32(((மாறிலி ggml_fp16_t *) drowf)[j]);
                            எனில் (val <= maxval) {
                                தொடர்;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    எனில் (kxmax == -1 || kymax == -1) {
                        தொடர்;
                    }

                    வெறுமை * drow = (வெறுமை *)(cdata + dst->nb[1] * (iy + kymax));
                    மாறிலி முழு j = ix + kxmax;
                    எனில் (dst->type == GGML_TYPE_F32) {
                        ((தசம *) drow)[j] += grad0;
                    } இல்லையெனில் {
                        ((ggml_fp16_t *) drow)[j] = GGML_CPU_FP32_TO_FP16(grad0 + GGML_CPU_FP16_TO_FP32(((மாறிலி ggml_fp16_t *) drow)[j]));
                    }
                } இல்லையெனில் எனில் (op == GGML_OP_POOL_AVG) {
                    மாறிலி தசம grad = grad0 / ka;

                    க்கு (முழு ky = 0; ky < k1; ++ky) {
                        எனில் (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            தொடர்;
                        }
                        வெறுமை * drow = (வெறுமை *)(cdata + dst->nb[1] * (iy + ky));
                        க்கு (முழு kx = 0; kx < k0; ++kx) {
                            முழு j = ix + kx;
                            எனில் (j < 0 || j >= dst->ne[0]) {
                                தொடர்;
                            }

                            எனில் (dst->type == GGML_TYPE_F32) {
                                ((தசம *) drow)[j] += grad;
                            } இல்லையெனில் {
                                ((ggml_fp16_t *) drow)[j] += GGML_CPU_FP32_TO_FP16(grad);
                            }
                        }
                    }
                } இல்லையெனில் {
                    GGML_ASSERT(பொய்);
                }
            }
        }

        cdata  += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}

// ggml_compute_forward_upscale

நிலையான வெறுமை ggml_compute_forward_upscale_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    தசம sf0 = (தசம)ne0/src0->ne[0];
    தசம sf1 = (தசம)ne1/src0->ne[1];
    தசம sf2 = (தசம)ne2/src0->ne[2];
    தசம sf3 = (தசம)ne3/src0->ne[3];
    தசம pixel_offset = 0.5f;

    மாறிலி int32_t mode_flags = ggml_get_op_params_i32(dst, 0);
    மாறிலி ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    எனில் (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        pixel_offset = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (தசம)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (தசம)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    எனில் (mode == GGML_SCALE_MODE_NEAREST) {
        க்கு (int64_t i3 = 0; i3 < ne3; i3++) {
            மாறிலி int64_t i03 = i3 / sf3;
            க்கு (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                மாறிலி int64_t i02 = i2 / sf2;
                க்கு (int64_t i1 = 0; i1 < ne1; i1++) {
                    மாறிலி int64_t i01 = i1 / sf1;
                    க்கு (int64_t i0 = 0; i0 < ne0; i0++) {
                        மாறிலி int64_t i00 = i0 / sf0;

                        மாறிலி தசம * x = (தசம *)((எழுத்து *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                              தசம * y = (தசம *)((எழுத்து *)  dst->data +  i0*nb0  +  i1*nb1  +  i2*nb2  +  i3*nb3);

                        *y = *x;
                    }
                }
            }
        }
    } இல்லையெனில் எனில் (mode == GGML_SCALE_MODE_BILINEAR && (mode_flags & GGML_SCALE_FLAG_ANTIALIAS)) {
        // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
        // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
        auto triangle_filter = [](தசம x) -> தசம {
            திரும்பு std::max(1.0f - fabsf(x), 0.0f);
        };

        // support and invscale, minimum 1 pixel for bilinear
        மாறிலி தசம support1  = std::max(1.0f, 1.0f / sf1);
        மாறிலி தசம invscale1 = 1.0f / support1;
        மாறிலி தசம support0  = std::max(1.0f, 1.0f / sf0);
        மாறிலி தசம invscale0 = 1.0f / support0;

        க்கு (int64_t i3 = 0; i3 < ne3; i3++) {
            மாறிலி int64_t i03 = i3 / sf3;
            க்கு (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                மாறிலி int64_t i02 = i2 / sf2;
                க்கு (int64_t i1 = 0; i1 < ne1; i1++) {
                    மாறிலி தசம y = ((தசம) i1 + pixel_offset) / sf1;
                    க்கு (int64_t i0 = 0; i0 < ne0; i0++) {
                        மாறிலி தசம x = ((தசம) i0 + pixel_offset) / sf0;

                        // the range of source pixels that contribute
                        மாறிலி int64_t x_min = std::max<int64_t>(x - support0 + pixel_offset, 0);
                        மாறிலி int64_t x_max = std::min<int64_t>(x + support0 + pixel_offset, ne00);
                        மாறிலி int64_t y_min = std::max<int64_t>(y - support1 + pixel_offset, 0);
                        மாறிலி int64_t y_max = std::min<int64_t>(y + support1 + pixel_offset, ne01);

                        // bilinear filter with antialiasing
                        தசம val = 0.0f;
                        தசம total_weight = 0.0f;

                        க்கு (int64_t sy = y_min; sy < y_max; sy++) {
                            மாறிலி தசம weight_y = triangle_filter((sy - y + pixel_offset) * invscale1);

                            க்கு (int64_t sx = x_min; sx < x_max; sx++) {
                                மாறிலி தசம weight_x = triangle_filter((sx - x + pixel_offset) * invscale0);
                                மாறிலி தசம weight = weight_x * weight_y;

                                எனில் (weight <= 0.0f) {
                                    தொடர்;
                                }

                                மாறிலி தசம pixel = *(மாறிலி தசம *)((மாறிலி எழுத்து *)src0->data + sx*nb00 + sy*nb01 + i02*nb02 + i03*nb03);
                                val += pixel * weight;
                                total_weight += weight;
                            }
                        }

                        எனில் (total_weight > 0.0f) {
                            val /= total_weight;
                        }

                        தசம * dst_ptr = (தசம *)((எழுத்து *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *dst_ptr = val;
                    }
                }
            }
        }
    } இல்லையெனில் எனில் (mode == GGML_SCALE_MODE_BILINEAR) {
        க்கு (int64_t i3 = 0; i3 < ne3; i3++) {
            மாறிலி int64_t i03 = i3 / sf3;
            க்கு (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                மாறிலி int64_t i02 = i2 / sf2;
                க்கு (int64_t i1 = 0; i1 < ne1; i1++) {
                    மாறிலி தசம y = ((தசம)i1 + pixel_offset) / sf1 - pixel_offset;
                    int64_t y0 = (int64_t)floorf(y);
                    int64_t y1 = y0 + 1;

                    y0 = std::max(int64_t(0), std::min(y0, ne01 - 1));
                    y1 = std::max(int64_t(0), std::min(y1, ne01 - 1));

                    தசம dy = y - (தசம)y0;
                    dy = std::max(0.0f, std::min(dy, 1.0f));

                    க்கு (int64_t i0 = 0; i0 < ne0; i0++) {
                        மாறிலி தசம x = ((தசம)i0 + pixel_offset) / sf0 - pixel_offset;
                        int64_t x0 = (int64_t)floorf(x);
                        int64_t x1 = x0 + 1;

                        x0 = std::max(int64_t(0), std::min(x0, ne00 - 1));
                        x1 = std::max(int64_t(0), std::min(x1, ne00 - 1));

                        தசம dx = x - (தசம)x0;
                        dx = std::max(0.0f, std::min(dx, 1.0f));

                        // fetch the four surrounding pixel values and interpolate
                        மாறிலி தசம a = *(மாறிலி தசம *)((மாறிலி எழுத்து *)src0->data + x0*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        மாறிலி தசம b = *(மாறிலி தசம *)((மாறிலி எழுத்து *)src0->data + x1*nb00 + y0*nb01 + i02*nb02 + i03*nb03);
                        மாறிலி தசம c = *(மாறிலி தசம *)((மாறிலி எழுத்து *)src0->data + x0*nb00 + y1*nb01 + i02*nb02 + i03*nb03);
                        மாறிலி தசம d = *(மாறிலி தசம *)((மாறிலி எழுத்து *)src0->data + x1*nb00 + y1*nb01 + i02*nb02 + i03*nb03);

                        மாறிலி தசம val = a*(1 - dx)*(1 - dy) + b*dx*(1 - dy) + c*(1 - dx)*dy + d*dx*dy;

                        தசம * y_dst = (தசம *)((எழுத்து *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } இல்லையெனில் எனில் (mode == GGML_SCALE_MODE_BICUBIC) {
        // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
        மாறிலி தசம a = -0.75f; // use alpha = -0.75 (same as PyTorch)
        auto weight1 = [a](தசம x) { திரும்பு ((a + 2) * x - (a + 3)) * x * x + 1; };
        auto weight2 = [a](தசம x) { திரும்பு ((a * x - 5 * a) * x + 8 * a) * x - 4 * a; };
        auto bicubic = [=](தசம p0, தசம p1, தசம p2, தசம p3, தசம x) {
            மாறிலி தசம w0 = weight2(x + 1);
            மாறிலி தசம w1 = weight1(x + 0);
            மாறிலி தசம w2 = weight1(1 - x);
            மாறிலி தசம w3 = weight2(2 - x);
            திரும்பு p0*w0 + p1*w1 + p2*w2 + p3*w3;
        };

        க்கு (int64_t i3 = 0; i3 < ne3; i3++) {
            மாறிலி int64_t i03 = i3 / sf3;
            க்கு (int64_t i2 = ith; i2 < ne2; i2 += nth) {
                மாறிலி int64_t i02 = i2 / sf2;
                க்கு (int64_t i1 = 0; i1 < ne1; i1++) {
                    மாறிலி தசம y = ((தசம)i1 + pixel_offset) / sf1 - pixel_offset;
                    மாறிலி int64_t y0 = (int64_t)floorf(y);
                    மாறிலி தசம dy = y - (தசம)y0;

                    க்கு (int64_t i0 = 0; i0 < ne0; i0++) {
                        மாறிலி தசம x = ((தசம)i0 + pixel_offset) / sf0 - pixel_offset;
                        மாறிலி int64_t x0 = (int64_t)floorf(x);
                        மாறிலி தசம dx = x - (தசம)x0;

                        auto p = [=](int64_t x_off, int64_t y_off) -> தசம {
                            int64_t i00 = std::max(int64_t(0), std::min(x0 + x_off, ne00 - 1));
                            int64_t i01 = std::max(int64_t(0), std::min(y0 + y_off, ne01 - 1));
                            திரும்பு *(மாறிலி தசம *)((மாறிலி எழுத்து *)src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);
                        };

                        மாறிலி தசம val = bicubic(
                            bicubic(p(-1,-1), p(0,-1), p(1,-1), p(2,-1), dx),
                            bicubic(p(-1, 0), p(0, 0), p(1, 0), p(2, 0), dx),
                            bicubic(p(-1, 1), p(0, 1), p(1, 1), p(2, 1), dx),
                            bicubic(p(-1, 2), p(0, 2), p(1, 2), p(2, 2), dx), dy);

                        தசம * y_dst = (தசம *)((எழுத்து *)dst->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3);
                        *y_dst = val;
                    }
                }
            }
        }
    } இல்லையெனில் {
        GGML_ABORT("unsupported upscale mode");
    }
}

வெறுமை ggml_compute_forward_upscale(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_upscale_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_pad

template<தருக்க circular_t>
நிலையான வெறுமை ggml_compute_forward_pad_f32(
    மாறிலி ggml_compute_params * params,
          ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    ப்ரதிஜ்ஞா(dst->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    தசம * dst_ptr = (தசம *) dst->data;
    மாறிலி int32_t lp0 = ggml_get_op_params_i32(dst, 0);
    மாறிலி int32_t rp0 = ggml_get_op_params_i32(dst, 1);
    மாறிலி int32_t lp1 = ggml_get_op_params_i32(dst, 2);
    மாறிலி int32_t rp1 = ggml_get_op_params_i32(dst, 3);
    மாறிலி int32_t lp2 = ggml_get_op_params_i32(dst, 4);
    மாறிலி int32_t rp2 = ggml_get_op_params_i32(dst, 5);
    மாறிலி int32_t lp3 = ggml_get_op_params_i32(dst, 6);
    மாறிலி int32_t rp3 = ggml_get_op_params_i32(dst, 7);

    மாறிலி int64_t nrows = ne1 * ne2 * ne3;
    மாறிலி தருக்க src_row_contig = src0->nb[0] == அளவு(தசம);

    க்கு (int64_t row = ith; row < nrows; row += nth) {
        மாறிலி int64_t i1 = row % ne1;
        மாறிலி int64_t i2 = (row / ne1) % ne2;
        மாறிலி int64_t i3 = row / (ne1 * ne2);

        தசம * dst_row = dst_ptr + row * ne0;

        எனில் constexpr (circular_t) {
            // circular padding wraps around on a torus
            மாறிலி int64_t src_i1 = ggml_wrap_around(i1 - lp1, ne01);
            மாறிலி int64_t src_i2 = ggml_wrap_around(i2 - lp2, ne02);
            மாறிலி int64_t src_i3 = ggml_wrap_around(i3 - lp3, ne03);

            க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                மாறிலி int64_t src_i0 = ggml_wrap_around(i0 - lp0, ne00);
                dst_row[i0] = *(மாறிலி தசம *)((மாறிலி எழுத்து *) src0->data + src_i3*nb03 + src_i2*nb02 + src_i1*nb01 + src_i0*nb00);
            }
        } இல்லையெனில் {
            மாறிலி தருக்க interior = (i1 >= lp1 && i1 < ne1 - rp1)
                               && (i2 >= lp2 && i2 < ne2 - rp2)
                               && (i3 >= lp3 && i3 < ne3 - rp3);
            எனில் (!interior) {
                memset(dst_row, 0, ne0 * அளவு(தசம));
                தொடர்;
            }

            மாறிலி எழுத்து * src_row = (மாறிலி எழுத்து *) src0->data + (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01;

            எனில் (src_row_contig && lp0 >= 0 && rp0 >= 0) {
                memset(dst_row, 0, lp0 * அளவு(தசம));
                memcpy(dst_row + lp0, src_row, ne00 * அளவு(தசம));
                memset(dst_row + lp0 + ne00, 0, rp0 * அளவு(தசம));
            } இல்லையெனில் {
                க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                    எனில் (i0 >= lp0 && i0 < ne0 - rp0) {
                        dst_row[i0] = *(மாறிலி தசம *)(src_row + (i0 - lp0)*nb00);
                    } இல்லையெனில் {
                        dst_row[i0] = 0.0f;
                    }
                }
            }
        }
    }
}


வெறுமை ggml_compute_forward_pad(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி தருக்க circular = (தருக்க) ggml_get_op_params_i32(dst, 8);
    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                எனில் (circular) {
                    ggml_compute_forward_pad_f32<உண்மை>(params, dst);
                } இல்லையெனில் {
                    ggml_compute_forward_pad_f32<பொய்>(params, dst);
                }
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_pad_reflect_1d

வெறுமை ggml_compute_forward_pad_reflect_1d(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int32_t * opts = (மாறிலி int32_t *) dst->op_params;
    மாறிலி முழு p0 = opts[0];
    மாறிலி முழு p1 = opts[1];

    GGML_TENSOR_UNARY_OP_LOCALS

    க்கு (int64_t i3 = 0; i3 < ne3; i3++) {
        க்கு (int64_t i2 = 0; i2 < ne2; i2++) {
            க்கு (int64_t i1 = ith; i1 < ne1; i1 += nth) {
                தசம * left  = (தசம *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 +         p0*nb0);
                தசம * right = (தசம *) ((எழுத்து *) dst->data + i3*nb3 + i2*nb2 + i1*nb1 + (ne0-p1-1)*nb0);

                ggml_vec_cpy_f32(ne00, left, (தசம *) ((எழுத்து *) src0->data + i3*nb03 + i2*nb02 + i1*nb01));

                க்கு (முழு i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0];   }
                க்கு (முழு i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
            }
        }
    }
}

// ggml_compute_forward_roll

நிலையான int64_t ggml_wrap_index(int64_t i, int64_t ne) {
    எனில் (i < 0) {
        திரும்பு i + ne;
    } இல்லையெனில் எனில் (i >= ne) {
        திரும்பு i - ne;
    }
    திரும்பு i;
}

நிலையான வெறுமை ggml_compute_forward_roll_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி தசம * src_data = (மாறிலி தசம *) src0->data;
    தசம * dst_data = (தசம *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி முழு s0 = ggml_get_op_params_i32(dst, 0);
    மாறிலி முழு s1 = ggml_get_op_params_i32(dst, 1);
    மாறிலி முழு s2 = ggml_get_op_params_i32(dst, 2);
    மாறிலி முழு s3 = ggml_get_op_params_i32(dst, 3);

    மாறிலி int64_t total = ne1 * ne2 * ne3;
    மாறிலி int64_t per_thread = (total + params->nth) / params->nth;
    மாறிலி int64_t start = params->ith * per_thread;
    மாறிலி int64_t end   = std::min(start + per_thread, total);

    க்கு (int64_t i = start; i < end; ++i) {
        மாறிலி int64_t i1 = i % ne1;
        மாறிலி int64_t i2 = (i / ne1) % ne2;
        மாறிலி int64_t i3 = i / (ne2 * ne1);
        தசம * dst_row = dst_data + (i3*nb3 + i2*nb2 + i1*nb1) / அளவு(தசம);

        மாறிலி int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
        மாறிலி int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
        மாறிலி int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
        மாறிலி தசம * src_row = src_data + (i03*nb03 + i02*nb02 + i01*nb01) / அளவு(தசம);

        மாறிலி int64_t s = ggml_wrap_index(-s0, ne00);
        மாறிலி int64_t n = ne00 - s;
        ggml_vec_cpy_f32(n, dst_row,     src_row + s);
        ggml_vec_cpy_f32(s, dst_row + n, src_row);
    }
}

வெறுமை ggml_compute_forward_roll(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_roll_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_arange

நிலையான வெறுமை ggml_compute_forward_arange_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    GGML_ASSERT(dst->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி தசம start = ggml_get_op_params_f32(dst, 0);
    மாறிலி தசம stop  = ggml_get_op_params_f32(dst, 1);
    மாறிலி தசம step  = ggml_get_op_params_f32(dst, 2);

    மாறிலி int64_t steps = (int64_t) ceilf((stop - start) / step);

    GGML_ASSERT(ggml_nelements(dst) == steps);

    க்கு (int64_t i = ith; i < steps; i+= nth) {
        தசம value = start + step * i;
        ((தசம *)dst->data)[i] = value;
    }
}

வெறுமை ggml_compute_forward_arange(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {
    தேர்வு (dst->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_arange_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_compute_forward_timestep_embedding_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->nb[0] == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி முழு dim = ggml_get_op_params_i32(dst, 0);
    மாறிலி முழு max_period = ggml_get_op_params_i32(dst, 1);

    முழு half = dim / 2;

    க்கு (int64_t i = 0; i < ne00; i++) {
        தசம * embed_data = (தசம *)((எழுத்து *)  dst->data +  i*nb1);
        க்கு (int64_t j = ith; j < half; j += nth) {
            தசம timestep = ((தசம *)src0->data)[i];
            தசம freq = (தசம)expf(-logf(max_period) * j / half);
            தசம arg = timestep * freq;
            embed_data[j] = cosf(arg);
            embed_data[j + half] = sinf(arg);
        }
        எனில் (dim % 2 != 0 && ith == 0) {
            embed_data[2 * half] = 0.f;
        }
    }
}

வெறுமை ggml_compute_forward_timestep_embedding(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_timestep_embedding_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_argsort

template<எண்ணிக்கை ggml_sort_order order>
கட்டமைப்பு cmp_argsort {
    மாறிலி தசம * data;
    தருக்க operator()(int32_t a, int32_t b) மாறிலி {
        எனில் constexpr (order == GGML_SORT_ORDER_ASC) {
            திரும்பு data[a] < data[b];
        } இல்லையெனில் {
            திரும்பு data[a] > data[b];
        }
    }
};

நிலையான வெறுமை ggml_compute_forward_argsort_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t nr = ggml_nrows(src0);

    ggml_sort_order order = (ggml_sort_order) ggml_get_op_params_i32(dst, 0);

    க்கு (int64_t i = ith; i < nr; i += nth) {
        மாறிலி தசம * src_data = (தசம *)((எழுத்து *) src0->data + i*nb01);

        int32_t * dst_data = (int32_t *)((எழுத்து *) dst->data + i*nb1);

        க்கு (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        தேர்வு (order) {
            நிலை GGML_SORT_ORDER_ASC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{src_data});
                நிறுத்து;

            நிலை GGML_SORT_ORDER_DESC:
                std::sort(dst_data, dst_data + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{src_data});
                நிறுத்து;

            இயல்பு:
                GGML_ABORT("invalid sort order");
        }
    }
}

வெறுமை ggml_compute_forward_argsort(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_argsort_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k

கட்டமைப்பு cmp_உச்சி_k {
    மாறிலி தசம * data;
    தருக்க operator()(int32_t a, int32_t b) மாறிலி {
        திரும்பு data[a] > data[b];
    }
};

நிலையான வெறுமை ggml_compute_forward_உச்சி_k_f32(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_UNARY_OP_LOCALS

    GGML_ASSERT(nb0 == அளவு(தசம));

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t nr = ggml_nrows(src0);

    மாறிலி முழு உச்சி_k = ne0;

    int32_t * tmp = (int32_t *) params->wdata + (ne00 + CACHE_வரி_SIZE_F32) * ith;

    க்கு (int64_t i = ith; i < nr; i += nth) {
        மாறிலி தசம * src_data = (தசம *)((எழுத்து *) src0->data + i*nb01);

        க்கு (int64_t j = 0; j < ne00; j++) {
            tmp[j] = j;
        }

        std::partial_sort(tmp, tmp + உச்சி_k, tmp + ne00, cmp_உச்சி_k{src_data});

        int32_t * dst_data = (int32_t *)((எழுத்து *) dst->data + i*nb1);

        std::copy(tmp, tmp + உச்சி_k, dst_data);

        // emphasize that the order is not important
        எனில் (உச்சி_k > 1) {
            std::swap(dst_data[0], dst_data[1]);
        }
    }
}

வெறுமை ggml_compute_forward_உச்சி_k(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_உச்சி_k_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_compute_forward_flash_attn_ext_f16_one_chunk(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst,
        முழு ir0, முழு ir1,
        int64_t ic_start, int64_t ic_end,
        தசம * partials, int64_t partial_stride) {

    மாறிலி தருக்க write_partials = (partials != nullptr);
    மாறிலி ggml_tensor * q     = dst->src[0];
    மாறிலி ggml_tensor * k     = dst->src[1];
    மாறிலி ggml_tensor * v     = dst->src[2];
    மாறிலி ggml_tensor * mask  = dst->src[3];
    மாறிலி ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    மாறிலி int64_t DK = nek0;
    மாறிலி int64_t DV = nev0;
    மாறிலி int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // broadcast factors
    மாறிலி int64_t rk2 = neq2/nek2;
    மாறிலி int64_t rk3 = neq3/nek3;

    மாறிலி int64_t rv2 = neq2/nev2;
    மாறிலி int64_t rv3 = neq3/nev3;

    // parallelize by q rows using ggml_vec_dot_f32

    தசம scale         = 1.0f;
    தசம max_bias      = 0.0f;
    தசம logit_softcap = 0.0f;

    memcpy(&scale,         (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max_bias,      (தசம *) dst->op_params + 1, அளவு(தசம));
    memcpy(&logit_softcap, (தசம *) dst->op_params + 2, அளவு(தசம));

    எனில் (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    மாறிலி uint32_t n_head      = neq2;
    மாறிலி uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    மாறிலி தசம m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    மாறிலி தசம m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    ggml_type         மாறிலி k_vec_dot_type = ggml_get_type_traits_cpu(k->type)->vec_dot_type;
    ggml_from_float_t மாறிலி q_to_vec_dot   = ggml_get_type_traits_cpu(k_vec_dot_type)->from_float;
    ggml_vec_dot_t    மாறிலி kq_vec_dot     = ggml_get_type_traits_cpu(k->type)->vec_dot;
    ggml_to_float_t   மாறிலி v_to_float     = ggml_get_type_traits(v->type)->to_float;

    GGML_ASSERT((                            q_to_vec_dot) && "fattn: unsupported K-type");
    GGML_ASSERT((v->type == GGML_TYPE_F32 || v_to_float  ) && "fattn: unsupported V-type");

    முழு ith = params->ith;

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // q indices
        மாறிலி முழு iq3 = ir/(neq2*neq1);
        மாறிலி முழு iq2 = (ir - iq3*neq2*neq1)/neq1;
        மாறிலி முழு iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        மாறிலி uint32_t h = iq2; // head index
        மாறிலி தசம slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        தசம S = 0.0f;      // sum
        தசம M = -INFINITY; // maximum KQ value

        தசம       * VKQ32 = (தசம       *) params->wdata + ith*(1*DK + 2*DV + CACHE_வரி_SIZE_F32); // FP32 VKQ accumulator
        தசம       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        எனில் (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*அளவு(ggml_fp16_t));
        } இல்லையெனில் {
            memset(VKQ32, 0, DV*அளவு(தசம));
        }

        மாறிலி ggml_fp16_t * mp = mask ? (ggml_fp16_t *)((எழுத்து *) mask->data + iq1*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]) : வெற்று;

        // k indices
        மாறிலி முழு ik3 = iq3 / rk3;
        மாறிலி முழு ik2 = iq2 / rk2;

        // v indices
        மாறிலி முழு iv3 = iq3 / rv3;
        மாறிலி முழு iv2 = iq2 / rv2;

        மாறிலி தசம * pq = (மாறிலி தசம *) ((எழுத்து *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3));
        q_to_vec_dot(pq, Q_q, DK);

        // online softmax / attention
        // loop over n_kv and n_head_kv
        // ref: https://arxiv.org/pdf/2112.05682.pdf

        க்கு (int64_t ic = ic_start; ic < ic_end; ++ic) {
            மாறிலி தசம mv = mp ? slope*GGML_CPU_FP16_TO_FP32(mp[ic]) : 0.0f;
            எனில் (mv == -INFINITY) {
                தொடர்;
            }

            தசம s; // KQ value

            மாறிலி எழுத்து * k_data = (மாறிலி எழுத்து *) k->data + ( ic*nbk1 + ik2*nbk2 + ik3*nbk3);
            kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);

            s = s*scale; // scale KQ value

            எனில் (logit_softcap != 0.0f) {
                s = logit_softcap*tanhf(s);
            }

            s += mv; // apply mask

            மாறிலி தசம Mold = M;

            தசம ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
            தசம vs = 1.0f; // post-softmax KQ value, expf(s - M)

            மாறிலி எழுத்து * v_data = ((மாறிலி எழுத்து *) v->data + (ic*nbv1 + iv2*nbv2 + iv3*nbv3));

            எனில் (v->type == GGML_TYPE_F16) {
                எனில் (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f16(DV, VKQ16, ms);
                } இல்லையெனில் {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                ggml_vec_mad_f16(DV, VKQ16, (மாறிலி ggml_fp16_t *) v_data, vs);
            } இல்லையெனில் {
                எனில் (s > M) {
                    // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
                    M = s;
                    ms = expf(Mold - M);

                    // V = V*expf(Mold - M)
                    ggml_vec_scale_f32(DV, VKQ32, ms);
                } இல்லையெனில் {
                    // no new maximum, ms == 1.0f, vs != 1.0f
                    vs = expf(s - M);
                }

                // V += v*expf(s - M)
                எனில் (v_to_float) {
                    v_to_float(v_data, V32, DV);
                    ggml_vec_mad_f32(DV, VKQ32, V32, vs);
                } இல்லையெனில் {
                    // V is F32
                    ggml_vec_mad_f32(DV, VKQ32, (மாறிலி தசம *) v_data, vs);
                }
            }

            S = S*ms + vs; // scale and increment sum with partial sum
        }

        எனில் (v->type == GGML_TYPE_F16) {
            க்கு (int64_t d = 0; d < DV; ++d) {
                VKQ32[d] = GGML_CPU_FP16_TO_FP32(VKQ16[d]);
            }
        }

        // sinks - apply only on the first kv-chunk
        எனில் (sinks && ic_start == 0) {
            மாறிலி தசம s = ((தசம *)((எழுத்து *) sinks->data))[h];

            தசம ms = 1.0f;
            தசம vs = 1.0f;

            எனில் (s > M) {
                ms = expf(M - s);
                M = s;
                ggml_vec_scale_f32(DV, VKQ32, ms);
            } இல்லையெனில் {
                vs = expf(s - M);
            }

            S = S*ms + vs;
        }

        எனில் (write_partials) {
            // Write M, S, VKQ to partials for later reduction
            // partials layout: [M, S, VKQ[DV]] per query head
            தசம * partial = partials + ir * partial_stride;
            partial[0] = M;
            partial[1] = S;
            memcpy(partial + 2, VKQ32, DV * அளவு(தசம));
        } இல்லையெனில் {
            // V /= S
            மாறிலி தசம S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            ggml_vec_scale_f32(DV, VKQ32, S_inv);

            // dst indices
            மாறிலி முழு i1 = iq1;
            மாறிலி முழு i2 = iq2;
            மாறிலி முழு i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((எழுத்து *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32, nb1);
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_flash_attn_ext_tiled(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst,
        முழு ir0, முழு ir1) {
    மாறிலி ggml_tensor * q     = dst->src[0];
    மாறிலி ggml_tensor * k     = dst->src[1];
    மாறிலி ggml_tensor * v     = dst->src[2];
    மாறிலி ggml_tensor * mask  = dst->src[3];
    மாறிலி ggml_tensor * sinks = dst->src[4];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    மாறிலி int64_t DK = nek0;
    மாறிலி int64_t DV = nev0;
    மாறிலி int64_t N  = neq1;

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
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    GGML_ASSERT(k->type == v->type);
    மாறிலி ggml_type kv_type = k->type;


    // broadcast factors
    மாறிலி int64_t rk2 = neq2/nek2;
    மாறிலி int64_t rk3 = neq3/nek3;

    மாறிலி int64_t rv2 = neq2/nev2;
    மாறிலி int64_t rv3 = neq3/nev3;

    தசம scale         = 1.0f;
    தசம max_bias      = 0.0f;
    தசம logit_softcap = 0.0f;

    memcpy(&scale,         (தசம *) dst->op_params + 0, அளவு(தசம));
    memcpy(&max_bias,      (தசம *) dst->op_params + 1, அளவு(தசம));
    memcpy(&logit_softcap, (தசம *) dst->op_params + 2, அளவு(தசம));

    எனில் (logit_softcap != 0) {
        scale /= logit_softcap;
    }

    மாறிலி uint32_t n_head      = neq2;
    மாறிலி uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    மாறிலி தசம m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    மாறிலி தசம m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    முழு ith = params->ith;

    நிலையான constexpr முழு Q_TILE_SZ  = ggml_fa_tile_config::Q;
    நிலையான constexpr முழு KV_TILE_SZ = ggml_fa_tile_config::KV;

    முழு ir = ir0;
    போது (ir < ir1) {
        // q indices for the start of this tile
        மாறிலி முழு iq3 = ir/(neq2*neq1);
        மாறிலி முழு iq2 = (ir - iq3*neq2*neq1)/neq1;
        மாறிலி முழு iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        // Number of valid rows in this tile:
        // - limited by tile size (Q_TILE_SZ)
        // - limited by chunk boundary (ir1 - ir)
        // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
        மாறிலி முழு tile_rows = MIN(Q_TILE_SZ, MIN((முழு)(ir1 - ir), (முழு)(neq1 - iq1)));
        GGML_ASSERT(tile_rows > 0);

        மாறிலி uint32_t h = iq2; // head index
        மாறிலி தசம slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1) : 1.0f;

        தசம S[Q_TILE_SZ];
        தசம M[Q_TILE_SZ];

        க்கு (முழு i = 0 ; i < Q_TILE_SZ; ++i) {
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
        தசம * base  = (தசம *) params->wdata + ith*(Q_TILE_SZ*DK + 2*Q_TILE_SZ*KV_TILE_SZ + Q_TILE_SZ*DV + KV_TILE_SZ*DV + KV_TILE_SZ*DK + CACHE_வரி_SIZE_F32);

        வெறுமை  * Q_q    = base;
        தசம * KQ     = (தசம *)((எழுத்து *)base + Q_TILE_SZ * DK * அளவு(தசம));
        தசம * mask32 = KQ + Q_TILE_SZ * KV_TILE_SZ;
        தசம * VKQ32  = mask32 + Q_TILE_SZ * KV_TILE_SZ;
        தசம * V32    = VKQ32 + Q_TILE_SZ * DV;
        தசம * K_f32  = V32 + KV_TILE_SZ * DV;

        memset(VKQ32, 0, Q_TILE_SZ * DV * அளவு(தசம));
        memset(mask32, 0, Q_TILE_SZ * KV_TILE_SZ * அளவு(தசம));

        // k indices
        மாறிலி முழு ik3 = iq3 / rk3;
        மாறிலி முழு ik2 = iq2 / rk2;

        // v indices
        மாறிலி முழு iv3 = iq3 / rv3;
        மாறிலி முழு iv2 = iq2 / rv2;

        {
            தசம * Q_f32 = (தசம *)Q_q;
            க்கு (முழு tq = 0; tq < tile_rows; tq++) {
                மாறிலி தசம * pq = (மாறிலி தசம *) ((எழுத்து *) q->data + ((iq1 + tq)*nbq1 + iq2*nbq2 + iq3*nbq3));
                memcpy(Q_f32 + tq * DK, pq, DK * அளவு(தசம));
            }
            க்கு (முழு tq = tile_rows; tq < Q_TILE_SZ; tq++) {
                memset(Q_f32 + tq * DK, 0, DK * அளவு(தசம));
            }
        }

        memset(K_f32, 0, DK * KV_TILE_SZ * அளவு(தசம));
        memset(V32,   0, KV_TILE_SZ * DV * அளவு(தசம));

        க்கு (int64_t ic = 0; ic < nek1; ic += KV_TILE_SZ) {
            மாறிலி முழு kv_tile = (முழு)std::min((int64_t)KV_TILE_SZ, nek1 - ic);

            // skip the tile entirely if all the masks are -inf
            எனில் (mask) {
                தருக்க can_skip = உண்மை;
                க்கு (முழு tq = 0; tq < tile_rows; tq++) {
                    மாறிலி ggml_fp16_t * mp_row = (மாறிலி ggml_fp16_t *)((மாறிலி எழுத்து *) mask->data + (iq1 + tq)*mask->nb[1] + (iq2%mask->ne[2])*mask->nb[2] + (iq3%mask->ne[3])*mask->nb[3]);
                    க்கு (முழு tk = 0; tk < kv_tile; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = slope * GGML_CPU_FP16_TO_FP32(mp_row[ic + tk]);
                        எனில் (mask32[tq * KV_TILE_SZ + tk] != -INFINITY) {
                            can_skip = பொய்;
                        }
                    }
                    // Pad remaining mask entries with -inf
                    க்கு (முழு tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        mask32[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }

                எனில் (can_skip) {
                    தொடர்;
                }
            }

            // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
            // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
            க்கு (முழு tk = 0; tk < kv_tile; tk++) {
                மாறிலி எழுத்து * k_data = (மாறிலி எழுத்து *)k->data + (ic + tk)*nbk1 + ik2*nbk2 + ik3*nbk3;
                எனில் (kv_type == GGML_TYPE_F16) {
                    மாறிலி ggml_fp16_t * k_f16 = (மாறிலி ggml_fp16_t *)k_data;
                    க்கு (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = GGML_CPU_FP16_TO_FP32(k_f16[dk]);
                    }
                } இல்லையெனில் {
                    மாறிலி தசம * k_f32_src = (மாறிலி தசம *)k_data;
                    க்கு (int64_t dk = 0; dk < DK; dk++) {
                        K_f32[dk * KV_TILE_SZ + tk] = k_f32_src[dk];
                    }
                }
            }
            memset(KQ, 0, Q_TILE_SZ * KV_TILE_SZ * அளவு(தசம));
            simd_gemm(KQ, (மாறிலி தசம *)Q_q, K_f32, Q_TILE_SZ, DK, KV_TILE_SZ);
            ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, scale);

            எனில் (logit_softcap != 0.0f) {
                ggml_vec_tanh_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, KQ);
                ggml_vec_scale_f32(Q_TILE_SZ * KV_TILE_SZ, KQ, logit_softcap);
            }

            // Set padded KQ entries to -inf so softmax gives them zero weight
            எனில் (kv_tile < KV_TILE_SZ) {
                க்கு (முழு tq = 0; tq < Q_TILE_SZ; tq++) {
                    க்கு (முழு tk = kv_tile; tk < KV_TILE_SZ; tk++) {
                        KQ[tq * KV_TILE_SZ + tk] = -INFINITY;
                    }
                }
            }

            எனில் (mask) {
                ggml_vec_add_f32(tile_rows * KV_TILE_SZ, KQ, KQ, mask32);
            }

            தருக்க skip[Q_TILE_SZ] = {};

            க்கு (முழு tq = 0; tq < Q_TILE_SZ; tq++) {
                தசம * kq_row = KQ + tq * KV_TILE_SZ;

                தசம tile_max;
                ggml_vec_max_f32(KV_TILE_SZ, &tile_max, kq_row);

                எனில் (tile_max == -INFINITY) {
                    skip[tq] = உண்மை;
                    தொடர்;
                }

                மாறிலி தசம Mold = M[tq];
                மாறிலி தசம Mnew = fmaxf(Mold, tile_max);

                எனில் (Mnew > Mold) {
                    மாறிலி தசம ms = expf(Mold - Mnew);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                    S[tq] *= ms;
                }
                M[tq] = Mnew;


                S[tq] += ggml_vec_soft_max_f32(KV_TILE_SZ, kq_row, kq_row, Mnew);
            }

            // V accumulation: VKQ32 += softmax(KQ) * V
            // Pack V tile to contiguous F32, zero-padded
            க்கு (முழு tk = 0; tk < kv_tile; tk++) {
                மாறிலி எழுத்து * v_data = (மாறிலி எழுத்து *)v->data + (ic + tk)*nbv1 + iv2*nbv2 + iv3*nbv3;
                எனில் (kv_type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((மாறிலி ggml_fp16_t *)v_data, V32 + tk * DV, DV);
                } இல்லையெனில் {
                    memcpy(V32 + tk * DV, v_data, DV * அளவு(தசம));
                }
            }
            க்கு (முழு tq = 0; tq < Q_TILE_SZ; tq++) {
                எனில் (skip[tq]) {
                    memset(KQ + tq * KV_TILE_SZ, 0, KV_TILE_SZ * அளவு(தசம));
                }
            }
            simd_gemm(VKQ32, KQ, V32, Q_TILE_SZ, KV_TILE_SZ, DV);
        }

        // sinks (apply only to valid rows in the tile)
        எனில் (sinks) {
            மாறிலி தசம s = ((தசம *)((எழுத்து *) sinks->data))[h];

            க்கு (முழு tq = 0; tq < tile_rows; tq++) {
                தசம ms = 1.0f;
                தசம vs = 1.0f;

                எனில் (s > M[tq]) {
                    ms = expf(M[tq] - s);
                    ggml_vec_scale_f32(DV, VKQ32 + tq * DV, ms);
                } இல்லையெனில் {
                    vs = expf(s - M[tq]);
                }

                S[tq] = S[tq] * ms + vs;
            }
        }

        க்கு (முழு tq = 0; tq < tile_rows; tq++) {
            // V /= S
            மாறிலி தசம S_inv = S[tq] == 0.0f ? 0.0f : 1.0f / S[tq];
            ggml_vec_scale_f32(DV, VKQ32 + tq * DV, S_inv);

            // dst indices
            மாறிலி முழு i1 = iq1 + tq;
            மாறிலி முழு i2 = iq2;
            மாறிலி முழு i3 = iq3;

            // permute(0, 2, 1, 3)
            memcpy((எழுத்து *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1, VKQ32 + tq * DV, nb1);
        }

        ir += tile_rows;
    }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
நிலையான வெறுமை ggml_flash_attn_ext_reduce_partials(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst,
        மாறிலி int64_t n_chunks,
        மாறிலி int64_t chunk_size) {

    மாறிலி ggml_tensor * q = dst->src[0];
    மாறிலி ggml_tensor * k = dst->src[1];
    மாறிலி ggml_tensor * v = dst->src[2];

    மாறிலி int64_t DK        = k->ne[0];
    மாறிலி int64_t DV        = v->ne[0];
    மாறிலி int64_t nek1      = k->ne[1];
    மாறிலி int64_t n_q_heads = q->ne[2];

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t wdata_per_thread = DK + 2*DV + CACHE_வரி_SIZE_F32;
    தசம *       thread_wdata     = (தசம *) params->wdata + ith * wdata_per_thread;

    மாறிலி int64_t partials_offset  = nth * (DK + 2*DV + CACHE_வரி_SIZE_F32);
    மாறிலி int64_t partial_size     = 2 + DV;
    மாறிலி தசம * partials_base    = (மாறிலி தசம *) params->wdata + partials_offset;

    // Output layout
    மாறிலி int64_t ne1 = dst->ne[1];
    மாறிலி int64_t ne2 = dst->ne[2];
    மாறிலி size_t  nb1 = dst->nb[1];

    // Each thread reduces a subset of query heads
    க்கு (int64_t q_head = ith; q_head < n_q_heads; q_head += nth) {
        தசம   M_final   = -INFINITY;
        தசம   S_final   = 0.0f;
        தசம * VKQ_final = thread_wdata;
        memset(VKQ_final, 0, DV * அளவு(தசம));

        // Combine partials from all chunks
        க்கு (int64_t chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
            மாறிலி int64_t ic_start = chunk_idx * chunk_size;
            எனில் (ic_start >= nek1) தொடர்;

            மாறிலி தசம * partial   = partials_base + (q_head * n_chunks + chunk_idx) * partial_size;
            மாறிலி தசம   M_chunk   = partial[0];
            மாறிலி தசம   S_chunk   = partial[1];
            மாறிலி தசம * VKQ_chunk = partial + 2;

            எனில் (S_chunk == 0.0f) தொடர்;

            மாறிலி தசம M_new     = fmaxf(M_final, M_chunk);
            மாறிலி தசம scale_old = expf(M_final - M_new);
            மாறிலி தசம scale_new = expf(M_chunk - M_new);

            க்கு (int64_t d = 0; d < DV; ++d) {
                VKQ_final[d] = VKQ_final[d] * scale_old + VKQ_chunk[d] * scale_new;
            }
            S_final = S_final * scale_old + S_chunk * scale_new;
            M_final = M_new;
        }

        // Normalize and write to output
        எனில் (S_final != 0.0f) {
            மாறிலி தசம S_inv = 1.0f / S_final;
            ggml_vec_scale_f32(DV, VKQ_final, S_inv);
        }
        // iq1=0, iq3=0 for decode
        memcpy((எழுத்து *) dst->data + (0*ne2*ne1 + q_head + 0*ne1)*nb1, VKQ_final, nb1);
    }
}

நிலையான வெறுமை ggml_compute_forward_flash_attn_ext_f16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * q     = dst->src[0];
    மாறிலி ggml_tensor * k     = dst->src[1];
    மாறிலி ggml_tensor * v     = dst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, q,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbq, q,   nb)
    GGML_TENSOR_LOCALS(int64_t, nek, k,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbk, k,   nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v,   ne)
    GGML_TENSOR_LOCALS(size_t,  nbv, v,   nb)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb,  dst, nb)

    மாறிலி int64_t DK = nek0;
    மாறிலி int64_t DV = nev0;
    மாறிலி int64_t N  = neq1;


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
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
    மாறிலி தருக்க use_ref = params->use_ref;

    மாறிலி தருக்க kv_is_f32_or_f16 = (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_F16);
    மாறிலி தருக்க use_split_kv_path = !use_ref && (neq1 == 1 && neq3 == 1) && kv_is_f32_or_f16 && (k->type == v->type) && q->type == GGML_TYPE_F32 && nek1 >= 512;

    எனில் (use_split_kv_path) {
        மாறிலி int64_t chunk_size = (nek1 + nth - 1) / nth;

        // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
        மாறிலி int64_t partial_size  = 2 + DV;
        தசம *       partials_base = (தசம *) params->wdata + nth * (DK + 2*DV + CACHE_வரி_SIZE_F32);

        மாறிலி int64_t ic_start = ith * chunk_size;
        மாறிலி int64_t ic_end   = std::min(ic_start + chunk_size, nek1);

        மாறிலி int64_t partial_stride = nth * partial_size;
        தசம *       chunk_partials = partials_base + ith * partial_size;

        எனில் (ic_start < nek1) {
            க்கு (int64_t q_head = 0; q_head < neq2; q_head++) {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(
                    params, dst, q_head, q_head + 1, ic_start, ic_end,
                    chunk_partials, partial_stride);
            }
        } இல்லையெனில் {
            க்கு (int64_t q_head = 0; q_head < neq2; q_head++) {
                தசம * q_partials = chunk_partials + q_head * partial_stride;
                q_partials[0] = -INFINITY;  // M
                q_partials[1] = 0.0f;       // S
            }
        }

        ggml_barrier(params->threadpool);
        ggml_flash_attn_ext_reduce_partials(params, dst, nth, chunk_size);
    } இல்லையெனில் {

        // total rows in q
        மாறிலி int64_t nr = neq1*neq2*neq3;

        // disable for NUMA
        மாறிலி தருக்க disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        முழு nth_scaled = nth * 4;
        int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        எனில் (nth == 1 || nchunk < nth || disable_chunking) {
            nchunk = nth;
        }

        எனில் (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        மாறிலி int64_t dr = (nr + nchunk - 1) / nchunk;

        நிலையான constexpr int64_t Q_TILE_SZ  = ggml_fa_tile_config::Q;
        தருக்க use_tiled = !use_ref &&
                               (q->type == GGML_TYPE_F32 &&
                                kv_is_f32_or_f16 &&
                                k->type == v->type &&
                                neq1 >= Q_TILE_SZ);
#எனில் defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#எனில் defined(__ARM_FEATURE_SVE)
        மாறிலி int64_t f32_epr = svcntw();
#இல்லையெனில்
        மாறிலி int64_t f32_epr = GGML_F32_EPR;
#endif
        use_tiled &= (DV % f32_epr == 0);
#endif
        முழு current_chunk = ith;

        போது (current_chunk < nchunk) {
            மாறிலி int64_t ir0 = dr * current_chunk;
            மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

            எனில் (use_tiled) {
                ggml_compute_forward_flash_attn_ext_tiled(params, dst, ir0, ir1);
            } இல்லையெனில் {
                ggml_compute_forward_flash_attn_ext_f16_one_chunk(params, dst, ir0, ir1, 0, nek1, nullptr, 0);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }
}

வெறுமை ggml_compute_forward_flash_attn_ext(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    தேர்வு (dst->op_params[3]) {
        நிலை GGML_PREC_DEFAULT:
        நிலை GGML_PREC_F32:
            {
                // uses F32 accumulators
                ggml_compute_forward_flash_attn_ext_f16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_flash_attn_back

நிலையான வெறுமை ggml_compute_forward_flash_attn_back_f32(
        மாறிலி ggml_compute_params * params,
        மாறிலி தருக்க masked,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * q = dst->src[0];
    மாறிலி ggml_tensor * k = dst->src[1];
    மாறிலி ggml_tensor * v = dst->src[2];
    மாறிலி ggml_tensor * d = dst->src[3];

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

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t D = neq0;
    மாறிலி int64_t N = neq1;
    மாறிலி int64_t P = nek1 - N;
    மாறிலி int64_t M = P + N;

    மாறிலி முழு Mup  = ggml_up(M, GGML_SOFT_MAX_UNROLL);
    மாறிலி முழு mxDM = MAX(D, Mup);

    // GGML_ASSERT(ne0 == D);
    // GGML_ASSERT(ne1 == N);
    GGML_ASSERT(P >= 0);

    GGML_ASSERT(nbq0 == அளவு(தசம));
    GGML_ASSERT(nbk0 == அளவு(தசம));
    GGML_ASSERT(nbv0 == அளவு(தசம));

    GGML_ASSERT(neq0 == D);
    GGML_ASSERT(nek0 == D);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned0 == D);

    GGML_ASSERT(neq1 == N);
    GGML_ASSERT(nek1 == N + P);
    GGML_ASSERT(nev1 == D);
    GGML_ASSERT(ned1 == N);

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == அளவு(தசம));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    எனில் (ith == 0) {
        memset(dst->data, 0, nb0*ne0*ne1*ne2*ne3);
    }
    ggml_barrier(params->threadpool);

    மாறிலி int64_t elem_q = ggml_nelements(q);
    மாறிலி int64_t elem_k = ggml_nelements(k);

    ggml_type result_type = dst->type;
    GGML_ASSERT(ggml_blck_size(result_type) == 1);
    மாறிலி size_t tsize = ggml_type_size(result_type);

    மாறிலி size_t offs_q = 0;
    மாறிலி size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
    மாறிலி size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

    வெறுமை * grad_q = (எழுத்து *) dst->data;
    வெறுமை * grad_k = (எழுத்து *) dst->data + offs_k;
    வெறுமை * grad_v = (எழுத்து *) dst->data + offs_v;

    மாறிலி size_t nbgq1 = nb0*neq0;
    மாறிலி size_t nbgq2 = nb0*neq0*neq1;
    மாறிலி size_t nbgq3 = nb0*neq0*neq1*neq2;

    மாறிலி size_t nbgk1 = nb0*nek0;
    மாறிலி size_t nbgk2 = nb0*nek0*nek1;
    மாறிலி size_t nbgk3 = nb0*nek0*nek1*neq2;

    மாறிலி size_t nbgv1 = nb0*nev0;
    மாறிலி size_t nbgv2 = nb0*nev0*nev1;
    மாறிலி size_t nbgv3 = nb0*nev0*nev1*neq2;

    // parallelize by k rows using ggml_vec_dot_f32

    // total rows in k
    மாறிலி முழு nr = nek2*nek3;

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    மாறிலி தசம scale = 1.0f/sqrtf(D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    // how often k2 (and v2) is repeated in q2
    முழு nrep = neq2/nek2;

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        // q indices
        மாறிலி முழு ik3 = ir/(nek2);
        மாறிலி முழு ik2 = ir - ik3*nek2;

        மாறிலி முழு iq3 = ik3;
        மாறிலி முழு id3 = ik3;
        மாறிலி முழு iv3 = ik3;
        மாறிலி முழு iv2 = ik2;

        க்கு (முழு irep = 0; irep < nrep; ++irep) {
            மாறிலி முழு iq2 = ik2 + irep*nek2;
            மாறிலி முழு id2 = iq2;

            // (ik2 + irep*nek2) % nek2 == ik2
            க்கு (முழு iq1 = 0; iq1 < neq1; ++iq1) {
                மாறிலி முழு id1 = iq1;

                // not sure about CACHE_LINE_SIZE_F32..
                // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
                தசம * S  = (தசம *) params->wdata + ith*2*(mxDM + CACHE_வரி_SIZE_F32) + 0*(mxDM+CACHE_வரி_SIZE_F32);
                தசம * SM = (தசம *) params->wdata + ith*2*(mxDM + CACHE_வரி_SIZE_F32) + 1*(mxDM+CACHE_வரி_SIZE_F32);

                க்கு (முழு i = M; i < Mup; ++i) {
                    S[i] = -INFINITY;
                }

                மாறிலி int64_t masked_begin = masked ? (P + iq1 + 1) : M;
                க்கு (int64_t ic = 0; ic < masked_begin; ++ic) {
                    // k indices
                    மாறிலி முழு ik1 = ic;

                    // S indices
                    மாறிலி முழு i1 = ik1;

                    ggml_vec_dot_f32(neq0,
                            S + i1, 0,
                            (தசம *) ((எழுத்து *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)), 0,
                            (தசம *) ((எழுத்து *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)), 0, 1);
                }

                // scale
                ggml_vec_scale_f32(masked_begin, S, scale);

                க்கு (int64_t i = masked_begin; i < M; i++) {
                    S[i] = -INFINITY;
                }

                // softmax
                // exclude known -INF S[..] values from max and loop
                // dont forget to set their SM values to zero
                {
                    தசம max = -INFINITY;
                    ggml_vec_max_f32(masked_begin, &max, S);

                    ggml_float sum = 0.0;
                    {
#ifdef GGML_SOFT_MAX_ACCELERATE
                        max = -max;
                        vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
                        vvexpf(SM, SM, &Mup);
                        ggml_vec_sum_f32(Mup, &sum, SM);
#இல்லையெனில்
                        sum = ggml_vec_soft_max_f32(Mup, SM, S, max);
#endif
                    }

                    ப்ரதிஜ்ஞா(sum > 0.0);

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
                க்கு (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            S,
                             (தசம *) ((எழுத்து *) v->data + (          ic*nbv1  + iv2*nbv2 + iv3*nbv3)),
                            *(தசம *) ((எழுத்து *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2 + id3*nbd3)));
                }

                // S = SM * (S - dot(SM, S))
                தசம dot_SM_gradSM = 0;
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
                க்கு (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (தசம *) ((எழுத்து *) grad_q  + (iq1*nbgq1 + iq2*nbgq2  + iq3*nbgq3)),
                            (தசம *) ((எழுத்து *) k->data + (ic*nbk1   + ik2*nbk2   + ik3*nbk3)),
                            S[ic]);
                }

                // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
                // for ic:
                //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
                //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
                // exclude known zero S[..] values from loop
                க்கு (int64_t ic = 0; ic < masked_begin; ++ic) {
                    ggml_vec_mad_f32(D,
                            (தசம *) ((எழுத்து *) grad_k  + (ic*nbgk1  + ik2*nbgk2  + ik3*nbgk3)),
                            (தசம *) ((எழுத்து *) q->data + (iq1*nbq1  + iq2*nbq2   + iq3*nbq3)),
                            S[ic]);
                }

                // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
                // for ic:
                //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
                //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
                // exclude known zero SM[..] values from mad
                க்கு (int64_t ic = 0; ic < D; ++ic) {
                    ggml_vec_mad_f32(masked_begin,
                            (தசம *) ((எழுத்து *) grad_v   + (          ic*nbgv1 + iv2*nbgv2 + iv3*nbgv3)),
                            SM,
                            *(தசம *) ((எழுத்து *) d->data + (ic*nbd0 + id1*nbd1 + id2*nbd2  + id3*nbd3)));
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_flash_attn_back(
        மாறிலி ggml_compute_params * params,
        மாறிலி தருக்க masked,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * q = dst->src[0];

    தேர்வு (q->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_flash_attn_back_f32(params, masked, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_conv

நிலையான வெறுமை ggml_compute_forward_ssm_conv_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0]; // conv_x
    மாறிலி ggml_tensor * src1 = dst->src[1]; // conv1d.weight

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nc  = src1->ne[0]; // d_conv
    மாறிலி முழு ncs = src0->ne[0]; // d_conv - 1 + n_t
    மாறிலி முழு nr  = src0->ne[1]; // d_inner
    மாறிலி முழு n_t =  dst->ne[1]; // tokens per sequence
    மாறிலி முழு n_s =  dst->ne[2]; // number of sequences in the batch

    GGML_ASSERT( dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(src1->nb[0] == அளவு(தசம));
    GGML_ASSERT(src0->nb[1] == src0->ne[0]*அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);
    மாறிலி முழு ir  = ir1 - ir0;

    க்கு (முழு i3 = 0; i3 < n_s; ++i3) {
        க்கு (முழு i2 = 0; i2 < n_t; ++i2) {
            // {d_conv - 1 + n_t, d_inner, n_seqs}
            // sliding window
            மாறிலி தசம * s = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2])); // {d_conv, d_inner, n_s}
            மாறிலி தசம * c = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1->data + ir0*(src1->nb[1])); // {d_conv, d_inner}
            தசம * x = (தசம *) ((எழுத்து *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2])); // {d_inner, n_t, n_s}

            // TODO: transpose the output for smaller strides for big batches?
            // d_inner
            க்கு (முழு i1 = 0; i1 < ir; ++i1) {
                // rowwise dot product
                // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
                தசம sumf = 0.0f;

                // d_conv
                க்கு (முழு i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
        }
    }
}

வெறுமை ggml_compute_forward_ssm_conv(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    தேர்வு (dst->src[0]->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_conv_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_ssm_scan

நிலையான வெறுமை ggml_compute_forward_ssm_scan_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
    மாறிலி ggml_tensor * src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
    மாறிலி ggml_tensor * src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
    மாறிலி ggml_tensor * src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
    மாறிலி ggml_tensor * src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
    மாறிலி ggml_tensor * src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
    மாறிலி ggml_tensor * src6 = dst->src[6]; // ids {n_seqs}

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t nc = src0->ne[0]; // d_state
    மாறிலி int64_t nr = src0->ne[1]; // dim
    மாறிலி int64_t nh = src1->ne[1]; // n_head
    மாறிலி int64_t ng = src4->ne[1];
    மாறிலி int64_t nt = src1->ne[2]; // number of tokens per sequence
    மாறிலி int64_t ns = src1->ne[3]; // number of sequences in the batch
    மாறிலி int64_t K  = ggml_get_op_params_i32(dst, 0);

    // can't use ggml_nbytes because src1 is not necessarily contiguous
    மாறிலி int64_t s_off = ggml_nelements(src1) * ggml_element_size(src1);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(src1) + K*nc*nr*nh*ns == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == அளவு(தசம));
    GGML_ASSERT(src1->nb[0] == அளவு(தசம));
    GGML_ASSERT(src2->nb[0] == அளவு(தசம));
    GGML_ASSERT(src3->nb[0] == அளவு(தசம));
    GGML_ASSERT(src4->nb[0] == அளவு(தசம));
    GGML_ASSERT(src5->nb[0] == அளவு(தசம));
    GGML_ASSERT(src6->nb[0] == அளவு(int32_t));
    GGML_ASSERT(nh % ng == 0);
    GGML_ASSERT(src3->ne[0] == 1 || K == 1);

    // heads per thread
    மாறிலி முழு dh = (nh + nth - 1)/nth;

    // head range for this thread
    மாறிலி முழு ih0 = dh*ith;
    மாறிலி முழு ih1 = MIN(ih0 + dh, nh);

    மாறிலி int32_t * ids = (மாறிலி int32_t *) src6->data;

    க்கு (முழு i3 = 0; i3 < ns; ++i3) {
        மாறிலி தசம * s0 = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0->data + ids[i3]*(src0->nb[3])); // {d_state, dim, nh, ns}
              தசம * s  = (      தசம *) ((      எழுத்து *) dst->data  + i3*(src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

        க்கு (முழு i2 = 0; i2 < nt; ++i2) {
            மாறிலி தசம * x  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1->data + i2*(src1->nb[2]) + i3*(src1->nb[3])); // {dim, nh, nt, ns}
            மாறிலி தசம * dt = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src2->data + i2*(src2->nb[1]) + i3*(src2->nb[2])); // {nh, nt, ns}
            மாறிலி தசம * A  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src3->data); // {d_state, nh} or {1, nh}
            மாறிலி தசம * B  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src4->data + i2*(src4->nb[2]) + i3*(src4->nb[3])); // {d_state, ng, nt, ns}
            மாறிலி தசம * C  = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src5->data + i2*(src5->nb[2]) + i3*(src5->nb[3])); // {d_state, ng, nt, ns}
                  தசம * y  = (      தசம *) ((      எழுத்து *) dst->data + i2*(nh*nr*அளவு(தசம)) + i3*(nt*nh*nr*அளவு(தசம))); // {dim, nh, nt, ns}

            எனில் (src3->ne[0] == 1) {
                // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

                // n_head
                க்கு (முழு h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    மாறிலி தசம dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    மாறிலி தசம dA = expf(dt_soft_plus * A[h]);
                    மாறிலி முழு g = h / (nh / ng); // repeat_interleave

                    // dim
                    க்கு (முழு i1 = 0; i1 < nr; ++i1) {
                        மாறிலி முழு ii = i1 + h*nr;
                        மாறிலி தசம x_dt = x[ii] * dt_soft_plus;
                        தசம sumf = 0.0f;
#எனில் defined(GGML_SIMD)
    #எனில் defined(__ARM_FEATURE_SVE)
                        மாறிலி முழு ggml_f32_epr = svcntw();
                        மாறிலி முழு ggml_f32_step = 1 * ggml_f32_epr;

                        மாறிலி முழு np = (nc & ~(ggml_f32_step - 1));

                        GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        க்கு (முழு i = 0; i < np; i += ggml_f32_step) {
                            // TODO: maybe unroll more?
                            க்கு (முழு j = 0; j < 1; j++) {
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
    #அல்லதெனில் defined(__riscv_v_intrinsic)
                        // todo: RVV implementation
                        மாறிலி முழு np = 0;
    #இல்லையெனில்
                        மாறிலி முழு np = (nc & ~(GGML_F32_STEP - 1));

                        GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

                        GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
                        GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

                        GGML_F32_VEC ax[GGML_F32_ARR];
                        GGML_F32_VEC ay[GGML_F32_ARR];
                        GGML_F32_VEC az[GGML_F32_ARR];

                        க்கு (முழு i = 0; i < np; i += GGML_F32_STEP) {
                            க்கு (முழு j = 0; j < GGML_F32_ARR; j++) {
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
#இல்லையெனில்
                        மாறிலி முழு np = 0;
#endif
                        // d_state
                        க்கு (முழு i0 = np; i0 < nc; ++i0) {
                            மாறிலி முழு i = i0 + ii*nc;
                            மாறிலி முழு ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            மாறிலி தசம state = (s0[i] * dA) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
                    }
                }
            } இல்லையெனில் {
                // Mamba-1 has an element-wise decay factor for the states

                // n_head
                க்கு (முழு h = ih0; h < ih1; ++h) {
                    // ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
                    மாறிலி தசம dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    மாறிலி முழு g = h / (nh / ng); // repeat_interleave

                    // dim
                    க்கு (முழு i1 = 0; i1 < nr; ++i1) {
                        மாறிலி முழு ii = i1 + h*nr;
                        மாறிலி தசம x_dt = x[ii] * dt_soft_plus;
#எனில் defined(__ARM_FEATURE_SVE)
                        svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
                        svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
                        svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

                        // d_state
                        // TODO: what happens when (d_state % svcntw()) != 0?
                        க்கு (int64_t k = 0; k < nc; k += svcntw()) {
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
#இல்லையெனில்
                        தசம sumf = 0.0f;
                        // NOTE: can't really use GGML_SIMD here because d_state is usually 16
                        //       and also because expf is used within the loop.
                        // d_state
                        க்கு (முழு i0 = 0; i0 < nc; ++i0) {
                            மாறிலி முழு i = i0 + ii*nc;
                            மாறிலி முழு ig = i0 + g*nc;
                            // state = prev_state * dA + dB * x
                            மாறிலி தசம state = (s0[i] * expf(dt_soft_plus * A[i0 + h*nc])) + (B[ig] * x_dt);
                            // y = rowwise_dotprod(state, C)
                            sumf += state * C[ig];
                            s[i] = state;
                        }
                        y[ii] = sumf;
#endif
                    }
                }
            }
            மாறிலி int64_t slot = nt - 1 - i2;
            எனில் (K > 1 && slot > 0 && slot < K) {
                தசம * s_snapshot = (தசம *) ((எழுத்து *) dst->data + s_off + (slot*ns + i3)*(src0->nb[3]));
                க்கு (முழு h = ih0; h < ih1; ++h) {
                    memcpy((எழுத்து *) s_snapshot + h*src0->nb[2], (எழுத்து *) s + h*src0->nb[2], src0->nb[2]);
                }
            }
            // use the output as the source when it's not the first token-wise iteration
            s0 = s;
        }
    }
}

வெறுமை ggml_compute_forward_ssm_scan(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    தேர்வு (dst->src[0]->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_ssm_scan_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_part

நிலையான வெறுமை ggml_compute_forward_win_part_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    மாறிலி int32_t nep0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t nep1 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t w    = ((மாறிலி int32_t *)(dst->op_params))[2];

    ப்ரதிஜ்ஞா(ne00 == ne0);
    ப்ரதிஜ்ஞா(ne3  == nep0*nep1);

    // TODO: optimize / multi-thread
    க்கு (முழு py = 0; py < nep1; ++py) {
        க்கு (முழு px = 0; px < nep0; ++px) {
            மாறிலி int64_t i3 = py*nep0 + px;
            க்கு (int64_t i2 = 0; i2 < ne2; ++i2) {
                க்கு (int64_t i1 = 0; i1 < ne1; ++i1) {
                    க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                        மாறிலி int64_t i02 = py*w + i2;
                        மாறிலி int64_t i01 = px*w + i1;
                        மாறிலி int64_t i00 = i0;

                        மாறிலி int64_t i = i3*ne2*ne1*ne0 + i2*ne1*ne0    + i1*ne0   + i0;
                        மாறிலி int64_t j =                  i02*ne01*ne00 + i01*ne00 + i00;

                        எனில் (py*w + i2 >= ne02 || px*w + i1 >= ne01) {
                            ((தசம *) dst->data)[i] = 0.0f;
                        } இல்லையெனில் {
                            ((தசம *) dst->data)[i] = ((தசம *) src0->data)[j];
                        }
                    }
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_win_part(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_win_part_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_win_unpart

நிலையான வெறுமை ggml_compute_forward_win_unpart_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    மாறிலி ggml_tensor * src0 = dst->src[0];

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne)

    மாறிலி int32_t w = ((மாறிலி int32_t *)(dst->op_params))[0];

    // padding
    மாறிலி முழு px = (w - ne1%w)%w;
    //const int py = (w - ne2%w)%w;

    மாறிலி முழு npx = (px + ne1)/w;
    //const int npy = (py + ne2)/w;

    ப்ரதிஜ்ஞா(ne0 == ne00);

    // TODO: optimize / multi-thread
    க்கு (int64_t i2 = 0; i2 < ne2; ++i2) {
        க்கு (int64_t i1 = 0; i1 < ne1; ++i1) {
            க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                மாறிலி முழு ip2 = i2/w;
                மாறிலி முழு ip1 = i1/w;

                மாறிலி int64_t i02 = i2%w;
                மாறிலி int64_t i01 = i1%w;
                மாறிலி int64_t i00 = i0;

                மாறிலி int64_t i = (ip2*npx + ip1)*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00 + i00;
                மாறிலி int64_t j =                                  i2*ne1*ne0    + i1*ne0   + i0;

                ((தசம *) dst->data)[j] = ((தசம *) src0->data)[i];
            }
        }
    }
}

வெறுமை ggml_compute_forward_win_unpart(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_win_unpart_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_unary

வெறுமை ggml_compute_forward_unary(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_unary_op op = ggml_get_unary_op(dst);

    தேர்வு (op) {
        நிலை GGML_UNARY_OP_ABS:
            {
                ggml_compute_forward_abs(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_SGN:
            {
                ggml_compute_forward_sgn(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_NEG:
            {
                ggml_compute_forward_neg(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_STEP:
            {
                ggml_compute_forward_step(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_TANH:
            {
                ggml_compute_forward_tanh(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_ELU:
            {
                ggml_compute_forward_elu(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_RELU:
            {
                ggml_compute_forward_relu(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_SIGMOID:
            {
                ggml_compute_forward_sigmoid(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_GELU:
            {
                ggml_compute_forward_gelu(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_GELU_ERF:
            {
                ggml_compute_forward_gelu_erf(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_GELU_QUICK:
            {
                ggml_compute_forward_gelu_quick(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_SILU:
            {
                ggml_compute_forward_silu(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_HARDSWISH:
            {
                ggml_compute_forward_hardswish(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_HARDSIGMOID:
            {
                ggml_compute_forward_hardsigmoid(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_EXP:
            {
                ggml_compute_forward_exp(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_FLOOR:
            {
                ggml_compute_forward_floor(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_CEIL:
            {
                ggml_compute_forward_ceil(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_ROUND:
            {
                ggml_compute_forward_round(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_TRUNC:
            {
                ggml_compute_forward_trunc(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_XIELU:
            {
                ggml_compute_forward_xielu(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_EXPM1:
            {
                ggml_compute_forward_expm1(params, dst);
            } நிறுத்து;
        நிலை GGML_UNARY_OP_SOFTPLUS:
            {
                ggml_compute_forward_softplus(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

//ggml_compute_forward_glu

வெறுமை ggml_compute_forward_glu(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_glu_op op = ggml_get_glu_op(dst);

    தேர்வு (op) {
        நிலை GGML_GLU_OP_REGLU:
            {
                ggml_compute_forward_reglu(params, dst);
            } நிறுத்து;
        நிலை GGML_GLU_OP_GEGLU:
            {
                ggml_compute_forward_geglu(params, dst);
            } நிறுத்து;
        நிலை GGML_GLU_OP_SWIGLU:
            {
                ggml_compute_forward_swiglu(params, dst);
            } நிறுத்து;
        நிலை GGML_GLU_OP_SWIGLU_OAI:
            {
                ggml_compute_forward_swiglu_oai(params, dst);
            } நிறுத்து;
        நிலை GGML_GLU_OP_GEGLU_ERF:
            {
                ggml_compute_forward_geglu_erf(params, dst);
            } நிறுத்து;
        நிலை GGML_GLU_OP_GEGLU_QUICK:
            {
                ggml_compute_forward_geglu_quick(params, dst);
            } நிறுத்து;
        நிலை GGML_GLU_OP_SWIGLU_CLAMP:
            {
                ggml_compute_forward_swiglu_clamp(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_get_rel_pos

நிலையான வெறுமை ggml_compute_forward_get_rel_pos_f16(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    GGML_UNUSED(params);

    மாறிலி ggml_tensor * src0 = dst->src[0];

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

    GGML_TENSOR_UNARY_OP_LOCALS

    மாறிலி int64_t w = ne1;

    ggml_fp16_t * src0_data = (ggml_fp16_t *) src0->data;
    ggml_fp16_t * dst_data  = (ggml_fp16_t *) dst->data;

    க்கு (int64_t i2 = 0; i2 < ne2; ++i2) {
        க்கு (int64_t i1 = 0; i1 < ne1; ++i1) {
            மாறிலி int64_t pos = (w - i1 - 1) + i2;
            க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                dst_data[i2*ne1*ne0 + i1*ne0 + i0] = src0_data[pos*ne00 + i0];
            }
        }
    }
}

வெறுமை ggml_compute_forward_get_rel_pos(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F16:
        நிலை GGML_TYPE_BF16:
            {
                ggml_compute_forward_get_rel_pos_f16(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_add_rel_pos

நிலையான வெறுமை ggml_compute_forward_add_rel_pos_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];
    மாறிலி ggml_tensor * src2 = dst->src[2];

    மாறிலி தருக்க inplace = (தருக்க) ((int32_t *) dst->op_params)[0];
    எனில் (!inplace) {
        எனில் (params->ith == 0) {
            memcpy((எழுத்து *) dst->data, (எழுத்து *) src0->data, ggml_nbytes(dst));
        }
        ggml_barrier(params->threadpool);
    }
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

    தசம * src1_data = (தசம *) src1->data;
    தசம * src2_data = (தசம *) src2->data;
    தசம * dst_data  = (தசம *) dst->data;

    மாறிலி int64_t ne10 = src1->ne[0];
    மாறிலி int64_t ne11 = src1->ne[1];
    மாறிலி int64_t ne12 = src1->ne[2];
    மாறிலி int64_t ne13 = src1->ne[3];

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // total patches in dst
    மாறிலி முழு np = ne13;

    // patches per thread
    மாறிலி முழு dp = (np + nth - 1)/nth;

    // patch range for this thread
    மாறிலி முழு ip0 = dp*ith;
    மாறிலி முழு ip1 = MIN(ip0 + dp, np);

    க்கு (int64_t i13 = ip0; i13 < ip1; ++i13) {
        க்கு (int64_t i12 = 0; i12 < ne12; ++i12) {
            க்கு (int64_t i11 = 0; i11 < ne11; ++i11) {
                மாறிலி int64_t jp1 = i13*ne12*ne11*ne10 + i12*ne11*ne10 + i11*ne10;
                க்கு (int64_t i10 = 0; i10 < ne10; ++i10) {
                    மாறிலி int64_t jp0  = jp1 + i10;
                    மாறிலி தசம src1_e = src1_data[jp0];
                    மாறிலி தசம src2_e = src2_data[jp0];

                    மாறிலி int64_t jdh = jp0 * ne10;
                    மாறிலி int64_t jdw = jdh - (ne10 - 1) * i10;

                    க்கு (int64_t j = 0; j < ne10; ++j) {
                        dst_data[jdh + j     ] += src2_e;
                        dst_data[jdw + j*ne10] += src1_e;
                    }
                }
            }
        }
    }
}

வெறுமை ggml_compute_forward_add_rel_pos(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_add_rel_pos_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_rwkv_wkv6

நிலையான வெறுமை ggml_compute_forward_rwkv_wkv6_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி int64_t T = dst->src[1]->ne[2];
    மாறிலி int64_t C = dst->ne[0];
    மாறிலி int64_t HEADS = dst->src[1]->ne[1];
    மாறிலி int64_t n_seqs = dst->src[5]->ne[1];
    மாறிலி int64_t head_size = C / HEADS;

    தசம * dst_data = (தசம *) dst->data;
    தசம * state = ((தசம *) dst->data) + C * T;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு h_start =  (HEADS * (ith    )) / nth;
    மாறிலி முழு h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    தசம * k =          (தசம *) dst->src[0]->data;
    தசம * v =          (தசம *) dst->src[1]->data;
    தசம * r =          (தசம *) dst->src[2]->data;
    தசம * time_faaaa = (தசம *) dst->src[3]->data;
    தசம * time_decay = (தசம *) dst->src[4]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    எனில் (ith == 0) {
        memset(dst_data, 0, T * C * அளவு(தசம));
    }
    ggml_barrier(params->threadpool);


    #எனில் defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define WKV_VECTOR_SIZE 8
    #அல்லதெனில் defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define WKV_VECTOR_SIZE 16
    #அல்லதெனில் defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define WKV_VECTOR_SIZE 8
    #அல்லதெனில் defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define WKV_VECTOR_SIZE 4
    #endif

    #ifdef WKV_VECTOR_SIZE
        முழு wkv_vector_size;
        #எனில் defined(__ARM_FEATURE_SVE)
            wkv_vector_size = svcntw();
        #இல்லையெனில்
            wkv_vector_size = WKV_VECTOR_SIZE;
        #endif
        மாறிலி int64_t vec_மொத்தம் = head_size / wkv_vector_size;

        க்கு (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            தசம * state_cur = state + state_offset;
            தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[5]->data + state_offset;

            க்கு (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                க்கு (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    தசம k_val = k[t_h_i_offset];
                    தசம r_val = r[t_h_i_offset];
                    தசம time_faaaa_val = time_faaaa[h_i_offset];
                    தசம time_decay_val = time_decay[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X r_vec = GGML_F32X_SET1(r_val);
                    GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
                    GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

                    க்கு (int64_t j = 0; j < vec_மொத்தம்; j++) {
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
                    க்கு (int64_t j = vec_மொத்தம் * wkv_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        தசம v_val = v[t_h_j_offset];
                        தசம kv_val = v_val * k_val;
                        தசம prev_state_val = state_prev[h_2d_i_j_offset];
                        தசம temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }

    #இல்லையெனில்
        // basically fused operations:
        // dst = r @ (time_faaaa * (k @ v) + state),
        // state = time_decay * state + (k @ v),
        // recursive through each token
        க்கு (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            தசம * state_cur = state + state_offset;
            தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[5]->data + state_offset;

            க்கு (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                க்கு (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_i_offset = h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    தசம k_val = k[t_h_i_offset];
                    தசம r_val = r[t_h_i_offset];
                    தசம time_faaaa_val = time_faaaa[h_i_offset];
                    // RWKV v6: different time_decay for each token.
                    தசம time_decay_val = time_decay[t_h_i_offset];

                    க்கு (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        தசம v_val = v[t_h_j_offset];
                        தசம kv_val = v_val * k_val;
                        தசம prev_state_val = state_prev[h_2d_i_j_offset];
                        தசம temp_val = kv_val * time_faaaa_val + prev_state_val;
                        dst_data[t_h_j_offset] += temp_val * r_val;
                        state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
                    }
                }
            }
        }
    #endif
}


வெறுமை ggml_compute_forward_rwkv_wkv6(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv6_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_gla

நிலையான வெறுமை ggml_compute_forward_gla_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி int64_t T = dst->src[1]->ne[2];
    மாறிலி int64_t C = dst->ne[0];
    மாறிலி int64_t HEADS = dst->src[1]->ne[1];
    மாறிலி int64_t n_seqs = dst->src[4]->ne[1];
    மாறிலி int64_t head_size = C / HEADS;
    மாறிலி தசம scale = ggml_get_op_params_f32(dst, 0);

    தசம * dst_data = (தசம *) dst->data;
    தசம * state = ((தசம *) dst->data) + C * T;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு h_start =  (HEADS * (ith    )) / nth;
    மாறிலி முழு h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    தசம * k = (தசம *) dst->src[0]->data;
    தசம * v = (தசம *) dst->src[1]->data;
    தசம * q = (தசம *) dst->src[2]->data;
    தசம * g = (தசம *) dst->src[3]->data;

    size_t t_stride = HEADS * head_size; // Same to C

    size_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    size_t h_stride_2d = head_size * head_size;

    எனில் (ith == 0) {
        memset(dst_data, 0, T * C * அளவு(தசம));
    }
    ggml_barrier(params->threadpool);


    #எனில் defined(__AVX__) && !defined(__AVX512F__)
        #define GGML_F32X GGML_F32x8
        #define GGML_F32X_SET1 GGML_F32x8_SET1
        #define GGML_F32X_LOAD GGML_F32x8_LOAD
        #define GGML_F32X_STORE GGML_F32x8_STORE
        #define GGML_F32X_MUL GGML_F32x8_MUL
        #define GGML_F32X_FMA GGML_F32x8_FMA
        #define GLA_VECTOR_SIZE 8
    #அல்லதெனில் defined(__AVX512F__)
        #define GGML_F32X GGML_F32x16
        #define GGML_F32X_SET1 GGML_F32x16_SET1
        #define GGML_F32X_LOAD GGML_F32x16_LOAD
        #define GGML_F32X_STORE GGML_F32x16_STORE
        #define GGML_F32X_MUL GGML_F32x16_MUL
        #define GGML_F32X_FMA GGML_F32x16_FMA
        #define GLA_VECTOR_SIZE 16
    #அல்லதெனில் defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
        #define GGML_F32X GGML_F32xt
        #define GGML_F32X_SET1 GGML_F32xt_SET1
        #define GGML_F32X_LOAD GGML_F32xt_LOAD
        #define GGML_F32X_STORE GGML_F32xt_STORE
        #define GGML_F32X_MUL GGML_F32xt_MUL
        #define GGML_F32X_FMA GGML_F32xt_FMA
        #define GLA_VECTOR_SIZE 8
    #அல்லதெனில் defined(__ARM_NEON) && defined(__aarch64__)
        #define GGML_F32X GGML_F32x4
        #define GGML_F32X_SET1 GGML_F32x4_SET1
        #define GGML_F32X_LOAD GGML_F32x4_LOAD
        #define GGML_F32X_STORE GGML_F32x4_STORE
        #define GGML_F32X_MUL GGML_F32x4_MUL
        #define GGML_F32X_FMA GGML_F32x4_FMA
        #define GLA_VECTOR_SIZE 4
    #endif

    #ifdef GLA_VECTOR_SIZE
        முழு gla_vector_size;
        #எனில் defined(__ARM_FEATURE_SVE)
            gla_vector_size = svcntw();
        #இல்லையெனில்
            gla_vector_size = GLA_VECTOR_SIZE;
        #endif
        மாறிலி int64_t vec_மொத்தம் = head_size / gla_vector_size;

        க்கு (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            தசம * state_cur = state + state_offset;
            தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[4]->data + state_offset;

            க்கு (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                க்கு (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    தசம k_val = k[t_h_i_offset];
                    தசம q_val = q[t_h_i_offset] * scale;
                    தசம g_val = g[t_h_i_offset];

                    // Broadcast scalar values to vectors
                    GGML_F32X k_vec = GGML_F32X_SET1(k_val);
                    GGML_F32X q_vec = GGML_F32X_SET1(q_val);
                    GGML_F32X g_vec = GGML_F32X_SET1(g_val);

                    க்கு (int64_t j = 0; j < vec_மொத்தம்; j++) {
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
                    க்கு (int64_t j = vec_மொத்தம் * gla_vector_size; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;
                        தசம v_val = v[t_h_j_offset];
                        தசம kv_val = v_val * k_val;
                        தசம prev_state_val = state_prev[h_2d_i_j_offset];
                        தசம temp_val = kv_val + prev_state_val * g_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }

    #இல்லையெனில்
        க்கு (int64_t t = 0; t < T; t++) {
            size_t t_offset = t * t_stride;
            size_t state_offset = head_size * C * (t / (T / n_seqs));
            தசம * state_cur = state + state_offset;
            தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[4]->data + state_offset;

            க்கு (int64_t h = h_start; h < h_end; h++) {
                size_t h_offset = h * h_stride;
                size_t t_h_offset = t_offset + h_offset;
                size_t h_2d_offset = h * h_stride_2d;

                க்கு (int64_t i = 0; i < head_size; i++) {
                    size_t t_h_i_offset = t_h_offset + i;
                    size_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    தசம k_val = k[t_h_i_offset];
                    தசம q_val = q[t_h_i_offset] * scale;
                    தசம g_val = g[t_h_i_offset];

                    க்கு (int64_t j = 0; j < head_size; j++) {
                        size_t t_h_j_offset = t_h_offset + j;
                        size_t h_2d_i_j_offset = h_2d_i_offset + j;

                        தசம v_val = v[t_h_j_offset];
                        தசம kv_val = v_val * k_val;
                        தசம prev_state_val = state_prev[h_2d_i_j_offset];
                        தசம temp_val = prev_state_val * g_val + kv_val;
                        dst_data[t_h_j_offset] += temp_val * q_val;
                        state_cur[h_2d_i_j_offset] = temp_val;
                    }
                }
            }
        }
    #endif
}


வெறுமை ggml_compute_forward_gla(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_gla_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_compute_forward_solve_tri_f32(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst) {
    மாறிலி கட்டமைப்பு ggml_tensor * src0 = dst->src[0];  // A (lower triangular)
    மாறிலி கட்டமைப்பு ggml_tensor * src1 = dst->src[1];  // B (RHS)

    GGML_TENSOR_இருமம்_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ne00 == ne01); // A must be square
    GGML_ASSERT(ne0  == ne10); // solution cols == B cols
    GGML_ASSERT(ne1  == ne11); // solution rows == B rows

    GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
    GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t k = ne10;   // number of RHS columns
    மாறிலி int64_t n = ne11;   // A is n×n
    மாறிலி int64_t nr = ne02 * ne03 * k; // we're parallelizing on columns here, so seq x token x column will be the unit

    // chunks per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // chunk range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    மாறிலி தசம * A = (மாறிலி தசம *) src0->data;  // [n, n, B1, B2]
    மாறிலி தசம * B = (மாறிலி தசம *) src1->data;  // [n, k, B1, B2]
          தசம * X = (      தசம *) dst->data;   // [n, k, B1, B2]

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*k);
        மாறிலி int64_t i02 = (ir - i03*ne02*k)/k;
        மாறிலி int64_t i01 = (ir - i03*ne02*k - i02*k);

        மாறிலி தசம * A_batch = A + i02 * nb02 / அளவு(தசம) + i03 * nb03 / அளவு(தசம);
        மாறிலி தசம * B_batch = B + i02 * nb12 / அளவு(தசம) + i03 * nb13 / அளவு(தசம);

        தசம * X_batch = X + i02 * nb2 / அளவு(தசம) + i03 * nb3 / அளவு(தசம);

        க்கு (int64_t i00 = 0; i00 < n; ++i00) {
            தசம sum = 0.0f;
            க்கு (int64_t t = 0; t < i00; ++t) {
                sum += A_batch[i00 * n + t] * X_batch[t * k + i01];
            }

            மாறிலி தசம diag = A_batch[i00 * n + i00];
            ப்ரதிஜ்ஞா(diag != 0.0f && "Zero diagonal in triangular matrix");

            X_batch[i00 * k + i01] = (B_batch[i00 * k + i01] - sum) / diag;
        }
    }
}

வெறுமை ggml_compute_forward_solve_tri(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_solve_tri_f32(params, dst);
    } இல்லையெனில் {
        GGML_ABORT("fatal error");
    }
}

// ggml_compute_forward_gated_delta_net
நிலையான வெறுமை ggml_compute_forward_gated_delta_net_one_chunk(
    மாறிலி ggml_compute_params * params,
    ggml_tensor * dst,
    int64_t ir0,
    int64_t ir1) {

    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    மாறிலி int64_t S_v      = src_v->ne[0];
    மாறிலி int64_t H        = src_v->ne[1];
    மாறிலி int64_t n_tokens = src_v->ne[2];
    மாறிலி int64_t n_seqs   = src_v->ne[3];

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

    மாறிலி தருக்க kda = (neg0 == S_v);

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    மாறிலி int64_t K = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(K >= 1);
    // per-seq stride in floats (seq s starts at state + s * seq_stride)
    மாறிலி int64_t state_seq_stride = src_state->nb[3] / அளவு(தசம);

    மாறிலி int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
    மாறிலி முழு ith = params->ith;

    தசம * delta       = (தசம *)params->wdata + ith * per_thread + CACHE_வரி_SIZE_F32;
    தசம * state_work  = K > 1 ? (delta + S_v) : nullptr;

    // output layout: [attn_scores | new_states]
    // attn_scores: S_v * H * n_tokens * n_seqs    floats
    // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
    மாறிலி int64_t attn_score_elems    = S_v * H * n_tokens * n_seqs;
    மாறிலி int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    தசம * attn_out_base  = (தசம *)dst->data;
    தசம * state_out_base = (தசம *)dst->data + attn_score_elems;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    மாறிலி தசம * state_in_base = (மாறிலி தசம *)src_state->data;

  //const int64_t rq1 = nev1 / neq1;
  //const int64_t rk1 = nev1 / nek1;
    மாறிலி int64_t rq3 = nev3 / neq3;
    மாறிலி int64_t rk3 = nev3 / nek3;

    மாறிலி தசம scale = 1.0f / sqrtf((தசம) S_v);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t iv1 = ir % H; // head_index
        மாறிலி int64_t iv3 = ir / H; // sequence

        மாறிலி int64_t iq1 = iv1 % neq1;
        மாறிலி int64_t ik1 = iv1 % nek1;

        மாறிலி int64_t iq3 = iv3 / rq3;
        மாறிலி int64_t ik3 = iv3 / rk3;

        // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
        // For K>1, work in scratch and copy out per-token when the slot is in range.
        தசம * s_out = (K > 1)
            ? state_work
            : state_out_base + (iv3 * H + iv1) * S_v * S_v;

        // copy input state into the working buffer and operate in-place
        // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
        மாறிலி தசம * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        memcpy(s_out, s_in, S_v * S_v * அளவு(தசம));

        // attn output pointer for first token of this (head, seq)
        தசம * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;

        க்கு (int64_t t = 0; t < n_tokens; t++) {
            மாறிலி தசம * q_d = (மாறிலி தசம *)((மாறிலி எழுத்து *)src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1);
            மாறிலி தசம * k_d = (மாறிலி தசம *)((மாறிலி எழுத்து *)src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1);
            மாறிலி தசம * v_d = (மாறிலி தசம *)((மாறிலி எழுத்து *)src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1);

            மாறிலி தசம beta_val = *(மாறிலி தசம *)((மாறிலி எழுத்து *)src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
            மாறிலி தசம * g_d    =  (மாறிலி தசம *)((மாறிலி எழுத்து *)src_g->data    + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);

            // state is stored transposed: s_out[j*S_v + i] = S[i][j]
            // so row j of s_out = column j of S (contiguous access)

            எனில் (kda) {
                // precompute exp(g) into delta scratch (reused below)
                க்கு (int64_t i = 0; i < S_v; ++i) {
                    delta[i] = expf(g_d[i]);
                }
                // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
                க்கு (int64_t j = 0; j < S_v; ++j) {
                    ggml_vec_mul_f32(S_v, &s_out[j * S_v], &s_out[j * S_v], delta);
                }
            } இல்லையெனில் {
                ggml_vec_scale_f32(S_v * S_v, s_out, expf(g_d[0]));
            }

            // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
            க்கு (int64_t j = 0; j < S_v; ++j) {
                தசம sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, k_d, 0, 1);
                delta[j] = (v_d[j] - sum) * beta_val;
            }

            // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
            க்கு (int64_t j = 0; j < S_v; ++j) {
                ggml_vec_mad_f32(S_v, &s_out[j * S_v], k_d, delta[j]);
            }

            // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
            க்கு (int64_t j = 0; j < S_v; ++j) {
                தசம sum = 0.0f;
                ggml_vec_dot_f32(S_v, &sum, 0, &s_out[j * S_v], 0, q_d, 0, 1);
                attn_data[j] = sum * scale;
            }

            attn_data += S_v * H; // advance to next token

            எனில் (K > 1) {
                மாறிலி int64_t target_slot = n_tokens - 1 - t;
                எனில் (target_slot >= 0 && target_slot < K) {
                    தசம * curr_state_o = state_out_base + target_slot * state_size_per_snap +
                                     (iv3 * H + iv1) * S_v * S_v;
                    memcpy(curr_state_o, s_out, S_v * S_v * அளவு(தசம));
                }
            }
        }
    }
}


நிலையான வெறுமை ggml_compute_forward_gated_delta_net_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    ggml_tensor * V = dst->src[2];
    int64_t nr = V->ne[1] * V->ne[3];

    // disable for NUMA
    மாறிலி தருக்க disable_chunking = ggml_is_numa();

    முழு nth = params->nth;
    முழு ith = params->ith;

    // 4x chunks per thread
    முழு nth_scaled = nth * 4;
    int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    எனில் (nth == 1 || nchunk < nth || disable_chunking) {
      nchunk = nth;
    }

    எனில் (ith == 0) {
      ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    மாறிலி int64_t dr = (nr + nchunk - 1) / nchunk;

    முழு current_chunk = ith;

    போது (current_chunk < nchunk) {
        மாறிலி int64_t ir0 = dr * current_chunk;
        மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

        ggml_compute_forward_gated_delta_net_one_chunk(params, dst, ir0, ir1);
        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

வெறுமை ggml_compute_forward_gated_delta_net(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_gated_delta_net_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}


// ggml_compute_forward_dsv4_hc_comb

நிலையான வெறுமை ggml_dsv4_hc_comb_norm_cols(தசம * comb, தசம eps) {
    constexpr int64_t hc = 4;

    க்கு (int64_t idst = 0; idst < hc; ++idst) {
        தசம sum = eps;
        க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += comb[idst + hc*isrc];
        }

        மாறிலி தசம inv_sum = 1.0f / sum;
        க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

நிலையான வெறுமை ggml_dsv4_hc_comb_norm_rows(தசம * comb, தசம eps) {
    constexpr int64_t hc = 4;

    க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
        தசம sum = eps;
        க்கு (int64_t idst = 0; idst < hc; ++idst) {
            sum += comb[idst + hc*isrc];
        }

        மாறிலி தசம inv_sum = 1.0f / sum;
        க்கு (int64_t idst = 0; idst < hc; ++idst) {
            comb[idst + hc*isrc] *= inv_sum;
        }
    }
}

நிலையான வெறுமை ggml_compute_forward_dsv4_hc_comb_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * mixes = dst->src[0];
    மாறிலி ggml_tensor * scale = dst->src[1];
    மாறிலி ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc = 4;
    constexpr int64_t comb_offset = 2*hc;
    constexpr int64_t hc_mix_dim = (2 + hc)*hc;

    மாறிலி int64_t n_tokens = mixes->ne[1];

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

    மாறிலி தசம eps = ggml_get_op_params_f32(dst, 0);
    மாறிலி int32_t n_iter = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n_iter > 0);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t dr  = (n_tokens + nth - 1) / nth;
    மாறிலி int64_t it0 = dr * ith;
    மாறிலி int64_t it1 = MIN(it0 + dr, n_tokens);

    மாறிலி தசம scale_comb = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) scale->data + 2*nbs0);

    க்கு (int64_t it = it0; it < it1; ++it) {
        தசம comb[hc*hc];

        க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
            தசம max = -INFINITY;
            க்கு (int64_t idst = 0; idst < hc; ++idst) {
                மாறிலி int64_t idx = idst + hc*isrc;
                மாறிலி தசம xv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) mixes->data + (comb_offset + idx)*nbm0 + it*nbm1);
                மாறிலி தசம bv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) base->data  + (comb_offset + idx)*nbb0);
                மாறிலி தசம v = xv * scale_comb + bv;
                comb[idx] = v;
                max = MAX(max, v);
            }

            தசம sum = 0.0f;
            க்கு (int64_t idst = 0; idst < hc; ++idst) {
                மாறிலி int64_t idx = idst + hc*isrc;
                மாறிலி தசம v = expf(comb[idx] - max);
                comb[idx] = v;
                sum += v;
            }

            மாறிலி தசம inv_sum = 1.0f / sum;
            க்கு (int64_t idst = 0; idst < hc; ++idst) {
                மாறிலி int64_t idx = idst + hc*isrc;
                comb[idx] = comb[idx] * inv_sum + eps;
            }
        }

        ggml_dsv4_hc_comb_norm_cols(comb, eps);
        க்கு (int32_t i = 1; i < n_iter; ++i) {
            ggml_dsv4_hc_comb_norm_rows(comb, eps);
            ggml_dsv4_hc_comb_norm_cols(comb, eps);
        }

        க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
            க்கு (int64_t idst = 0; idst < hc; ++idst) {
                மாறிலி int64_t idx = idst + hc*isrc;
                *(தசம *) ((எழுத்து *) dst->data + idst*nbd0 + isrc*nbd1 + it*nbd2) = comb[idx];
            }
        }
    }
}

வெறுமை ggml_compute_forward_dsv4_hc_comb(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_comb_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_pre

நிலையான வெறுமை ggml_compute_forward_dsv4_hc_pre_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * x       = dst->src[0];
    மாறிலி ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    மாறிலி int64_t n_embd   = x->ne[0];
    மாறிலி int64_t hc       = x->ne[1];
    மாறிலி int64_t n_tokens = x->ne[2];

    மாறிலி தசம scale = ggml_get_op_params_f32(dst, 0);
    மாறிலி தருக்க  gated = ggml_get_op_params_i32(dst, 1) != 0;

    GGML_ASSERT(dst->ne[0] == n_embd);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    எனில் (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
    } இல்லையெனில் {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
    }

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t nr  = n_embd * n_tokens;
    மாறிலி int64_t dr  = (nr + nth - 1) / nth;
    மாறிலி int64_t ir0 = dr * ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i0 = ir % n_embd;
        மாறிலி int64_t it = ir / n_embd;

        தசம sum = 0.0f;
        க்கு (int64_t ih = 0; ih < hc; ++ih) {
            மாறிலி தசம xv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) x->data + i0*nbx0 + ih*nbx1 + it*nbx2);
            தசம wv;
            எனில் (gated) {
                மாறிலி தசம gv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) weights->data + i0*nbw0 + ih*nbw1 + it*nbw2);
                wv = 1.0f / (1.0f + expf(-gv));
            } இல்லையெனில் {
                wv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) weights->data + ih*nbw0 + it*nbw1);
            }
            sum += xv * wv;
        }

        *(தசம *) ((எழுத்து *) dst->data + i0*nbd0 + it*nbd1) = scale * sum;
    }
}

வெறுமை ggml_compute_forward_dsv4_hc_pre(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_pre_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_dsv4_hc_post

நிலையான வெறுமை ggml_compute_forward_dsv4_hc_post_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * x        = dst->src[0];
    மாறிலி ggml_tensor * residual = dst->src[1];
    மாறிலி ggml_tensor * post     = dst->src[2];
    மாறிலி ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    மாறிலி int64_t n_embd   = x->ne[0];
    மாறிலி int64_t n_tokens = x->ne[1];
    மாறிலி int64_t hc       = residual->ne[1];

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
    எனில் (comb) {
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

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t nr  = n_embd * hc * n_tokens;
    மாறிலி int64_t dr  = (nr + nth - 1) / nth;
    மாறிலி int64_t ir0 = dr * ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i0     = ir % n_embd;
        மாறிலி int64_t idst   = (ir / n_embd) % hc;
        மாறிலி int64_t it     = ir / (n_embd * hc);

        மாறிலி தசம xv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) x->data    + i0*nbx0 + it*nbx1);
        மாறிலி தசம pv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) post->data + idst*nbp0 + it*nbp1);

        தசம sum = xv * pv;
        எனில் (comb) {
            க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
                மாறிலி தசம rv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) residual->data + i0*nbr0 + isrc*nbr1 + it*nbr2);
                மாறிலி தசம cv = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) comb->data     + idst*nbc0 + isrc*nbc1 + it*nbc2);
                sum += rv * cv;
            }
        } இல்லையெனில் {
            sum += *(மாறிலி தசம *) ((மாறிலி எழுத்து *) residual->data + i0*nbr0 + idst*nbr1 + it*nbr2);
        }

        *(தசம *) ((எழுத்து *) dst->data + i0*nbd0 + idst*nbd1 + it*nbd2) = sum;
    }
}

வெறுமை ggml_compute_forward_dsv4_hc_post(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_dsv4_hc_post_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_top_k_mask

வெறுமை ggml_compute_forward_உச்சி_k_mask(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி ggml_tensor * mask   = dst->src[0];
    மாறிலி ggml_tensor * உச்சி_k  = dst->src[1];
    மாறிலி ggml_tensor * score  = dst->src[2];
    மாறிலி ggml_tensor * blocks = dst->src[3];
    மாறிலி ggml_tensor * extra  = dst->src[4];

    GGML_ASSERT(dst->type == mask->type);
    GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

    மாறிலி int64_t n_kv    = mask->ne[0];
    மாறிலி int64_t ne1     = dst->ne[1];
    மாறிலி int64_t ne2     = dst->ne[2];
    மாறிலி int64_t n_rows  = ggml_nrows(dst);
    மாறிலி int64_t n_உச்சி   = உச்சி_k->ne[0];
    மாறிலி int64_t n_blk   = blocks ? blocks->ne[0] : 1;
    மாறிலி int64_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    மாறிலி int64_t n_extra = extra ? extra->ne[0] : 0;

    மாறிலி size_t ts = ggml_type_size(dst->type);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    க்கு (int64_t ir = ith; ir < n_rows; ir += nth) {
        மாறிலி int64_t i1 = ir % ne1;
        மாறிலி int64_t i2 = (ir / ne1) % ne2;
        மாறிலி int64_t i3 = ir / (ne1*ne2);

        மாறிலி எழுத்து * mask_row = (மாறிலி எழுத்து *) mask->data
            + (i1 % mask->ne[1])*mask->nb[1] + (i2 % mask->ne[2])*mask->nb[2] + (i3 % mask->ne[3])*mask->nb[3];
        எழுத்து * dst_row = (எழுத்து *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3];

        எனில் (dst->type == GGML_TYPE_F16) {
            மாறிலி ggml_fp16_t ninf = GGML_CPU_FP32_TO_FP16(-INFINITY);
            க்கு (int64_t c = 0; c < n_kv; ++c) {
                ((ggml_fp16_t *) dst_row)[c] = ninf;
            }
        } இல்லையெனில் {
            க்கு (int64_t c = 0; c < n_kv; ++c) {
                ((தசம *) dst_row)[c] = -INFINITY;
            }
        }

        auto keep = [&](int32_t c) {
            எனில் (c >= 0 && c < n_kv) {
                memcpy(dst_row + c*ts, mask_row + c*ts, ts);
            }
        };

        மாறிலி எழுத்து * உச்சி_k_row = (மாறிலி எழுத்து *) உச்சி_k->data + i1*உச்சி_k->nb[1] + i2*உச்சி_k->nb[2] + i3*உச்சி_k->nb[3];

        க்கு (int64_t j = 0; j < n_உச்சி; ++j) {
            மாறிலி int32_t p = *(மாறிலி int32_t *) (உச்சி_k_row + j*உச்சி_k->nb[0]);
            எனில் (p < 0 || p >= n_pick) {
                தொடர்;
            }
            எனில் (score) {
                மாறிலி தசம s = *(மாறிலி தசம *) ((மாறிலி எழுத்து *) score->data
                    + p*score->nb[0] + i1*score->nb[1] + i2*score->nb[2] + i3*score->nb[3]);
                எனில் (!(s > -INFINITY)) {
                    தொடர்;
                }
            }
            எனில் (!blocks) {
                keep(p);
                தொடர்;
            }
            க்கு (int64_t k = 0; k < n_blk; ++k) {
                keep(*(மாறிலி int32_t *) ((மாறிலி எழுத்து *) blocks->data + k*blocks->nb[0] + p*blocks->nb[1]));
            }
        }

        எனில் (extra) {
            மாறிலி எழுத்து * extra_row = (மாறிலி எழுத்து *) extra->data + i1*extra->nb[1] + i2*extra->nb[2] + i3*extra->nb[3];
            க்கு (int64_t j = 0; j < n_extra; ++j) {
                keep(*(மாறிலி int32_t *) (extra_row + j*extra->nb[0]));
            }
        }
    }
}

// ggml_compute_forward_rwkv_wkv7

நிலையான வெறுமை ggml_compute_forward_rwkv_wkv7_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {
    மாறிலி int64_t T = dst->src[1]->ne[2];
    மாறிலி int64_t C = dst->ne[0];
    மாறிலி int64_t HEADS = dst->src[1]->ne[1];
    மாறிலி int64_t n_seqs = dst->src[6]->ne[1];
    மாறிலி int64_t head_size = C / HEADS;

    தசம * dst_data = (தசம *) dst->data;
    தசம * state = ((தசம *) dst->data) + C * T;

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு h_start =  (HEADS * (ith    )) / nth;
    மாறிலி முழு h_end   = ((HEADS * (ith + 1)) / nth < HEADS) ?
                         (HEADS * (ith + 1)) / nth : HEADS;

    தசம * r = (தசம *) dst->src[0]->data;
    தசம * w = (தசம *) dst->src[1]->data;
    தசம * k = (தசம *) dst->src[2]->data;
    தசம * v = (தசம *) dst->src[3]->data;
    தசம * a = (தசம *) dst->src[4]->data;
    தசம * b = (தசம *) dst->src[5]->data;

    int64_t t_stride = HEADS * head_size; // Same to C

    int64_t h_stride = C / HEADS;
    GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
    int64_t h_stride_2d = head_size * head_size;

    #எனில் defined(GGML_SIMD)
        #எனில் defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
            // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
            க்கு (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                தசம * state_cur = state + state_offset;
                தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[6]->data + state_offset;

                க்கு (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    க்கு (int64_t i = 0; i < head_size; i++) {
                        int64_t t_h_i_offset = t_h_offset + i;
                        int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                        தசம v_val = v[t_h_i_offset];

                        தசம sa = 0, result = 0;
                        க்கு (int64_t j = 0; j < head_size; j++) {
                            sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                        }

                        க்கு (int64_t j = 0; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            தசம r_val = r[t_h_j_offset];
                            தசம w_val = w[t_h_j_offset];
                            தசம k_val = k[t_h_j_offset];
                            தசம b_val = b[t_h_j_offset];
                            தசம kv_val = v_val * k_val;
                            தசம prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            result += state_cur[h_2d_i_j_offset] * r_val;
                        }
                        dst_data[t_h_i_offset] = result;
                    }
                }
            }
        #இல்லையெனில்
            க்கு (int64_t t = 0; t < T; t++) {
                int64_t t_offset = t * t_stride;
                int64_t state_offset = head_size * C * (t / (T / n_seqs));
                தசம * state_cur = state + state_offset;
                தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[6]->data + state_offset;

                க்கு (int64_t h = h_start; h < h_end; h++) {
                    int64_t h_offset = h * h_stride;
                    int64_t t_h_offset = t_offset + h_offset;
                    int64_t h_2d_offset = h * h_stride_2d;

                    க்கு (int64_t ii = 0; ii < head_size; ii++) {
                        int64_t t_h_i_offset = t_h_offset + ii;
                        int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

                        GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

                        தசம sa = 0;
                        {
                            GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
                            GGML_F32_VEC ax[GGML_F32_ARR];
                            GGML_F32_VEC ay[GGML_F32_ARR];
                            க்கு (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
                                க்கு (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        க்கு (; j < head_size; j += GGML_F32_STEP) {
                            க்கு (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
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
                        க்கு (; j < head_size; j++) {
                            int64_t t_h_j_offset = t_h_offset + j;
                            int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                            தசம r_val = r[t_h_j_offset];
                            தசம w_val = w[t_h_j_offset];
                            தசம k_val = k[t_h_j_offset];
                            தசம b_val = b[t_h_j_offset];
                            தசம kv_val = v[t_h_i_offset] * k_val;

                            தசம prev_state_val = state_prev[h_2d_i_j_offset];
                            state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                            dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
                        }
                    }
                }
            }
        #endif
    #இல்லையெனில்
        க்கு (int64_t t = 0; t < T; t++) {
            int64_t t_offset = t * t_stride;
            int64_t state_offset = head_size * C * (t / (T / n_seqs));
            தசம * state_cur = state + state_offset;
            தசம * state_prev = t % (T / n_seqs) ? state_cur : (தசம*)dst->src[6]->data + state_offset;

            க்கு (int64_t h = h_start; h < h_end; h++) {
                int64_t h_offset = h * h_stride;
                int64_t t_h_offset = t_offset + h_offset;
                int64_t h_2d_offset = h * h_stride_2d;

                க்கு (int64_t i = 0; i < head_size; i++) {
                    int64_t t_h_i_offset = t_h_offset + i;
                    int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

                    தசம v_val = v[t_h_i_offset];

                    தசம sa = 0, result = 0;
                    க்கு (int64_t j = 0; j < head_size; j++) {
                        sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
                    }

                    க்கு (int64_t j = 0; j < head_size; j++) {
                        int64_t t_h_j_offset = t_h_offset + j;
                        int64_t h_2d_i_j_offset = h_2d_i_offset + j;

                        தசம r_val = r[t_h_j_offset];
                        தசம w_val = w[t_h_j_offset];
                        தசம k_val = k[t_h_j_offset];
                        தசம b_val = b[t_h_j_offset];
                        தசம kv_val = v_val * k_val;
                        தசம prev_state_val = state_prev[h_2d_i_j_offset];
                        state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
                        result += state_cur[h_2d_i_j_offset] * r_val;
                    }
                    dst_data[t_h_i_offset] = result;
                }
            }
        }
    #endif
}


வெறுமை ggml_compute_forward_rwkv_wkv7(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_rwkv_wkv7_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_map_custom1

வெறுமை ggml_compute_forward_map_custom1(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * a = dst->src[0];

    கட்டமைப்பு ggml_map_custom1_op_params p;
    memcpy(&p, dst->op_params, அளவு(p));

    p.fun(dst, a, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom2

வெறுமை ggml_compute_forward_map_custom2(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * a = dst->src[0];
    மாறிலி ggml_tensor * b = dst->src[1];

    கட்டமைப்பு ggml_map_custom2_op_params p;
    memcpy(&p, dst->op_params, அளவு(p));

    p.fun(dst, a, b, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_map_custom3

வெறுமை ggml_compute_forward_map_custom3(
        மாறிலி ggml_compute_params * params,
              ggml_tensor * dst) {

    மாறிலி ggml_tensor * a = dst->src[0];
    மாறிலி ggml_tensor * b = dst->src[1];
    மாறிலி ggml_tensor * c = dst->src[2];

    கட்டமைப்பு ggml_map_custom3_op_params p;
    memcpy(&p, dst->op_params, அளவு(p));

    p.fun(dst, a, b, c, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_custom

வெறுமை ggml_compute_forward_custom(
    மாறிலி கட்டமைப்பு ggml_compute_params * params,
          கட்டமைப்பு ggml_tensor * dst) {

    கட்டமைப்பு ggml_custom_op_params p;
    memcpy(&p, dst->op_params, அளவு(p));

    p.fun(dst, params->ith, params->nth, p.userdata);
}

// ggml_compute_forward_cross_entropy_loss

நிலையான வெறுமை ggml_compute_forward_cross_entropy_loss_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_scalar(dst));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // TODO: handle transposed/permuted matrices
    மாறிலி int64_t nc = src0->ne[0];
    மாறிலி int64_t nr = ggml_nrows(src0);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    தசம * sums =  (தசம *) params->wdata;
    தசம * st   = ((தசம *) params->wdata) + nth + ith*nc;
    தசம sum_thread = 0.0f;

    GGML_ASSERT(params->wsize >= அளவு(தசம) * (nth + nth * nc));

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    க்கு (int64_t i1 = ir0; i1 < ir1; ++i1) {
        மாறிலி தசம * s0 = (மாறிலி தசம *)((மாறிலி எழுத்து *) src0->data + i1*src0->nb[1]);
        மாறிலி தசம * s1 = (மாறிலி தசம *)((மாறிலி எழுத்து *) src1->data + i1*src1->nb[1]);

#ifndef NDEBUG
        க்கு (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ப்ரதிஜ்ஞா(!isnan(s0[i]));
            ப்ரதிஜ்ஞா(!isnan(s1[i]));
        }
#endif // NDEBUG

        தசம max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        மாறிலி ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, st, s0, max);
        ப்ரதிஜ்ஞா(sum_softmax >= 0.0);

        ggml_vec_add1_f32(nc, st, st, -sum_softmax);
        ggml_vec_mul_f32(nc, st, st, s1);

        தசம sum_st = 0.0f;
        ggml_vec_sum_f32(nc, &sum_st, st);
        sum_thread += sum_st;

#ifndef NDEBUG
        க்கு (int64_t i = 0; i < nc; ++i) {
            ப்ரதிஜ்ஞா(!isnan(st[i]));
            ப்ரதிஜ்ஞா(!isinf(st[i]));
        }
#endif // NDEBUG
    }
    sums[ith] = sum_thread;
    ggml_barrier(params->threadpool);

    எனில் (ith == 0) {
        தசம * dp = (தசம *) dst->data;
        ggml_vec_sum_f32(nth, dp, sums);
        dp[0] *= -1.0f / (தசம) nr;
    }
}

வெறுமை ggml_compute_forward_cross_entropy_loss(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// ggml_compute_forward_cross_entropy_loss_back

நிலையான வெறுமை ggml_compute_forward_cross_entropy_loss_back_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * grad  = dst->src[0]; // gradient of forward pass output
    மாறிலி ggml_tensor * src0f = dst->src[1]; // src0 of forward pass
    மாறிலி ggml_tensor * src1f = dst->src[2]; // src1 of forward pass

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(grad));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

    மாறிலி int64_t ith = params->ith;
    மாறிலி int64_t nth = params->nth;

    // TODO: handle transposed/permuted matrices
    மாறிலி int64_t nc = src0f->ne[0];
    மாறிலி int64_t nr = ggml_nrows(src0f);

    // rows per thread
    மாறிலி int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி int64_t ir0 = dr*ith;
    மாறிலி int64_t ir1 = MIN(ir0 + dr, nr);

    மாறிலி தசம d_by_nr = ((மாறிலி தசம *) grad->data)[0] / (தசம) nr;

    க்கு (int64_t i1 = ir0; i1 < ir1; i1++) {
        தசம       * ds0 = (தசம       *)((எழுத்து       *) dst->data   + i1*dst->nb[1]);
        மாறிலி தசம * s0  = (மாறிலி தசம *)((மாறிலி எழுத்து *) src0f->data + i1*src0f->nb[1]);
        மாறிலி தசம * s1  = (மாறிலி தசம *)((மாறிலி எழுத்து *) src1f->data + i1*src1f->nb[1]);

#ifndef NDEBUG
        க்கு (int64_t i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            ப்ரதிஜ்ஞா(!isnan(s0[i]));
            ப்ரதிஜ்ஞா(!isnan(s1[i]));
        }
#endif // NDEBUG

        // soft_max
        தசம max = -INFINITY;
        ggml_vec_max_f32(nc, &max, s0);
        மாறிலி ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
        ப்ரதிஜ்ஞா(sum > 0.0);
        ggml_vec_scale_f32(nc, ds0, 1.0/sum);

        // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
        ggml_vec_sub_f32(nc, ds0, ds0, s1);
        ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
        க்கு (int64_t i = 0; i < nc; ++i) {
            ப்ரதிஜ்ஞா(!isnan(ds0[i]));
            ப்ரதிஜ்ஞா(!isinf(ds0[i]));
        }
#endif // NDEBUG
    }
}

வெறுமை ggml_compute_forward_cross_entropy_loss_back(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_cross_entropy_loss_back_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_compute_forward_opt_step_adamw_f32(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0         = dst->src[0];
    மாறிலி ggml_tensor * src0_grad    = dst->src[1];
    மாறிலி ggml_tensor * src0_grad_m  = dst->src[2];
    மாறிலி ggml_tensor * src0_grad_v  = dst->src[3];
    மாறிலி ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr  = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    மாறிலி தசம * adamw_params_ptr = ggml_get_data_f32(adamw_params);

    மாறிலி தசம alpha  = adamw_params_ptr[0];
    மாறிலி தசம beta1  = adamw_params_ptr[1];
    மாறிலி தசம beta2  = adamw_params_ptr[2];
    மாறிலி தசம eps    = adamw_params_ptr[3];
    மாறிலி தசம wd     = adamw_params_ptr[4];
    மாறிலி தசம beta1h = adamw_params_ptr[5];
    மாறிலி தசம beta2h = adamw_params_ptr[6];
    மாறிலி தசம keep   = 1.f - alpha * wd;
    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir/(ne02*ne01);
        மாறிலி int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        மாறிலி int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        மாறிலி size_t offset = i03*nb03 + i02*nb02 + i01*nb01;

        தசம       * w = (தசம       *) ((எழுத்து       *) src0->data        + offset); // weight
        மாறிலி தசம * g = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_grad->data   + offset); // grad
        தசம       * m = (தசம       *) ((எழுத்து       *) src0_grad_m->data + offset);
        தசம       * v = (தசம       *) ((எழுத்து       *) src0_grad_v->data + offset);

        க்கு (முழு i00 = 0; i00 < ne00; ++i00) {
            m[i00] = m[i00]*beta1 +        g[i00]*(1.0f - beta1);
            v[i00] = v[i00]*beta2 + g[i00]*g[i00]*(1.0f - beta2);

            மாறிலி தசம mh =       m[i00]*beta1h;
            மாறிலி தசம vh = sqrtf(v[i00]*beta2h) + eps;

            // The weight decay is applied independently of the Adam momenta m and v.
            // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
            // See: https://arxiv.org/pdf/1711.05101v3.pdf
            w[i00] = w[i00] * keep - alpha * mh / vh;
        }
    }
}

வெறுமை ggml_compute_forward_opt_step_adamw(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_adamw_f32(params, dst);
            } நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error");
            }
    }
}

நிலையான வெறுமை ggml_compute_forward_opt_step_sgd_f32(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0       = dst->src[0];
    மாறிலி ggml_tensor * src0_grad  = dst->src[1];
    மாறிலி ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி முழு nr = ggml_nrows(src0);

    GGML_TENSOR_UNARY_OP_LOCALS
    GGML_ASSERT(nb00 == அளவு(தசம));

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1) / nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr * ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    // using adamw param subset we care about - alpha, wd - could have a separate struct
    மாறிலி தசம * sgd_params_ptr   = ggml_get_data_f32(sgd_params);
    மாறிலி தசம   alpha            = sgd_params_ptr[0];
    மாறிலி தசம   keep             = 1.f - alpha * sgd_params_ptr[1];

    க்கு (முழு ir = ir0; ir < ir1; ++ir) {
        மாறிலி int64_t i03 = ir / (ne02 * ne01);
        மாறிலி int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        மாறிலி int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

        மாறிலி size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

        தசம *       w = (தசம *) ((எழுத்து *) src0->data + offset);                   // weight
        மாறிலி தசம * g = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_grad->data + offset);  // grad

        க்கு (முழு i00 = 0; i00 < ne00; ++i00) {
            w[i00] = w[i00] * keep - alpha * g[i00];
        }
    }
}

வெறுமை ggml_compute_forward_opt_step_sgd(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];

    தேர்வு (src0->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_opt_step_sgd_f32(params, dst);
            }
            நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error - sgd is F32 only");
            }
    }
}

நிலையான உள்வரி தசம ggml_fwht_load(மாறிலி தசம value) {
    திரும்பு value;
}

நிலையான உள்வரி தசம ggml_fwht_load(மாறிலி ggml_fp16_t value) {
    திரும்பு ggml_fp16_to_fp32(value);
}

template<typename src_t>
நிலையான வெறுமை ggml_compute_forward_fwht_impl(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, தசம> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_இருமம்_OP_LOCALS

    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    மாறிலி int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    மாறிலி int64_t nr = ne11 * ne12 * ne13;
    மாறிலி int64_t rows_per_thread = (nr + nth - 1) / nth;
    மாறிலி int64_t start_row = ith * rows_per_thread;
    மாறிலி int64_t end_row = MIN(start_row + rows_per_thread, nr);

    மாறிலி தசம scale = 1.0f / sqrtf((தசம)n);

#எனில் defined(GGML_SIMD)
    மாறிலி GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    க்கு (int64_t r = start_row; r < end_row; r++) {
        மாறிலி int64_t i13 = r / (ne11 * ne12);
        மாறிலி int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        மாறிலி int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        மாறிலி src_t * src_row = (மாறிலி src_t *) ((மாறிலி எழுத்து *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        தசம * dst_row = (தசம *) ((எழுத்து *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        க்கு (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#எனில் defined(GGML_SIMD)
#எனில் defined(__ARM_FEATURE_SVE)
        மாறிலி முழு step = svcntw();
#இல்லையெனில்
        மாறிலி முழு step = GGML_F32_EPR;
#endif
#இல்லையெனில்
        மாறிலி முழு step = n;
#endif
        க்கு (int64_t len = 1; len < step && len < n; len <<= 1) {
            க்கு (int64_t i = 0; i < n; i += 2 * len) {
                க்கு (int64_t j = 0; j < len; j++) {
                    தசம u = dst_row[i + j];
                    தசம v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#எனில் defined(GGML_SIMD)
        க்கு (int64_t len = step; len < n; len <<= 1) {
            க்கு (int64_t i = 0; i < n; i += 2 * len) {
                க்கு (int64_t j = 0; j < len; j += step) {
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

வெறுமை ggml_compute_forward_fwht(மாறிலி ggml_compute_params * params, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src1 = dst->src[1];

    தேர்வு (src1->type) {
        நிலை GGML_TYPE_F32:
            {
                ggml_compute_forward_fwht_impl<தசம>(params, dst);
            }
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                ggml_compute_forward_fwht_impl<ggml_fp16_t>(params, dst);
            }
            நிறுத்து;
        இயல்பு:
            {
                GGML_ABORT("fatal error - fwht supports F32 and F16 input");
            }
    }
}

// ggml_compute_forward_lightning_indexer

வெறுமை ggml_compute_forward_lightning_indexer(
        மாறிலி ggml_compute_params * params,
        ggml_tensor * dst) {

    மாறிலி ggml_tensor * q = dst->src[0];
    மாறிலி ggml_tensor * k = dst->src[1];
    மாறிலி ggml_tensor * w = dst->src[2]; // weights
    மாறிலி ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(   q->type == GGML_TYPE_F32);
    GGML_ASSERT(   w->type == GGML_TYPE_F32);
    GGML_ASSERT(   m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, புதிய,  w, ne)
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

    மாறிலி முழு n_embd    = q->ne[0];
    மாறிலி முழு n_head    = q->ne[1];
    மாறிலி முழு n_tokens  = q->ne[2];
    மாறிலி முழு n_stream  = q->ne[3];
    மாறிலி முழு n_kv      = k->ne[2];

    ggml_to_float_t மாறிலி k_to_float = ggml_get_type_traits(k->type)->to_float;
    GGML_ASSERT((k->type == GGML_TYPE_F32 || k_to_float) && "lightning indexer: unsupported K-type");

    மாறிலி முழு nr  = n_kv;
    மாறிலி முழு ith = params->ith;
    மாறிலி முழு nth = params->nth;

    // (temporary) buffer for K converted to float
    தசம * k_row_f32 = (தசம *) params->wdata + ith*(1*n_embd + CACHE_வரி_SIZE_F32);

    // rows per thread
    மாறிலி முழு dr = (nr + nth - 1)/nth;

    // row range for this thread
    மாறிலி முழு ir0 = dr*ith;
    மாறிலி முழு ir1 = MIN(ir0 + dr, nr);

    க்கு (முழு s = 0; s < n_stream; ++s) {
        க்கு (முழு t = 0; t < n_tokens; ++t) {
            மாறிலி தசம       *   w_row =       (தசம *) ((எழுத்து *)   w->data + t*nbw1 +        s*nbw3);
            மாறிலி ggml_fp16_t *   m_row = (ggml_fp16_t *) ((எழுத்து *)   m->data + t*nbm1 + (s%nem3)*nbm3);
            தசம             * dst_row =       (தசம *) ((எழுத்து *) dst->data + t*nb1  +        s*nb3 );
            க்கு (முழு ik = ir0; ik < ir1; ++ik) {
                எழுத்து * k_row = (எழுத்து *) k->data + ik*nbk2 + s*nbk3;
                எனில் (k_to_float) {
                    k_to_float(k_row, k_row_f32, n_embd);
                } இல்லையெனில் {
                    k_row_f32 = (தசம *) k_row;
                }
                தசம score = 0.0f;
                க்கு (முழு h = 0; h < n_head; ++h) {
                    // dot product of q and k for head h
                    தசம qk = 0.0f;
                    மாறிலி தசம * q_row = (தசம *) ((எழுத்து *) q->data + h*nbq1 + t*nbq2 + s*nbq3);
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
